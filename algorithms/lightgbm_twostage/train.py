"""LightGBM-TwoStage — 单次训练-预测（与 lightgbm_baseline/train.py 口径对齐）。

< test_start 的样本用于训练，末尾 val_days 天作 val（早停 + 后处理调参），
[test_start, test_end] 一次性预测。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MarketConfig
from .model import (
    QUANTILES,
    apply_adaptive_naive_blend,
    apply_residual_correction,
    compute_dynamic_floor_value,
    compute_time_decay_weights,
    evaluate,
    find_best_threshold,
    fit_floor_classifier,
    fit_lgbm,
    fit_quantile_lgbm,
    quantile_combine,
    select_top_k_features,
    tune_adaptive_naive_blend,
    tune_quantile_params,
    tune_residual_gamma,
    two_stage_predict,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """单次训练-预测的可配参数。"""
    val_days: int = 7
    params: Optional[Dict] = None
    feature_select_top_k: int = 0
    two_stage: bool = False
    quantile_mode: bool = False
    quantile_two_stage: bool = False
    time_decay_half_life: int = 0
    adaptive_naive_blend: bool = True
    dynamic_floor_value: bool = True
    residual_correction: bool = True


def single_pass_predict(
    feat: pd.DataFrame, cfg: MarketConfig, tc: TrainConfig,
    test_start: str, test_end: str,
) -> Tuple[Dict[str, Any], Optional[Dict], Optional[pd.DataFrame]]:
    """单次训练 + 单次预测（与其他算法的口径一致）。

    Args:
        feat: 由 build_features 产出，含 trade_date / hour / step / y / 特征列
        cfg: MarketConfig
        tc:  TrainConfig
        test_start / test_end: 由 market yaml 的 splits 定义

    Returns:
        info:        汇总 dict（含 mae/rmse/mape/profile_corr/后处理参数等）
        last_payload: 模型 + 特征列（供 run.py 保存 feature_importance）
        all_preds:   pd.DataFrame[trade_date, hour, step, y, pred, (q10..q90)]
    """
    params = {**(tc.params or {})}
    if "step" not in feat.columns:
        feat = feat.copy()
        feat["step"] = feat["hour"].astype(int)

    dates = sorted(feat["trade_date"].unique())
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)
    dates = [d for d in dates if pd.Timestamp(d) <= test_end_ts]
    if not dates:
        raise ValueError(f"feat 中无 ≤ test_end ({test_end}) 的样本")

    test_dates = [d for d in dates if test_start_ts <= pd.Timestamp(d) <= test_end_ts]
    pre_test_dates = [d for d in dates if pd.Timestamp(d) < test_start_ts]
    if not test_dates:
        raise ValueError(f"test_start={test_start} 之后无样本")
    if len(pre_test_dates) <= tc.val_days:
        raise ValueError(
            f"训练集过小：< test_start 仅 {len(pre_test_dates)} 天，"
            f"不足以切出 val_days={tc.val_days} 天"
        )

    val_dates = pre_test_dates[-tc.val_days:]
    train_dates = pre_test_dates[:-tc.val_days]
    logger.info(
        "single-pass split: train=%d 天 (%s ~ %s), val=%d 天 (%s ~ %s), "
        "test=%d 天 (%s ~ %s)",
        len(train_dates), train_dates[0], train_dates[-1],
        len(val_dates), val_dates[0], val_dates[-1],
        len(test_dates), test_dates[0], test_dates[-1],
    )

    x_cols = [c for c in feat.columns if c not in ("trade_date", "y", "step")]
    if tc.feature_select_top_k > 0 and tc.feature_select_top_k < len(x_cols):
        # 用 train+val 之外的尾部做 holdout，与原 CV 设计保持一致
        x_cols = select_top_k_features(
            feat, x_cols, min_train_days=len(train_dates),
            val_days=tc.val_days, top_k=tc.feature_select_top_k, params=params,
        )

    train_mask = feat["trade_date"].isin(train_dates)
    val_mask = feat["trade_date"].isin(val_dates)
    test_mask = feat["trade_date"].isin(test_dates)

    X_train = feat.loc[train_mask, x_cols]
    y_train = feat.loc[train_mask, "y"].values
    X_val = feat.loc[val_mask, x_cols]
    y_val = feat.loc[val_mask, "y"].values
    X_test = feat.loc[test_mask, x_cols]
    y_test = feat.loc[test_mask, "y"].values

    sw = None
    if tc.time_decay_half_life > 0:
        sw = compute_time_decay_weights(
            feat.loc[train_mask, "trade_date"], tc.time_decay_half_life,
        )

    floor_value = (
        compute_dynamic_floor_value(y_train, cfg.floor_price)
        if tc.dynamic_floor_value else cfg.floor_pred_value
    )

    naive_test_arr = _get_naive(feat, dates, test_dates)
    naive_val_arr = _get_naive(feat, dates, val_dates)

    clf = None
    threshold = 1.1
    q_models = None
    q_test_preds: Optional[Dict[float, np.ndarray]] = None
    qc_params: Dict = {}

    if tc.quantile_mode or tc.quantile_two_stage:
        q_models = {}
        for alpha in QUANTILES:
            q_models[alpha] = fit_quantile_lgbm(
                X_train, y_train, X_val, y_val, alpha, params,
                sample_weight=sw,
            )
        model = q_models[0.50]
        q_val_preds = {a: m.predict(X_val) for a, m in q_models.items()}
        qc_params = tune_quantile_params(q_val_preds, naive_val_arr, y_val)
        q_test_preds = {a: m.predict(X_test) for a, m in q_models.items()}
        pred_test = quantile_combine(q_test_preds, naive_test_arr, **qc_params)

        if tc.quantile_two_stage:
            clf = fit_floor_classifier(
                X_train, y_train, X_val, y_val, cfg.floor_price,
                params, sample_weight=sw,
            )
            if clf is not None:
                val_qc_pred = quantile_combine(q_val_preds, naive_val_arr, **qc_params)
                val_floor_prob = clf.predict_proba(X_val)[:, 1]
                threshold, _ = find_best_threshold(
                    val_floor_prob, val_qc_pred, y_val,
                    cfg.floor_price, floor_value,
                )
                if threshold < 1.0:
                    test_floor_prob = clf.predict_proba(X_test)[:, 1]
                    pred_test = np.where(
                        test_floor_prob >= threshold, floor_value, pred_test,
                    )

    elif tc.two_stage:
        clf = fit_floor_classifier(
            X_train, y_train, X_val, y_val, cfg.floor_price,
            params, sample_weight=sw,
        )
        model = fit_lgbm(X_train, y_train, X_val, y_val, params, sample_weight=sw)
        if clf is not None:
            val_floor_prob = clf.predict_proba(X_val)[:, 1]
            val_reg_pred = model.predict(X_val)
            threshold, _ = find_best_threshold(
                val_floor_prob, val_reg_pred, y_val,
                cfg.floor_price, floor_value,
            )
            pred_test = two_stage_predict(clf, model, X_test, threshold, floor_value)
        else:
            pred_test = model.predict(X_test)

    else:
        model = fit_lgbm(X_train, y_train, X_val, y_val, params, sample_weight=sw)
        pred_test = model.predict(X_test)

    pred_val: Optional[np.ndarray] = None
    if tc.adaptive_naive_blend or tc.residual_correction:
        if tc.quantile_mode or tc.quantile_two_stage:
            q_val_p = {a: q_models[a].predict(X_val) for a in QUANTILES}
            pred_val = quantile_combine(q_val_p, naive_val_arr, **qc_params)
            if tc.quantile_two_stage and clf is not None and threshold < 1.0:
                vfp = clf.predict_proba(X_val)[:, 1]
                pred_val = np.where(vfp >= threshold, floor_value, pred_val)
        elif tc.two_stage and clf is not None and threshold < 1.0:
            pred_val = two_stage_predict(clf, model, X_val, threshold, floor_value)
        else:
            pred_val = model.predict(X_val)

    naive_blend_alpha = 0.0
    if tc.adaptive_naive_blend and pred_val is not None:
        naive_blend_alpha, _ = tune_adaptive_naive_blend(pred_val, naive_val_arr, y_val)
        if naive_blend_alpha > 0:
            pred_val = apply_adaptive_naive_blend(pred_val, naive_val_arr, naive_blend_alpha)
            pred_test = apply_adaptive_naive_blend(pred_test, naive_test_arr, naive_blend_alpha)

    residual_gamma = 0.0
    residual_bias: Dict[int, float] = {}
    if tc.residual_correction and pred_val is not None:
        hh_val = feat.loc[val_mask, "step"].values
        hh_test = feat.loc[test_mask, "step"].values
        residual_gamma, residual_bias = tune_residual_gamma(pred_val, y_val, hh_val)
        if residual_gamma > 0:
            pred_test = apply_residual_correction(
                pred_test, hh_test, residual_bias, residual_gamma,
            )

    test_metrics = evaluate(y_test, pred_test)

    test_df = feat.loc[test_mask, ["trade_date", "hour", "step", "y"]].copy()
    test_df["pred"] = pred_test
    if (tc.quantile_mode or tc.quantile_two_stage) and q_test_preds is not None:
        for alpha in QUANTILES:
            test_df[f"q{int(alpha * 100):02d}"] = q_test_preds[alpha]

    naive_valid = ~np.isnan(naive_test_arr)
    naive_metrics = (
        evaluate(y_test[naive_valid], naive_test_arr[naive_valid])
        if naive_valid.any() else None
    )

    by_date = test_df.groupby("trade_date")
    daily_corrs: List[float] = []
    for _, g in by_date:
        if len(g) < 3 or g["y"].std() < 1e-6 or g["pred"].std() < 1e-6:
            continue
        daily_corrs.append(float(np.corrcoef(g["y"].values, g["pred"].values)[0, 1]))
    profile_corr = round(float(np.mean(daily_corrs)), 4) if daily_corrs else None

    info: Dict[str, Any] = {
        "train_days": len(train_dates),
        "val_days": len(val_dates),
        "test_days": len(test_dates),
        "n_train_samples": int(train_mask.sum()),
        "n_val_samples": int(val_mask.sum()),
        "n_test_samples": int(test_mask.sum()),
        "test_date_start": str(test_dates[0])[:10],
        "test_date_end": str(test_dates[-1])[:10],
        "feature_count": len(x_cols),
        "mae": round(test_metrics["mae"], 4),
        "rmse": round(test_metrics["rmse"], 4),
        "mape": round(test_metrics["mape"], 6),
        "profile_corr": profile_corr,
        "naive_yesterday": naive_metrics,
        "floor_pred_value": floor_value if tc.dynamic_floor_value else None,
        "naive_blend_alpha": naive_blend_alpha if tc.adaptive_naive_blend else None,
        "residual_gamma": residual_gamma if tc.residual_correction else None,
    }
    if tc.two_stage or tc.quantile_two_stage:
        info["floor_threshold"] = float(threshold)
        info["floor_actual"] = int((y_test <= cfg.floor_price).sum())
        info["floor_pred_count"] = int((pred_test <= cfg.floor_price + 1).sum())
    if tc.quantile_mode or tc.quantile_two_stage:
        info["quantile_params"] = qc_params

    last_payload: Dict[str, Any] = {
        "model": model,
        "feature_columns": list(x_cols),
    }
    if clf is not None:
        last_payload["classifier"] = clf
        last_payload["threshold"] = threshold
    if q_models is not None:
        last_payload["quantile_models"] = q_models
        last_payload["quantile_params"] = qc_params

    return info, last_payload, test_df.reset_index(drop=True)


def _get_naive(
    feat: pd.DataFrame, all_dates: List, target_dates: List,
) -> np.ndarray:
    """构建 naive baseline（昨日同时段，按 step 索引）。"""
    naive_vals = []
    for td in target_dates:
        td_idx = list(all_dates).index(td)
        cur = feat.loc[feat["trade_date"] == td].sort_values("step")
        if td_idx > 0:
            prev_d = all_dates[td_idx - 1]
            prev = feat.loc[
                feat["trade_date"] == prev_d, ["step", "y"]
            ].set_index("step")["y"]
            naive_vals.extend(cur["step"].map(prev).values)
        else:
            naive_vals.extend([np.nan] * len(cur))
    return np.array(naive_vals, dtype=float)


def compute_baselines(feat: pd.DataFrame, test_start: str, test_end: str) -> Dict:
    """Naive baselines: 昨日同时段 + 上周同时段（test_start ~ test_end，按 step 索引）。"""
    if "step" not in feat.columns:
        feat = feat.copy()
        feat["step"] = feat["hour"].astype(int)
    dates = sorted(feat["trade_date"].unique())
    ts_start = pd.Timestamp(test_start)
    ts_end = pd.Timestamp(test_end)
    test_dates = [d for d in dates if ts_start <= pd.Timestamp(d) <= ts_end]
    if not test_dates:
        return {}
    results = {}

    naive_records = []
    naive_week_records = []
    for td in test_dates:
        td_idx = list(dates).index(td)
        cur = feat.loc[feat["trade_date"] == td, ["step", "y"]].copy()
        if td_idx > 0:
            prev_d = dates[td_idx - 1]
            prev_vals = feat.loc[
                feat["trade_date"] == prev_d, ["step", "y"]
            ].set_index("step")["y"]
            cur["naive_prev"] = cur["step"].map(prev_vals)
            naive_records.append(cur)
        if td_idx >= 7:
            prev_w = dates[td_idx - 7]
            prev_w_vals = feat.loc[
                feat["trade_date"] == prev_w, ["step", "y"]
            ].set_index("step")["y"]
            cur_w = cur.copy()
            cur_w["naive_week"] = cur_w["step"].map(prev_w_vals)
            naive_week_records.append(cur_w)

    if naive_records:
        ndf = pd.concat(naive_records).dropna(subset=["naive_prev"])
        if len(ndf) > 0:
            results["naive_yesterday"] = evaluate(
                ndf["y"].values, ndf["naive_prev"].values
            )
    if naive_week_records:
        wdf = pd.concat(naive_week_records).dropna(subset=["naive_week"])
        if len(wdf) > 0:
            results["naive_lastweek"] = evaluate(
                wdf["y"].values, wdf["naive_week"].values
            )
    return results
