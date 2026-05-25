"""LightGBM-TwoStage — Expanding Window Cross-Validation。

移植自 jiangsu_prj/scripts/train_dayahead.py expanding_window_cv，
完整实现按 trade_date 的扩展窗口交叉验证：
  - 训练窗口持续扩大
  - val 窗口紧跟训练窗口末尾（用于 early stopping + 后处理参数调优）
  - test 窗口紧跟 val 窗口
  - 逐步滑动 step_days
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

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
class CVConfig:
    """Expanding Window CV 全部可配参数。"""
    val_days: int = 7
    test_days: int = 7
    step_days: int = 7
    params: Optional[Dict] = None
    feature_select_top_k: int = 0
    two_stage: bool = False
    quantile_mode: bool = False
    quantile_two_stage: bool = False
    time_decay_half_life: int = 0
    adaptive_naive_blend: bool = True
    dynamic_floor_value: bool = True
    residual_correction: bool = True


def expanding_window_cv(
    feat: pd.DataFrame, cfg: MarketConfig, cv: CVConfig,
    test_start: str, test_end: str,
) -> Tuple[List[Dict], Optional[Dict], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Expanding window CV by trade_date。

    test_start / test_end 由 config/markets/*.yaml 统一定义。
    CV 的 test 窗口从 test_start 开始向后滑动直到 test_end，
    train 窗口使用 test_start 之前的全部数据（逐步扩大），
    val 窗口紧贴 test 窗口前 val_days 天。

    Returns (fold_results, last_model_payload, last_fold_test_df, all_preds).
    """
    params = {**(cv.params or {})}
    if "step" not in feat.columns:
        feat = feat.copy()
        feat["step"] = feat["hour"].astype(int)
    dates = sorted(feat["trade_date"].unique())

    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)
    dates = [d for d in dates if pd.Timestamp(d) <= test_end_ts]
    n_dates = len(dates)
    x_cols = [c for c in feat.columns if c not in ("trade_date", "y")]

    test_start_idx = None
    for idx, d in enumerate(dates):
        if pd.Timestamp(d) >= test_start_ts:
            test_start_idx = idx
            break
    if test_start_idx is None:
        raise ValueError(f"test_start {test_start} 超出数据范围")

    date_set = set(dates)
    feat = feat.loc[feat["trade_date"].isin(date_set)].copy()

    x_cols = [c for c in x_cols if c != "step"]
    min_train_for_select = max(30, test_start_idx - cv.val_days)
    if cv.feature_select_top_k > 0 and cv.feature_select_top_k < len(x_cols):
        x_cols = select_top_k_features(
            feat, x_cols, min_train_for_select, cv.val_days,
            cv.feature_select_top_k, params,
        )

    folds: List[Dict] = []
    all_test_dfs: List[pd.DataFrame] = []
    last_payload: Optional[Dict] = None
    last_test_df: Optional[pd.DataFrame] = None

    fold_idx = 0
    test_window_start = test_start_idx

    while test_window_start < n_dates:
        test_end_idx = min(test_window_start + cv.test_days, n_dates)
        test_dates = dates[test_window_start:test_end_idx]
        val_end_idx = test_window_start
        val_start_idx = max(0, val_end_idx - cv.val_days)
        val_dates = dates[val_start_idx:val_end_idx]
        train_dates = dates[:val_start_idx]

        train_mask = feat["trade_date"].isin(train_dates)
        val_mask = feat["trade_date"].isin(val_dates)
        test_mask = feat["trade_date"].isin(test_dates)

        X_train = feat.loc[train_mask, x_cols]
        y_train = feat.loc[train_mask, "y"].values
        X_val = feat.loc[val_mask, x_cols]
        y_val = feat.loc[val_mask, "y"].values
        X_test = feat.loc[test_mask, x_cols]
        y_test = feat.loc[test_mask, "y"].values

        if len(X_train) < 24 or len(X_val) < 24 or len(X_test) < 24:
            test_window_start += cv.step_days
            continue

        sw = None
        if cv.time_decay_half_life > 0:
            sw = compute_time_decay_weights(
                feat.loc[train_mask, "trade_date"], cv.time_decay_half_life
            )

        fold_floor_value = (
            compute_dynamic_floor_value(y_train, cfg.floor_price)
            if cv.dynamic_floor_value else cfg.floor_pred_value
        )

        naive_test_arr = _get_naive(feat, dates, test_dates)
        naive_val_arr = _get_naive(feat, dates, val_dates)

        # ── 模型训练 ──
        clf = None
        threshold = 1.1
        q_models = None
        q_test_preds = None
        qc_params: Dict = {}

        if cv.quantile_mode or cv.quantile_two_stage:
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

            if cv.quantile_two_stage:
                clf = fit_floor_classifier(
                    X_train, y_train, X_val, y_val, cfg.floor_price,
                    params, sample_weight=sw,
                )
                if clf is not None:
                    val_qc_pred = quantile_combine(q_val_preds, naive_val_arr, **qc_params)
                    val_floor_prob = clf.predict_proba(X_val)[:, 1]
                    threshold, _ = find_best_threshold(
                        val_floor_prob, val_qc_pred, y_val,
                        cfg.floor_price, fold_floor_value,
                    )
                    if threshold < 1.0:
                        test_floor_prob = clf.predict_proba(X_test)[:, 1]
                        pred_test = np.where(test_floor_prob >= threshold, fold_floor_value, pred_test)

        elif cv.two_stage:
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
                    cfg.floor_price, fold_floor_value,
                )
                pred_test = two_stage_predict(clf, model, X_test, threshold, fold_floor_value)
            else:
                pred_test = model.predict(X_test)

        else:
            model = fit_lgbm(X_train, y_train, X_val, y_val, params, sample_weight=sw)
            pred_test = model.predict(X_test)

        # ── val 集预测（用于后处理参数调优） ──
        pred_val = None
        if cv.adaptive_naive_blend or cv.residual_correction:
            if cv.quantile_mode or cv.quantile_two_stage:
                q_val_p = {a: q_models[a].predict(X_val) for a in QUANTILES}
                pred_val = quantile_combine(q_val_p, naive_val_arr, **qc_params)
                if cv.quantile_two_stage and clf is not None and threshold < 1.0:
                    vfp = clf.predict_proba(X_val)[:, 1]
                    pred_val = np.where(vfp >= threshold, fold_floor_value, pred_val)
            elif cv.two_stage and clf is not None and threshold < 1.0:
                pred_val = two_stage_predict(clf, model, X_val, threshold, fold_floor_value)
            else:
                pred_val = model.predict(X_val)

        # ── 后处理：model-naive 自适应混合 ──
        naive_blend_alpha = 0.0
        if cv.adaptive_naive_blend and pred_val is not None:
            naive_blend_alpha, _ = tune_adaptive_naive_blend(pred_val, naive_val_arr, y_val)
            if naive_blend_alpha > 0:
                pred_val = apply_adaptive_naive_blend(pred_val, naive_val_arr, naive_blend_alpha)
                pred_test = apply_adaptive_naive_blend(pred_test, naive_test_arr, naive_blend_alpha)

        # ── 后处理：残差校正（按 step 分组，对粒度通用） ──
        residual_gamma = 0.0
        residual_bias: Dict[int, float] = {}
        if cv.residual_correction and pred_val is not None:
            hh_val = feat.loc[val_mask, "step"].values
            hh_test = feat.loc[test_mask, "step"].values
            residual_gamma, residual_bias = tune_residual_gamma(pred_val, y_val, hh_val)
            if residual_gamma > 0:
                pred_test = apply_residual_correction(pred_test, hh_test, residual_bias, residual_gamma)

        test_metrics = evaluate(y_test, pred_test)

        test_df = feat.loc[test_mask, ["trade_date", "hour", "step", "y"]].copy()
        test_df["pred"] = pred_test
        if cv.quantile_mode and q_test_preds is not None:
            for alpha in QUANTILES:
                test_df[f"q{int(alpha * 100):02d}"] = q_test_preds[alpha]

        naive_valid = ~np.isnan(naive_test_arr)
        naive_metrics = (
            evaluate(y_test[naive_valid], naive_test_arr[naive_valid])
            if naive_valid.any() else None
        )

        fold_info: Dict[str, Any] = {
            "fold": fold_idx,
            "train_days": len(train_dates),
            "val_days": len(val_dates),
            "test_days": len(test_dates),
            "test_date_start": str(test_dates[0])[:10],
            "test_date_end": str(test_dates[-1])[:10],
            "test": test_metrics,
            "naive_yesterday": naive_metrics,
        }
        if cv.two_stage or cv.quantile_two_stage:
            fold_info["floor_threshold"] = threshold
            fold_info["floor_actual"] = int((y_test <= cfg.floor_price).sum())
            fold_info["floor_pred"] = int((pred_test <= cfg.floor_price + 1).sum())
        if cv.quantile_mode:
            fold_info["quantile_params"] = qc_params
        if cv.adaptive_naive_blend:
            fold_info["naive_blend_alpha"] = naive_blend_alpha
        if cv.dynamic_floor_value:
            fold_info["floor_pred_value"] = fold_floor_value
        if cv.residual_correction:
            fold_info["residual_gamma"] = residual_gamma

        logger.info(
            "Fold %d: MAE=%.2f RMSE=%.2f (%s ~ %s)",
            fold_idx, test_metrics["mae"], test_metrics["rmse"],
            test_dates[0], test_dates[-1],
        )

        folds.append(fold_info)
        all_test_dfs.append(test_df.copy())
        last_payload = {
            "model": model,
            "feature_columns": list(x_cols),
        }
        if clf is not None:
            last_payload["classifier"] = clf
            last_payload["threshold"] = threshold
        if q_models is not None:
            last_payload["quantile_models"] = q_models
            last_payload["quantile_params"] = qc_params
        last_test_df = test_df.copy()
        fold_idx += 1
        test_window_start += cv.step_days

    all_preds = (
        pd.concat(all_test_dfs, ignore_index=True)
        .drop_duplicates(subset=["trade_date", "step"], keep="last")
        .sort_values(["trade_date", "step"])
        .reset_index(drop=True)
    ) if all_test_dfs else None

    return folds, last_payload, last_test_df, all_preds


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
            prev = feat.loc[feat["trade_date"] == prev_d, ["step", "y"]].set_index("step")["y"]
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
            results["naive_yesterday"] = evaluate(ndf["y"].values, ndf["naive_prev"].values)

    if naive_week_records:
        wdf = pd.concat(naive_week_records).dropna(subset=["naive_week"])
        if len(wdf) > 0:
            results["naive_lastweek"] = evaluate(wdf["y"].values, wdf["naive_week"].values)

    return results
