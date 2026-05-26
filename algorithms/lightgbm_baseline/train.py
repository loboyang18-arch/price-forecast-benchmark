"""LightGBM baseline — 训练、预测、评估。

流程：
  1. 加载统一数据 → hourly 特征表
  2. 按 test_start 切分 train/test（val 从 train 末尾取 7 天用于 early stopping）
  3. LightGBM 训练（early stopping on val MAE）
  4. Test 集预测 + 指标计算
  5. 输出 predictions CSV + metrics JSON
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import MarketConfig
from .features import build_features

logger = logging.getLogger(__name__)

VAL_DAYS = 7
LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "seed": 42,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50


def run_experiment(
    df_15min: pd.DataFrame,
    cfg: MarketConfig,
    output_dir: Path,
    test_start: str,
    test_end: str,
    freq: str = "1h",
    feature_spec: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """完整实验：特征构建 → 切分 → 训练 → 预测 → 评估 → 保存。

    freq: "1h" 或 "15min"，决定特征粒度与预测粒度。
    feature_spec: 由 ResolvedSpec.to_dict() 提供，写入 metrics.json 做实验追溯。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    feat = build_features(df_15min, cfg, freq=freq)
    train_df, val_df, test_df = _split(feat, test_start, test_end)
    model, importance = _train(train_df, val_df)
    preds, metrics = _evaluate(model, test_df, cfg, test_start, freq)
    if feature_spec is not None:
        metrics["feature_spec"] = feature_spec

    _save_results(preds, metrics, importance, cfg, output_dir, freq)
    return metrics


def _split(
    feat: pd.DataFrame, test_start: str, test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按 test_start/test_end 切分，val 从 train 末尾取 VAL_DAYS 天。"""
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end) + pd.Timedelta(days=1)
    train_all = feat.loc[feat.index < test_start]
    test_df = feat.loc[(feat.index >= test_start) & (feat.index < test_end)]

    val_start = test_start - pd.Timedelta(days=VAL_DAYS)
    train_df = train_all.loc[train_all.index < val_start]
    val_df = train_all.loc[train_all.index >= val_start]

    logger.info(
        "切分: train=%d (..%s), val=%d (%s..%s), test=%d (%s..)",
        len(train_df), train_df.index.max().date(),
        len(val_df), val_df.index.min().date(), val_df.index.max().date(),
        len(test_df), test_df.index.min().date(),
    )
    return train_df, val_df, test_df


def _get_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """分离特征和标签。"""
    feature_cols = [c for c in df.columns if c != "y"]
    X = df[feature_cols]
    y = df["y"].values
    return X, y


def _train(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Tuple[lgb.Booster, pd.DataFrame]:
    """LightGBM 训练，early stopping on val MAE。"""
    X_train, y_train = _get_xy(train_df)
    X_val, y_val = _get_xy(val_df)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(100),
    ]

    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    importance = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    logger.info(
        "训练完成: best_iteration=%d, val_mae=%.4f",
        model.best_iteration, model.best_score["val"]["l1"],
    )
    return model, importance


def _evaluate(
    model: lgb.Booster, test_df: pd.DataFrame, cfg: MarketConfig,
    test_start: str, freq: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Test 集预测 + 指标计算。"""
    X_test, y_test = _get_xy(test_df)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    preds = pd.DataFrame({
        "ts": test_df.index,
        "actual": y_test,
        "predicted": y_pred,
    })

    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    valid_mask = np.abs(y_test) > 1e-9
    mape = float(np.mean(np.abs((y_test[valid_mask] - y_pred[valid_mask]) / y_test[valid_mask]))) * 100

    by_date = preds.groupby(preds["ts"].dt.date)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
        daily_corrs = by_date.apply(lambda g: g["actual"].corr(g["predicted"]))
    profile_corr = float(daily_corrs.mean())

    metrics = {
        "market_id": cfg.market_id,
        "target_col": cfg.target_col,
        "algorithm": "LightGBM",
        "freq": freq,
        "test_start": test_start,
        "test_rows": int(len(y_test)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape_pct": round(mape, 2),
        "profile_corr": round(profile_corr, 4),
        "best_iteration": model.best_iteration,
        "n_features": int(X_test.shape[1]),
    }

    logger.info(
        "%s: MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%  ProfileCorr=%.4f",
        cfg.market_id, mae, rmse, mape, profile_corr,
    )
    return preds, metrics


def _save_results(
    preds: pd.DataFrame,
    metrics: Dict[str, Any],
    importance: pd.DataFrame,
    cfg: MarketConfig,
    output_dir: Path,
    freq: str,
) -> None:
    """保存预测结果、指标、特征重要性。"""
    fname = "test_predictions_15min.csv" if freq == "15min" else "test_predictions_hourly.csv"
    pred_path = output_dir / fname
    preds.to_csv(pred_path, index=False)
    try:
        from pfbench.metrics import evaluate_predictions_csv
        metrics["extended_metrics"] = evaluate_predictions_csv(pred_path)
    except Exception as exc:
        logger.warning("extended_metrics 计算失败: %s", exc)
        metrics["extended_metrics"] = None
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    logger.info("结果已保存到 %s", output_dir)
