"""点预测与形态类指标（不依赖具体模型）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def point_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / (np.abs(actual) + 1e-6)) * 100)
    bias = float(np.mean(err))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape_pct": round(mape, 2),
        "bias": round(bias, 4),
        "valid_point_count": int(len(actual)),
    }


def shape_metrics(df: pd.DataFrame) -> dict:
    """按日聚合：曲线相关、涨跌方向准确率。"""
    daily = df.copy()
    daily["date"] = daily.index.normalize()
    corrs = []
    dir_hits = []
    for _, g in daily.groupby("date"):
        a = g["actual"].values
        p = g["pred"].values
        if len(a) < 3 or np.std(a) < 1e-6 or np.std(p) < 1e-6:
            continue
        corrs.append(float(np.corrcoef(a, p)[0, 1]))
        da = np.diff(a)
        dp = np.diff(p)
        if len(da):
            dir_hits.append(float(np.mean(np.sign(da) == np.sign(dp))))

    if not corrs:
        return {
            "profile_corr": None,
            "neg_corr_day_ratio": None,
            "direction_acc": None,
            "n_days": 0,
        }
    corrs_arr = np.array(corrs)
    return {
        "profile_corr": round(float(np.nanmean(corrs_arr)), 4),
        "neg_corr_day_ratio": round(float(np.mean(corrs_arr < 0)), 4),
        "direction_acc": round(float(np.mean(dir_hits)), 4) if dir_hits else None,
        "n_days": len(corrs),
    }


def evaluate_frame(df: pd.DataFrame) -> dict:
    a = df["actual"].astype(float).values
    p = df["pred"].astype(float).values
    return {
        "point_metrics": point_metrics(a, p),
        "shape_metrics": shape_metrics(df),
    }


def try_extended_eval(df: pd.DataFrame, task: str = "da") -> dict | None:
    """若已安装 price_forecast_eval，返回扩展指标。"""
    try:
        from price_forecast_eval import evaluate_model_predictions, from_result_columns
    except ImportError:
        return None
    try:
        frame = df.reset_index()
        if "ts" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "ts"})
        ef = from_result_columns(
            frame, actual_col="actual", pred_col="pred", ts_index=False,
        )
        return evaluate_model_predictions(ef, task_type=task, include_extended=True)
    except Exception:
        return None
