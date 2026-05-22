"""预测结果读取与标准化。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PRED_COL_CANDIDATES = ("pred", "predicted", "pred_lgb", "pred_v12", "forecast", "y_pred")
ACTUAL_COL_CANDIDATES = ("actual", "y_true", "target", "price")


def infer_pred_col(df: pd.DataFrame) -> str | None:
    for c in PRED_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def infer_actual_col(df: pd.DataFrame) -> str | None:
    for c in ACTUAL_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def load_prediction_csv(
    path: Path,
    *,
    test_start: str | None = None,
    test_end: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts").sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    else:
        raise ValueError(f"{path}: 需要 ts 列或 DatetimeIndex")

    pred_col = infer_pred_col(df)
    actual_col = infer_actual_col(df)
    if pred_col is None or actual_col is None:
        raise ValueError(f"{path}: 无法识别 pred/actual 列")

    out = df[[actual_col, pred_col]].rename(
        columns={actual_col: "actual", pred_col: "pred"},
    )
    out = out.loc[out["actual"].notna() & out["pred"].notna()]

    if test_start:
        out = out.loc[out.index >= pd.Timestamp(test_start)]
    if test_end:
        out = out.loc[out.index <= pd.Timestamp(test_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

    if len(out) == 0:
        raise ValueError(f"{path}: 过滤后无有效样本")
    return out
