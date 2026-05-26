"""三种朴素基准策略的预测实现。"""
from __future__ import annotations

import pandas as pd

STRATEGIES = ("lag_1d", "lag_7d", "rolling_7d_mean")


def aggregate_target(
    df_15min: pd.DataFrame,
    target_col: str,
    freq: str,
) -> pd.Series:
    """把市场 15min 长表的 target 列聚合到目标粒度。

    1h 模式：4 个 15min 取均值；
    15min 模式：原样返回。
    """
    if target_col not in df_15min.columns:
        raise KeyError(
            f"target_col={target_col} 不在长表列中；可用列示例："
            f"{list(df_15min.columns[:10])}..."
        )
    y = df_15min[target_col].astype(float)
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("df_15min 需有 DatetimeIndex")
    if freq == "1h":
        return y.resample("1h").mean()
    if freq == "15min":
        return y
    raise ValueError(f"freq 仅支持 1h / 15min, got {freq!r}")


def _steps_per_day(freq: str) -> int:
    return 24 if freq == "1h" else 96


def naive_predict(
    y_full: pd.Series,
    strategy: str,
    freq: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    """生成 test 区间 [test_start, test_end] 内的朴素预测。

    Returns:
        DataFrame[ts, actual, predicted]，按 ts 升序。
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy 必须是 {STRATEGIES}, got {strategy!r}")

    s1d = _steps_per_day(freq)
    if strategy == "lag_1d":
        pred = y_full.shift(s1d)
    elif strategy == "lag_7d":
        pred = y_full.shift(s1d * 7)
    else:
        pred = pd.concat(
            [y_full.shift(s1d * k) for k in range(1, 8)], axis=1
        ).mean(axis=1)

    test_end_inclusive = pd.Timestamp(test_end) + pd.Timedelta(hours=23, minutes=45)
    mask = (y_full.index >= pd.Timestamp(test_start)) & (y_full.index <= test_end_inclusive)
    out = pd.DataFrame({
        "ts": y_full.index[mask],
        "actual": y_full.values[mask],
        "predicted": pred.values[mask],
    }).dropna(subset=["actual", "predicted"]).reset_index(drop=True)
    return out
