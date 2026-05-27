"""LightGBM baseline — 特征工程（5 lag-bucket 设计）。

每个 group 按其 ``window_lag_days`` 单独 shift，避免 v1 老路径下"日前出清价被
shift 1d 当 lag1 特征"导致的数据泄漏：

  - lag 0d → 不 shift（如 BOUNDARY）
  - lag 1d → shift steps_per_day（如 BOUNDARY_DM1）
  - lag 2d → shift 2 * steps_per_day（如 CLEARING_RT / ACTUAL）
  - lag 3d → shift 3 * steps_per_day（如 CLEARING_DA）
  - lag 4d → shift 4 * steps_per_day（如 CLEARING_RT_NODAL）

target 自身的多级历史 lag 由 ``cfg.price_lag_hours`` 给出，最小值 = target
所属 group 的 ``window_lag_days`` × 24h（如内蒙 target lag=4d → 起点 96h）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import MarketConfig

logger = logging.getLogger(__name__)

STEPS_PER_DAY = {"1h": 24, "15min": 96}


def build_features(
    df_15min: pd.DataFrame, cfg: MarketConfig, freq: str = "1h",
) -> pd.DataFrame:
    """从 15min 原始数据构建特征表（freq=1h 或 15min）。

    Returns:
        DataFrame, index=ts (按 freq 重采样), 包含所有特征列 + target 列 'y'。
    """
    if freq not in STEPS_PER_DAY:
        raise ValueError(f"不支持的 freq={freq}，可选 {list(STEPS_PER_DAY)}")
    steps_per_day = STEPS_PER_DAY[freq]

    df_h = _resample(df_15min, freq)
    feat = pd.DataFrame(index=df_h.index)

    feat["y"] = df_h[cfg.target_col]

    _add_calendar(feat, freq)
    _add_group_lags(feat, df_h, cfg, steps_per_day)
    _add_price_lags(feat, df_h, cfg, steps_per_day)
    _add_rolling_stats(feat, df_h, cfg, steps_per_day)
    _add_derived(feat, df_h, cfg, steps_per_day)

    n_before = len(feat)
    feat = feat.dropna(subset=["y"])
    logger.info(
        "%s [%s]: %d 特征列, %d -> %d 行 (去掉 y=NaN)；target_lag=%dd, price_lag_h=%s",
        cfg.market_id, freq, feat.shape[1] - 1, n_before, len(feat),
        cfg.target_lag_days, cfg.price_lag_hours,
    )
    return feat


def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """15min → freq（mean 聚合），仅保留数值列。1h 模式下相当于 4 点取平均。"""
    numeric = df.select_dtypes(include=["number"])
    if freq == "15min":
        return numeric
    return numeric.resample(freq).mean()


def _add_calendar(feat: pd.DataFrame, freq: str) -> None:
    """日历特征（已知，不涉及泄漏）。15min 模式下额外加 minute 与 slot。"""
    idx = feat.index
    feat["hour"] = idx.hour
    feat["dow"] = idx.dayofweek
    feat["month"] = idx.month
    feat["is_weekend"] = (idx.dayofweek >= 5).astype(np.float32)
    feat["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    if freq == "15min":
        slot = idx.hour * 4 + idx.minute // 15
        feat["slot"] = slot
        feat["slot_sin"] = np.sin(2 * np.pi * slot / 96)
        feat["slot_cos"] = np.cos(2 * np.pi * slot / 96)


def _add_group_lags(
    feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig, steps_per_day: int,
) -> None:
    """按每个 group 的 window_lag_days 单独 shift 该组所有列。

    lag_days=0：不 shift，列名前缀 lag0_；
    lag_days=k>0：shift k * steps_per_day，列名前缀 lag{k}d_。
    """
    for lag_days, cols in cfg.cols_by_lag_days.items():
        shift_n = lag_days * steps_per_day
        prefix = "lag0" if lag_days == 0 else f"lag{lag_days}d"
        for col in cols:
            if col in df_h.columns:
                feat[f"{prefix}_{col}"] = df_h[col].shift(shift_n) if shift_n > 0 else df_h[col]


def _add_price_lags(
    feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig, steps_per_day: int,
) -> None:
    """target 价格多级滞后。price_lag_hours 表达的是"小时数"，按粒度换算为步数。

    最小 lag_h = target_lag_days * 24，保证 D 日预测时 target 历史值已可见。
    """
    target = df_h[cfg.target_col]
    multiplier = steps_per_day // 24
    for lag_h in cfg.price_lag_hours:
        feat[f"target_lag{lag_h}h"] = target.shift(lag_h * multiplier)


def _add_rolling_stats(
    feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig, steps_per_day: int,
) -> None:
    """基于已 shift target_lag_days 天的 target 序列计算滚动统计量。

    target 自身 lag=target_lag_days（如内蒙 v2 = 4d），早期版本固定 shift 1d 会
    在 v2 数据上引入未来信息泄漏。
    """
    lag_base = max(cfg.target_lag_days, 1) * steps_per_day
    lagged = df_h[cfg.target_col].shift(lag_base)
    day = steps_per_day
    week = 7 * steps_per_day

    feat["target_roll24h_mean"] = lagged.rolling(day, min_periods=day // 2).mean()
    feat["target_roll24h_std"] = lagged.rolling(day, min_periods=day // 2).std()
    feat["target_roll168h_mean"] = lagged.rolling(week, min_periods=week // 2).mean()
    feat["target_roll168h_std"] = lagged.rolling(week, min_periods=week // 2).std()

    by_slot = lagged.groupby([lagged.index.hour, lagged.index.minute])
    feat["target_hourly_roll7d_mean"] = by_slot.transform(
        lambda s: s.rolling(7, min_periods=3).mean()
    )


def _add_derived(
    feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig, steps_per_day: int,
) -> None:
    """衍生特征：差分、比率、峰谷价差。

    最小可用 lag = target_lag_days；接下来 +1d；以及 7d（与 weekly 比较）。
    """
    target = df_h[cfg.target_col]
    lag_a_days = max(cfg.target_lag_days, 1)
    lag_b_days = lag_a_days + 1
    lag_c_days = max(lag_a_days, 7)

    lag_a = target.shift(lag_a_days * steps_per_day)
    lag_b = target.shift(lag_b_days * steps_per_day)
    lag_c = target.shift(lag_c_days * steps_per_day)

    feat[f"target_diff_{lag_a_days}d_{lag_b_days}d"] = lag_a - lag_b
    feat[f"target_ratio_{lag_a_days}d_{lag_c_days}d"] = lag_a / lag_c.replace(0, np.nan)

    peak_hours = set(range(8, 12)) | set(range(17, 21))
    valley_hours = set(range(0, 8)) | {23}

    peak_mask = lag_a.index.hour.isin(peak_hours)
    valley_mask = lag_a.index.hour.isin(valley_hours)

    by_date_lag = lag_a.groupby(lag_a.index.date)
    feat[f"target_lag{lag_a_days}d_daily_max"] = by_date_lag.transform("max")
    feat[f"target_lag{lag_a_days}d_daily_min"] = by_date_lag.transform("min")
    feat[f"target_lag{lag_a_days}d_amplitude"] = (
        feat[f"target_lag{lag_a_days}d_daily_max"]
        - feat[f"target_lag{lag_a_days}d_daily_min"]
    )

    lag_a_peak = lag_a.where(peak_mask)
    lag_a_valley = lag_a.where(valley_mask)
    feat[f"target_lag{lag_a_days}d_peak_mean"] = lag_a_peak.groupby(
        lag_a_peak.index.date
    ).transform("mean")
    feat[f"target_lag{lag_a_days}d_valley_mean"] = lag_a_valley.groupby(
        lag_a_valley.index.date
    ).transform("mean")
