"""LightGBM baseline — 特征工程。

支持 1h / 15min 两种粒度，逻辑通过 freq 参数统一：
  - 1h: steps_per_day=24, D-1 = shift(24), D-2 = shift(48), D-7 = shift(168)
  - 15min: steps_per_day=96, D-1 = shift(96), D-2 = shift(192), D-7 = shift(672)

严格遵循 lag0/lag1/lag2 规则防止未来信息泄漏：
  - lag0：D 日预测/计划值，不 shift（D-1 已发布）
  - lag1：D-1 日数据，shift steps_per_day
  - lag2：D-2 日数据，shift 2 * steps_per_day
  - 衍生特征均基于已 shift 的序列计算
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
    _add_lag0(feat, df_h, cfg)
    _add_lag1(feat, df_h, cfg, steps_per_day)
    _add_lag2(feat, df_h, cfg, steps_per_day)
    _add_price_lags(feat, df_h, cfg, steps_per_day)
    _add_rolling_stats(feat, df_h, cfg, steps_per_day)
    _add_derived(feat, df_h, cfg, steps_per_day)

    n_before = len(feat)
    feat = feat.dropna(subset=["y"])
    logger.info(
        "%s [%s]: %d 特征列, %d -> %d 行 (去掉 y=NaN)",
        cfg.market_id, freq, feat.shape[1] - 1, n_before, len(feat),
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


def _add_lag0(feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig) -> None:
    """lag0：D日已知预测/计划值，不 shift。"""
    for col in cfg.lag0_cols:
        if col in df_h.columns:
            feat[f"lag0_{col}"] = df_h[col]


def _add_lag1(
    feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig, steps_per_day: int,
) -> None:
    """lag1：D-1日数据，shift steps_per_day。"""
    for col in cfg.lag1_cols:
        if col in df_h.columns:
            feat[f"lag1_{col}"] = df_h[col].shift(steps_per_day)


def _add_lag2(
    feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig, steps_per_day: int,
) -> None:
    """lag2：D-2日数据，shift 2 * steps_per_day。"""
    for col in cfg.lag2_cols:
        if col in df_h.columns:
            feat[f"lag2_{col}"] = df_h[col].shift(2 * steps_per_day)


def _add_price_lags(
    feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig, steps_per_day: int,
) -> None:
    """target 价格多级滞后。price_lag_hours 表达的是"小时数"，按粒度换算为步数。"""
    target = df_h[cfg.target_col]
    multiplier = steps_per_day // 24
    for lag_h in cfg.price_lag_hours:
        feat[f"target_lag{lag_h}h"] = target.shift(lag_h * multiplier)


def _add_rolling_stats(
    feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig, steps_per_day: int,
) -> None:
    """基于已 shift 的 lag1 target 序列计算滚动统计量。"""
    lagged = df_h[cfg.target_col].shift(steps_per_day)
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
    """衍生特征：差分、比率、峰谷价差。"""
    target = df_h[cfg.target_col]
    lag1d = target.shift(steps_per_day)
    lag2d = target.shift(2 * steps_per_day)
    lag7d = target.shift(7 * steps_per_day)

    feat["target_diff_24h"] = lag1d - lag2d
    feat["target_ratio_24h_168h"] = lag1d / lag7d.replace(0, np.nan)

    peak_hours = set(range(8, 12)) | set(range(17, 21))
    valley_hours = set(range(0, 8)) | {23}

    peak_mask = lag1d.index.hour.isin(peak_hours)
    valley_mask = lag1d.index.hour.isin(valley_hours)

    by_date_lag = lag1d.groupby(lag1d.index.date)
    feat["target_lag1d_daily_max"] = by_date_lag.transform("max")
    feat["target_lag1d_daily_min"] = by_date_lag.transform("min")
    feat["target_lag1d_amplitude"] = (
        feat["target_lag1d_daily_max"] - feat["target_lag1d_daily_min"]
    )

    lag1d_peak = lag1d.where(peak_mask)
    lag1d_valley = lag1d.where(valley_mask)
    feat["target_lag1d_peak_mean"] = lag1d_peak.groupby(
        lag1d_peak.index.date
    ).transform("mean")
    feat["target_lag1d_valley_mean"] = lag1d_valley.groupby(
        lag1d_valley.index.date
    ).transform("mean")
