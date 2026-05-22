"""LightGBM baseline — 特征工程。

严格遵循 lag0/lag1/lag2 规则防止未来信息泄漏：
  - lag0：D日预测/计划值，不 shift（D-1 已发布）
  - lag1：D-1日数据，shift 24h
  - lag2：D-2日数据，shift 48h
  - 衍生特征均基于已 shift 的序列计算
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import MarketConfig

logger = logging.getLogger(__name__)


def build_features(df_15min: pd.DataFrame, cfg: MarketConfig) -> pd.DataFrame:
    """从 15min 原始数据构建小时级特征表。

    Returns:
        DataFrame, index=ts(hourly), 包含所有特征列 + target 列 'y'。
    """
    df_h = _resample_hourly(df_15min)
    feat = pd.DataFrame(index=df_h.index)

    feat["y"] = df_h[cfg.target_col]

    _add_calendar(feat)
    _add_lag0(feat, df_h, cfg)
    _add_lag1(feat, df_h, cfg)
    _add_lag2(feat, df_h, cfg)
    _add_price_lags(feat, df_h, cfg)
    _add_rolling_stats(feat, df_h, cfg)
    _add_derived(feat, df_h, cfg)

    n_before = len(feat)
    feat = feat.dropna(subset=["y"])
    logger.info(
        "%s: %d 特征列, %d -> %d 行 (去掉 y=NaN)",
        cfg.market_id, feat.shape[1] - 1, n_before, len(feat),
    )
    return feat


def _resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """15min → hourly (mean4)，仅保留数值列。"""
    numeric = df.select_dtypes(include=["number"])
    return numeric.resample("1h").mean()


def _add_calendar(feat: pd.DataFrame) -> None:
    """日历特征（已知，不涉及泄漏）。"""
    idx = feat.index
    feat["hour"] = idx.hour
    feat["dow"] = idx.dayofweek
    feat["month"] = idx.month
    feat["is_weekend"] = (idx.dayofweek >= 5).astype(np.float32)
    feat["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)


def _add_lag0(feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig) -> None:
    """lag0：D日已知预测/计划值，不 shift。"""
    for col in cfg.lag0_cols:
        if col in df_h.columns:
            feat[f"lag0_{col}"] = df_h[col]


def _add_lag1(feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig) -> None:
    """lag1：D-1日数据，shift 24h。"""
    for col in cfg.lag1_cols:
        if col in df_h.columns:
            feat[f"lag1_{col}"] = df_h[col].shift(24)


def _add_lag2(feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig) -> None:
    """lag2：D-2日数据，shift 48h。"""
    for col in cfg.lag2_cols:
        if col in df_h.columns:
            feat[f"lag2_{col}"] = df_h[col].shift(48)


def _add_price_lags(feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig) -> None:
    """target 价格多级滞后。"""
    target = df_h[cfg.target_col]
    for lag_h in cfg.price_lag_hours:
        feat[f"target_lag{lag_h}h"] = target.shift(lag_h)


def _add_rolling_stats(feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig) -> None:
    """基于已 shift 的 lag1 target 序列计算滚动统计量。"""
    lagged = df_h[cfg.target_col].shift(24)

    feat["target_roll24h_mean"] = lagged.rolling(24, min_periods=12).mean()
    feat["target_roll24h_std"] = lagged.rolling(24, min_periods=12).std()
    feat["target_roll168h_mean"] = lagged.rolling(168, min_periods=84).mean()
    feat["target_roll168h_std"] = lagged.rolling(168, min_periods=84).std()

    by_hour = lagged.groupby(lagged.index.hour)
    feat["target_hourly_roll7d_mean"] = by_hour.transform(
        lambda s: s.rolling(7, min_periods=3).mean()
    )


def _add_derived(feat: pd.DataFrame, df_h: pd.DataFrame, cfg: MarketConfig) -> None:
    """衍生特征：差分、比率、峰谷价差。"""
    target = df_h[cfg.target_col]
    lag24 = target.shift(24)
    lag48 = target.shift(48)
    lag168 = target.shift(168)

    feat["target_diff_24h"] = lag24 - lag48
    feat["target_ratio_24h_168h"] = lag24 / lag168.replace(0, np.nan)

    peak_hours = set(range(8, 12)) | set(range(17, 21))
    valley_hours = set(range(0, 8)) | {23}

    peak_mask = lag24.index.hour.isin(peak_hours)
    valley_mask = lag24.index.hour.isin(valley_hours)

    by_date_lag = lag24.groupby(lag24.index.date)
    feat["target_lag1d_daily_max"] = by_date_lag.transform("max")
    feat["target_lag1d_daily_min"] = by_date_lag.transform("min")
    feat["target_lag1d_amplitude"] = (
        feat["target_lag1d_daily_max"] - feat["target_lag1d_daily_min"]
    )

    lag24_peak = lag24.where(peak_mask)
    lag24_valley = lag24.where(valley_mask)
    feat["target_lag1d_peak_mean"] = lag24_peak.groupby(
        lag24_peak.index.date
    ).transform("mean")
    feat["target_lag1d_valley_mean"] = lag24_valley.groupby(
        lag24_valley.index.date
    ).transform("mean")
