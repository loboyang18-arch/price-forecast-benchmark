"""LightGBM-TwoStage — 特征工程（5 lag-bucket 设计）。

每个特征视图（boundary / price / realtime / actual / target）按所属 group 的
``window_lag_days`` 决定 ``dates[i - lag_d]`` 偏移，避免 v1 老路径下"日前出清价
被当作 D-1 历史"导致的数据泄漏。

历史多档 lag 命名 ``dm{k}`` 表示 D-k 天的实际 lag（如 v2 内蒙 target_lag=4d 时，
原 "dm1 历史价" 自动变为 ``dm4``）。

特征模块：
  1. Boundary 曲线（lag0 only）：当日值 + 差分 + 日统计 + rank + 净负荷 + 新能源渗透率
  2. 历史日前价格（取 max(1,2,7) 与 price_lag_days 的 clamp，去重）
  3. 历史实时价格（取 max(1,2) 与 realtime_lag_days 的 clamp，去重）
  4. 日前-实时价差（lag = max(price, realtime)）
  5. 历史实际运行值（lag = actual_lag_days）
  6. 地板价结构（lag = target_lag_days）
  7. 近3天/7天趋势与波动率（窗口 ending at D - target_lag_days）
  8. 同时段历史地板价频率（近7天，ending at D - target_lag_days）
  9. 日历特征
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import MarketConfig

logger = logging.getLogger(__name__)

STEPS_PER_DAY = {"1h": 24, "15min": 96}


def build_features(
    df_15min: pd.DataFrame, cfg: MarketConfig, freq: str = "1h",
) -> pd.DataFrame:
    """从 15min 数据构建特征表（freq=1h 或 15min）。

    Returns:
        DataFrame, 每行 = (trade_date, step), 包含 y + 所有特征。
    """
    if freq not in STEPS_PER_DAY:
        raise ValueError(f"不支持的 freq={freq}，可选 {list(STEPS_PER_DAY)}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
        return _build_features_impl(df_15min, cfg, freq)


def _build_features_impl(
    df_15min: pd.DataFrame, cfg: MarketConfig, freq: str,
) -> pd.DataFrame:
    steps_per_day = STEPS_PER_DAY[freq]
    numeric = df_15min.select_dtypes(include=["number"])
    df_h = numeric.resample("1h").mean() if freq == "1h" else numeric

    dates = sorted(df_h.index.normalize().unique())
    records: List[pd.DataFrame] = []

    for i, d in enumerate(dates):
        day_mask = df_h.index.normalize() == d
        day = df_h.loc[day_mask]
        if len(day) != steps_per_day:
            continue

        step_idx = np.arange(steps_per_day)
        hour_in_day = day.index.hour.values
        row: Dict = {}
        row["trade_date"] = [d] * steps_per_day
        row["hour"] = hour_in_day
        row["step"] = step_idx
        row["y"] = (
            day[cfg.target_col].values if cfg.target_col in day.columns
            else np.full(steps_per_day, np.nan)
        )

        _add_boundary_features(row, day, cfg)
        _add_price_lag_features(row, df_h, dates, i, cfg, step_idx, steps_per_day)
        _add_realtime_lag_features(row, df_h, dates, i, cfg, step_idx, steps_per_day)
        _add_spread_features(row, df_h, dates, i, cfg, step_idx, steps_per_day)
        _add_actual_features(row, df_h, dates, i, cfg, step_idx, steps_per_day)
        _add_floor_stats(row, df_h, dates, i, cfg, steps_per_day)
        _add_trend_features(row, df_h, dates, i, cfg, steps_per_day)
        _add_hourly_floor_rate(row, df_h, dates, i, cfg, hour_in_day, steps_per_day)
        _add_calendar(row, d, hour_in_day, step_idx, freq)

        records.append(pd.DataFrame(row))

    if not records:
        raise ValueError(f"{cfg.market_id}: 无有效交易日可构建特征")

    feat = pd.concat(records, ignore_index=True)
    feat["trade_date"] = pd.to_datetime(feat["trade_date"])
    feat = feat.dropna(subset=["y"]).reset_index(drop=True)

    x_cols = [c for c in feat.columns if c not in ("trade_date", "y")]
    logger.info(
        "%s [%s]: %d 特征列, %d 行, %d 交易日",
        cfg.market_id, freq, len(x_cols), len(feat), feat["trade_date"].nunique(),
    )
    return feat


def _get_day(df_h: pd.DataFrame, d, steps_per_day: int) -> Optional[pd.DataFrame]:
    mask = df_h.index.normalize() == d
    day = df_h.loc[mask]
    return day if len(day) == steps_per_day else None


def _clamp_lags(orig_lags: List[int], group_lag_days: int) -> List[tuple]:
    """把原始历史 lag 列表按 group_lag_days clamp 到最小可见 lag，去重保序。

    例如 orig_lags=[1, 2, 7], group_lag_days=3 → [(3, "dm3"), (7, "dm7")]。
    """
    seen = set()
    out: List[tuple] = []
    for L in orig_lags:
        actual = max(L, group_lag_days)
        if actual in seen:
            continue
        seen.add(actual)
        out.append((actual, f"dm{actual}"))
    return out


def _short_name(col: str) -> str:
    return (col
            .replace("price_dayahead_", "p_")
            .replace("price_realtime_", "rt_")
            .replace("market_clearing_", "mcp_")
            .replace("realtime_clearing_", "rtc_")
            .replace("reliability_clearing_", "rel_")
            .replace("_forecast", "_fcst")
            .replace("_boundary", "_bnd")
            .replace("_actual", "_act")
            .replace("_汇总", "")
            .replace("_江南", "_jn")
            .replace("_江北", "_jb")
            .replace("_华东", "")
            .replace("_final", ""))


# ── 1. Boundary 特征（lag0） ──────────────────────────────────

def _add_boundary_features(row: Dict, day: pd.DataFrame, cfg: MarketConfig) -> None:
    available = [c for c in cfg.boundary_cols if c in day.columns]
    re_cols, load_col = [], None

    for col in available:
        short = _short_name(col)
        vals = day[col].values
        row[f"bnd_{short}"] = vals
        row[f"bnd_{short}_diff"] = np.diff(vals, prepend=vals[0])
        row[f"bnd_{short}_day_mean"] = np.nanmean(vals)
        row[f"bnd_{short}_day_std"] = np.nanstd(vals)
        row[f"bnd_{short}_rank"] = pd.Series(vals).rank(pct=True).values

        cn = col.lower()
        if "load" in cn and "fcst" in short or "load" in cn and "forecast" in col:
            load_col = col
        if any(k in cn for k in ["wind", "solar", "pv", "renewable"]):
            re_cols.append(col)

    if load_col and re_cols:
        load_v = day[load_col].values
        re_total = sum(day[c].values for c in re_cols)
        net_load = load_v - re_total
        row["net_load"] = net_load
        row["net_load_day_mean"] = np.nanmean(net_load)
        row["net_load_day_std"] = np.nanstd(net_load)
        row["net_load_rank"] = pd.Series(net_load).rank(pct=True).values
        row["re_penetration"] = np.where(load_v > 0, re_total / load_v, 0.0)
        row["re_total"] = re_total
        row["re_total_day_mean"] = np.nanmean(re_total)


# ── 2. 历史日前价格（按 price_lag_days clamp） ────────────────

def _add_price_lag_features(
    row: Dict, df_h: pd.DataFrame, dates, i: int, cfg: MarketConfig,
    step_idx: np.ndarray, steps_per_day: int,
) -> None:
    for lag_d, tag in _clamp_lags([1, 2, 7], cfg.price_lag_days):
        if i < lag_d:
            continue
        prev = _get_day(df_h, dates[i - lag_d], steps_per_day)
        if prev is None:
            continue
        for col in cfg.price_cols:
            if col not in prev.columns:
                continue
            short = _short_name(col)
            vals = prev[col].values
            row[f"{short}_{tag}_mean"] = np.nanmean(vals)
            row[f"{short}_{tag}_std"] = np.nanstd(vals)
            row[f"{short}_{tag}_min"] = np.nanmin(vals)
            row[f"{short}_{tag}_max"] = np.nanmax(vals)
            row[f"{short}_{tag}_same_hh"] = vals[step_idx]
            row[f"{short}_{tag}_same_hh_diff"] = vals[step_idx] - np.nanmean(vals)


# ── 3. 历史实时价格（按 realtime_lag_days clamp） ─────────────

def _add_realtime_lag_features(
    row: Dict, df_h: pd.DataFrame, dates, i: int, cfg: MarketConfig,
    step_idx: np.ndarray, steps_per_day: int,
) -> None:
    if not cfg.realtime_cols:
        return
    for lag_d, tag in _clamp_lags([1, 2], cfg.realtime_lag_days):
        if i < lag_d:
            continue
        prev = _get_day(df_h, dates[i - lag_d], steps_per_day)
        if prev is None:
            continue
        for col in cfg.realtime_cols:
            if col not in prev.columns:
                continue
            short = _short_name(col)
            vals = prev[col].values
            row[f"{short}_{tag}_mean"] = np.nanmean(vals)
            row[f"{short}_{tag}_std"] = np.nanstd(vals)
            row[f"{short}_{tag}_min"] = np.nanmin(vals)
            row[f"{short}_{tag}_max"] = np.nanmax(vals)
            row[f"{short}_{tag}_same_hh"] = vals[step_idx]
            row[f"{short}_{tag}_floor_rate"] = (vals <= cfg.floor_price).mean()
            row[f"{short}_{tag}_spike_rate"] = (vals >= cfg.floor_price * 8).mean()


# ── 4. 日前-实时价差（lag = max(price, realtime)） ───────────

def _add_spread_features(
    row: Dict, df_h: pd.DataFrame, dates, i: int, cfg: MarketConfig,
    step_idx: np.ndarray, steps_per_day: int,
) -> None:
    if not cfg.realtime_cols or not cfg.price_cols:
        return
    lag_d = max(cfg.price_lag_days, cfg.realtime_lag_days, 1)
    if i < lag_d:
        return
    prev = _get_day(df_h, dates[i - lag_d], steps_per_day)
    if prev is None:
        return

    tag = f"dm{lag_d}"
    for da_col, rt_col in zip(cfg.price_cols, cfg.realtime_cols):
        if da_col not in prev.columns or rt_col not in prev.columns:
            continue
        spread = prev[da_col].values - prev[rt_col].values
        label = _short_name(da_col).split("_")[1] if "_" in _short_name(da_col) else "spread"
        row[f"{tag}_spread_{label}_mean"] = np.mean(spread)
        row[f"{tag}_spread_{label}_std"] = np.std(spread)
        row[f"{tag}_spread_{label}_abs_mean"] = np.mean(np.abs(spread))
        row[f"{tag}_spread_{label}_same_hh"] = spread[step_idx]


# ── 5. 历史实际运行值（lag = actual_lag_days） ───────────────

def _add_actual_features(
    row: Dict, df_h: pd.DataFrame, dates, i: int, cfg: MarketConfig,
    step_idx: np.ndarray, steps_per_day: int,
) -> None:
    lag_d = max(cfg.actual_lag_days, 1)
    if i < lag_d:
        return
    prev = _get_day(df_h, dates[i - lag_d], steps_per_day)
    if prev is None:
        return

    tag = f"dm{lag_d}"
    re_cols_actual = []
    load_actual_vals = None

    for orig in cfg.actual_cols:
        if orig not in prev.columns:
            continue
        short = _short_name(orig)
        vals = prev[orig].values
        row[f"{short}_{tag}_mean"] = np.nanmean(vals)
        row[f"{short}_{tag}_std"] = np.nanstd(vals)
        row[f"{short}_{tag}_same_hh"] = vals[step_idx]

        if "load" in orig.lower():
            load_actual_vals = vals
        if any(k in orig.lower() for k in ["wind", "solar", "pv", "renewable"]):
            re_cols_actual.append(vals)

    if re_cols_actual:
        re_actual = sum(re_cols_actual)
        row[f"re_actual_{tag}_mean"] = np.nanmean(re_actual)
        if load_actual_vals is not None:
            net = load_actual_vals - re_actual
            row[f"net_load_actual_{tag}_mean"] = np.nanmean(net)
            row[f"net_load_actual_{tag}_min"] = np.nanmin(net)
            row[f"net_load_actual_{tag}_same_hh"] = net[step_idx]
            row[f"re_penetration_actual_{tag}"] = np.where(
                load_actual_vals > 0, re_actual / load_actual_vals, 0.0
            ).mean()

    if load_actual_vals is not None:
        bnd_load = [c for c in cfg.boundary_cols if "load" in c.lower()]
        if bnd_load:
            bnd_col = bnd_load[0]
            if bnd_col in prev.columns:
                row[f"load_fcst_err_{tag}_mean"] = np.mean(prev[bnd_col].values - load_actual_vals)

    re_bnd_cols = [c for c in cfg.boundary_cols
                   if any(k in c.lower() for k in ["wind", "solar", "pv", "renewable"])]
    if re_bnd_cols and re_cols_actual:
        re_actual = sum(re_cols_actual)
        re_bnd = sum(prev[c].values for c in re_bnd_cols if c in prev.columns)
        if isinstance(re_bnd, np.ndarray):
            row[f"re_fcst_err_{tag}_mean"] = np.mean(re_bnd - re_actual)
            row[f"re_fcst_err_{tag}_std"] = np.std(re_bnd - re_actual)


# ── 6. 地板价结构（lag = target_lag_days） ───────────────────

def _add_floor_stats(
    row: Dict, df_h: pd.DataFrame, dates, i: int, cfg: MarketConfig,
    steps_per_day: int,
) -> None:
    lag_d = max(cfg.target_lag_days, 1)
    if i < lag_d or cfg.target_col not in df_h.columns:
        return
    prev = _get_day(df_h, dates[i - lag_d], steps_per_day)
    if prev is None or cfg.target_col not in prev.columns:
        return
    tag = f"dm{lag_d}"
    vals = prev[cfg.target_col].values
    row[f"{tag}_floor_rate"] = (vals <= cfg.floor_price).mean()
    row[f"{tag}_floor_count"] = (vals <= cfg.floor_price).sum()
    row[f"{tag}_spike_rate"] = (vals >= cfg.floor_price * 8).mean()


# ── 7. 近3天/7天趋势与波动率（窗口 ending at D - target_lag） ──

def _add_trend_features(
    row: Dict, df_h: pd.DataFrame, dates, i: int, cfg: MarketConfig,
    steps_per_day: int,
) -> None:
    """趋势/波动取自 [D - target_lag - n + 1, D - target_lag] 区间，
    确保所有历史 target 都已可见。"""
    if cfg.target_col not in df_h.columns:
        return
    lag_d = max(cfg.target_lag_days, 1)

    # 近 3 天：dates[i - lag - 2 .. i - lag]
    end_idx = i - lag_d  # 含
    start_idx = end_idx - 2
    if start_idx >= 0:
        recent = dates[start_idx:end_idx + 1]
        means = []
        for d in recent:
            day = _get_day(df_h, d, steps_per_day)
            if day is not None and cfg.target_col in day.columns:
                means.append(day[cfg.target_col].mean())
        if len(means) > 1:
            row["target_3d_avg"] = np.mean(means)
            row["target_3d_trend"] = means[-1] - means[0]

    # 近 7 天：dates[i - lag - 6 .. i - lag]
    start_idx = end_idx - 6
    if start_idx >= 0:
        week_means = []
        for j in range(7):
            day = _get_day(df_h, dates[start_idx + j], steps_per_day)
            if day is not None and cfg.target_col in day.columns:
                week_means.append(day[cfg.target_col].mean())
        if len(week_means) == 7:
            x = np.arange(7, dtype=float)
            slope = np.polyfit(x, week_means, 1)[0]
            row["target_7d_trend"] = slope
            row["target_7d_vol"] = np.std(week_means)


# ── 8. 同时段历史地板价频率（近 7 天，ending at D - target_lag） ──

def _add_hourly_floor_rate(
    row: Dict, df_h: pd.DataFrame, dates, i: int, cfg: MarketConfig,
    hour_in_day: np.ndarray, steps_per_day: int,
) -> None:
    """同小时历史地板价率：按 hour 0..23 聚合，每个 step 取对应小时。"""
    if cfg.target_col not in df_h.columns:
        return
    lag_d = max(cfg.target_lag_days, 1)
    end_idx = i - lag_d
    start_idx = end_idx - 6
    if start_idx < 0:
        return
    hist_dates = dates[start_idx:end_idx + 1]
    rates_by_hour: Dict[int, List[float]] = {}
    for d in hist_dates:
        day = _get_day(df_h, d, steps_per_day)
        if day is None or cfg.target_col not in day.columns:
            continue
        for h in range(24):
            mask_h = day.index.hour == h
            if mask_h.any():
                hourly_min = day.loc[mask_h, cfg.target_col].min()
                rates_by_hour.setdefault(h, []).append(
                    1.0 if hourly_min <= cfg.floor_price else 0.0,
                )

    if rates_by_hour:
        row["hh_floor_rate_7d"] = np.array([
            np.mean(rates_by_hour.get(h, [0.0])) for h in hour_in_day
        ])


# ── 9. 日历特征 ──────────────────────────────────────────────

def _add_calendar(
    row: Dict, d, hour_in_day: np.ndarray, step_idx: np.ndarray, freq: str,
) -> None:
    dt = pd.Timestamp(d)
    row["dow"] = dt.dayofweek
    row["month"] = dt.month
    row["is_weekend"] = int(dt.dayofweek >= 5)
    row["hour_sin"] = np.sin(2 * np.pi * hour_in_day / 24)
    row["hour_cos"] = np.cos(2 * np.pi * hour_in_day / 24)
    row["dow_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
    row["dow_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)
    if freq == "15min":
        n_steps = len(step_idx)
        row["step_sin"] = np.sin(2 * np.pi * step_idx / n_steps)
        row["step_cos"] = np.cos(2 * np.pi * step_idx / n_steps)
