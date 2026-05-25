"""Conv2D-MultiTask — 数据准备与 Dataset。

按日切成 96×15min，构建 lag0/lag1/lag2 + 时间编码张量；
按 (date, slot) 构建样本，输入形状 (C, H_SLOTS, LOOKBACK_DAYS)：
  - 1h 模式：H_SLOTS = (CONTEXT_BEFORE + 1 + CONTEXT_AFTER) * 4
  - 15min 模式：H_SLOTS = SLOTS_BEFORE + 1 + SLOTS_AFTER
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    CONTEXT_AFTER,
    CONTEXT_BEFORE,
    LOOKBACK_DAYS,
    MarketConfig,
    N_TIME_ENC,
    SLOTS_AFTER,
    SLOTS_BEFORE,
    SLOTS_PER_DAY,
    SLOTS_PER_HOUR,
)

logger = logging.getLogger(__name__)


def build_daily_arrays(
    df_15min: pd.DataFrame, cfg: MarketConfig, freq: str = "1h",
) -> Tuple[List, Dict, Dict, Dict, Dict, int, int]:
    """从 15min 长表构建按日切片字典。

    Returns:
        valid_dates: 有完整 target 的日期列表
        day_lag0: {date: (96, C_LAG0_RAW + N_TIME_ENC) ndarray}
        day_lag1: {date: (96, C_LAG1) ndarray}
        day_lag2: {date: (96, C_LAG2) ndarray}
        day_targets: {date: (steps_per_day,) ndarray}  (steps_per_day=24 for 1h, 96 for 15min)
        c_total: 总通道数 = len(lag0) + N_TIME_ENC + len(lag1) + len(lag2)
        steps_per_day: 24 或 96
    """
    df = df_15min.sort_index()
    start = df.index.min().normalize().date()
    end = df.index.max().date()
    date_range = pd.date_range(start, end, freq="D")

    lag0_present = [c for c in cfg.lag0_cols if c in df.columns]
    lag1_present = [c for c in cfg.lag1_cols if c in df.columns]
    lag2_present = [c for c in cfg.lag2_cols if c in df.columns]
    missing_lag0 = set(cfg.lag0_cols) - set(lag0_present)
    missing_lag1 = set(cfg.lag1_cols) - set(lag1_present)
    missing_lag2 = set(cfg.lag2_cols) - set(lag2_present)
    if missing_lag0 or missing_lag1 or missing_lag2:
        logger.warning(
            "%s: 缺失列 lag0=%s lag1=%s lag2=%s",
            cfg.market_id, missing_lag0, missing_lag1, missing_lag2,
        )

    c_lag0 = len(lag0_present) + N_TIME_ENC
    c_lag1 = len(lag1_present)
    c_lag2 = len(lag2_present)
    c_total = c_lag0 + c_lag1 + c_lag2
    steps_per_day = SLOTS_PER_DAY if freq == "15min" else 24

    day_lag0: Dict = {}
    day_lag1: Dict = {}
    day_lag2: Dict = {}
    day_targets: Dict = {}
    valid: List = []

    for d_ts in date_range:
        d = d_ts.date()
        grid = pd.date_range(pd.Timestamp(d), periods=SLOTS_PER_DAY, freq="15min")
        raw = df.reindex(grid).copy()
        if raw.isna().to_numpy().all():
            continue

        l0 = (raw[lag0_present].values.astype(np.float32)
              if lag0_present else np.zeros((SLOTS_PER_DAY, 0), dtype=np.float32))
        steps = np.arange(SLOTS_PER_DAY, dtype=np.float32)
        dow = float(pd.Timestamp(d).dayofweek)
        te = np.column_stack([
            np.sin(2 * np.pi * steps / SLOTS_PER_DAY),
            np.cos(2 * np.pi * steps / SLOTS_PER_DAY),
            np.full(SLOTS_PER_DAY, np.sin(2 * np.pi * dow / 7), dtype=np.float32),
            np.full(SLOTS_PER_DAY, np.cos(2 * np.pi * dow / 7), dtype=np.float32),
        ])
        day_lag0[d] = np.concatenate([l0, te], axis=1).astype(np.float32)
        day_lag1[d] = (raw[lag1_present].values.astype(np.float32)
                       if lag1_present else np.zeros((SLOTS_PER_DAY, 0), dtype=np.float32))
        day_lag2[d] = (raw[lag2_present].values.astype(np.float32)
                       if lag2_present else np.zeros((SLOTS_PER_DAY, 0), dtype=np.float32))

        if cfg.target_col not in raw.columns:
            continue
        tgt_96 = raw[cfg.target_col].values.astype(np.float32)

        if freq == "15min":
            if np.isfinite(tgt_96).all():
                day_targets[d] = tgt_96
                valid.append(d)
        else:
            hourly_y = tgt_96.reshape(24, SLOTS_PER_HOUR).mean(axis=1).astype(np.float32)
            if np.isfinite(hourly_y).all():
                day_targets[d] = hourly_y
                valid.append(d)

    valid = sorted(valid)
    logger.info(
        "%s [%s]: %d 个日历日，%d 天有有效 target，C_total=%d (lag0=%d+%d_te, lag1=%d, lag2=%d)",
        cfg.market_id, freq, len(day_lag0), len(valid),
        c_total, len(lag0_present), N_TIME_ENC, c_lag1, c_lag2,
    )
    return valid, day_lag0, day_lag1, day_lag2, day_targets, c_total, steps_per_day


def _hour_four_slots(arr: np.ndarray, hh: int) -> np.ndarray:
    s = SLOTS_PER_HOUR * hh
    return arr[s:s + SLOTS_PER_HOUR].copy()


def _get_hour_slots(
    day_arrays: Dict, d, h: int,
    ctx_before: int = CONTEXT_BEFORE,
    ctx_after: int = CONTEXT_AFTER,
) -> np.ndarray:
    """日 d 第 h 小时上下文 [h-ctx_before, h+ctx_after] 的 (n_slots, C) 切片。"""
    n_slots = (ctx_before + 1 + ctx_after) * SLOTS_PER_HOUR
    arr = day_arrays[d]
    c = arr.shape[1]

    start_slot = (h - ctx_before) * SLOTS_PER_HOUR
    end_slot = (h + ctx_after + 1) * SLOTS_PER_HOUR

    if 0 <= start_slot and end_slot <= SLOTS_PER_DAY:
        return arr[start_slot:end_slot]

    result = np.zeros((n_slots, c), dtype=np.float32)
    out_idx = 0
    for hh_ in range(h - ctx_before, h + ctx_after + 1):
        cur_d = d
        cur_h = hh_
        if cur_h < 0:
            cur_d = (pd.Timestamp(d) - pd.Timedelta(days=1)).date()
            cur_h += 24
        elif cur_h >= 24:
            cur_d = (pd.Timestamp(d) + pd.Timedelta(days=1)).date()
            cur_h -= 24
        if cur_d in day_arrays:
            result[out_idx:out_idx + SLOTS_PER_HOUR] = _hour_four_slots(day_arrays[cur_d], cur_h)
        else:
            result[out_idx:out_idx + SLOTS_PER_HOUR] = (
                arr[0:SLOTS_PER_HOUR] if hh_ < 0
                else arr[SLOTS_PER_DAY - SLOTS_PER_HOUR:SLOTS_PER_DAY]
            )
        out_idx += SLOTS_PER_HOUR
    return result


def _get_context_slots(day_arrays: Dict, center_d, slot_idx: int) -> np.ndarray:
    """15min 模式：中心 slot_idx ∈ [0, 95] 上下文 [slot-BEFORE, slot+AFTER]。"""
    window = SLOTS_BEFORE + 1 + SLOTS_AFTER
    arr0 = day_arrays[center_d]
    c = arr0.shape[1]
    out = np.zeros((window, c), dtype=np.float32)

    for j, rel in enumerate(range(-SLOTS_BEFORE, SLOTS_AFTER + 1)):
        idx = slot_idx + rel
        cur_d = center_d
        while idx < 0:
            cur_d = (pd.Timestamp(cur_d) - pd.Timedelta(days=1)).date()
            idx += SLOTS_PER_DAY
        while idx >= SLOTS_PER_DAY:
            cur_d = (pd.Timestamp(cur_d) + pd.Timedelta(days=1)).date()
            idx -= SLOTS_PER_DAY
        if cur_d in day_arrays:
            out[j] = day_arrays[cur_d][idx]
    return out


def compute_norm(
    day_lag0: Dict, day_lag1: Dict, day_lag2: Dict, train_days: List,
) -> Tuple[np.ndarray, np.ndarray]:
    """训练集均值方差归一化参数。"""
    rows = []
    for d in train_days:
        if d in day_lag0 and d in day_lag1 and d in day_lag2:
            row = np.concatenate([day_lag0[d], day_lag1[d], day_lag2[d]], axis=1)
            rows.append(row)
    stack = np.concatenate(rows, axis=0)
    mean = np.nanmean(stack, axis=0).astype(np.float32)
    std = np.nanstd(stack, axis=0).astype(np.float32) + 1e-8
    return mean, std


class Conv2dDataset(Dataset):
    """按 (date, slot/hour) 一个样本 → (C, H_SLOTS, LOOKBACK_DAYS) + 归一化 target + 方向标签。"""

    def __init__(
        self,
        sample_dates: List,
        day_lag0: Dict, day_lag1: Dict, day_lag2: Dict, day_targets: Dict,
        norm_mean: np.ndarray, norm_std: np.ndarray,
        y_mean: float, y_std: float,
        c_total: int, steps_per_day: int, freq: str,
    ):
        self.freq = freq
        self.steps_per_day = steps_per_day
        self.items: List[Tuple[np.ndarray, float, int]] = []
        self.meta: List[Tuple[object, int]] = []

        a0 = set(day_lag0.keys())
        a1 = set(day_lag1.keys())
        a2 = set(day_lag2.keys())

        for d in sample_dates:
            if d not in day_targets:
                continue
            dates0 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS - 1, -1, -1)]
            dates1 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS, 0, -1)]
            dates2 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS + 1, 1, -1)]

            if not (all(dd in a0 for dd in dates0)
                    and all(dd in a1 for dd in dates1)
                    and all(dd in a2 for dd in dates2)):
                continue

            d_prev = (pd.Timestamp(d) - pd.Timedelta(days=1)).date()

            for idx in range(steps_per_day):
                layers = []
                for k in range(LOOKBACK_DAYS):
                    if freq == "15min":
                        s0 = _get_context_slots(day_lag0, dates0[k], idx)
                        s1 = _get_context_slots(day_lag1, dates1[k], idx)
                        s2 = _get_context_slots(day_lag2, dates2[k], idx)
                    else:
                        s0 = _get_hour_slots(day_lag0, dates0[k], idx)
                        s1 = _get_hour_slots(day_lag1, dates1[k], idx)
                        s2 = _get_hour_slots(day_lag2, dates2[k], idx)
                    layers.append(np.concatenate([s0, s1, s2], axis=1))

                grid = np.stack(layers, axis=-1).transpose(1, 0, 2)  # (C, H_SLOTS, LOOKBACK)
                grid = np.nan_to_num(grid, nan=0.0)
                grid = ((grid - norm_mean.reshape(c_total, 1, 1))
                        / norm_std.reshape(c_total, 1, 1)).astype(np.float32)

                tgt = np.float32((day_targets[d][idx] - y_mean) / y_std)

                if idx > 0:
                    diff = day_targets[d][idx] - day_targets[d][idx - 1]
                elif d_prev in day_targets:
                    diff = day_targets[d][0] - day_targets[d_prev][-1]
                else:
                    diff = 0.0
                dir_label = 2 if diff > 0 else (0 if diff < 0 else 1)

                self.items.append((grid, tgt, dir_label))
                self.meta.append((d, idx))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        grid, tgt, dl = self.items[idx]
        return (
            torch.from_numpy(grid),
            torch.tensor(tgt, dtype=torch.float32),
            torch.tensor(dl, dtype=torch.long),
        )
