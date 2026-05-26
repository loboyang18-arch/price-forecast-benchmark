"""Conv2D-MultiTask — 数据准备与 Dataset。

按日切成 96×15min，按 feature_registry 中的 ``BOUNDARY / CLEARING_* / ACTUAL`` 类别
组成 3 个 stream（事前 / 历史价 / 实际值），每个 stream 7 天回看：

  stream_boundary: 列 = BOUNDARY + BOUNDARY_CLEARED + WEATHER + 4 时间编码
  stream_history:  列 = CLEARING_DA + CLEARING_RT
  stream_actual:   列 = ACTUAL

按 (date, slot/hour) 构建样本，输入形状 (C, H_SLOTS, LOOKBACK_DAYS)：
  - 1h 模式：H_SLOTS = (CONTEXT_BEFORE + 1 + CONTEXT_AFTER) * 4 = 12
  - 15min 模式：H_SLOTS = SLOTS_BEFORE + 1 + SLOTS_AFTER = 12

时间相位（避免未来泄漏）：
  - boundary stream 取 [D-6, D]（D 日 BOUNDARY 列在 D-1 末已可知，是 lag0）
  - history  stream 取 [D-7, D-1]（D 日的历史价 → shift 1d 起）
  - actual   stream 取 [D-8, D-2]（D 日的实际值 → shift 2d 起）
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pfbench.feature_registry import ResolvedSpec

from .config import (
    CONTEXT_AFTER,
    CONTEXT_BEFORE,
    LOOKBACK_DAYS,
    N_TIME_ENC,
    SLOTS_AFTER,
    SLOTS_BEFORE,
    SLOTS_PER_DAY,
    SLOTS_PER_HOUR,
    STREAM_ACTUAL,
    STREAM_BOUNDARY,
    STREAM_DAY_OFFSET,
    STREAM_HISTORY,
)

logger = logging.getLogger(__name__)


def _resolve_stream_cols(spec: ResolvedSpec) -> Dict[str, List[str]]:
    """从 ResolvedSpec 中按 conv2d 的 3-stream 语义聚合列名。"""
    def collect(group_names):
        out = []
        for n in group_names:
            g = spec.groups.get(n)
            if g is None:
                continue
            for c in g.cols:
                if c not in out:
                    out.append(c)
        return out

    return {
        "boundary": collect(STREAM_BOUNDARY),
        "history":  collect(STREAM_HISTORY),
        "actual":   collect(STREAM_ACTUAL),
    }


def build_daily_arrays(
    df_15min: pd.DataFrame, spec: ResolvedSpec, freq: str = "1h",
) -> Tuple[List, Dict, Dict, Dict, Dict, int, int, Dict[str, List[str]]]:
    """从 15min 长表按 feature_registry 类别构建按日切片字典。

    Returns:
        valid_dates: 有完整 target 的日期列表
        day_boundary: {date: (96, C_BOUNDARY + N_TIME_ENC) ndarray}
        day_history:  {date: (96, C_HISTORY) ndarray}
        day_actual:   {date: (96, C_ACTUAL) ndarray}
        day_targets: {date: (steps_per_day,) ndarray}
        c_total: 总通道数 = C_BOUNDARY + N_TIME_ENC + C_HISTORY + C_ACTUAL
        steps_per_day: 24 或 96
        stream_cols: {"boundary": [...], "history": [...], "actual": [...]}（含 time_enc 已加在 boundary）
    """
    df = df_15min.sort_index()
    start = df.index.min().normalize().date()
    end = df.index.max().date()
    date_range = pd.date_range(start, end, freq="D")

    stream_cols = _resolve_stream_cols(spec)
    boundary_cols_present = [c for c in stream_cols["boundary"] if c in df.columns]
    history_cols_present = [c for c in stream_cols["history"] if c in df.columns]
    actual_cols_present = [c for c in stream_cols["actual"] if c in df.columns]
    target_col = spec.target

    missing = (set(stream_cols["boundary"]) | set(stream_cols["history"]) | set(stream_cols["actual"])) - set(
        boundary_cols_present + history_cols_present + actual_cols_present
    )
    if missing:
        logger.warning("%s: 数据中缺失列（已跳过）: %s", spec.market_id, sorted(missing))
    if target_col not in df.columns:
        raise ValueError(f"{spec.market_id}: target {target_col!r} 不在数据集中")

    c_boundary = len(boundary_cols_present) + N_TIME_ENC
    c_history = len(history_cols_present)
    c_actual = len(actual_cols_present)
    c_total = c_boundary + c_history + c_actual
    steps_per_day = SLOTS_PER_DAY if freq == "15min" else 24

    day_boundary: Dict = {}
    day_history: Dict = {}
    day_actual: Dict = {}
    day_targets: Dict = {}
    valid: List = []

    for d_ts in date_range:
        d = d_ts.date()
        grid = pd.date_range(pd.Timestamp(d), periods=SLOTS_PER_DAY, freq="15min")
        raw = df.reindex(grid).copy()
        if raw.isna().to_numpy().all():
            continue

        l0 = (raw[boundary_cols_present].values.astype(np.float32)
              if boundary_cols_present else np.zeros((SLOTS_PER_DAY, 0), dtype=np.float32))
        steps = np.arange(SLOTS_PER_DAY, dtype=np.float32)
        dow = float(pd.Timestamp(d).dayofweek)
        te = np.column_stack([
            np.sin(2 * np.pi * steps / SLOTS_PER_DAY),
            np.cos(2 * np.pi * steps / SLOTS_PER_DAY),
            np.full(SLOTS_PER_DAY, np.sin(2 * np.pi * dow / 7), dtype=np.float32),
            np.full(SLOTS_PER_DAY, np.cos(2 * np.pi * dow / 7), dtype=np.float32),
        ])
        day_boundary[d] = np.concatenate([l0, te], axis=1).astype(np.float32)
        day_history[d] = (raw[history_cols_present].values.astype(np.float32)
                          if history_cols_present else np.zeros((SLOTS_PER_DAY, 0), dtype=np.float32))
        day_actual[d] = (raw[actual_cols_present].values.astype(np.float32)
                         if actual_cols_present else np.zeros((SLOTS_PER_DAY, 0), dtype=np.float32))

        tgt_96 = raw[target_col].values.astype(np.float32)

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
        "%s [%s, freq=%s]: %d 日历日，%d 天有 target；通道 c_total=%d "
        "(boundary=%d+%d_te, history=%d, actual=%d)",
        spec.market_id, target_col, freq, len(day_boundary), len(valid),
        c_total, len(boundary_cols_present), N_TIME_ENC, c_history, c_actual,
    )
    resolved_stream_cols = {
        "boundary": boundary_cols_present + ["_te_slot_sin", "_te_slot_cos", "_te_dow_sin", "_te_dow_cos"],
        "history": history_cols_present,
        "actual": actual_cols_present,
    }
    return (valid, day_boundary, day_history, day_actual, day_targets,
            c_total, steps_per_day, resolved_stream_cols)


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
    day_boundary: Dict, day_history: Dict, day_actual: Dict, train_days: List,
) -> Tuple[np.ndarray, np.ndarray]:
    """训练集均值方差归一化参数。"""
    rows = []
    for d in train_days:
        if d in day_boundary and d in day_history and d in day_actual:
            row = np.concatenate([day_boundary[d], day_history[d], day_actual[d]], axis=1)
            rows.append(row)
    stack = np.concatenate(rows, axis=0)
    mean = np.nanmean(stack, axis=0).astype(np.float32)
    std = np.nanstd(stack, axis=0).astype(np.float32) + 1e-8
    return mean, std


class Conv2dDataset(Dataset):
    """按 (date, slot/hour) 一个样本 → (C, H_SLOTS, LOOKBACK_DAYS) + 归一化 target + 方向标签。

    boundary / history / actual 三个 stream 各自按相应的"时间相位"取 7 天回看窗口，
    保证 D 日预测时只使用 D-1 末（含）之前的信息。
    """

    def __init__(
        self,
        sample_dates: List,
        day_boundary: Dict, day_history: Dict, day_actual: Dict, day_targets: Dict,
        norm_mean: np.ndarray, norm_std: np.ndarray,
        y_mean: float, y_std: float,
        c_total: int, steps_per_day: int, freq: str,
    ):
        self.freq = freq
        self.steps_per_day = steps_per_day
        self.items: List[Tuple[np.ndarray, float, int]] = []
        self.meta: List[Tuple[object, int]] = []

        a_b = set(day_boundary.keys())
        a_h = set(day_history.keys())
        a_a = set(day_actual.keys())

        off_b = STREAM_DAY_OFFSET["boundary"]
        off_h = STREAM_DAY_OFFSET["history"]
        off_a = STREAM_DAY_OFFSET["actual"]

        for d in sample_dates:
            if d not in day_targets:
                continue
            # 三个 stream 的 7 天回看，按相位偏移
            dates_b = [(pd.Timestamp(d) - pd.Timedelta(days=off_b + k)).date()
                       for k in range(LOOKBACK_DAYS - 1, -1, -1)]
            dates_h = [(pd.Timestamp(d) - pd.Timedelta(days=off_h + k)).date()
                       for k in range(LOOKBACK_DAYS - 1, -1, -1)]
            dates_a = [(pd.Timestamp(d) - pd.Timedelta(days=off_a + k)).date()
                       for k in range(LOOKBACK_DAYS - 1, -1, -1)]

            if not (all(dd in a_b for dd in dates_b)
                    and all(dd in a_h for dd in dates_h)
                    and all(dd in a_a for dd in dates_a)):
                continue

            d_prev = (pd.Timestamp(d) - pd.Timedelta(days=1)).date()

            for idx in range(steps_per_day):
                layers = []
                for k in range(LOOKBACK_DAYS):
                    if freq == "15min":
                        s0 = _get_context_slots(day_boundary, dates_b[k], idx)
                        s1 = _get_context_slots(day_history, dates_h[k], idx)
                        s2 = _get_context_slots(day_actual, dates_a[k], idx)
                    else:
                        s0 = _get_hour_slots(day_boundary, dates_b[k], idx)
                        s1 = _get_hour_slots(day_history, dates_h[k], idx)
                        s2 = _get_hour_slots(day_actual, dates_a[k], idx)
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
