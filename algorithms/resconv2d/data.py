"""ResConv2D — 数据准备与 Dataset。

张量构建逻辑与 ``algorithms.conv2d_multitask.data`` 完全一致，直接复用底层函数。
唯一新增点：Δ价目标 —— 邻时段差，首时段 anchor 取前一日末时段。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from algorithms.conv2d_multitask.data import (
    _get_context_slots,
    _get_hour_slots,
    build_daily_arrays,
    compute_norm,
)

from .config import (
    CONTEXT_AFTER,
    CONTEXT_BEFORE,
    LOOKBACK_DAYS,
)

logger = logging.getLogger(__name__)


def build_delta_targets(day_targets: Dict) -> Dict:
    """对每一天构造 (steps_per_day,) Δ价向量。

    delta[0] = y[0] - y_prev_last       （anchor=前一日末时段；没有前一日则 anchor=y[0]）
    delta[t] = y[t] - y[t-1]            （t ≥ 1）
    """
    sorted_days = sorted(day_targets.keys())
    out: Dict = {}
    for i, d in enumerate(sorted_days):
        abs_vals = day_targets[d]
        prev_d = sorted_days[i - 1] if i > 0 else None
        if prev_d is not None and prev_d in day_targets:
            anchor = float(day_targets[prev_d][-1])
        else:
            anchor = float(abs_vals[0])
        delta = np.empty_like(abs_vals)
        delta[0] = abs_vals[0] - anchor
        delta[1:] = abs_vals[1:] - abs_vals[:-1]
        out[d] = delta.astype(np.float32)
    return out


class ResConv2dDataset(Dataset):
    """(date, slot/hour) → (C, H_SLOTS, LOOKBACK_DAYS) + tgt_norm + delta_norm。"""

    def __init__(
        self,
        sample_dates: List,
        day_boundary: Dict, day_history: Dict, day_actual: Dict,
        day_targets: Dict, day_delta_targets: Dict,
        norm_mean: np.ndarray, norm_std: np.ndarray,
        y_mean: float, y_std: float,
        delta_y_mean: float, delta_y_std: float,
        c_total: int, steps_per_day: int, freq: str,
    ):
        self.freq = freq
        self.steps_per_day = steps_per_day
        self.items: List[Tuple[np.ndarray, float, float]] = []
        self.meta: List[Tuple[object, int]] = []

        a_b = set(day_boundary.keys())
        a_h = set(day_history.keys())
        a_a = set(day_actual.keys())

        for d in sample_dates:
            if d not in day_targets or d not in day_delta_targets:
                continue
            # 5 lag-bucket：三 stream 统一回看 [D-6, D]
            dates_lb = [(pd.Timestamp(d) - pd.Timedelta(days=k)).date()
                        for k in range(LOOKBACK_DAYS - 1, -1, -1)]

            if not (all(dd in a_b for dd in dates_lb)
                    and all(dd in a_h for dd in dates_lb)
                    and all(dd in a_a for dd in dates_lb)):
                continue

            for idx in range(steps_per_day):
                layers = []
                for k in range(LOOKBACK_DAYS):
                    if freq == "15min":
                        s0 = _get_context_slots(day_boundary, dates_lb[k], idx)
                        s1 = _get_context_slots(day_history, dates_lb[k], idx)
                        s2 = _get_context_slots(day_actual, dates_lb[k], idx)
                    else:
                        s0 = _get_hour_slots(
                            day_boundary, dates_lb[k], idx,
                            ctx_before=CONTEXT_BEFORE, ctx_after=CONTEXT_AFTER,
                        )
                        s1 = _get_hour_slots(
                            day_history, dates_lb[k], idx,
                            ctx_before=CONTEXT_BEFORE, ctx_after=CONTEXT_AFTER,
                        )
                        s2 = _get_hour_slots(
                            day_actual, dates_lb[k], idx,
                            ctx_before=CONTEXT_BEFORE, ctx_after=CONTEXT_AFTER,
                        )
                    layers.append(np.concatenate([s0, s1, s2], axis=1))

                grid = np.stack(layers, axis=-1).transpose(1, 0, 2)
                grid = np.nan_to_num(grid, nan=0.0)
                grid = ((grid - norm_mean.reshape(c_total, 1, 1))
                        / norm_std.reshape(c_total, 1, 1)).astype(np.float32)

                tgt = np.float32((day_targets[d][idx] - y_mean) / y_std)
                dtgt = np.float32(
                    (day_delta_targets[d][idx] - delta_y_mean) / delta_y_std,
                )

                self.items.append((grid, tgt, dtgt))
                self.meta.append((d, idx))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        grid, tgt, dtgt = self.items[idx]
        return (
            torch.from_numpy(grid),
            torch.tensor(tgt, dtype=torch.float32),
            torch.tensor(dtgt, dtype=torch.float32),
        )


__all__ = [
    "ResConv2dDataset",
    "build_daily_arrays",
    "build_delta_targets",
    "compute_norm",
]
