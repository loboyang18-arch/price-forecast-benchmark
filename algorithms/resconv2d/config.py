"""ResConv2D — 架构常量 + group→stream 映射。

特征列名 / target / alt_targets 全部由 ``pfbench.feature_registry`` 从
``config/markets/<id>.yaml`` 读取，本文件不再硬写 per-market 列表。

ResConv2D 和 Conv2D-MultiTask 共用 3-stream 语义、7 天回看窗口与时间相位偏移，
区别仅在：
  - CONTEXT_BEFORE=5, CONTEXT_AFTER=0（v25 原版 ctx=5+0，h_slots=24）
  - 网络为 10+ 层 ResBlock 加深版，参数量 ~2.5M
  - 副头从「方向 3 分类」改为「Δ价回归」
"""
from __future__ import annotations

from typing import Tuple

STREAM_BOUNDARY: Tuple[str, ...] = ("BOUNDARY", "BOUNDARY_CLEARED", "WEATHER")
STREAM_HISTORY: Tuple[str, ...] = ("CLEARING_DA", "CLEARING_RT")
STREAM_ACTUAL: Tuple[str, ...] = ("ACTUAL",)

STREAM_DAY_OFFSET = {
    "boundary": 0,
    "history": 1,
    "actual": 2,
}

LOOKBACK_DAYS = 7
SLOTS_PER_HOUR = 4
SLOTS_PER_DAY = 96

# v25 原版默认 ctx=5+0（h_slots=24，1h 粒度）。
CONTEXT_BEFORE = 5
CONTEXT_AFTER = 0

# 15min 模式：v25 网络池化两次后还要 conv 两次（kernel=3, padding=0），h_out=h//4 − 4 必须 ≥1。
# 因此 15min 至少需 h_slots≥20。这里给出与 conv2d 不同的默认值。
SLOTS_BEFORE = 11
SLOTS_AFTER = 8

N_TIME_ENC = 4
