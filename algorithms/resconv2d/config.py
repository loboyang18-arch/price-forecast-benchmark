"""ResConv2D — 架构常量 + group→stream 映射。

特征列名 / target / alt_targets 全部由 ``pfbench.feature_registry`` 从
``config/markets/<id>.yaml`` 读取，本文件不再硬写 per-market 列表。

ResConv2D 和 Conv2D-MultiTask 共用 5 lag-bucket 设计、3-stream 语义、7 天回看，
区别仅在：
  - CONTEXT_BEFORE=5, CONTEXT_AFTER=0（v25 原版 ctx=5+0，h_slots=24）
  - 网络为 10+ 层 ResBlock 加深版，参数量 ~2.5M
  - 副头从「方向 3 分类」改为「Δ价回归」

stream 归属与 Conv2D 一致：每列按 feature_registry 中所属 group 的 window_lag_days 单独 shift，
shift 后所有 stream 统一从 [D-6, D] 取 7 天窗口（STREAM_DAY_OFFSET 已废弃）。
"""
from __future__ import annotations

from typing import Tuple

STREAM_BOUNDARY: Tuple[str, ...] = ("BOUNDARY", "BOUNDARY_CLEARED", "WEATHER")
STREAM_HISTORY: Tuple[str, ...] = (
    "BOUNDARY_DM1",
    "CLEARING_DA",
    "CLEARING_RT",
    "CLEARING_RT_NODAL",
)
STREAM_ACTUAL: Tuple[str, ...] = ("ACTUAL",)

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
