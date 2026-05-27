"""Conv2D-MultiTask — 架构常量 + group→stream 映射。

特征列名 / target / alt_targets 全部由 ``pfbench.feature_registry`` 从
``config/markets/<id>.yaml`` 读取，本文件不再硬写 per-market 列表。

Conv2D 的特殊之处：它使用**连续 7 天回看窗口**。新设计（5 lag-bucket）下：

  - 每个特征列按所属 group 的 ``window_lag_days`` 单独 shift（在 data.py 端完成）
  - shift 后所有 stream 统一从 [D-6, D] 取 7 天窗口（``STREAM_DAY_OFFSET`` 已废弃）
  - target 真值从**未 shift 的原始 df** 取，避免被污染

stream 仍然存在（控制网络分支归属），但只表达"特征语义分组"，不再表达"时间相位"。

stream 归属：
  - ``STREAM_BOUNDARY``：D 日已知的事前预测/标签（window_lag=0）
  - ``STREAM_HISTORY``：D-k 已知的历史/出清/参考（多种 window_lag，如 1d/2d/3d/4d）
  - ``STREAM_ACTUAL``：D-2 已知的实测（window_lag=2d）

此分组只决定网络的 3 个 stream 分支，时间相位由每列的 window_lag_days 自行决定。
"""
from __future__ import annotations

from typing import Tuple

# ── feature_registry 类别 → conv2d 3-stream 映射 ─────────────
STREAM_BOUNDARY: Tuple[str, ...] = ("BOUNDARY", "BOUNDARY_CLEARED", "WEATHER")
STREAM_HISTORY:  Tuple[str, ...] = (
    "BOUNDARY_DM1",       # lag=1d
    "CLEARING_DA",        # lag=3d
    "CLEARING_RT",        # lag=2d
    "CLEARING_RT_NODAL",  # lag=4d
)
STREAM_ACTUAL:   Tuple[str, ...] = ("ACTUAL",)

# ── Conv2D 架构参数（市场无关） ───────────────────────────────
LOOKBACK_DAYS = 7        # 回看 7 天
SLOTS_PER_HOUR = 4       # 15min 粒度
SLOTS_PER_DAY = 96
CONTEXT_BEFORE = 1       # 1h 模式：前后各 1 小时上下文
CONTEXT_AFTER = 1
SLOTS_BEFORE = 7         # 15min 模式：前 7 步 + 后 4 步
SLOTS_AFTER = 4

# 时间编码通道数（sin/cos hour + sin/cos dow）
N_TIME_ENC = 4
