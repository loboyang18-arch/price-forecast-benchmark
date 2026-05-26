"""Conv2D-MultiTask — 架构常量 + group→stream 映射。

特征列名 / target / alt_targets 全部由 ``pfbench.feature_registry`` 从
``config/markets/<id>.yaml`` 读取，本文件不再硬写 per-market 列表。

Conv2D 的特殊之处：它使用**连续 7 天回看窗口**，而不是离散 lag shift。所以这里
把 feature_registry 的 7 个语义类别（BOUNDARY / CLEARING_DA / ACTUAL 等）按
"时间相位"映射到 3 个 stream：

  - ``STREAM_BOUNDARY``：D 日已知的事前数据 → 7 天窗口为 [D-6, D]
  - ``STREAM_HISTORY``：D-1 末已知的历史价格/出清 → 7 天窗口为 [D-7, D-1]
  - ``STREAM_ACTUAL``：D-2 末已知的实际运行值 → 7 天窗口为 [D-8, D-2]

这样保证整个张量沿着「目标日 D」对齐，且不引入未来信息。
"""
from __future__ import annotations

from typing import Tuple

# ── feature_registry 类别 → conv2d 3-stream 映射 ─────────────
# 每个 stream 对应「时间相位」相同的若干 feature_registry 类别。
STREAM_BOUNDARY: Tuple[str, ...] = ("BOUNDARY", "BOUNDARY_CLEARED", "WEATHER")
STREAM_HISTORY:  Tuple[str, ...] = ("CLEARING_DA", "CLEARING_RT")
STREAM_ACTUAL:   Tuple[str, ...] = ("ACTUAL",)

# 各 stream 的"距离今日的起点偏移天数"（与 _build_daily_arrays 中 LOOKBACK 配对）
STREAM_DAY_OFFSET = {
    "boundary": 0,    # 7 天回看 = [D-6, D]
    "history":  1,    # 7 天回看 = [D-7, D-1]
    "actual":   2,    # 7 天回看 = [D-8, D-2]
}

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
