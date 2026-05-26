"""LightGBM-Baseline — 市场特征视图（由 Feature Registry 派生）。

本算法不再 hardcode 特征列。所有列名通过 ``pfbench.feature_registry`` 从
``config/markets/<market>.yaml`` 的 ``features`` 块解析得到，运行时由
``MarketConfig.from_resolved_spec()`` 派生为本算法所需的 lag0/1/2 视图。

类别 → lag 视图的映射：
    lag0 = BOUNDARY + BOUNDARY_CLEARED + WEATHER   (D 日已知，不 shift)
    lag1 = CLEARING_DA + CLEARING_RT               (D-1 价格)
    lag2 = ACTUAL                                  (D-2 实际)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from pfbench.feature_registry import ResolvedSpec

LAG0_GROUPS = ("BOUNDARY", "BOUNDARY_CLEARED", "WEATHER")
LAG1_GROUPS = ("CLEARING_DA", "CLEARING_RT")
LAG2_GROUPS = ("ACTUAL",)


@dataclass
class MarketConfig:
    """LightGBM-Baseline 在 features.py 内部使用的特征视图。"""

    market_id: str
    target_col: str
    lag0_cols: List[str]
    lag1_cols: List[str]
    lag2_cols: List[str]
    price_lag_hours: List[int] = field(default_factory=lambda: [24, 48, 168])

    @classmethod
    def from_resolved_spec(cls, resolved: ResolvedSpec) -> "MarketConfig":
        """从 Feature Registry 的 ResolvedSpec 派生本算法所需视图。

        ``price_lag_hours`` 反推自 CLEARING_DA 的 lag_periods + freq，保持"小时数"语义，
        使得 features.py 中 ``_add_price_lags`` 的 multiplier 逻辑无需改动。
        """
        def _gather(group_names):
            cols: List[str] = []
            for name in group_names:
                g = resolved.groups.get(name)
                if g is None:
                    continue
                cols.extend(g.cols)
            return cols

        step_minutes = 60 if resolved.freq == "1h" else 15
        da = resolved.groups.get("CLEARING_DA")
        if da is not None and da.lag_periods:
            price_lag_hours = [int(p * step_minutes / 60) for p in da.lag_periods]
        else:
            price_lag_hours = [24, 48, 168]

        return cls(
            market_id=resolved.market_id,
            target_col=resolved.target,
            lag0_cols=_gather(LAG0_GROUPS),
            lag1_cols=_gather(LAG1_GROUPS),
            lag2_cols=_gather(LAG2_GROUPS),
            price_lag_hours=price_lag_hours,
        )
