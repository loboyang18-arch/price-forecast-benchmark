"""LightGBM-TwoStage — 市场特征视图（由 Feature Registry 派生）。

本算法不再 hardcode 特征列。所有列名通过 ``pfbench.feature_registry`` 从
``config/markets/<market>.yaml`` 的 ``features`` 块解析得到，运行时由
``MarketConfig.from_resolved_spec()`` 派生为本算法所需的视图。

类别 → 视图映射：
    boundary_cols  = BOUNDARY + BOUNDARY_CLEARED + WEATHER   (D 日已知边界)
    price_cols     = CLEARING_DA                             (历史日前价格)
    realtime_cols  = CLEARING_RT                             (历史实时价格)
    actual_cols    = ACTUAL                                  (历史实际运行)

注意：``actual_cols`` 由 ``List[str]`` 表达原始列名；features.py 内部用
``_short_name(orig)`` 自动生成短名，不再需要手工 orig→short 映射。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pfbench.feature_registry import ResolvedSpec

BOUNDARY_GROUPS = ("BOUNDARY", "BOUNDARY_CLEARED", "WEATHER")
PRICE_GROUPS = ("CLEARING_DA",)
REALTIME_GROUPS = ("CLEARING_RT",)
ACTUAL_GROUPS = ("ACTUAL",)


@dataclass
class MarketConfig:
    """LightGBM-TwoStage 在 features.py / train.py 内部使用的特征视图。"""

    market_id: str
    target_col: str
    boundary_cols: List[str]
    price_cols: List[str]
    realtime_cols: List[str]
    actual_cols: List[str]
    floor_price: float = 50.0
    floor_pred_value: float = 30.0
    val_days: int = 7
    step_days: int = 7

    @classmethod
    def from_resolved_spec(cls, resolved: ResolvedSpec) -> "MarketConfig":
        """从 Feature Registry 的 ResolvedSpec 派生本算法所需视图。"""
        def _gather(group_names):
            cols: List[str] = []
            for name in group_names:
                g = resolved.groups.get(name)
                if g is None:
                    continue
                cols.extend(g.cols)
            return cols

        return cls(
            market_id=resolved.market_id,
            target_col=resolved.target,
            boundary_cols=_gather(BOUNDARY_GROUPS),
            price_cols=_gather(PRICE_GROUPS),
            realtime_cols=_gather(REALTIME_GROUPS),
            actual_cols=_gather(ACTUAL_GROUPS),
        )
