"""LightGBM-TwoStage — 市场特征视图（由 Feature Registry 派生）。

5 lag-bucket 设计下，每个 "视图" 携带其所属 group 的 ``window_lag_days``，
features.py 内部按此决定 ``dates[i - lag_d]`` 偏移，避免 v1 老路径下
"日前出清价被当作 D-1 历史"导致的数据泄漏。

视图 → group 映射（``*_lag_days`` 从对应 group 的 window_lag_days 读取）：
    boundary_cols  ← BOUNDARY + BOUNDARY_CLEARED + WEATHER + BOUNDARY_DM1
                     （boundary_lag_days 取 max，通常 0 或 1）
    price_cols     ← CLEARING_DA                              (price_lag_days)
    realtime_cols  ← CLEARING_RT                              (realtime_lag_days)
    actual_cols    ← ACTUAL                                   (actual_lag_days)
    target         ← 所属 group                                (target_lag_days)

历史价多档 lag（dm1/dm2/dm7 等命名）改造：``dm{k}`` 表示 D-k 天的实际 lag，
而非原代码硬编码的 "近 1 天"——例如 v2 内蒙 target_lag_days=4，则
``_add_floor_stats`` 用 ``dates[i - 4]`` 而非 ``dates[i - 1]``。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pfbench.feature_registry import ResolvedSpec

BOUNDARY_GROUPS = ("BOUNDARY", "BOUNDARY_CLEARED", "WEATHER", "BOUNDARY_DM1")
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
    boundary_lag_days: int       # 取所有 boundary group 的 max（D 日可见取 0，D-1 取 1）
    price_lag_days: int          # CLEARING_DA group window_lag_days
    realtime_lag_days: int       # CLEARING_RT group window_lag_days
    actual_lag_days: int         # ACTUAL group window_lag_days
    target_lag_days: int         # target 所属 group window_lag_days
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
                if g is not None:
                    cols.extend(g.cols)
            return cols

        def _max_lag(group_names) -> int:
            lags = [resolved.groups[n].window_lag_days
                    for n in group_names if n in resolved.groups]
            return max(lags) if lags else 0

        # 注意：boundary 视图同时容纳 lag0 BOUNDARY 与 lag1d BOUNDARY_DM1，
        # 但 features._add_boundary_features 不做 shift，所以两者必须 lag 相同。
        # 仅当所有 boundary group 的 window_lag_days 一致时才合并，否则只保留
        # lag=0 的列以避免泄漏。
        boundary_groups_kept = [n for n in BOUNDARY_GROUPS
                                if n in resolved.groups
                                and resolved.groups[n].window_lag_days == 0]
        boundary_cols = _gather(boundary_groups_kept)

        target_lag_days = 0
        for g in resolved.groups.values():
            if resolved.target in g.cols:
                target_lag_days = g.window_lag_days
                break

        return cls(
            market_id=resolved.market_id,
            target_col=resolved.target,
            boundary_cols=boundary_cols,
            price_cols=_gather(PRICE_GROUPS),
            realtime_cols=_gather(REALTIME_GROUPS),
            actual_cols=_gather(ACTUAL_GROUPS),
            boundary_lag_days=0,  # 仅保留 lag0 列
            price_lag_days=_max_lag(PRICE_GROUPS),
            realtime_lag_days=_max_lag(REALTIME_GROUPS),
            actual_lag_days=_max_lag(ACTUAL_GROUPS),
            target_lag_days=target_lag_days,
        )
