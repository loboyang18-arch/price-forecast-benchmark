"""LightGBM-Baseline — 市场特征视图（由 Feature Registry 派生）。

本算法不再 hardcode 特征列。所有列名通过 ``pfbench.feature_registry`` 从
``config/markets/<market>.yaml`` 的 ``features`` 块解析得到，运行时由
``MarketConfig.from_resolved_spec()`` 派生为本算法所需的视图。

5 lag-bucket 设计下：
    每个 group 的 ``window_lag_days`` 直接决定该 group 内所有列的 shift 量。
    本算法把列按 ``window_lag_days`` 归桶到 ``cols_by_lag_days``：
      - lag 0d → 不 shift（D 日已知，如 BOUNDARY）
      - lag 1d → shift 1 day（如 BOUNDARY_DM1）
      - lag 2d → shift 2 days（如 CLEARING_RT / ACTUAL）
      - lag 3d → shift 3 days（如 CLEARING_DA）
      - lag 4d → shift 4 days（如 CLEARING_RT_NODAL）
    target 自身的 lag 由 ``target_lag_days`` 给出（target 所属 group 的 window_lag_days）。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from pfbench.feature_registry import ResolvedSpec


@dataclass
class MarketConfig:
    """LightGBM-Baseline 在 features.py 内部使用的特征视图。"""

    market_id: str
    target_col: str
    cols_by_lag_days: Dict[int, List[str]]  # {0: [...], 1: [...], 2: [...], 3: [...], 4: [...]}
    target_lag_days: int                    # target 所属 group 的 window_lag_days
    price_lag_hours: List[int] = field(default_factory=list)

    @classmethod
    def from_resolved_spec(cls, resolved: ResolvedSpec) -> "MarketConfig":
        """从 Feature Registry 的 ResolvedSpec 派生本算法所需视图。

        ``price_lag_hours``（target 多级历史价 lag，单位小时）自动从
        ``target_lag_days`` 推导：[lag, lag+1, max(lag, 7)] 取 unique 后升序。
        例如 target_lag_days=4 → [96, 120, 168]（D-4 / D-5 / D-7）。
        """
        cols_by_lag: Dict[int, List[str]] = defaultdict(list)
        target_lag_days = 0
        for g in resolved.groups.values():
            if not g.cols:
                continue
            cols_by_lag[g.window_lag_days].extend(g.cols)
            if resolved.target in g.cols:
                target_lag_days = g.window_lag_days

        base_d = max(target_lag_days, 1)
        lag_days_list = sorted({base_d, base_d + 1, max(base_d, 7)})
        price_lag_hours = [d * 24 for d in lag_days_list]

        return cls(
            market_id=resolved.market_id,
            target_col=resolved.target,
            cols_by_lag_days=dict(sorted(cols_by_lag.items())),
            target_lag_days=target_lag_days,
            price_lag_hours=price_lag_hours,
        )
