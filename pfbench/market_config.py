"""市场配置公共接口 — 所有算法从此处获取统一的测试集划分等参数。

单一数据源：config/markets/<market_id>.yaml
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml

from .paths import MARKETS_DIR


@dataclass(frozen=True)
class MarketSplit:
    """市场级的统一测试集划分，所有算法共享。"""
    market_id: str
    test_start: str
    test_end: str


def load_market_split(market_id: str) -> MarketSplit:
    """从 config/markets/<market_id>.yaml 读取统一的测试集划分。"""
    path = MARKETS_DIR / f"{market_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"市场配置不存在: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return MarketSplit(
        market_id=cfg["market_id"],
        test_start=str(cfg["test_start"]),
        test_end=str(cfg["test_end"]),
    )


_CACHE: Dict[str, MarketSplit] = {}


def get_market_split(market_id: str) -> MarketSplit:
    """带缓存的 load_market_split。"""
    if market_id not in _CACHE:
        _CACHE[market_id] = load_market_split(market_id)
    return _CACHE[market_id]
