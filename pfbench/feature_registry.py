"""Feature registry — 跨算法共享的特征本体。

从 ``config/markets/<market_id>.yaml`` 的 ``features`` 块读取，提供统一的查询接口：

  - 类别枚举（``FEATURE_GROUPS``）：
        BOUNDARY / BOUNDARY_CLEARED / WEATHER / CLEARING_DA / CLEARING_RT / ACTUAL
        / CALENDAR / DERIVED
  - 每市场列出每个类别下的具体列 + 是否默认 enabled
  - target_default + alt_targets
  - 每个类别的默认 lag（语义化，由 ``pfbench.lag_resolver`` 在 runtime 按 freq 解析）

算法侧用 ``FeatureSpec`` 声明想用哪几组、想覆盖哪些 lag、target 取哪个；
``resolve_columns`` 返回每个类别下"启用的列 + shift periods 列表"，
shift 由 ``pfbench.lag_resolver.lag_to_periods`` 统一计算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from .lag_resolver import lag_label_to_days, lag_labels_to_periods
from .paths import MARKETS_DIR

FEATURE_GROUPS = (
    "BOUNDARY",
    "BOUNDARY_CLEARED",
    "BOUNDARY_DM1",        # D-1 当天数据（可用性 lag=1d），如备用、平衡裕度
    "WEATHER",
    "CLEARING_DA",
    "CLEARING_RT",
    "CLEARING_RT_NODAL",   # 节点价实测（可用性 lag=4d，与一般 RT 区分）
    "ACTUAL",
    "CALENDAR",
    "DERIVED",
)


@dataclass
class FeatureGroup:
    """单个特征类别的注册信息。"""
    name: str
    enabled: bool
    cols: List[str]
    lag_labels: List[str]            # （兼容字段）算法侧用于派生 shift 副本时使用
    window_lag: str = "0"            # 数据可用性 lag 标签：'0' / '1d' / '2d' / '3d' / '4d'


@dataclass
class FeatureRegistry:
    """单市场的全部特征注册信息。"""
    market_id: str
    target_default: str
    alt_targets: List[str]
    groups: Dict[str, FeatureGroup]

    def all_default_cols(self) -> List[str]:
        """返回所有 enabled 类别下的列，按 FEATURE_GROUPS 顺序拼接。"""
        out: List[str] = []
        for g_name in FEATURE_GROUPS:
            g = self.groups.get(g_name)
            if g is None or not g.enabled:
                continue
            out.extend(g.cols)
        return out


@dataclass
class FeatureSpec:
    """算法声明它要用哪些特征 + lag。

    Attributes:
        target: 显式指定 target 列（必须在 ``alt_targets`` / target_default 之中）。
            如果 None，使用 registry.target_default。
        groups: 想启用的类别列表，例如 ``["BOUNDARY", "CLEARING_DA", "ACTUAL"]``。
            None 表示使用 registry 中 enabled=true 的所有类别。
        lag_overrides: 覆盖 yaml 默认 lag；形如
            ``{"CLEARING_DA": ["1d", "2d"], "ACTUAL": []}``。
            ``[]`` 表示该类别强制不 shift。
    """
    target: Optional[str] = None
    groups: Optional[List[str]] = None
    lag_overrides: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ResolvedGroup:
    """解析后的单类别信息。"""
    name: str
    cols: List[str]
    lag_periods: List[int]           # 已按 freq 解析为整数步数（兼容字段）
    window_lag_days: int = 0         # 可用性 lag 转整数天（用于深度模型 5 lag-bucket 取 7 天窗口）


@dataclass
class ResolvedSpec:
    """解析后的完整特征规格，可序列化到 metrics.json。"""
    market_id: str
    freq: str
    target: str
    groups: Dict[str, ResolvedGroup]

    @property
    def all_cols(self) -> List[str]:
        out: List[str] = []
        for g in self.groups.values():
            out.extend(g.cols)
        return out

    def to_dict(self) -> Dict:
        """便于持久化到 metrics.json / 实验追溯。"""
        return {
            "market_id": self.market_id,
            "freq": self.freq,
            "target": self.target,
            "groups": {
                name: {
                    "cols": g.cols,
                    "lag_periods": g.lag_periods,
                    "window_lag_days": g.window_lag_days,
                }
                for name, g in self.groups.items()
            },
        }


def _validate_group_name(name: str) -> None:
    if name not in FEATURE_GROUPS:
        raise ValueError(
            f"未知 feature group: {name!r}；可选 {list(FEATURE_GROUPS)}"
        )


_REGISTRY_CACHE: Dict[str, FeatureRegistry] = {}


def load_feature_registry(market_id: str) -> FeatureRegistry:
    """读 ``config/markets/<market_id>.yaml`` 的 ``features`` 块。"""
    if market_id in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[market_id]

    path = MARKETS_DIR / f"{market_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"市场配置不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    feats = raw.get("features")
    if feats is None:
        raise ValueError(
            f"{path} 中未定义 'features' 块；请先在 yaml 中按 schema 声明 features"
        )

    target_default = feats.get("target_default")
    if not target_default:
        raise ValueError(f"{path}: features.target_default 必填")
    alt_targets = list(feats.get("alt_targets") or [])
    if target_default not in alt_targets:
        alt_targets = [target_default] + alt_targets

    default_lags: Dict[str, List[str]] = dict(feats.get("default_lag_labels") or {})

    raw_groups = feats.get("groups") or {}
    groups: Dict[str, FeatureGroup] = {}
    for g_name in FEATURE_GROUPS:
        spec = raw_groups.get(g_name, {})
        enabled = bool(spec.get("enabled", False))
        cols = list(spec.get("cols") or [])
        lag_labels = list(spec.get("lag_labels") or default_lags.get(g_name) or [])
        window_lag = str(spec.get("window_lag", "0"))
        # 提前校验，让 yaml 错误尽早暴露
        lag_label_to_days(window_lag)
        groups[g_name] = FeatureGroup(
            name=g_name, enabled=enabled, cols=cols,
            lag_labels=lag_labels, window_lag=window_lag,
        )

    reg = FeatureRegistry(
        market_id=market_id,
        target_default=target_default,
        alt_targets=alt_targets,
        groups=groups,
    )
    _REGISTRY_CACHE[market_id] = reg
    return reg


def list_alt_targets(market_id: str) -> List[str]:
    return load_feature_registry(market_id).alt_targets


def resolve_columns(market_id: str, spec: FeatureSpec, freq: str) -> ResolvedSpec:
    """按 spec + freq 解析出每个启用类别下的列 + 步数。

    Args:
        market_id: 市场 ID
        spec: 算法侧的 FeatureSpec
        freq: ``"1h"`` 或 ``"15min"``

    Returns:
        ResolvedSpec，包含每类的列名与对应 ``shift periods``。
    """
    reg = load_feature_registry(market_id)

    if spec.target is None:
        target = reg.target_default
    else:
        if spec.target not in reg.alt_targets:
            raise ValueError(
                f"{market_id}: target={spec.target!r} 不在 alt_targets 中；"
                f"可选: {reg.alt_targets}"
            )
        target = spec.target

    if spec.groups is None:
        wanted = [name for name, g in reg.groups.items() if g.enabled]
    else:
        for name in spec.groups:
            _validate_group_name(name)
        wanted = list(spec.groups)

    for name in spec.lag_overrides:
        _validate_group_name(name)

    resolved: Dict[str, ResolvedGroup] = {}
    for name in wanted:
        g = reg.groups[name]
        if not g.cols:
            continue
        if name in spec.lag_overrides:
            lag_labels = spec.lag_overrides[name]
        else:
            lag_labels = g.lag_labels
        periods = lag_labels_to_periods(lag_labels, freq)
        window_lag_days = lag_label_to_days(g.window_lag)
        resolved[name] = ResolvedGroup(
            name=name, cols=list(g.cols),
            lag_periods=periods, window_lag_days=window_lag_days,
        )

    return ResolvedSpec(
        market_id=market_id, freq=freq, target=target, groups=resolved,
    )
