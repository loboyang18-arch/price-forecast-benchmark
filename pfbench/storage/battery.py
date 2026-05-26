# Aligned with NM battery_cfg + run_dashboard_hongjing MILP_ENV
"""储能电站物理参数配置（红井 2h 系统为默认）。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# 红井 2h 系统默认（200 MW / 400 MWh / 1.5×日充 / η=0.91 / 280 元/MWh 补偿）
HONGJING_2H_DEFAULTS = {
    "p_max_mw": 200.0,
    "cap_mwh": 400.0,
    "max_charge_mwh": 600.0,
    "eta_rt": 0.910,
    "aux_mwh": 13.03,
    "dp_ramp_mw": 66.67,
    "l_min": 4,
    "cap_comp_per_mwh": 280.0,
}


@dataclass
class BatteryConfig:
    """与 MILP 约束一致的电池参数。"""

    p_max_mw: float = HONGJING_2H_DEFAULTS["p_max_mw"]
    cap_mwh: float = HONGJING_2H_DEFAULTS["cap_mwh"]
    max_charge_mwh: float = HONGJING_2H_DEFAULTS["max_charge_mwh"]
    eta_rt: float = HONGJING_2H_DEFAULTS["eta_rt"]
    dt: float = 0.25
    dp_ramp_mw: float = HONGJING_2H_DEFAULTS["dp_ramp_mw"]
    l_min: int = HONGJING_2H_DEFAULTS["l_min"]
    aux_mwh: float = HONGJING_2H_DEFAULTS["aux_mwh"]
    cap_comp_per_mwh: float = HONGJING_2H_DEFAULTS["cap_comp_per_mwh"]
    comp_mode: str = "in_objective"
    carry_soc: bool = False
    terminal_discount: float = 0.95
    enforce_switch_gap: bool = True
    time_limit: float = 120.0
    T: int = 96

    @property
    def eta_c(self) -> float:
        return math.sqrt(self.eta_rt)

    @property
    def eta_d(self) -> float:
        return math.sqrt(self.eta_rt)

    def milp_cap_comp(self) -> float:
        """MILP 目标中计入的容量补偿（元/MWh）。"""
        if self.comp_mode == "in_objective":
            return self.cap_comp_per_mwh
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_max_mw": self.p_max_mw,
            "cap_mwh": self.cap_mwh,
            "max_charge_mwh": self.max_charge_mwh,
            "eta_rt": self.eta_rt,
            "aux_mwh": self.aux_mwh,
            "dp_ramp_mw": self.dp_ramp_mw,
            "l_min": self.l_min,
            "cap_comp_per_mwh": self.cap_comp_per_mwh,
            "comp_mode": self.comp_mode,
            "carry_soc": self.carry_soc,
        }

    @classmethod
    def from_yaml_dict(cls, raw: dict[str, Any] | None) -> "BatteryConfig":
        if not raw:
            return cls()
        merged = {**HONGJING_2H_DEFAULTS}
        battery = raw.get("battery", raw)
        if isinstance(battery, dict):
            key_map = {
                "p_max_mw": "p_max_mw",
                "cap_mwh": "cap_mwh",
                "max_charge_mwh": "max_charge_mwh",
                "eta_rt": "eta_rt",
                "aux_mwh": "aux_mwh",
                "dp_ramp_mw": "dp_ramp_mw",
                "l_min": "l_min",
                "cap_comp_per_mwh": "cap_comp_per_mwh",
            }
            for yaml_k, attr in key_map.items():
                if yaml_k in battery:
                    merged[attr] = battery[yaml_k]
        cfg = cls(
            p_max_mw=float(merged["p_max_mw"]),
            cap_mwh=float(merged["cap_mwh"]),
            max_charge_mwh=float(merged["max_charge_mwh"]),
            eta_rt=float(merged["eta_rt"]),
            aux_mwh=float(merged["aux_mwh"]),
            dp_ramp_mw=float(merged["dp_ramp_mw"]),
            l_min=int(merged["l_min"]),
            cap_comp_per_mwh=float(merged["cap_comp_per_mwh"]),
        )
        if isinstance(battery, dict):
            if "comp_mode" in battery:
                cfg.comp_mode = str(battery["comp_mode"])
            if "carry_soc" in battery:
                cfg.carry_soc = bool(battery["carry_soc"])
            if "terminal_discount" in battery:
                cfg.terminal_discount = float(battery["terminal_discount"])
            if "enforce_switch_gap" in battery:
                cfg.enforce_switch_gap = bool(battery["enforce_switch_gap"])
            if "time_limit" in battery:
                cfg.time_limit = float(battery["time_limit"])
        if "comp_mode" in raw:
            cfg.comp_mode = str(raw["comp_mode"])
        if "carry_soc" in raw:
            cfg.carry_soc = bool(raw["carry_soc"])
        return cfg


@dataclass
class StorageMarketConfig:
    """市场级储能评估配置（来自 config/markets/*.yaml storage 节）。"""

    station_name: str = "hongjing_220kv1m"
    settlement_price_col: str = "price_hongjing_220kv1m_nodal"
    battery: BatteryConfig = field(default_factory=BatteryConfig)

    @classmethod
    def from_yaml_dict(cls, raw: dict[str, Any] | None) -> "StorageMarketConfig":
        if not raw:
            return cls()
        return cls(
            station_name=str(raw.get("station_name", "hongjing_220kv1m")),
            settlement_price_col=str(
                raw.get("settlement_price_col", "price_hongjing_220kv1m_nodal")
            ),
            battery=BatteryConfig.from_yaml_dict(raw),
        )
