# Aligned with NM strategy_milp_15min L521-550
"""储能日收益评估与 §9.4 决策指标。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .battery import BatteryConfig


def eval_day_revenue(
    c: np.ndarray,
    d: np.ndarray,
    actual_96: np.ndarray,
    cfg: BatteryConfig,
) -> dict[str, float]:
    """用真实结算价计算当日净收益（套利 + 含补偿）。"""
    prices = np.asarray(actual_96, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    dt = cfg.dt

    e_c = c * dt
    e_d = d * dt

    discharge_rev = float(np.nansum(prices * e_d))
    charge_cost = float(np.nansum(prices * e_c))
    gross = discharge_rev - charge_cost

    avg_price = float(np.nanmean(prices))
    aux_cost = avg_price * cfg.aux_mwh
    net_arbitrage = gross - aux_cost
    capacity_comp = float(e_d.sum()) * cfg.cap_comp_per_mwh
    net_with_comp = net_arbitrage + capacity_comp

    return {
        "charge_mwh": round(float(e_c.sum()), 2),
        "discharge_mwh": round(float(e_d.sum()), 2),
        "charge_cost": round(charge_cost, 0),
        "discharge_rev": round(discharge_rev, 0),
        "gross": round(gross, 0),
        "aux_cost": round(aux_cost, 0),
        "net": round(net_arbitrage, 0),
        "net_arbitrage": round(net_arbitrage, 0),
        "capacity_comp": round(capacity_comp, 0),
        "net_with_comp": round(net_with_comp, 0),
    }


def _max_drawdown(cum: np.ndarray) -> tuple[float, float]:
    if len(cum) == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(np.max(dd)) if len(dd) else 0.0
    idx = int(np.argmax(dd)) if len(dd) else 0
    peak_val = float(peak[idx]) if idx < len(peak) else 0.0
    pct = max_dd / peak_val if peak_val > 1e-6 else 0.0
    return max_dd, pct


def _var_cvar(daily: np.ndarray, q: float) -> tuple[float, float]:
    if len(daily) == 0:
        return 0.0, 0.0
    var = float(np.quantile(daily, q))
    tail = daily[daily <= var]
    cvar = float(np.mean(tail)) if len(tail) else var
    return var, cvar


def _metric_block(
    daily_net: np.ndarray,
    daily_pf_net: np.ndarray,
    statuses: list[int],
) -> dict[str, Any]:
    n = len(daily_net)
    if n == 0:
        return {
            "total_net_yuan": 0.0,
            "mean_daily_net": 0.0,
            "std_daily_net": 0.0,
            "positive_day_ratio": 0.0,
            "max_drawdown_abs": 0.0,
            "max_drawdown_pct": 0.0,
            "var_5pct": 0.0,
            "var_1pct": 0.0,
            "cvar_5pct": 0.0,
            "cvar_1pct": 0.0,
            "sharpe_annual": 0.0,
            "regret_vs_pf_abs": 0.0,
            "regret_vs_pf_pct": 0.0,
            "realization_rate": 0.0,
            "infeasible_days": 0,
            "n_timeout": 0,
            "n_zero_solution": 0,
        }

    total = float(np.sum(daily_net))
    mean = float(np.mean(daily_net))
    std = float(np.std(daily_net, ddof=1)) if n > 1 else 0.0
    pos_ratio = float(np.mean(daily_net > 0))
    cum = np.cumsum(daily_net)
    max_dd, max_dd_pct = _max_drawdown(cum)
    var5, cvar5 = _var_cvar(daily_net, 0.05)
    var1, cvar1 = _var_cvar(daily_net, 0.01)
    sharpe = (mean / std * np.sqrt(365)) if std > 1e-6 else 0.0

    pf_total = float(np.sum(daily_pf_net))
    regret_abs = pf_total - total
    regret_pct = regret_abs / pf_total if abs(pf_total) > 1 else 0.0
    realization = total / pf_total if abs(pf_total) > 1 else 0.0

    infeasible = sum(1 for s in statuses if s not in (0, 3))
    n_timeout = sum(1 for s in statuses if s == 3)
    n_zero = sum(1 for s in statuses if s not in (0, 3))

    return {
        "total_net_yuan": round(total, 0),
        "mean_daily_net": round(mean, 0),
        "std_daily_net": round(std, 0),
        "positive_day_ratio": round(pos_ratio, 4),
        "max_drawdown_abs": round(max_dd, 0),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "var_5pct": round(var5, 0),
        "var_1pct": round(var1, 0),
        "cvar_5pct": round(cvar5, 0),
        "cvar_1pct": round(cvar1, 0),
        "sharpe_annual": round(float(sharpe), 4),
        "regret_vs_pf_abs": round(regret_abs, 0),
        "regret_vs_pf_pct": round(regret_pct, 4),
        "realization_rate": round(realization, 4),
        "infeasible_days": infeasible,
        "n_timeout": n_timeout,
        "n_zero_solution": n_zero,
    }


def compute_decision_metrics(
    df: pd.DataFrame,
    cfg: BatteryConfig,
) -> dict[str, Any]:
    """对 strategy_results 汇总 §9.4 决策指标（arbitrage / with_comp 两组）。"""
    if df.empty:
        empty = _metric_block(np.array([]), np.array([]), [])
        return {
            "arbitrage": empty,
            "with_comp": empty,
            "comp_mode": cfg.comp_mode,
            "carry_soc": cfg.carry_soc,
            "battery_params": cfg.to_dict(),
        }

    daily_arb = df["net_arbitrage"].astype(float).values
    daily_comp = df["net_with_comp"].astype(float).values
    daily_pf_arb = df["pf_net_arbitrage"].astype(float).values
    daily_pf_comp = df["pf_net_with_comp"].astype(float).values
    statuses = df.get("milp_status", pd.Series([0] * len(df))).astype(int).tolist()

    return {
        "arbitrage": _metric_block(daily_arb, daily_pf_arb, statuses),
        "with_comp": _metric_block(daily_comp, daily_pf_comp, statuses),
        "comp_mode": cfg.comp_mode,
        "carry_soc": cfg.carry_soc,
        "battery_params": cfg.to_dict(),
    }
