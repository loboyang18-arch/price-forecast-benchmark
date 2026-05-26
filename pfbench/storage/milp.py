# Aligned with NM strategy_milp_15min L73-340, L474-517
"""15 分钟粒度储能 MILP 充放电调度（scipy.optimize.milp / HiGHS）。"""
from __future__ import annotations

import logging
import warnings
from typing import Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, lil_matrix

from .battery import BatteryConfig

LOG = logging.getLogger(__name__)

OBJ_SCALE = 1000.0
SolveResult = Tuple[np.ndarray, np.ndarray, np.ndarray, int]


class MILPSolverFailure(RuntimeError):
    """MILP 求解器在时限内未找到可行解。"""


def _min_run_count(tt: int, lm: int) -> int:
    cnt = 0
    for t in range(tt):
        for k in range(1, lm):
            if t + k < tt:
                cnt += 1
    return cnt


def _build_milp_15min(
    prices_96: np.ndarray,
    cfg: BatteryConfig,
    *,
    prices_discharge_96: np.ndarray | None = None,
    soc_init: float = 0.0,
    force_zero_end: bool = True,
    next_day_avg_price: float = 0.0,
    cap_comp_per_mwh: float | None = None,
    min_charge_mwh: float = 0.0,
    raise_on_failure: bool = False,
) -> SolveResult:
    """构建并求解 15 分钟粒度日前充放电 MILP。

    Returns:
        (c, d, soc, status): 各长度 96；status 0=最优, 3=超时有解, 其他=失败。
    """
    T = cfg.T
    n = 5 * T
    ic, id_, iyc, iyd, is_ = 0, T, 2 * T, 3 * T, 4 * T

    eta_c = cfg.eta_c
    eta_d = cfg.eta_d
    p_max = cfg.p_max_mw
    cap_mwh = cfg.cap_mwh
    dp_ramp = cfg.dp_ramp_mw
    l_min = cfg.l_min
    max_charge_mwh = cfg.max_charge_mwh
    dt = cfg.dt
    terminal_discount = cfg.terminal_discount
    enforce_switch_gap = cfg.enforce_switch_gap
    time_limit = cfg.time_limit

    if cap_comp_per_mwh is None:
        cap_comp_per_mwh = cfg.milp_cap_comp()

    p_charge = np.asarray(prices_96, dtype=float)
    p_discharge = p_charge if prices_discharge_96 is None else np.asarray(
        prices_discharge_96, dtype=float
    )

    obj = np.zeros(n)
    obj[ic : ic + T] = +p_charge * dt / OBJ_SCALE
    obj[id_ : id_ + T] = -(p_discharge + cap_comp_per_mwh) * dt / OBJ_SCALE
    if not force_zero_end and next_day_avg_price > 0:
        terminal_value_per_mwh = next_day_avg_price * eta_d * terminal_discount
        obj[is_ + T - 1] = -terminal_value_per_mwh / OBJ_SCALE

    lb = np.zeros(n)
    ub = np.zeros(n)
    ub[ic : ic + T] = p_max
    ub[id_ : id_ + T] = p_max
    ub[iyc : iyc + T] = 1.0
    ub[iyd : iyd + T] = 1.0
    ub[is_ : is_ + T] = cap_mwh
    if force_zero_end:
        ub[is_ + T - 1] = 0.0
        lb[is_ + T - 1] = 0.0

    integ = np.zeros(n)
    integ[iyc : iyc + T] = 1
    integ[iyd : iyd + T] = 1

    n_minrun = _min_run_count(T, l_min)
    has_min_charge = float(min_charge_mwh) > 0
    n_switch_rows = 2 * (T - 1) if enforce_switch_gap else 0
    n_con = (
        4 * T
        + 4 * (T - 1)
        + n_switch_rows
        + 2 * n_minrun
        + 1
        + (1 if has_min_charge else 0)
    )
    a_mat = lil_matrix((n_con, n), dtype=float)
    lb_c = np.full(n_con, -np.inf)
    ub_c = np.zeros(n_con)
    row = 0

    for t in range(T):
        a_mat[row, ic + t] = +eta_c * dt
        a_mat[row, id_ + t] = -dt / eta_d
        if t > 0:
            a_mat[row, is_ + t - 1] = +1.0
        a_mat[row, is_ + t] = -1.0
        lb_c[row] = ub_c[row] = (-soc_init if t == 0 else 0.0)
        row += 1

    for t in range(T):
        a_mat[row, ic + t] = +1.0
        a_mat[row, iyc + t] = -p_max
        ub_c[row] = 0.0
        row += 1

    for t in range(T):
        a_mat[row, id_ + t] = +1.0
        a_mat[row, iyd + t] = -p_max
        ub_c[row] = 0.0
        row += 1

    for t in range(T):
        a_mat[row, iyc + t] = 1.0
        a_mat[row, iyd + t] = 1.0
        ub_c[row] = 1.0
        row += 1

    if enforce_switch_gap:
        for t in range(T - 1):
            a_mat[row, iyc + t] = 1.0
            a_mat[row, iyd + t + 1] = 1.0
            ub_c[row] = 1.0
            row += 1
        for t in range(T - 1):
            a_mat[row, iyd + t] = 1.0
            a_mat[row, iyc + t + 1] = 1.0
            ub_c[row] = 1.0
            row += 1

    for t in range(1, T):
        a_mat[row, ic + t] = +1.0
        a_mat[row, ic + t - 1] = -1.0
        ub_c[row] = dp_ramp
        row += 1

    for t in range(1, T):
        a_mat[row, ic + t - 1] = +1.0
        a_mat[row, ic + t] = -1.0
        ub_c[row] = dp_ramp
        row += 1

    for t in range(1, T):
        a_mat[row, id_ + t] = +1.0
        a_mat[row, id_ + t - 1] = -1.0
        ub_c[row] = dp_ramp
        row += 1

    for t in range(1, T):
        a_mat[row, id_ + t - 1] = +1.0
        a_mat[row, id_ + t] = -1.0
        ub_c[row] = dp_ramp
        row += 1

    for t in range(T):
        for k in range(1, l_min):
            if t + k >= T:
                break
            a_mat[row, iyc + t + k] = -1.0
            a_mat[row, iyc + t] = +1.0
            if t > 0:
                a_mat[row, iyc + t - 1] = -1.0
            ub_c[row] = 0.0
            row += 1

    for t in range(T):
        for k in range(1, l_min):
            if t + k >= T:
                break
            a_mat[row, iyd + t + k] = -1.0
            a_mat[row, iyd + t] = +1.0
            if t > 0:
                a_mat[row, iyd + t - 1] = -1.0
            ub_c[row] = 0.0
            row += 1

    for t in range(T):
        a_mat[row, ic + t] = dt
    ub_c[row] = max_charge_mwh
    row += 1

    if has_min_charge:
        for t in range(T):
            a_mat[row, ic + t] = -dt
        ub_c[row] = -float(min_charge_mwh)
        row += 1

    assert row == n_con, f"约束行计数错误: {row} != {n_con}"

    if cap_comp_per_mwh > 0:
        solver_opts = {
            "disp": False,
            "time_limit": float(time_limit),
            "mip_rel_gap": 1e-3,
        }
    else:
        solver_opts = {"disp": False, "time_limit": float(time_limit)}

    res = milp(
        obj,
        constraints=LinearConstraint(csc_matrix(a_mat), lb_c, ub_c),
        integrality=integ,
        bounds=Bounds(lb, ub),
        options=solver_opts,
    )

    if res.status not in (0, 3) or res.x is None:
        msg = (
            f"MILP 求解未找到可行解 (status={res.status}, "
            f"time_limit={time_limit}s, cap_comp={cap_comp_per_mwh})"
        )
        if raise_on_failure:
            raise MILPSolverFailure(msg)
        warnings.warn(msg + "，返回零方案", RuntimeWarning, stacklevel=2)
        LOG.warning(msg)
        return np.zeros(T), np.zeros(T), np.zeros(T), int(res.status)

    x = res.x
    return (
        x[ic : ic + T],
        x[id_ : id_ + T],
        x[is_ : is_ + T],
        int(res.status),
    )


def _build_milp_cbc(
    prices_96: np.ndarray,
    cfg: BatteryConfig,
    *,
    prices_discharge_96: np.ndarray | None = None,
    soc_init: float = 0.0,
    force_zero_end: bool = True,
    next_day_avg_price: float = 0.0,
    cap_comp_per_mwh: float | None = None,
    min_charge_mwh: float = 0.0,
    raise_on_failure: bool = False,
) -> SolveResult:
    """PuLP + CBC，与内蒙 strategy_milp_15min 默认求解器一致。"""
    import pulp

    T = cfg.T
    dt = cfg.dt
    eta_c = cfg.eta_c
    eta_d = cfg.eta_d
    p_max = cfg.p_max_mw
    cap_mwh = cfg.cap_mwh
    dp_ramp = cfg.dp_ramp_mw
    l_min = cfg.l_min
    max_charge_mwh = cfg.max_charge_mwh
    terminal_discount = cfg.terminal_discount
    enforce_switch_gap = cfg.enforce_switch_gap
    time_limit = cfg.time_limit

    if cap_comp_per_mwh is None:
        cap_comp_per_mwh = cfg.milp_cap_comp()

    p_charge = np.asarray(prices_96, dtype=float)
    p_discharge = p_charge if prices_discharge_96 is None else np.asarray(
        prices_discharge_96, dtype=float
    )

    prob = pulp.LpProblem("battery_dispatch_15min", pulp.LpMaximize)
    c_vars = [pulp.LpVariable(f"c_{t}", 0, p_max) for t in range(T)]
    d_vars = [pulp.LpVariable(f"d_{t}", 0, p_max) for t in range(T)]
    y_c = [pulp.LpVariable(f"yc_{t}", 0, 1, cat=pulp.LpBinary) for t in range(T)]
    y_d = [pulp.LpVariable(f"yd_{t}", 0, 1, cat=pulp.LpBinary) for t in range(T)]
    soc_vars = [pulp.LpVariable(f"soc_{t}", 0, cap_mwh) for t in range(T)]
    if force_zero_end:
        soc_vars[T - 1].upBound = 0.0

    revenue = pulp.lpSum(
        (p_discharge[t] + cap_comp_per_mwh) * d_vars[t] * dt - p_charge[t] * c_vars[t] * dt
        for t in range(T)
    )
    if not force_zero_end and next_day_avg_price > 0:
        revenue += next_day_avg_price * eta_d * terminal_discount * soc_vars[T - 1]
    prob += revenue

    for t in range(T):
        prev_soc = soc_init if t == 0 else soc_vars[t - 1]
        prob += soc_vars[t] == prev_soc + eta_c * c_vars[t] * dt - d_vars[t] * dt / eta_d
    for t in range(T):
        prob += c_vars[t] <= p_max * y_c[t]
        prob += d_vars[t] <= p_max * y_d[t]
        prob += y_c[t] + y_d[t] <= 1
    if enforce_switch_gap:
        for t in range(T - 1):
            prob += y_c[t] + y_d[t + 1] <= 1
            prob += y_d[t] + y_c[t + 1] <= 1
    for t in range(1, T):
        prob += c_vars[t] - c_vars[t - 1] <= dp_ramp
        prob += c_vars[t - 1] - c_vars[t] <= dp_ramp
        prob += d_vars[t] - d_vars[t - 1] <= dp_ramp
        prob += d_vars[t - 1] - d_vars[t] <= dp_ramp
    for t in range(T):
        prev_c = 0 if t == 0 else y_c[t - 1]
        prev_d = 0 if t == 0 else y_d[t - 1]
        for k in range(1, l_min):
            if t + k < T:
                prob += y_c[t + k] >= y_c[t] - prev_c
                prob += y_d[t + k] >= y_d[t] - prev_d
    prob += pulp.lpSum(c_vars[t] * dt for t in range(T)) <= max_charge_mwh
    if float(min_charge_mwh) > 0:
        prob += pulp.lpSum(c_vars[t] * dt for t in range(T)) >= float(min_charge_mwh)

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=int(time_limit), gapRel=0.05)
    prob.solve(solver)

    has_solution = (
        prob.status == pulp.LpStatusOptimal
        or (
            prob.status == pulp.LpStatusNotSolved
            and prob.sol_status == pulp.constants.LpSolutionFeasible
        )
    )
    if not has_solution:
        msg = (
            f"CBC 求解失败 (status={prob.status}: {pulp.LpStatus[prob.status]}, "
            f"time_limit={time_limit}s)"
        )
        if raise_on_failure:
            raise MILPSolverFailure(msg)
        warnings.warn(msg + "，返回零方案", RuntimeWarning, stacklevel=2)
        LOG.warning(msg)
        return np.zeros(T), np.zeros(T), np.zeros(T), -1

    status = 0 if prob.status == pulp.LpStatusOptimal else 3
    c_out = np.array([v.varValue or 0.0 for v in c_vars])
    d_out = np.array([v.varValue or 0.0 for v in d_vars])
    soc_out = np.array([v.varValue or 0.0 for v in soc_vars])
    return c_out, d_out, soc_out, status


def _solve_dispatch(
    prices_96: np.ndarray,
    cfg: BatteryConfig,
    **kwargs,
) -> SolveResult:
    """默认 scipy HiGHS（与 neimeng _build_milp_15min 逐行对齐）。"""
    cap = kwargs.pop("cap_comp_per_mwh", None)
    if cap is None:
        cap = cfg.milp_cap_comp()
    return _build_milp_15min(
        np.asarray(prices_96, dtype=float),
        cfg,
        cap_comp_per_mwh=cap,
        **kwargs,
    )


def solve_day_milp_15min(
    prices_96: np.ndarray,
    cfg: BatteryConfig,
    **kwargs,
) -> SolveResult:
    """基于预测电价求解最优充放电计划。"""
    return _solve_dispatch(prices_96, cfg, **kwargs)


def solve_pf_day_15min(
    actual_96: np.ndarray,
    cfg: BatteryConfig,
    **kwargs,
) -> SolveResult:
    """完全预知基准：用真实 15 分钟电价求解同一 MILP。"""
    return _solve_dispatch(actual_96, cfg, **kwargs)


def solve_day_milp_15min_robust(
    p10_96: np.ndarray,
    p50_96: np.ndarray,
    p90_96: np.ndarray,
    cfg: BatteryConfig,
    alpha: float = 0.5,
    **kwargs,
) -> SolveResult:
    """鲁棒 MILP：充电按偏高价、放电按偏低价估计。"""
    p10 = np.asarray(p10_96, dtype=float)
    p50 = np.asarray(p50_96, dtype=float)
    p90 = np.asarray(p90_96, dtype=float)
    p_charge = (1.0 - alpha) * p50 + alpha * p90
    p_discharge = (1.0 - alpha) * p50 + alpha * p10
    cap = kwargs.pop("cap_comp_per_mwh", None)
    if cap is None:
        cap = cfg.milp_cap_comp()
    return _build_milp_15min(
        prices_96=p_charge,
        cfg=cfg,
        prices_discharge_96=p_discharge,
        cap_comp_per_mwh=cap,
        **kwargs,
    )
