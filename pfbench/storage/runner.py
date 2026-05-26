# Aligned with NM strategy_milp_15min L576-717
"""储能评估主流程：预测 CSV → MILP → 收益 → 产物。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pfbench.data import load_market_data
from pfbench.market_config import get_market_split
from pfbench.paths import LOCAL_PREDICTIONS_DIR, MARKETS_DIR, RUNS_DIR

from .battery import BatteryConfig, StorageMarketConfig
from .milp import solve_day_milp_15min, solve_pf_day_15min
from .plot import plot_weekly
from .report import print_report, write_markdown_summary
from .revenue import compute_decision_metrics, eval_day_revenue

LOG = logging.getLogger(__name__)
STORAGE_RUNS_DIR = RUNS_DIR / "storage"
PLOT_COLS = ["_c", "_d", "_actual", "_pred", "_soc"]


def storage_output_dirname(algo_name: str, freq: str = "15min") -> str:
    """储能产物目录名：15min 与预测目录同名；hourly 加 ``_hourly`` 后缀。"""
    if freq == "15min":
        return algo_name
    base = algo_name.removesuffix("_15min")
    if base.endswith("_hourly"):
        return base
    return f"{base}_hourly"


def load_storage_market_config(market_id: str) -> StorageMarketConfig:
    path = MARKETS_DIR / f"{market_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"市场配置不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return StorageMarketConfig.from_yaml_dict(raw.get("storage"))


def _pred_column(df: pd.DataFrame) -> str:
    if "predicted" in df.columns:
        return "predicted"
    if "pred" in df.columns:
        return "pred"
    raise KeyError("预测 CSV 需含 predicted 或 pred 列")


def _slots_summary(c: np.ndarray, d: np.ndarray, threshold: float = 0.5) -> tuple[str, str]:
    chg_slots = [t for t in range(96) if c[t] > threshold]
    dis_slots = [t for t in range(96) if d[t] > threshold]
    chg_str = (
        f"{chg_slots[0] * 15 // 60:02d}:{chg_slots[0] * 15 % 60:02d}–"
        f"{(chg_slots[-1] + 1) * 15 // 60:02d}:{(chg_slots[-1] + 1) * 15 % 60:02d}"
        if chg_slots
        else "-"
    )
    dis_str = (
        f"{dis_slots[0] * 15 // 60:02d}:{dis_slots[0] * 15 % 60:02d}–"
        f"{(dis_slots[-1] + 1) * 15 // 60:02d}:{(dis_slots[-1] + 1) * 15 % 60:02d}"
        if dis_slots
        else "-"
    )
    return chg_str, dis_str


def _expand_pred_to_96(day_pred: pd.DataFrame, pred_col: str) -> tuple[np.ndarray, bool]:
    n = len(day_pred)
    if n == 96:
        return day_pred[pred_col].values.astype(float), True
    if n >= 90:
        return day_pred[pred_col].values[:96].astype(float), True
    if n < 24:
        raise ValueError(f"当日预测点数不足: {n}")
    pred_hourly = day_pred[pred_col].values.astype(float)
    return np.repeat(pred_hourly, 4), False


def _load_settlement_series(
    market_id: str,
    settlement_col: str,
    test_start: str,
    test_end: str,
) -> pd.Series:
    df, _meta = load_market_data(market_id)
    if settlement_col not in df.columns:
        raise KeyError(f"结算价列不存在: {settlement_col}")
    s = df[settlement_col].copy()
    s.index = pd.to_datetime(s.index)
    mask = (s.index >= pd.Timestamp(test_start)) & (s.index <= pd.Timestamp(test_end) + pd.Timedelta(hours=23, minutes=45))
    return s.loc[mask]


def run_storage_eval(
    market_id: str,
    algorithm: str,
    pred_csv: Path,
    *,
    out_dir: Path | None = None,
    cfg: BatteryConfig | None = None,
    storage_cfg: StorageMarketConfig | None = None,
    label: str | None = None,
    plot: bool = True,
    verbose: bool = True,
    freq: str = "15min",
) -> dict[str, Any]:
    """对单个预测文件运行储能 MILP 评估。"""
    storage_cfg = storage_cfg or load_storage_market_config(market_id)
    cfg = cfg or storage_cfg.battery
    split = get_market_split(market_id)

    pred_path = Path(pred_csv)
    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)

    pred = pd.read_csv(pred_path, parse_dates=["ts"])
    pred_col = _pred_column(pred)
    pred = pred.rename(columns={pred_col: "predicted"})
    pred["date"] = pred["ts"].dt.date.astype(str)
    pred = pred[
        (pred["ts"] >= pd.Timestamp(split.test_start))
        & (pred["ts"] <= pd.Timestamp(split.test_end) + pd.Timedelta(hours=23, minutes=45))
    ]

    settlement = _load_settlement_series(
        market_id,
        storage_cfg.settlement_price_col,
        split.test_start,
        split.test_end,
    )

    storage_algo = storage_output_dirname(algorithm, freq)
    out_dir = out_dir or (STORAGE_RUNS_DIR / market_id / storage_algo)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_kind = "hourly→15m" if freq != "15min" else "15m-native"
    label = label or f"MILP-15min/{storage_algo} ({pred_kind})"

    dates = sorted(pred["date"].unique())
    rows: list[dict[str, Any]] = []
    soc_carry = 0.0
    soc_carry_pf = 0.0
    carry_soc = cfg.carry_soc

    if verbose:
        mode = "carry SOC" if carry_soc else "daily reset"
        LOG.info("%s/%s: %d days, %s", market_id, algorithm, len(dates), mode)

    for i, date in enumerate(dates):
        day_pred = pred[pred["date"] == date].sort_values("ts")
        try:
            pred_96, pred_native_15m = _expand_pred_to_96(day_pred, "predicted")
        except ValueError:
            continue

        day_settle = settlement[settlement.index.date.astype(str) == date]
        if len(day_settle) < 90:
            if verbose:
                LOG.warning("[%s] settlement missing, skip", date)
            continue
        actual_96 = day_settle.values[:96].astype(float)

        is_last = i == len(dates) - 1
        if carry_soc:
            force_end = is_last
            if not is_last:
                next_date = dates[i + 1]
                next_pred = pred[pred["date"] == next_date]["predicted"].values
                next_avg = float(np.mean(next_pred)) if len(next_pred) else 0.0
            else:
                next_avg = 0.0
        else:
            force_end = True
            next_avg = 0.0

        c, d, soc, milp_status = solve_day_milp_15min(
            pred_96,
            cfg,
            soc_init=soc_carry,
            force_zero_end=force_end,
            next_day_avg_price=next_avg,
        )
        c_pf, d_pf, soc_pf, pf_status = solve_pf_day_15min(
            actual_96,
            cfg,
            soc_init=soc_carry_pf,
            force_zero_end=force_end,
            next_day_avg_price=float(np.mean(actual_96)),
        )

        soc_end = float(soc[-1]) if soc.sum() > 0 else 0.0
        soc_end_pf = float(soc_pf[-1]) if soc_pf.sum() > 0 else 0.0
        if carry_soc:
            soc_carry = soc_end
            soc_carry_pf = soc_end_pf
        else:
            soc_carry = soc_carry_pf = 0.0

        rev = eval_day_revenue(c, d, actual_96, cfg)
        rev_pf = eval_day_revenue(c_pf, d_pf, actual_96, cfg)
        chg_str, dis_str = _slots_summary(c, d)

        rows.append({
            "date": date,
            "pred_native_15m": pred_native_15m,
            "charge_window": chg_str,
            "discharge_window": dis_str,
            "charge_mwh": rev["charge_mwh"],
            "discharge_mwh": rev["discharge_mwh"],
            "soc_end": round(soc_end, 1),
            "gross": rev["gross"],
            "aux_cost": rev["aux_cost"],
            "net": rev["net_arbitrage"],
            "net_arbitrage": rev["net_arbitrage"],
            "capacity_comp": rev["capacity_comp"],
            "net_with_comp": rev["net_with_comp"],
            "pf_gross": rev_pf["gross"],
            "pf_aux_cost": rev_pf["aux_cost"],
            "pf_net": rev_pf["net_arbitrage"],
            "pf_net_arbitrage": rev_pf["net_arbitrage"],
            "pf_net_with_comp": rev_pf["net_with_comp"],
            "milp_status": milp_status,
            "pf_status": pf_status,
            "_c": c.tolist(),
            "_d": d.tolist(),
            "_actual": actual_96.tolist(),
            "_pred": pred_96.tolist(),
            "_soc": soc.tolist(),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"{market_id}/{algorithm}: 无有效评估日")

    decision_metrics = compute_decision_metrics(df, cfg)
    metrics = {
        "market": market_id,
        "algorithm": algorithm,
        "storage_dir": storage_algo,
        "pred_freq": freq,
        "pred_csv": str(pred_path),
        "n_days": len(df),
        "decision_metrics": decision_metrics,
    }

    csv_out = out_dir / "strategy_results.csv"
    df.drop(columns=PLOT_COLS, errors="ignore").to_csv(csv_out, index=False, encoding="utf-8-sig")
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_summary(
        df,
        metrics,
        out_dir / "summary.md",
        label=label,
        market_id=market_id,
        algorithm=algorithm,
    )

    if plot:
        plot_weekly(df, out_dir / "plots", label=label)

    if verbose:
        print_report(df, label=label, carry_soc=carry_soc)

    return {
        "market_id": market_id,
        "algorithm": algorithm,
        "storage_dir": storage_algo,
        "pred_freq": freq,
        "out_dir": str(out_dir),
        "metrics": metrics,
        "df": df,
    }


def discover_prediction_files(
    market_id: str,
    algorithm: str = "all",
    freq: str = "15min",
) -> list[tuple[str, Path]]:
    """扫描 runs/predictions/<market>/<algo>/test_predictions_*.csv。"""
    base = LOCAL_PREDICTIONS_DIR / market_id
    if not base.is_dir():
        return []

    fname = "test_predictions_15min.csv" if freq == "15min" else "test_predictions_hourly.csv"
    results: list[tuple[str, Path]] = []

    if algorithm == "all":
        algo_dirs = sorted(d for d in base.iterdir() if d.is_dir())
    else:
        algo_dirs = [base / algorithm]
        if not algo_dirs[0].is_dir():
            alt = base / f"{algorithm}_15min" if freq == "15min" else base / algorithm
            algo_dirs = [alt] if alt.is_dir() else []

    for algo_dir in algo_dirs:
        pred_path = algo_dir / fname
        if pred_path.is_file():
            results.append((algo_dir.name, pred_path))
    return results


def run_storage_eval_batch(
    markets: list[str],
    algorithm: str = "all",
    freq: str = "15min",
    *,
    comp_mode: str | None = None,
    carry_soc: bool | None = None,
    plot: bool = True,
) -> list[dict[str, Any]]:
    """批量运行多市场、多算法储能评估。"""
    all_results: list[dict[str, Any]] = []
    for market_id in markets:
        storage_cfg = load_storage_market_config(market_id)
        cfg = storage_cfg.battery
        if comp_mode is not None:
            cfg.comp_mode = comp_mode
        if carry_soc is not None:
            cfg.carry_soc = carry_soc

        pairs = discover_prediction_files(market_id, algorithm=algorithm, freq=freq)
        if not pairs:
            LOG.warning("no predictions for market=%s freq=%s", market_id, freq)
            continue

        for algo_name, pred_path in pairs:
            try:
                r = run_storage_eval(
                    market_id,
                    algo_name,
                    pred_path,
                    cfg=cfg,
                    storage_cfg=storage_cfg,
                    plot=plot,
                    verbose=False,
                    freq=freq,
                )
                all_results.append(r)
            except Exception as exc:
                LOG.error("failed %s/%s: %s", market_id, algo_name, exc)
    return all_results
