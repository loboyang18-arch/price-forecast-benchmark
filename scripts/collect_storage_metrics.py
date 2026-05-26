#!/usr/bin/env python3
"""汇总 runs/storage 下各算法 metrics.json 为 Markdown 表。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORAGE = ROOT / "runs" / "storage"

MARKETS = ["neimeng", "chongqing", "jiangsu"]

ALGO_ORDER_15MIN = [
    ("naive_lag_1d_15min", "naive_lag_1d"),
    ("naive_lag_7d_15min", "naive_lag_7d"),
    ("naive_rolling_7d_mean_15min", "naive_rolling_7d_mean"),
    ("lightgbm_baseline_15min", "LightGBM-Baseline"),
    ("lightgbm_twostage_15min", "LightGBM-TwoStage"),
    ("conv2d_multitask_15min", "Conv2D-MultiTask"),
]

ALGO_ORDER_HOURLY = [
    ("naive_lag_1d_hourly", "naive_lag_1d"),
    ("naive_lag_7d_hourly", "naive_lag_7d"),
    ("naive_rolling_7d_mean_hourly", "naive_rolling_7d_mean"),
    ("lightgbm_baseline_hourly", "LightGBM-Baseline"),
    ("lightgbm_twostage_hourly", "LightGBM-TwoStage"),
    ("conv2d_multitask_hourly", "Conv2D-MultiTask"),
    ("resconv2d_hourly", "ResConv2D"),
]


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _wan(yuan: float) -> str:
    return f"{yuan / 1e4:.1f}"


def _print_market(market: str, algo_order: list[tuple[str, str]]) -> None:
    print(f"\n### {market}\n")
    print("| 算法 | net_arbitrage(万) | net_with_comp(万) | realization_rate |")
    print("|------|-------------------|-------------------|------------------|")
    pf_arb = pf_comp = None
    for algo_dir, label in algo_order:
        m = _load(STORAGE / market / algo_dir / "metrics.json")
        if not m:
            print(f"| {label} | — | — | — |")
            continue
        dm = m["decision_metrics"]
        arb = dm["arbitrage"]
        comp = dm["with_comp"]
        if pf_arb is None:
            pf_arb = arb["total_net_yuan"] + arb["regret_vs_pf_abs"]
            pf_comp = comp["total_net_yuan"] + comp["regret_vs_pf_abs"]
        print(
            f"| {label} | {_wan(arb['total_net_yuan'])} | "
            f"{_wan(comp['total_net_yuan'])} | {comp['realization_rate']:.1%} |"
        )
    if pf_arb is not None:
        print(
            f"| **PF（完全预知）** | **{_wan(pf_arb)}** | "
            f"**{_wan(pf_comp)}** | **100%** |"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总储能评估 metrics")
    ap.add_argument(
        "--freq",
        choices=("15min", "hourly"),
        default="15min",
        help="产物目录后缀：15min 为原生目录名，hourly 为 *_hourly",
    )
    args = ap.parse_args()
    order = ALGO_ORDER_15MIN if args.freq == "15min" else ALGO_ORDER_HOURLY
    title = "15min 原生预测" if args.freq == "15min" else "hourly 平铺预测"
    print(f"# 储能评估汇总 ({title})\n")
    for market in MARKETS:
        _print_market(market, order)


if __name__ == "__main__":
    main()
