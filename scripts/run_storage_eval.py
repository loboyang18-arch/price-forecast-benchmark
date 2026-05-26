#!/usr/bin/env python3
"""储能 MILP 决策与收益评估 CLI。

示例：
  python scripts/run_storage_eval.py --market neimeng --algorithm conv2d_multitask_15min --freq 15min
  python scripts/run_storage_eval.py --market all --algorithm all --freq 15min
  python scripts/run_storage_eval.py --market neimeng --algorithm all --comp-mode ex_post
  python scripts/run_storage_eval.py --market neimeng --algorithm all --carry-soc
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfbench.markets import list_markets
from pfbench.storage.runner import run_storage_eval, run_storage_eval_batch

SUPPORTED_MARKETS = ["neimeng", "chongqing", "jiangsu"]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_markets(market_arg: str) -> list[str]:
    if market_arg == "all":
        return [m for m in list_markets() if m in SUPPORTED_MARKETS]
    markets = [m.strip() for m in market_arg.split(",") if m.strip()]
    for m in markets:
        if m not in SUPPORTED_MARKETS:
            raise ValueError(f"不支持的市场: {m}，可选: {SUPPORTED_MARKETS}")
    return markets


def main() -> None:
    ap = argparse.ArgumentParser(description="储能 MILP 决策与收益评估")
    ap.add_argument(
        "--market",
        default="neimeng",
        help="市场 ID，逗号分隔或 all（neimeng/chongqing/jiangsu）",
    )
    ap.add_argument("--algorithm", default="all", help="算法目录名或 all")
    ap.add_argument(
        "--freq",
        choices=("15min", "hourly", "1h"),
        default="15min",
        help="预测频率（默认 15min）",
    )
    ap.add_argument(
        "--comp-mode",
        choices=("in_objective", "ex_post"),
        default=None,
        help="容量补偿口径：入 MILP 目标或事后计入",
    )
    ap.add_argument("--carry-soc", action="store_true", help="启用跨日 SOC 传递")
    ap.add_argument("--pred-csv", type=Path, default=None, help="直接指定预测 CSV（单文件模式）")
    ap.add_argument("--no-plot", action="store_true", help="不生成周图")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _setup_logging(args.verbose)
    freq = "15min" if args.freq == "15min" else "hourly"
    markets = _resolve_markets(args.market)

    if args.pred_csv:
        if len(markets) != 1 or args.algorithm == "all":
            raise SystemExit("--pred-csv 需配合单一 --market 与 --algorithm")
        run_storage_eval(
            markets[0],
            args.algorithm,
            args.pred_csv,
            plot=not args.no_plot,
            verbose=True,
            freq=freq,
        )
        return

    results = run_storage_eval_batch(
        markets,
        algorithm=args.algorithm,
        freq=freq,
        comp_mode=args.comp_mode,
        carry_soc=True if args.carry_soc else None,
        plot=not args.no_plot,
    )
    print(f"\n完成 {len(results)} 个储能评估任务")


if __name__ == "__main__":
    main()
