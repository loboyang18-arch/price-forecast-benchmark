#!/usr/bin/env python3
"""LightGBM baseline — CLI 入口。

用法：
  python algorithms/lightgbm_baseline/run.py --market all
  python algorithms/lightgbm_baseline/run.py --market neimeng
  python algorithms/lightgbm_baseline/run.py --market neimeng,chongqing
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pfbench.data import load_market_data

from algorithms.lightgbm_baseline.config import MARKET_CONFIGS
from algorithms.lightgbm_baseline.train import run_experiment


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="LightGBM baseline 实验")
    ap.add_argument(
        "--market", "-m", default="all",
        help="市场 ID（逗号分隔）或 all",
    )
    ap.add_argument(
        "--output-root", type=Path,
        default=ROOT / "runs" / "predictions",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _setup_logging(args.verbose)

    if args.market == "all":
        targets = list(MARKET_CONFIGS.keys())
    else:
        targets = [m.strip() for m in args.market.split(",")]

    failed = []
    results = []
    for mid in targets:
        if mid not in MARKET_CONFIGS:
            print(f"未知市场: {mid}，可选 {list(MARKET_CONFIGS.keys())}")
            failed.append(mid)
            continue

        cfg = MARKET_CONFIGS[mid]
        out_dir = args.output_root / mid / "lightgbm"
        print(f"\n{'='*60}")
        print(f"  {mid} → target={cfg.target_col}, test_start={cfg.test_start}")
        print(f"{'='*60}")

        try:
            df, meta = load_market_data(mid)
            metrics = run_experiment(df, cfg, out_dir)
            results.append(metrics)
            print(f"  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
                  f"MAPE={metrics['mape_pct']:.2f}%  ProfileCorr={metrics['profile_corr']:.4f}")
        except Exception as e:
            print(f"  FAIL: {e}")
            logging.exception("experiment failed for %s", mid)
            failed.append(mid)

    print(f"\n{'='*60}")
    print("汇总:")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['market_id']:12s}  MAE={r['mae']:8.4f}  RMSE={r['rmse']:8.4f}  "
              f"MAPE={r['mape_pct']:6.2f}%  Corr={r['profile_corr']:.4f}")
    if failed:
        print(f"\n失败: {failed}")
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
