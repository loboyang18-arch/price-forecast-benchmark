#!/usr/bin/env python3
"""朴素基准算法 — CLI 入口。

用法：
  python algorithms/naive_baseline/run.py --market all --freq 1h
  python algorithms/naive_baseline/run.py --market neimeng --freq 15min \
      --strategies lag_1d,lag_7d
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pfbench.data import load_market_data
from pfbench.exp_meta import make_experiment_id, save_config_snapshot
from pfbench.feature_registry import load_feature_registry
from pfbench.market_config import get_market_split
from pfbench.metrics import evaluate_predictions_csv
from pfbench.plotting import plot_weekly_predictions

from algorithms.naive_baseline.predict import (
    STRATEGIES,
    aggregate_target,
    naive_predict,
)

SUPPORTED_MARKETS = ["neimeng", "chongqing", "jiangsu"]
LOG = logging.getLogger("naive_baseline")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _algo_dirname(strategy: str, freq: str) -> str:
    base = f"naive_{strategy}"
    return f"{base}_15min" if freq == "15min" else base


def run_single(
    market_id: str,
    strategy: str,
    freq: str,
    output_root: Path,
    target: str = None,
) -> dict:
    reg = load_feature_registry(market_id)
    target_col = target or reg.target_default
    split = get_market_split(market_id)
    test_start = pd.Timestamp(split.test_start)
    test_end = pd.Timestamp(split.test_end)

    algo_dirname = _algo_dirname(strategy, freq)
    out_dir = output_root / market_id / algo_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_id = make_experiment_id(market_id, algo_dirname, target=target_col)
    save_config_snapshot(
        out_dir, exp_id,
        algorithm=f"naive_{strategy}",
        market=market_id, target=target_col, freq=freq,
        extra={"strategy": strategy,
               "test_start": str(test_start.date()),
               "test_end": str(test_end.date())},
    )

    df, _meta = load_market_data(market_id)
    y_full = aggregate_target(df, target_col, freq=freq)
    preds = naive_predict(y_full, strategy, freq, test_start, test_end)
    if preds.empty:
        raise RuntimeError(
            f"{market_id}/{strategy}/{freq}: 预测为空（测试区间无可用历史 lag）"
        )

    pred_fname = "test_predictions_15min.csv" if freq == "15min" else "test_predictions_hourly.csv"
    pred_path = out_dir / pred_fname
    preds.to_csv(pred_path, index=False)

    a = preds["actual"].values
    p = preds["predicted"].values
    mae = float(np.mean(np.abs(a - p)))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    corr = float(np.corrcoef(a, p)[0, 1]) if a.std() > 1e-6 and p.std() > 1e-6 else float("nan")

    metrics = {
        "experiment_id": exp_id,
        "market": market_id,
        "algorithm": f"naive_{strategy}",
        "strategy": strategy,
        "freq": freq,
        "target": target_col,
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "n_test_points": int(len(preds)),
        "test_mae": round(mae, 3),
        "test_rmse": round(rmse, 3),
        "test_profile_corr": round(corr, 4) if not np.isnan(corr) else None,
    }
    try:
        metrics["extended_metrics"] = evaluate_predictions_csv(pred_path)
    except Exception as exc:
        LOG.warning("extended_metrics 计算失败: %s", exc)
        metrics["extended_metrics"] = None

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    label = f"Naive-{strategy}" + ("-15min" if freq == "15min" else "")
    try:
        plot_dir = out_dir / "plots"
        plots = plot_weekly_predictions(
            preds, plot_dir, market_id, label,
            target_col=target_col, freq=freq,
        )
        LOG.info("图表 → %s/ (%d 张)", plot_dir, len(plots))
    except Exception as exc:
        LOG.warning("绘图失败: %s", exc)

    LOG.info(
        "Naive-%s | %s [%s]  MAE=%.2f  RMSE=%.2f  corr=%s",
        strategy, market_id, freq, mae, rmse,
        f"{corr:.4f}" if not np.isnan(corr) else "-",
    )
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="朴素基准实验")
    ap.add_argument("--market", "-m", default="all",
                    help="市场 ID（逗号分隔）或 all")
    ap.add_argument("--freq", choices=["1h", "15min"], default="1h")
    ap.add_argument("--strategies", default=",".join(STRATEGIES),
                    help=f"逗号分隔的策略列表，可选 {STRATEGIES}（默认全跑）")
    ap.add_argument("--target", default=None,
                    help="覆盖 yaml 中的默认 target")
    ap.add_argument("--output-root", type=Path,
                    default=ROOT / "runs" / "predictions")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _setup_logging(args.verbose)

    markets = SUPPORTED_MARKETS if args.market == "all" else \
        [m.strip() for m in args.market.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in STRATEGIES:
            raise SystemExit(f"未知 strategy: {s}，可选 {STRATEGIES}")

    failed = []
    results = []
    for mid in markets:
        if mid not in SUPPORTED_MARKETS:
            print(f"未知市场: {mid}，可选 {SUPPORTED_MARKETS}")
            failed.append(mid)
            continue
        for strategy in strategies:
            try:
                m = run_single(
                    mid, strategy, args.freq, args.output_root,
                    target=args.target,
                )
                results.append(m)
            except Exception as exc:
                print(f"  FAIL: {mid}/{strategy}/{args.freq}: {exc}")
                logging.exception("naive exp failed")
                failed.append((mid, strategy, args.freq))

    print(f"\n{'='*60}\n汇总 [freq={args.freq}]:\n{'='*60}")
    print(f"{'market':12s} {'strategy':18s} {'MAE':>8s} {'RMSE':>8s} {'corr':>8s}")
    print("-" * 60)
    for r in results:
        print(f"{r['market']:12s} {r['strategy']:18s} "
              f"{r['test_mae']:>8.2f} {r['test_rmse']:>8.2f} "
              f"{(r['test_profile_corr'] or 0):>8.4f}")
    if failed:
        print(f"\n失败: {failed}")
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
