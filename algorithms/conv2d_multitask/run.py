#!/usr/bin/env python3
"""Conv2D-MultiTask — CLI 入口。

用法：
  python algorithms/conv2d_multitask/run.py --market all
  python algorithms/conv2d_multitask/run.py --market neimeng --freq 15min
  python algorithms/conv2d_multitask/run.py --market chongqing,jiangsu --epochs 40
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pfbench.data import load_market_data
from pfbench.exp_meta import make_experiment_id, save_config_snapshot
from pfbench.feature_registry import FeatureSpec, load_feature_registry
from pfbench.market_config import get_market_split
from pfbench.metrics import evaluate_predictions_csv
from pfbench.plotting import plot_weekly_predictions

from algorithms.conv2d_multitask.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    run_experiment,
)

SUPPORTED_MARKETS = ["neimeng", "chongqing", "jiangsu"]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_single_market(
    market_id: str, output_root: Path, freq: str,
    epochs: int, batch_size: int, lr: float, val_days: int,
    target: str = None,
    groups: list = None,
    early_stop: bool = False,
    patience: int = 10,
    restore_best: bool = True,
    output_suffix: str = None,
) -> dict:
    reg = load_feature_registry(market_id)
    split = get_market_split(market_id)
    test_start = pd.Timestamp(split.test_start)
    test_end = pd.Timestamp(split.test_end)
    algo_dir = "conv2d_multitask_15min" if freq == "15min" else "conv2d_multitask"
    if output_suffix:
        algo_dir = f"{algo_dir}_{output_suffix}"
    out_dir = output_root / market_id / algo_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = FeatureSpec(target=target, groups=groups)
    resolved_target = target or reg.target_default
    exp_id = make_experiment_id(market_id, algo_dir, target=resolved_target)
    save_config_snapshot(
        out_dir, exp_id,
        algorithm="conv2d_multitask",
        market=market_id, target=resolved_target, freq=freq,
        extra={
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "val_days": val_days,
            "early_stop": early_stop, "patience": patience,
            "restore_best": restore_best,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
        },
    )
    print(f"\n{'=' * 60}")
    print(f"  {market_id} [{freq}] → target={target or reg.target_default}")
    print(f"  test_start={test_start.date()}  test_end={test_end.date()}")
    print(f"  out_dir={out_dir}")
    if early_stop:
        print(f"  early_stop=ON  patience={patience}  restore_best={restore_best}")
    print(f"{'=' * 60}")

    df, meta = load_market_data(market_id)
    result = run_experiment(
        df, market_id, test_start=test_start, test_end=test_end,
        freq=freq, spec=spec,
        epochs=epochs, batch_size=batch_size, lr=lr,
        val_days=val_days,
        early_stop=early_stop, patience=patience, restore_best=restore_best,
    )

    metrics = result["metrics"]
    pred_df = result["predictions"]

    pred_fname = "test_predictions_15min.csv" if freq == "15min" else "test_predictions_hourly.csv"
    pred_csv = out_dir / pred_fname
    pred_df.to_csv(pred_csv, index=False)

    try:
        metrics["extended_metrics"] = evaluate_predictions_csv(pred_csv)
    except Exception as exc:
        logging.warning("extended_metrics 计算失败: %s", exc)
        metrics["extended_metrics"] = None
    metrics["experiment_id"] = exp_id

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(result["train_log"], f, ensure_ascii=False, indent=2)

    import numpy as np
    import torch
    weights_path = out_dir / "model_weights.pt"
    torch.save(result["model"].state_dict(), weights_path)
    np.save(out_dir / "norm_mean.npy", result["norm_mean"])
    np.save(out_dir / "norm_std.npy", result["norm_std"])
    np.savez(out_dir / "target_stats.npz", y_mean=result["y_mean"], y_std=result["y_std"])

    print(
        f"  test_MAE={metrics['test_mae']:.2f}  RMSE={metrics['test_rmse']:.2f}  "
        f"profile_corr={metrics['test_profile_corr']}  dir_acc={metrics['test_dir_acc']:.3f}"
    )

    if len(pred_df) > 0:
        plot_dir = out_dir / "plots"
        algo_label = "Conv2D-MultiTask-15min" if freq == "15min" else "Conv2D-MultiTask"
        plots = plot_weekly_predictions(
            pred_df, plot_dir, market_id, algo_label,
            target_col=metrics["target"], freq=freq,
        )
        print(f"  图表 → {plot_dir}/ ({len(plots)} 张)")

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Conv2D-MultiTask 实验")
    ap.add_argument("--market", "-m", default="all",
                    help="市场 ID（逗号分隔）或 all")
    ap.add_argument("--freq", choices=["1h", "15min"], default="1h",
                    help="预测粒度：1h（默认）或 15min")
    ap.add_argument("--target", default=None,
                    help="覆盖默认 target；必须在该市场 yaml 的 alt_targets 中")
    ap.add_argument("--groups", default=None,
                    help="逗号分隔的 feature groups（默认使用 yaml 中 enabled=true 的类别）")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--val-days", type=int, default=7,
                    help="从训练集末尾切出多少天做 val")
    ap.add_argument("--output-root", type=Path,
                    default=ROOT / "runs" / "predictions")
    ap.add_argument("--early-stop", action="store_true",
                    help="启用基于 val_mae 的早停（默认关闭，保持向后兼容）")
    ap.add_argument("--patience", type=int, default=10,
                    help="早停 patience（仅在 --early-stop 时生效）")
    ap.add_argument("--no-restore-best", action="store_true",
                    help="不恢复 best-val 权重（仅在 --early-stop 时生效）")
    ap.add_argument("--output-suffix", default=None,
                    help="输出目录后缀（如 'es'→conv2d_multitask_es/）。"
                         "未指定时启用 --early-stop 自动用 'es'")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _setup_logging(args.verbose)

    if args.market == "all":
        targets = list(SUPPORTED_MARKETS)
    else:
        targets = [m.strip() for m in args.market.split(",")]

    groups = None
    if args.groups:
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    output_suffix = args.output_suffix
    if args.early_stop and not output_suffix:
        output_suffix = "es"

    failed = []
    results = []
    for mid in targets:
        if mid not in SUPPORTED_MARKETS:
            print(f"未知市场: {mid}，可选 {SUPPORTED_MARKETS}")
            failed.append(mid)
            continue
        try:
            metrics = run_single_market(
                mid, args.output_root, args.freq,
                args.epochs, args.batch_size, args.lr, args.val_days,
                target=args.target, groups=groups,
                early_stop=args.early_stop,
                patience=args.patience,
                restore_best=not args.no_restore_best,
                output_suffix=output_suffix,
            )
            results.append(metrics)
        except Exception as e:
            print(f"  FAIL: {e}")
            logging.exception("experiment failed for %s", mid)
            failed.append(mid)

    print(f"\n{'=' * 60}\n汇总 [freq={args.freq}]:\n{'=' * 60}")
    for r in results:
        print(
            f"  {r['market']:12s}  test_MAE={r['test_mae']:8.2f}  "
            f"RMSE={r['test_rmse']:8.2f}  "
            f"corr={r['test_profile_corr']}  dir_acc={r['test_dir_acc']:.3f}"
        )
    if failed:
        print(f"\n失败: {failed}")
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
