#!/usr/bin/env python3
"""LightGBM-TwoStage — CLI 入口。

用法（2026-05-26 起改为单次训练-预测，与 `lightgbm_baseline` / `conv2d_multitask` 口径对齐）：
  # 全部市场，默认配置（Two-Stage + 后处理）
  python algorithms/lightgbm_twostage/run.py --market all

  # 单市场，启用分位数模式
  python algorithms/lightgbm_twostage/run.py --market jiangsu --mode quantile

  # 自定义参数
  python algorithms/lightgbm_twostage/run.py --market neimeng --mode two_stage --time-decay 30 --top-k 80
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pfbench.data import load_market_data
from pfbench.exp_meta import make_experiment_id, save_config_snapshot
from pfbench.feature_registry import FeatureSpec, resolve_columns
from pfbench.market_config import get_market_split

from algorithms.lightgbm_twostage.config import MarketConfig
from algorithms.lightgbm_twostage.train import (
    TrainConfig,
    compute_baselines,
    single_pass_predict,
)
from algorithms.lightgbm_twostage.features import build_features

SUPPORTED_MARKETS = ["neimeng", "chongqing", "jiangsu"]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_single_market(
    market_id: str, mode: str, output_root: Path,
    time_decay: int, top_k: int, freq: str = "1h",
    target: str = None, groups: list = None,
) -> Dict[str, Any]:
    """运行单市场实验，返回汇总指标。"""
    spec = FeatureSpec(target=target, groups=groups)
    resolved = resolve_columns(market_id, spec, freq=freq)
    cfg = MarketConfig.from_resolved_spec(resolved)
    split = get_market_split(market_id)
    algo_dir = "lightgbm_twostage_15min" if freq == "15min" else "lightgbm_twostage"
    algo_label = "LightGBM-TwoStage-15min" if freq == "15min" else "LightGBM-TwoStage"
    out_dir = output_root / market_id / algo_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_id = make_experiment_id(market_id, algo_dir, target=cfg.target_col)
    save_config_snapshot(
        out_dir, exp_id,
        algorithm="lightgbm_twostage",
        market=market_id, target=cfg.target_col, freq=freq,
        extra={
            "feature_groups": {n: len(g.cols) for n, g in resolved.groups.items()},
            "test_start": str(split.test_start),
            "test_end": str(split.test_end),
        },
    )

    print(f"\n{'='*60}")
    print(f"  {market_id} — {algo_label} ({mode})")
    print(f"  target={cfg.target_col}, test_start={split.test_start} (from market yaml)")
    print(f"  feature_groups: { {n: len(g.cols) for n, g in resolved.groups.items()} }")
    print(f"{'='*60}")

    df, meta = load_market_data(market_id)
    print(f"  数据加载: {len(df)} 行 (15min), {meta.get('time_range', 'N/A')}")

    feat = build_features(df, cfg, freq=freq)
    x_cols = [c for c in feat.columns if c not in ("trade_date", "y")]
    print(f"  特征表: {len(feat)} 行 × {len(x_cols)} 特征, {feat['trade_date'].nunique()} 交易日")
    print(f"  特征前10: {x_cols[:10]}...")

    nan_rate = feat[x_cols].isnull().mean()
    high_nan = nan_rate[nan_rate > 0.1]
    if len(high_nan) > 0:
        print(f"  ⚠ 高 NaN 特征: {dict(high_nan.round(3))}")

    tc = TrainConfig(
        val_days=cfg.val_days,
        two_stage=(mode == "two_stage"),
        quantile_mode=(mode == "quantile"),
        quantile_two_stage=(mode == "quantile_two_stage"),
        time_decay_half_life=time_decay,
        feature_select_top_k=top_k,
        adaptive_naive_blend=True,
        dynamic_floor_value=True,
        residual_correction=True,
    )

    print(f"\n  Single train-predict: test_start={split.test_start}, val={tc.val_days}d")
    if tc.two_stage:
        print(f"  Two-Stage: floor_price={cfg.floor_price}")
    if tc.quantile_mode:
        print(f"  Quantile mode: {[0.10, 0.25, 0.50, 0.75, 0.90]}")
    if tc.time_decay_half_life > 0:
        print(f"  Time decay: half_life={tc.time_decay_half_life}d")
    if tc.feature_select_top_k > 0:
        print(f"  Feature selection: top-{tc.feature_select_top_k}")

    info, last_payload, all_preds = single_pass_predict(
        feat, cfg, tc,
        test_start=split.test_start, test_end=split.test_end,
    )

    print(f"\n  {'─'*40}")
    print(f"  Single-pass Results:")
    print(f"    Model MAE  = {info['mae']:.2f}")
    print(f"    Model RMSE = {info['rmse']:.2f}")
    print(f"    Model MAPE = {info['mape']:.4f}")
    if info.get("profile_corr") is not None:
        print(f"    Profile Corr = {info['profile_corr']:.4f}")
    if info.get("naive_yesterday"):
        n_mae = info["naive_yesterday"]["mae"]
        improv = (n_mae - info["mae"]) / n_mae * 100
        print(f"    Naive MAE  = {n_mae:.2f} (yesterday same hour, internal)")
        print(f"    Improvement vs naive = {improv:+.1f}%")

    baselines = compute_baselines(feat, split.test_start, split.test_end)

    if all_preds is not None:
        tdf = all_preds.copy()
        tdf["error"] = (tdf["pred"] - tdf["y"]).abs()
        floor_mask = tdf["y"] <= cfg.floor_price
        normal_mask = tdf["y"] > cfg.floor_price * 4
        if floor_mask.any():
            print(f"    地板价时段 (y<={cfg.floor_price}): "
                  f"MAE={tdf.loc[floor_mask, 'error'].mean():.1f} ({floor_mask.sum()} 样本)")
        if normal_mask.any():
            print(f"    正常价时段 (y>{cfg.floor_price * 4}): "
                  f"MAE={tdf.loc[normal_mask, 'error'].mean():.1f} ({normal_mask.sum()} 样本)")

    results: Dict[str, Any] = {
        "experiment_id": exp_id,
        "market_id": market_id,
        "algorithm": algo_label,
        "mode": mode,
        "freq": freq,
        "target_col": cfg.target_col,
        "feature_spec": resolved.to_dict(),
        "feature_count": info["feature_count"],
        "n_dates": int(feat["trade_date"].nunique()),
        "n_train_samples": info["n_train_samples"],
        "n_val_samples": info["n_val_samples"],
        "n_test_samples": info["n_test_samples"],
        "train_days": info["train_days"],
        "val_days": info["val_days"],
        "test_days": info["test_days"],
        "test_date_start": info["test_date_start"],
        "test_date_end": info["test_date_end"],
        "mae": info["mae"],
        "rmse": info["rmse"],
        "mape": info["mape"],
        "profile_corr": info.get("profile_corr"),
        "naive_yesterday": info.get("naive_yesterday"),
        "floor_pred_value": info.get("floor_pred_value"),
        "naive_blend_alpha": info.get("naive_blend_alpha"),
        "residual_gamma": info.get("residual_gamma"),
        "baselines": baselines,
    }
    for k in ("floor_threshold", "floor_actual", "floor_pred_count", "quantile_params"):
        if k in info:
            results[k] = info[k]

    pred_path = None
    if all_preds is not None:
        all_preds_out = all_preds.copy()
        if freq == "15min":
            all_preds_out["ts"] = (
                pd.to_datetime(all_preds_out["trade_date"])
                + pd.to_timedelta(all_preds_out["step"] * 15, unit="m")
            )
            pred_fname = "test_predictions_15min.csv"
        else:
            all_preds_out["ts"] = (
                pd.to_datetime(all_preds_out["trade_date"])
                + pd.to_timedelta(all_preds_out["hour"], unit="h")
            )
            pred_fname = "test_predictions_hourly.csv"
        all_preds_out = all_preds_out.rename(columns={"y": "actual", "pred": "predicted"})
        pred_path = out_dir / pred_fname
        all_preds_out[["ts", "actual", "predicted"]].to_csv(pred_path, index=False)
        print(f"  Predictions → {pred_path}")

    if pred_path is not None and pred_path.exists():
        try:
            from pfbench.metrics import evaluate_predictions_csv
            results["extended_metrics"] = evaluate_predictions_csv(pred_path)
        except Exception as exc:
            print(f"  WARN extended_metrics 计算失败: {exc}")
            results["extended_metrics"] = None

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  Metrics → {metrics_path}")

    if all_preds is not None:

        from pfbench.plotting import plot_weekly_predictions
        plot_dir = out_dir / "plots"
        plots = plot_weekly_predictions(
            all_preds_out, plot_dir, market_id, algo_label,
            target_col=cfg.target_col, freq=freq,
        )
        print(f"  图表 → {plot_dir}/ ({len(plots)} 张)")

    if last_payload:
        model_obj = last_payload.get("model")
        if model_obj and hasattr(model_obj, "feature_importances_"):
            imp_cols = last_payload.get("feature_columns", x_cols)
            imp_df = pd.DataFrame({
                "feature": imp_cols,
                "importance": model_obj.feature_importances_,
            }).sort_values("importance", ascending=False)
            imp_path = out_dir / "feature_importance.csv"
            imp_df.to_csv(imp_path, index=False)
            print(f"  Feature importance → {imp_path}")
            print(f"  Top-10 features:")
            for _, r in imp_df.head(10).iterrows():
                print(f"    {r['feature']:50s} {int(r['importance']):>6d}")

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="LightGBM-TwoStage 实验")
    ap.add_argument("--market", "-m", default="all", help="市场 ID（逗号分隔）或 all")
    ap.add_argument("--freq", choices=["1h", "15min"], default="1h",
                    help="预测粒度：1h（默认）或 15min")
    ap.add_argument("--output-root", type=Path, default=ROOT / "runs" / "predictions")
    ap.add_argument("--mode", choices=["regression", "two_stage", "quantile", "quantile_two_stage"],
                    default="two_stage", help="建模模式")
    ap.add_argument("--time-decay", type=int, default=30, help="时间衰减半衰期（天，0=关闭）")
    ap.add_argument("--top-k", type=int, default=0, help="特征选择 top-K（0=不选择）")
    ap.add_argument("--target", default=None,
                    help="覆盖默认 target；必须在该市场 yaml 的 alt_targets 中")
    ap.add_argument("--groups", default=None,
                    help="逗号分隔的 feature groups（默认使用 yaml 中 enabled=true 的类别）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _setup_logging(args.verbose)

    targets = (
        list(SUPPORTED_MARKETS) if args.market == "all"
        else [m.strip() for m in args.market.split(",")]
    )

    groups = None
    if args.groups:
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    failed = []
    all_results = []
    for mid in targets:
        if mid not in SUPPORTED_MARKETS:
            print(f"未知市场: {mid}，可选 {SUPPORTED_MARKETS}")
            failed.append(mid)
            continue
        try:
            r = run_single_market(
                mid, args.mode, args.output_root,
                args.time_decay, args.top_k, freq=args.freq,
                target=args.target, groups=groups,
            )
            all_results.append(r)
        except Exception as e:
            print(f"  FAIL: {e}")
            logging.exception("experiment failed for %s", mid)
            failed.append(mid)

    print(f"\n{'='*70}")
    print("汇总:")
    print(f"{'='*70}")
    for r in all_results:
        n = r.get("naive_yesterday") or {}
        naive_str = f"  Naive={n['mae']:.2f}" if n else ""
        corr = r.get("profile_corr")
        corr_str = f"  Corr={corr:.4f}" if corr is not None else ""
        print(f"  {r['market_id']:12s}  MAE={r['mae']:8.2f}  "
              f"RMSE={r['rmse']:8.2f}  MAPE={r['mape']:.4f}{corr_str}{naive_str}")
    if failed:
        print(f"\n失败: {failed}")
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
