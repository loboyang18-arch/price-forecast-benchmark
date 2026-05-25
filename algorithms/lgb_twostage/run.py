#!/usr/bin/env python3
"""LightGBM-TwoStage — CLI 入口。

用法：
  # 全部市场，默认配置（Two-Stage + 后处理 + Expanding Window CV）
  python algorithms/lgb_twostage/run.py --market all

  # 单市场，启用分位数模式
  python algorithms/lgb_twostage/run.py --market jiangsu --quantile

  # 自定义参数
  python algorithms/lgb_twostage/run.py --market neimeng --two-stage --time-decay 30 --top-k 80
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pfbench.data import load_market_data
from pfbench.market_config import get_market_split

from algorithms.lgb_twostage.config import MARKET_CONFIGS
from algorithms.lgb_twostage.cv import CVConfig, compute_baselines, expanding_window_cv
from algorithms.lgb_twostage.features import build_features


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_single_market(
    market_id: str, mode: str, output_root: Path,
    time_decay: int, top_k: int,
) -> Dict[str, Any]:
    """运行单市场实验，返回汇总指标。"""
    cfg = MARKET_CONFIGS[market_id]
    split = get_market_split(market_id)
    out_dir = output_root / market_id / "lgb_twostage"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {market_id} — LightGBM-TwoStage ({mode})")
    print(f"  target={cfg.target_col}, test_start={split.test_start} (from market yaml)")
    print(f"{'='*60}")

    df, meta = load_market_data(market_id)
    print(f"  数据加载: {len(df)} 行 (15min), {meta.get('time_range', 'N/A')}")

    feat = build_features(df, cfg)
    x_cols = [c for c in feat.columns if c not in ("trade_date", "y")]
    print(f"  特征表: {len(feat)} 行 × {len(x_cols)} 特征, {feat['trade_date'].nunique()} 交易日")
    print(f"  特征前10: {x_cols[:10]}...")

    nan_rate = feat[x_cols].isnull().mean()
    high_nan = nan_rate[nan_rate > 0.1]
    if len(high_nan) > 0:
        print(f"  ⚠ 高 NaN 特征: {dict(high_nan.round(3))}")

    cv = CVConfig(
        val_days=cfg.val_days,
        test_days=cfg.step_days,
        step_days=cfg.step_days,
        two_stage=(mode == "two_stage"),
        quantile_mode=(mode == "quantile"),
        quantile_two_stage=(mode == "quantile_two_stage"),
        time_decay_half_life=time_decay,
        feature_select_top_k=top_k,
        adaptive_naive_blend=True,
        dynamic_floor_value=True,
        residual_correction=True,
    )

    print(f"\n  Expanding Window CV: test_start={split.test_start}, "
          f"val={cv.val_days}d, test={cv.test_days}d, step={cv.step_days}d")
    if cv.two_stage:
        print(f"  Two-Stage: floor_price={cfg.floor_price}")
    if cv.quantile_mode:
        print(f"  Quantile mode: {[0.10, 0.25, 0.50, 0.75, 0.90]}")
    if cv.time_decay_half_life > 0:
        print(f"  Time decay: half_life={cv.time_decay_half_life}d")
    if cv.feature_select_top_k > 0:
        print(f"  Feature selection: top-{cv.feature_select_top_k}")

    folds, last_payload, last_test_df, all_preds = expanding_window_cv(
        feat, cfg, cv,
        test_start=split.test_start, test_end=split.test_end,
    )

    if not folds:
        raise RuntimeError(f"No valid CV folds for {market_id}")

    mean_mae = np.mean([f["test"]["mae"] for f in folds])
    mean_rmse = np.mean([f["test"]["rmse"] for f in folds])
    mean_mape = np.mean([f["test"]["mape"] for f in folds])
    naive_maes = [f["naive_yesterday"]["mae"] for f in folds if f.get("naive_yesterday")]
    mean_naive_mae = float(np.mean(naive_maes)) if naive_maes else None

    print(f"\n  {'─'*40}")
    print(f"  CV Results ({len(folds)} folds):")
    print(f"    Model MAE  = {mean_mae:.2f}")
    print(f"    Model RMSE = {mean_rmse:.2f}")
    print(f"    Model MAPE = {mean_mape:.4f}")
    if mean_naive_mae is not None:
        improv = (mean_naive_mae - mean_mae) / mean_naive_mae * 100
        print(f"    Naive MAE  = {mean_naive_mae:.2f} (yesterday same hour)")
        print(f"    Improvement = {improv:+.1f}%")

    baselines = compute_baselines(feat, split.test_start, split.test_end)

    if last_test_df is not None:
        tdf = last_test_df.copy()
        tdf["error"] = (tdf["pred"] - tdf["y"]).abs()
        floor_mask = tdf["y"] <= cfg.floor_price
        normal_mask = tdf["y"] > cfg.floor_price * 4
        if floor_mask.any():
            print(f"    地板价时段 (y<={cfg.floor_price}): "
                  f"MAE={tdf.loc[floor_mask, 'error'].mean():.1f} ({floor_mask.sum()} 样本)")
        if normal_mask.any():
            print(f"    正常价时段 (y>{cfg.floor_price * 4}): "
                  f"MAE={tdf.loc[normal_mask, 'error'].mean():.1f} ({normal_mask.sum()} 样本)")

    profile_corr = None
    if all_preds is not None and len(all_preds) > 0:
        by_date = all_preds.groupby("trade_date")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
            daily_corrs = by_date.apply(lambda g: g["y"].corr(g["pred"]))
        profile_corr = round(float(daily_corrs.mean()), 4)
        print(f"    Profile Corr = {profile_corr:.4f}")

    results: Dict[str, Any] = {
        "market_id": market_id,
        "algorithm": "LightGBM-TwoStage",
        "mode": mode,
        "target_col": cfg.target_col,
        "feature_count": len(x_cols),
        "n_dates": int(feat["trade_date"].nunique()),
        "cv_folds": len(folds),
        "mean_mae": round(mean_mae, 4),
        "mean_rmse": round(mean_rmse, 4),
        "mean_mape": round(mean_mape, 6),
        "profile_corr": profile_corr,
        "mean_naive_mae": round(mean_naive_mae, 4) if mean_naive_mae else None,
        "baselines": baselines,
        "folds": folds,
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  Metrics → {metrics_path}")

    if all_preds is not None:
        all_preds_out = all_preds.copy()
        all_preds_out["ts"] = pd.to_datetime(all_preds_out["trade_date"]) + pd.to_timedelta(all_preds_out["hour"], unit="h")
        all_preds_out = all_preds_out.rename(columns={"y": "actual", "pred": "predicted"})
        pred_path = out_dir / "test_predictions_hourly.csv"
        all_preds_out[["ts", "actual", "predicted"]].to_csv(pred_path, index=False)
        print(f"  Predictions (全量CV) → {pred_path}")

        from pfbench.plotting import plot_weekly_predictions
        plot_dir = out_dir / "plots"
        plots = plot_weekly_predictions(
            all_preds_out, plot_dir, market_id, "LightGBM-TwoStage",
            target_col=cfg.target_col,
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
    ap.add_argument("--output-root", type=Path, default=ROOT / "runs" / "predictions")
    ap.add_argument("--mode", choices=["regression", "two_stage", "quantile", "quantile_two_stage"],
                    default="two_stage", help="建模模式")
    ap.add_argument("--time-decay", type=int, default=30, help="时间衰减半衰期（天，0=关闭）")
    ap.add_argument("--top-k", type=int, default=0, help="特征选择 top-K（0=不选择）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _setup_logging(args.verbose)

    targets = (
        list(MARKET_CONFIGS.keys()) if args.market == "all"
        else [m.strip() for m in args.market.split(",")]
    )

    failed = []
    all_results = []
    for mid in targets:
        if mid not in MARKET_CONFIGS:
            print(f"未知市场: {mid}，可选 {list(MARKET_CONFIGS.keys())}")
            failed.append(mid)
            continue
        try:
            r = run_single_market(mid, args.mode, args.output_root, args.time_decay, args.top_k)
            all_results.append(r)
        except Exception as e:
            print(f"  FAIL: {e}")
            logging.exception("experiment failed for %s", mid)
            failed.append(mid)

    print(f"\n{'='*70}")
    print("汇总:")
    print(f"{'='*70}")
    for r in all_results:
        naive_str = f"  Naive={r['mean_naive_mae']:.2f}" if r.get("mean_naive_mae") else ""
        print(f"  {r['market_id']:12s}  MAE={r['mean_mae']:8.2f}  "
              f"RMSE={r['mean_rmse']:8.2f}  MAPE={r['mean_mape']:.4f}{naive_str}")
    if failed:
        print(f"\n失败: {failed}")
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
