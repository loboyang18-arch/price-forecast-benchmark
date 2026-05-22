#!/usr/bin/env python3
"""
跨市场电价预测批量评价（外部工程 + 本仓预测）。

示例：
  python scripts/run_benchmark.py --list-markets
  python scripts/run_benchmark.py --markets neimeng_sudun --sources external
  python scripts/run_benchmark.py --markets neimeng_sudun --sources local
  python scripts/run_benchmark.py --csv /path/to/pred.csv --market-id custom
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfbench.evaluate import (
    default_benchmark_output,
    evaluate_market,
    evaluate_markets,
    evaluate_predictions,
    save_results,
)
from pfbench.markets import DEFAULT_WORKSPACE, discover_predictions, list_markets, load_market
from pfbench.report import write_markdown_report


def main() -> None:
    ap = argparse.ArgumentParser(description="跨市场电价预测算法评价")
    ap.add_argument(
        "--markets", "-m",
        default="",
        help=f"逗号分隔市场 ID，可选: {', '.join(list_markets())}",
    )
    ap.add_argument("--list-markets", action="store_true", help="列出已注册市场")
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    ap.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="排行榜输出目录，默认 runs/benchmark/<name>",
    )
    ap.add_argument(
        "--run-name",
        default="latest",
        help="未指定 -o 时使用的子目录名（位于 runs/benchmark/）",
    )
    ap.add_argument(
        "--sources",
        choices=("all", "external", "local"),
        default="all",
        help="预测来源：兄弟工程 external / 本仓 runs/predictions local / all",
    )
    ap.add_argument("--csv", action="append", default=[], help="额外指定预测 CSV（可多次）")
    ap.add_argument("--market-id", default="custom", help="与 --csv 联用的市场 ID")
    ap.add_argument("--max-files", type=int, default=None, help="每市场最多评价文件数（调试）")
    ap.add_argument("--no-extended", action="store_true", help="不使用 price_forecast_eval 扩展指标")
    args = ap.parse_args()

    if args.list_markets:
        for mid in list_markets():
            m = load_market(mid, args.workspace)
            ext = len(discover_predictions(m, args.workspace, sources="external"))
            loc = len(discover_predictions(m, args.workspace, sources="local"))
            print(f"  {mid:20s}  {m.name:12s}  external={ext}  local={loc}")
        return

    out_dir = args.output or default_benchmark_output(args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_extended = not args.no_extended

    if args.csv:
        rows = []
        for c in args.csv:
            p = Path(c)
            if not p.is_file():
                print("SKIP (not found)", c)
                continue
            mcfg = load_market(args.market_id, args.workspace) if args.market_id in list_markets() else None
            ev = evaluate_predictions(
                p,
                market_id=args.market_id,
                test_start=mcfg.test_start if mcfg else None,
                test_end=mcfg.test_end if mcfg else None,
                task=mcfg.task if mcfg else "da",
                use_extended=use_extended,
                source="external",
            )
            pm, sm = ev["point_metrics"], ev["shape_metrics"]
            rows.append({
                "market_id": args.market_id,
                "market_name": mcfg.name if mcfg else args.market_id,
                "region": mcfg.region if mcfg else "custom",
                "source": ev.get("source", "external"),
                "model_key": ev["model_key"],
                "prediction_file": str(p),
                "mae": pm["mae"],
                "rmse": pm["rmse"],
                "mape_pct": pm["mape_pct"],
                "bias": pm["bias"],
                "valid_point_count": pm["valid_point_count"],
                "profile_corr": sm.get("profile_corr"),
                "direction_acc": sm.get("direction_acc"),
                "neg_corr_day_ratio": sm.get("neg_corr_day_ratio"),
                "n_days": sm.get("n_days"),
            })
            jpath = out_dir / f"metrics_{p.stem}.json"
            jpath.write_text(json.dumps(ev, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print("OK", p, f"MAE={pm['mae']:.2f}")

        import pandas as pd
        summary = pd.DataFrame(rows)
        save_results(summary, out_dir, None)
        write_markdown_report(summary, out_dir / "report.md")
        print("→", out_dir)
        return

    if not args.markets:
        ap.error("请指定 --markets 或 --csv")

    market_ids = [x.strip() for x in args.markets.split(",") if x.strip()]
    summary, errors = evaluate_markets(
        market_ids,
        args.workspace,
        use_extended=use_extended,
        max_files=args.max_files,
        sources=args.sources,
    )
    csv_path = save_results(summary, out_dir, errors)
    write_markdown_report(summary, out_dir / "report.md")

    print(summary.to_string(index=False) if len(summary) else "(empty)")
    print(f"\n→ {csv_path}")
    if errors:
        print(f"→ {out_dir / 'errors.txt'} ({sum(len(v) for v in errors.values())} errors)")


if __name__ == "__main__":
    main()
