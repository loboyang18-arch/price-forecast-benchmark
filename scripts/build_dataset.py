#!/usr/bin/env python3
"""
构建统一冻结数据集（runs/data/<market_id>/<version>/data.parquet）。

示例：
  python scripts/build_dataset.py --list
  python scripts/build_dataset.py --market all
  python scripts/build_dataset.py --market chongqing --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfbench.data import build_market, list_dataset_versions
from pfbench.data.checks import CheckFailure, check_dataset
from pfbench.markets import DEFAULT_WORKSPACE


SUPPORTED_MARKETS = (
    "chongqing",
    "jiangsu",
    "neimeng",
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="构建统一冻结数据集")
    ap.add_argument(
        "--market", "-m",
        default="all",
        help=f"市场 ID 或 all。可选: {', '.join(SUPPORTED_MARKETS)}",
    )
    ap.add_argument("--version", default="v1")
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    ap.add_argument("--force", action="store_true", help="覆盖已存在数据集")
    ap.add_argument("--list", action="store_true", help="列出已构建版本")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _setup_logging(args.verbose)

    if args.list:
        for m in SUPPORTED_MARKETS:
            versions = list_dataset_versions(m)
            print(f"  {m:20s}  versions={versions}")
        return

    if args.market == "all":
        targets = list(SUPPORTED_MARKETS)
    else:
        if args.market not in SUPPORTED_MARKETS:
            ap.error(f"未知 --market {args.market!r}，可选 {SUPPORTED_MARKETS} 或 all")
        targets = [args.market]

    failed: list[tuple[str, str]] = []
    for mid in targets:
        print(f"\n=== {mid} -> {args.version} ===")
        try:
            meta = build_market(
                mid,
                version=args.version,
                workspace=args.workspace,
                force=args.force,
            )
            print(
                f"  {meta['n_rows']} 行 × {meta['n_columns']} 列  "
                f"{meta['time_range'][0]} ~ {meta['time_range'][1]}  "
                f"irregular_steps={meta['irregular_steps']}  "
                f"numeric_na={meta['total_numeric_na']}"
            )

            print(f"\n  校验 {mid}/{args.version} ...")
            r = check_dataset(mid, version=args.version)
            print(
                f"    rows={r['n_rows']}  cols={r['n_columns']}  "
                f"irregular_steps={r['irregular_steps']}  "
                f"numeric_na={r['total_numeric_na']}"
            )
            if r["columns_with_na"]:
                for col, cnt in r["columns_with_na"].items():
                    print(f"    ⚠ {col}: {cnt} NaN")
        except (CheckFailure, Exception) as e:
            print(f"  FAIL: {e}")
            failed.append((mid, str(e)))

    if failed:
        print(f"\n失败 {len(failed)} 个: {failed}")
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
