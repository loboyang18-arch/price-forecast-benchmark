#!/usr/bin/env python3
"""
规范测试流程入口（占位）。

后续在此串联：数据准备 → 本仓算法推理 → 写入 runs/predictions → 调用评价。
当前请使用 scripts/run_benchmark.py 评价已有预测 CSV。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfbench.paths import PIPELINES_DIR, PIPELINES_SPEC_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description="电价预测规范测试流程（开发中）")
    ap.add_argument("--list", action="store_true", help="列出已注册的流程配置")
    ap.add_argument("pipeline_id", nargs="?", help="config/pipelines 下的流程 ID（不含扩展名）")
    args = ap.parse_args()

    if args.list:
        specs = sorted(PIPELINES_DIR.glob("*.yaml"))
        if not specs:
            print("(无流程配置，请在 config/pipelines/ 添加 YAML)")
        for p in specs:
            print(f"  {p.stem}")
        print(f"\n流程说明目录: {PIPELINES_SPEC_DIR}")
        return

    if not args.pipeline_id:
        ap.print_help()
        sys.exit(0)

    cfg = PIPELINES_DIR / f"{args.pipeline_id}.yaml"
    if not cfg.is_file():
        print(f"未找到流程配置: {cfg}")
        sys.exit(1)
    print("流程执行尚未实现。请先完成 algorithms/ 中的推理脚本，")
    print("将预测写入 runs/predictions/<market_id>/，再运行:")
    print(f"  python scripts/run_benchmark.py --markets <id> --sources local -o runs/benchmark/{args.pipeline_id}")


if __name__ == "__main__":
    main()
