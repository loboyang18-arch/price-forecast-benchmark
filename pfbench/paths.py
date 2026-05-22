"""工程路径常量（单一来源）。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
MARKETS_DIR = CONFIG_DIR / "markets"
PIPELINES_DIR = CONFIG_DIR / "pipelines"

# 运行产物：benchmark 排行榜、本仓预测 CSV 等
RUNS_DIR = ROOT / "runs"
BENCHMARK_RUNS_DIR = RUNS_DIR / "benchmark"
LOCAL_PREDICTIONS_DIR = RUNS_DIR / "predictions"
# 统一冻结数据集（runs/data/<market>/<version>/）
DATA_DIR = RUNS_DIR / "data"

ALGORITHMS_DIR = ROOT / "algorithms"
PIPELINES_SPEC_DIR = ROOT / "pipelines"
EXTERNAL_DOCS_DIR = ROOT / "external"
