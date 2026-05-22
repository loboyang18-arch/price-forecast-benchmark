"""读取 ``runs/data/<market>/<version>/data.parquet``。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..paths import DATA_DIR


def _dataset_dir(market_id: str, version: str) -> Path:
    return DATA_DIR / market_id / version


def load_meta(market_id: str, version: str = "v1") -> dict[str, Any]:
    path = _dataset_dir(market_id, version) / "meta.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"未找到数据集 meta: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_market_data(
    market_id: str,
    *,
    version: str = "v1",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取一个市场的冻结数据集（全量 15min）。

    返回 (DataFrame[index=ts], meta_dict)。
    """
    d = _dataset_dir(market_id, version)
    pq = d / "data.parquet"
    if not pq.is_file():
        raise FileNotFoundError(f"未找到 parquet: {pq}")
    df = pd.read_parquet(pq)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts").sort_index()
    meta = load_meta(market_id, version)
    return df, meta
