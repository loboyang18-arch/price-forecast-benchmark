"""读取 ``runs/data/<market>/<version>/data.parquet``。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from ..paths import DATA_DIR, MARKETS_DIR


def _dataset_dir(market_id: str, version: str) -> Path:
    return DATA_DIR / market_id / version


def _resolve_version(market_id: str, version: Optional[str]) -> str:
    """version 未显式指定时，从 ``config/markets/<id>.yaml`` 的 ``data.version`` 读取；
    yaml 也未指定时回落到 ``"v1"``。"""
    if version is not None:
        return version
    path = MARKETS_DIR / f"{market_id}.yaml"
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data_block = raw.get("data") or {}
            v = data_block.get("version")
            if v:
                return str(v)
    return "v1"


def load_meta(market_id: str, version: Optional[str] = None) -> dict[str, Any]:
    v = _resolve_version(market_id, version)
    path = _dataset_dir(market_id, v) / "meta.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"未找到数据集 meta: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_market_data(
    market_id: str,
    *,
    version: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取一个市场的冻结数据集（全量 15min）。

    version 未显式指定时，自动从 ``config/markets/<id>.yaml`` 的 ``data.version`` 读取。

    返回 (DataFrame[index=ts], meta_dict)。
    """
    v = _resolve_version(market_id, version)
    d = _dataset_dir(market_id, v)
    pq = d / "data.parquet"
    if not pq.is_file():
        raise FileNotFoundError(f"未找到 parquet: {pq}")
    df = pd.read_parquet(pq)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts").sort_index()
    meta = load_meta(market_id, v)
    return df, meta
