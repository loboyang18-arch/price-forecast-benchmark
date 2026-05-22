"""冻结数据集校验：时间完整性与缺测检查。"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .loader import load_market_data

logger = logging.getLogger(__name__)


class CheckFailure(RuntimeError):
    pass


def check_dataset(
    market_id: str,
    *,
    version: str = "v1",
    max_irregular_steps: int = 0,
) -> dict[str, Any]:
    """校验一个市场的冻结数据集。

    检查项：行数 > 0、时间严格递增、无重复、15min 步长连续、数值列缺测统计。
    """
    df, meta = load_market_data(market_id, version=version)
    res: dict[str, Any] = {
        "market_id": market_id,
        "version": version,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "ts_min": str(df.index.min()),
        "ts_max": str(df.index.max()),
    }

    if len(df) == 0:
        raise CheckFailure(f"{market_id}: 行数为 0")

    if not df.index.is_monotonic_increasing:
        raise CheckFailure(f"{market_id}: ts 非严格递增")
    if df.index.duplicated().any():
        raise CheckFailure(f"{market_id}: ts 有重复")

    step = pd.Timedelta(minutes=15)
    diffs = df.index.to_series().diff().dropna()
    irregular = int((diffs != step).sum())
    res["irregular_steps"] = irregular
    if irregular > max_irregular_steps:
        raise CheckFailure(
            f"{market_id}: irregular_steps={irregular} > {max_irregular_steps}"
        )

    num_na = df.select_dtypes(include=["number"]).isna().sum()
    cols_with_na = num_na[num_na > 0]
    res["total_numeric_na"] = int(num_na.sum())
    res["columns_with_na"] = {str(c): int(v) for c, v in cols_with_na.items()}

    return res
