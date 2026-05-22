"""统一冻结数据集：构建、加载、校验。

外部 API：
    load_market_data(market_id, version="v1") -> (df, meta)
    build_market(market_id, version="v1", force=False) -> meta
    list_dataset_versions(market_id) -> list[str]
"""

from .builder import BuildError, build_market, list_dataset_versions
from .loader import load_market_data, load_meta

__all__ = [
    "BuildError",
    "build_market",
    "list_dataset_versions",
    "load_market_data",
    "load_meta",
]
