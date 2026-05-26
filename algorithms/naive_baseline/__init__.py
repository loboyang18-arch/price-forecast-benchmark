"""朴素基准算法包。

提供三种零参数策略，作为算法对比的"零线"参考：
- lag_1d:           D-1 同时段
- lag_7d:           D-7 同时段（上周同日同时段）
- rolling_7d_mean:  D-1 ~ D-7 同时段均值

参考公司模型研发管理指南 §3.3 / §8.1。
"""
from .predict import (
    STRATEGIES,
    aggregate_target,
    naive_predict,
)

__all__ = ["STRATEGIES", "aggregate_target", "naive_predict"]
