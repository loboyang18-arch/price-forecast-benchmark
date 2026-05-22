"""江苏数据预处理钩子。

1. shift_ts_to_period_start — 时间戳从区间终点(period-end)转为区间起点(period-start)
2. drop_qflag_columns       — 去掉所有 _qflag 后缀的质量标记列
3. fill_reserve_from_prev_week — 正负备用 2025-11-16/11-23 整日缺失，用前一周同时段填补

不依赖 ``jiangsu_prj``；本仓 builder 通过预处理钩子调用。
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

RESERVE_COLS = (
    "reserve_negative_汇总",
    "reserve_positive_汇总",
)

_MISSING_DATES = [
    pd.Timestamp("2025-11-16"),
    pd.Timestamp("2025-11-23"),
]


def shift_ts_to_period_start(df: pd.DataFrame) -> pd.DataFrame:
    """将 period-end 时间戳左移 15 分钟，统一为 period-start 约定。

    江苏源数据 hh_index=1 对应 00:15（区间终点），重庆/内蒙第 1 时段为
    00:00（区间起点）。左移后三市场一致：一天 96 个时段 00:00 ~ 23:45。
    """
    out = df.copy()
    old_min, old_max = out.index.min(), out.index.max()
    out.index = out.index - pd.Timedelta(minutes=15)
    logger.info(
        "时间戳左移 15min (period-end → period-start): %s~%s → %s~%s",
        old_min, old_max, out.index.min(), out.index.max(),
    )
    return out


def drop_qflag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """删除所有 ``_qflag`` 后缀列，返回副本。"""
    qflag_cols = [c for c in df.columns if c.endswith("_qflag")]
    if not qflag_cols:
        return df.copy()
    out = df.drop(columns=qflag_cols)
    logger.info("去掉 %d 个 qflag 列，剩余 %d 列", len(qflag_cols), out.shape[1])
    return out


def fill_reserve_from_prev_week(df: pd.DataFrame) -> pd.DataFrame:
    """正负备用 11-16 / 11-23（周日）整日缺失，用前一周同时段填补。

    前置条件：shift_ts_to_period_start 已执行，一天的 96 时段落在同一
    日历日（00:00 ~ 23:45），可直接用 normalize() 匹配。
    """
    cols = [c for c in RESERVE_COLS if c in df.columns]
    if not cols:
        return df.copy()

    out = df.copy()
    total_filled = 0
    for date in _MISSING_DATES:
        prev = date - pd.Timedelta(days=7)
        day_mask = out.index.normalize() == date
        prev_mask = out.index.normalize() == prev
        day_idx = out.index[day_mask]
        prev_idx = out.index[prev_mask]
        if len(day_idx) == 0 or len(prev_idx) == 0:
            continue
        if len(day_idx) != len(prev_idx):
            logger.warning(
                "%s: 当日 %d 行 vs 前一周 %d 行，跳过",
                date.date(), len(day_idx), len(prev_idx),
            )
            continue
        for col in cols:
            na_count = int(out.loc[day_idx, col].isna().sum())
            if na_count > 0 and out.loc[prev_idx, col].notna().all():
                out.loc[day_idx, col] = out.loc[prev_idx, col].values
                total_filled += na_count
                logger.info(
                    "%s: %s 用 %s 同时段填补 %d 行",
                    col, date.date(), prev.date(), na_count,
                )

    if total_filled > 0:
        remaining = sum(int(out[c].isna().sum()) for c in cols)
        logger.info("备用填补完成: 共填 %d 值，剩余 NaN %d", total_filled, remaining)
    return out
