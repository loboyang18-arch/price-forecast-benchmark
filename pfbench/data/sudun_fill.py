"""苏敦节点电价缺测补齐（vendored from neimeng_prj/src/fill_sudun_dws_gaps.py）。

规则（按顺序执行）：

1. 同列、同小时内线性插值 + ffill/bfill
2. 整日全缺时，用 ``price_unified`` 代理苏敦三列

不依赖 ``neimeng_prj``；本仓 builder 通过预处理钩子调用。
"""
from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

SUDUN_COLS: Tuple[str, str, str] = (
    "price_sudun_500kv1m_nodal",
    "price_sudun_500kv1m_energy",
    "price_sudun_500kv1m_cong",
)
UNIFIED_COL = "price_unified"


def _fill_one_hour_series(s: pd.Series) -> pd.Series:
    x = s.copy()
    if x.notna().sum() > 0:
        x = x.interpolate(method="linear", limit_direction="both")
        x = x.ffill().bfill()
    return x


def fill_sudun_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """对 SUDUN_COLS 应用两步补齐，返回副本。索引需为 DatetimeIndex。"""
    cols_list = list(SUDUN_COLS)
    if not any(c in df.columns for c in cols_list):
        return df.copy()

    out = df.copy()
    before = int(out[cols_list].isna().to_numpy().sum())

    for col in cols_list:
        if col not in out.columns:
            continue
        out[col] = (
            out.groupby(pd.DatetimeIndex(out.index).normalize())[col]
            .transform(
                lambda g: g.groupby(pd.DatetimeIndex(g.index).hour)
                .transform(_fill_one_hour_series)
            )
        )

    nodal = cols_list[0]
    if nodal in out.columns and UNIFIED_COL in out.columns:
        days = pd.DatetimeIndex(out.index).normalize().unique()
        n_day_proxy = 0
        for d in days:
            mask = pd.DatetimeIndex(out.index).normalize() == d
            sub_idx = out.index[mask]
            if out.loc[sub_idx, nodal].isna().all():
                u = out.loc[sub_idx, UNIFIED_COL]
                if u.notna().any():
                    for c in cols_list:
                        if c in out.columns:
                            out.loc[sub_idx, c] = u
                    n_day_proxy += 1
        after = int(out[cols_list].isna().to_numpy().sum())
        logger.info(
            "苏敦缺测补齐: 15min NaN %d -> %d（填补 %d）；整日用 %s 共 %d 日",
            before, after, before - after, UNIFIED_COL, n_day_proxy,
        )
    return out


HONGJING_COLS: Tuple[str, str, str] = (
    "price_hongjing_220kv1m_nodal",
    "price_hongjing_220kv1m_energy",
    "price_hongjing_220kv1m_cong",
)


def fill_hongjing_from_unified(df: pd.DataFrame) -> pd.DataFrame:
    """红井节点价整日缺失时，用 price_unified 代理。"""
    cols_list = list(HONGJING_COLS)
    nodal = cols_list[0]
    if nodal not in df.columns or UNIFIED_COL not in df.columns:
        return df.copy()

    out = df.copy()
    before = int(out[nodal].isna().sum())
    if before == 0:
        return out

    days = pd.DatetimeIndex(out.index).normalize().unique()
    n_filled = 0
    for d in days:
        mask = pd.DatetimeIndex(out.index).normalize() == d
        sub_idx = out.index[mask]
        if out.loc[sub_idx, nodal].isna().all():
            u = out.loc[sub_idx, UNIFIED_COL]
            if u.notna().any():
                for c in cols_list:
                    if c in out.columns:
                        out.loc[sub_idx, c] = u
                n_filled += 1

    after = int(out[nodal].isna().sum())
    if n_filled > 0:
        logger.info(
            "红井缺测补齐: NaN %d -> %d（填补 %d）；整日用 %s 共 %d 日",
            before, after, before - after, UNIFIED_COL, n_filled,
        )
    return out


def fill_unified_from_sudun(df: pd.DataFrame) -> pd.DataFrame:
    """price_unified 整日缺失时，用 price_sudun_500kv1m_nodal 代替。"""
    nodal = SUDUN_COLS[0]
    if UNIFIED_COL not in df.columns or nodal not in df.columns:
        return df.copy()

    out = df.copy()
    before = int(out[UNIFIED_COL].isna().sum())
    if before == 0:
        return out

    days = pd.DatetimeIndex(out.index).normalize().unique()
    n_filled = 0
    for d in days:
        mask = pd.DatetimeIndex(out.index).normalize() == d
        sub_idx = out.index[mask]
        if out.loc[sub_idx, UNIFIED_COL].isna().all():
            proxy = out.loc[sub_idx, nodal]
            if proxy.notna().any():
                out.loc[sub_idx, UNIFIED_COL] = proxy
                n_filled += 1

    after = int(out[UNIFIED_COL].isna().sum())
    if n_filled > 0:
        logger.info(
            "统一出清价补齐: NaN %d -> %d（填补 %d）；整日用 %s 共 %d 日",
            before, after, before - after, nodal, n_filled,
        )
    return out
