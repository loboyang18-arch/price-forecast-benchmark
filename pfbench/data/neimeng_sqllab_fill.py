"""内蒙古 SQL 取数数据源（kronos_prod.dwd_logic_point_detail）预处理钩子。

按 `doc/内蒙SQL取数接入设计_V1.0.md` 实施，方案 B：

1. ``rename_neimeng_sqllab_columns`` — 中→英列名映射 + 剔除 8 列冗余/空列/短板
2. ``encode_renewable_surplus_level`` — 4 类中文标签 → int8 (0/1/2/3)
3. ``ffill_daily_bid_avg`` — 3 列日级申报电价当日 96 点 ffill

注：``GridSpotRtClearingPrice``（grid_spot_rt_clearing_price）按方案 B 在 step 1
直接 drop，因此不需要"D-1 7 点后填 0"的特殊钩子。
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# 中文列名 → 英文列名映射；值为 None 表示直接 drop
COLUMN_MAP: Dict[str, Optional[str]] = {
    # ── BOUNDARY (16 列，window_lag=0) ──────────────────────────────────
    "竞价空间预测D-1": "bidding_space_forecast_d1",
    "统调负荷预测D-1": "dispatched_load_forecast_d1",
    "统调负荷预测D-2": "dispatched_load_forecast_d2",
    "东送计划预测D-1": "eastward_delivery_plan_forecast_d1",
    "东送计划预测D-2": "eastward_delivery_plan_forecast_d2",
    "非市场出力计划D-1": "non_market_output_plan_d1",
    "非市场出力计划D-2": "non_market_output_plan_d2",
    "全网新能源预测D-1": "grid_renewable_forecast_d1",
    "全网新能源预测D-2": "grid_renewable_forecast_d2",
    "全网光伏预测D-1": "grid_pv_forecast_d1",
    "全网光伏预测D-2": "grid_pv_forecast_d2",
    "全网风电预测D-1": "grid_wind_power_forecast_d1",
    "全网风电预测D-2": "grid_wind_power_forecast_d2",
    "水电出力计划D-1": "hydro_output_plan_d1",
    "水电出力计划D-2": "hydro_output_plan_d2",
    "可再生能源富余程度": "renewable_energy_surplus_level",

    # ── BOUNDARY_DM1 (3 列，window_lag=1d) ──────────────────────────────
    "正备用容量": "upward_reserve_capacity",
    "负备用容量": "downward_reserve_capacity",
    "省内电力平衡裕度": "provincial_power_balance_margin",

    # ── ACTUAL (7 列，window_lag=2d) ────────────────────────────────────
    "统调负荷实测": "dispatched_load_actual",
    "东送计划实测": "eastward_delivery_plan_actual",
    "非市场出力计划实测": "non_market_output_plan_actual",
    "全网新能源实测": "grid_renewable_actual",
    "全网光伏实测": "grid_pv_actual",
    "全网风电实测": "grid_wind_power_actual",
    "水电出力计划实测": "hydro_output_plan_actual",

    # ── CLEARING_RT (4 列，window_lag=2d) ───────────────────────────────
    "实时节点电价参考_红井站1M": "rt_node_price_ref_nm_hongjing_sta_220kv_1m",
    "现货市场平均申报电价_燃煤": "spot_market_avg_declared_price_coal",
    "现货市场平均申报电价_风电": "spot_market_avg_declared_price_wind_power",
    "现货市场平均申报电价_光伏": "spot_market_avg_declared_price_pv",

    # ── CLEARING_DA (3 列，window_lag=3d) ───────────────────────────────
    "全网统一出清电价": "grid_unified_clearing_price",
    "呼包东统一出清电价": "hubaodong_unified_clearing_price",
    "呼包西统一出清电价": "hubaoxi_unified_clearing_price",

    # ── CLEARING_RT_NODAL (3 列，window_lag=4d；含 target) ──────────────
    "实时节点电价_红井站1M": "rt_node_price_nm_hongjing_sta_220kv_1m",
    "实时节点电能价格_红井站1M": "rt_node_energy_price_nm_hongjing_sta_220kv_1m",
    "实时节点阻塞价格_红井站1M": "rt_node_congestion_price_nm_hongjing_sta_220kv_1m",

    # ── 剔除列 (8 列) ────────────────────────────────────────────────
    "实际竞价空间":              None,  # 全空
    "实时节点电价_红井站2M":      None,  # Q3 与 1M 同值
    "实时节点电能价格_红井站2M":   None,
    "实时节点阻塞价格_红井站2M":   None,
    "实时节点电价参考_红井站2M":   None,  # Q3
    "是否弃风":                  None,  # Q2 冗余于 renewable_energy_surplus_level
    "是否弃光":                  None,
    "全网现货实时出清电价":       None,  # 方案 B：仅 7 个月覆盖 + 半天 0 填充
    # 时间索引相关
    "交易日": None,
    "时段":   None,
}


def rename_neimeng_sqllab_columns(df: pd.DataFrame) -> pd.DataFrame:
    """中→英列名映射，同时 drop 不入库的列。

    builder._read_source 已把 ts 列从 ``时间标签`` 改为 ``ts`` 并设为索引；
    本函数只处理 df.columns 中的数据列。
    """
    drop_cols = [c for c, v in COLUMN_MAP.items() if v is None and c in df.columns]
    rename_map = {c: v for c, v in COLUMN_MAP.items() if v is not None and c in df.columns}

    unknown = [c for c in df.columns if c not in COLUMN_MAP]
    if unknown:
        logger.warning("内蒙 SQL 数据出现未知列（未映射，将保留原名）：%s", unknown)

    out = df.drop(columns=drop_cols, errors="ignore").rename(columns=rename_map)
    logger.info(
        "内蒙 SQL 列名映射: 原 %d 列 -> 删 %d 列 / 改名 %d 列 -> 保留 %d 列",
        df.shape[1], len(drop_cols), len(rename_map), out.shape[1],
    )
    return out


# 4 类中文标签 → int8 编码；未知值（含 NaN/空）一律映射 0（"无弃用"）
_CURTAIL_MAP: Dict[str, int] = {
    "无弃风、弃光": 0,
    "无弃风、弃光、":  0,           # 容错：尾部冗余字符
    "弃风":         1,
    "只弃风":       1,
    "弃光":         2,
    "只弃光":       2,
    "弃风弃光":     3,
}


def encode_renewable_surplus_level(df: pd.DataFrame) -> pd.DataFrame:
    """``renewable_energy_surplus_level`` 4 类中文标签 → int8 (0/1/2/3)。"""
    col = "renewable_energy_surplus_level"
    if col not in df.columns:
        return df

    out = df.copy()
    raw = out[col]
    nunique_before = raw.dropna().nunique()
    distinct = sorted(raw.dropna().astype(str).unique())

    mapped = raw.astype(str).str.strip().map(_CURTAIL_MAP)
    n_unknown = int((mapped.isna() & raw.notna()).sum())
    if n_unknown > 0:
        unknown_vals = sorted({v for v in raw.dropna().astype(str)
                               if v.strip() not in _CURTAIL_MAP})
        logger.warning(
            "renewable_energy_surplus_level 有 %d 个值未在 _CURTAIL_MAP 中：%s "
            "（已映射为 0=无弃用）",
            n_unknown, unknown_vals,
        )
    out[col] = mapped.fillna(0).astype("int8")
    logger.info(
        "renewable_energy_surplus_level 编码: %d 种 distinct (%s) -> int8 (0~3)",
        nunique_before, distinct,
    )
    return out


_DAILY_BID_AVG_COLS = (
    "spot_market_avg_declared_price_coal",
    "spot_market_avg_declared_price_wind_power",
    "spot_market_avg_declared_price_pv",
)


def ffill_daily_bid_avg(df: pd.DataFrame) -> pd.DataFrame:
    """3 列日级申报电价 → 当日 96 点 ffill；跨日不传递。

    SQL 端这 3 列只在每天 00:00（或一日内某固定时刻）有 1 个非空值，需要
    在当日范围内 ffill 到 96 个 15min 槽位。
    """
    cols = [c for c in _DAILY_BID_AVG_COLS if c in df.columns]
    if not cols:
        return df

    out = df.copy()
    before = {c: int(out[c].notna().sum()) for c in cols}
    day_key = pd.DatetimeIndex(out.index).normalize()
    for c in cols:
        out[c] = out.groupby(day_key)[c].transform(lambda s: s.ffill())
    after = {c: int(out[c].notna().sum()) for c in cols}

    logger.info(
        "申报电价日级 ffill: %s",
        ", ".join(f"{c}: {before[c]} -> {after[c]}" for c in cols),
    )
    return out
