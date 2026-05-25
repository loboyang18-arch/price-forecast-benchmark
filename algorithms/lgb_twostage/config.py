"""LightGBM-TwoStage — 市场配置。

每个市场定义：
  target_col      — 预测目标列
  boundary_cols   — D 日 boundary / 预测曲线（lag0，不 shift）
  price_cols      — 历史日前出清价格（lag1+，需 shift）
  realtime_cols   — 历史实时出清价格（lag1+，需 shift）
  actual_cols     — 历史实际运行值（lag2，需 shift）
  floor_price     — 地板价阈值（用于 Two-Stage 分类）

注意：test_start 由 config/markets/<market_id>.yaml 统一定义，
      通过 pfbench.market_config.get_market_split() 获取。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MarketConfig:
    market_id: str
    target_col: str
    boundary_cols: List[str]
    price_cols: List[str]
    realtime_cols: List[str]
    actual_cols: Dict[str, str]  # orig_name -> short_name
    floor_price: float = 50.0
    floor_pred_value: float = 30.0
    val_days: int = 7
    step_days: int = 7


NEIMENG = MarketConfig(
    market_id="neimeng",
    target_col="price_unified",
    boundary_cols=[
        "load_forecast",
        "renewable_forecast",
        "wind_forecast",
        "solar_forecast",
        "east_send_forecast",
        "reserve_pos_capacity",
        "reserve_neg_capacity",
        "price_dayahead_preclear_energy",
    ],
    price_cols=[
        "price_unified",
        "price_sudun_500kv1m_nodal",
        "price_sudun_500kv1m_energy",
        "price_sudun_500kv1m_cong",
        "price_hongjing_220kv1m_nodal",
        "price_hbd",
        "price_hbx",
    ],
    realtime_cols=[],
    actual_cols={
        "load_actual": "load_actual",
        "renewable_actual": "re_actual",
        "wind_actual": "wind_actual",
        "solar_actual": "solar_actual",
    },
    floor_price=50.0,
)

CHONGQING = MarketConfig(
    market_id="chongqing",
    target_col="market_clearing_price",
    boundary_cols=[
        "total_load_pred_v1",
        "total_gen_pred_v1",
        "renewable_pred_v1",
        "solar_pred_v1",
        "wind_pred_v1",
        "hydro_pred_v1",
        "trans_pred_v1",
        "nonmarket_gen_pred_v1",
    ],
    price_cols=[
        "market_clearing_price",
        "market_clearing_power",
        "reliability_clearing_price",
    ],
    realtime_cols=[
        "realtime_clearing_price",
        "realtime_clearing_energy",
    ],
    actual_cols={
        "total_load_actual": "load_actual",
        "total_gen_actual": "gen_actual",
        "renewable_actual": "re_actual",
        "hydro_actual": "hydro_actual",
        "trans_actual": "trans_actual",
        "nonmarket_gen_actual": "nonmarket_actual",
    },
    floor_price=50.0,
)

JIANGSU = MarketConfig(
    market_id="jiangsu",
    target_col="price_dayahead_jn_node_江南",
    boundary_cols=[
        "load_forecast_boundary_汇总",
        "receive_plan_boundary_汇总",
        "gas_plan_boundary_江北",
        "gas_plan_boundary_江南",
        "pv_forecast_boundary_江北",
        "pv_forecast_boundary_江南",
        "wind_forecast_boundary_江北",
        "wind_forecast_boundary_江南",
        "reserve_positive_汇总",
        "reserve_negative_汇总",
    ],
    price_cols=[
        "price_dayahead_jn_node_江南",
        "price_dayahead_jn_江南",
        "price_dayahead_jb_node_江北",
        "price_dayahead_jb_江北",
    ],
    realtime_cols=[
        "price_realtime_jn_final_江南",
        "price_realtime_jb_final_江北",
        "price_realtime_jn_node_江南",
        "price_realtime_jb_node_江北",
    ],
    actual_cols={
        "load_actual_total_汇总": "load_actual",
        "wind_actual_江北": "wind_actual_jb",
        "wind_actual_江南": "wind_actual_jn",
        "pv_actual_江北": "pv_actual_jb",
        "pv_actual_江南": "pv_actual_jn",
        "gas_actual_江北": "gas_actual_jb",
        "gas_actual_江南": "gas_actual_jn",
        "receive_actual_huadong_华东": "receive_actual",
    },
    floor_price=50.0,
)

MARKET_CONFIGS: Dict[str, MarketConfig] = {
    "neimeng": NEIMENG,
    "chongqing": CHONGQING,
    "jiangsu": JIANGSU,
}
