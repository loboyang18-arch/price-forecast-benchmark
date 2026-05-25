"""LightGBM baseline — 市场特征映射配置。

每个市场定义：
  target_col   — 预测目标列
  lag0_cols    — D日已知预测/计划值（不 shift）
  lag1_cols    — D-1日出清/价格类（shift 24h）
  lag2_cols    — D-2日实际运行值（shift 48h）

注意：test_start 由 config/markets/<market_id>.yaml 统一定义，
      通过 pfbench.market_config.get_market_split() 获取。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MarketConfig:
    market_id: str
    target_col: str
    lag0_cols: List[str]
    lag1_cols: List[str]
    lag2_cols: List[str]
    price_lag_hours: List[int] = field(default_factory=lambda: [24, 48, 168])


NEIMENG = MarketConfig(
    market_id="neimeng",
    target_col="price_unified",
    lag0_cols=[
        "load_forecast",
        "renewable_forecast",
        "wind_forecast",
        "solar_forecast",
        "east_send_forecast",
        "reserve_pos_capacity",
        "reserve_neg_capacity",
        "price_dayahead_preclear_energy",
    ],
    lag1_cols=[
        "price_unified",
        "price_sudun_500kv1m_nodal",
        "price_sudun_500kv1m_energy",
        "price_sudun_500kv1m_cong",
        "price_hongjing_220kv1m_nodal",
        "price_hongjing_220kv1m_energy",
        "price_hongjing_220kv1m_cong",
        "price_hbd",
        "price_hbx",
    ],
    lag2_cols=[
        "load_actual",
        "renewable_actual",
        "wind_actual",
        "solar_actual",
    ],
)

CHONGQING = MarketConfig(
    market_id="chongqing",
    target_col="market_clearing_price",
    lag0_cols=[
        "total_load_pred_v1",
        "total_gen_pred_v1",
        "renewable_pred_v1",
        "solar_pred_v1",
        "wind_pred_v1",
        "hydro_pred_v1",
        "trans_pred_v1",
        "nonmarket_gen_pred_v1",
        "temperature_2m",
        "shortwave_radiation",
        "wind_speed_10m",
        "cloud_cover",
    ],
    lag1_cols=[
        "market_clearing_price",
        "market_clearing_power",
        "realtime_clearing_price",
        "realtime_clearing_energy",
        "reliability_clearing_price",
        "reliability_clearing_power",
    ],
    lag2_cols=[
        "total_load_actual",
        "total_gen_actual",
        "renewable_actual",
        "hydro_actual",
        "trans_actual",
        "nonmarket_gen_actual",
    ],
)

JIANGSU = MarketConfig(
    market_id="jiangsu",
    target_col="price_dayahead_jn_node_江南",
    lag0_cols=[
        "load_forecast_boundary_汇总",
        "wind_forecast_boundary_江北",
        "wind_forecast_boundary_江南",
        "pv_forecast_boundary_江北",
        "pv_forecast_boundary_江南",
        "gas_plan_boundary_江北",
        "gas_plan_boundary_江南",
        "receive_plan_boundary_汇总",
        "reserve_positive_汇总",
        "reserve_negative_汇总",
    ],
    lag1_cols=[
        "price_dayahead_jn_node_江南",
        "price_dayahead_jn_江南",
        "price_dayahead_jb_node_江北",
        "price_dayahead_jb_江北",
        "price_realtime_jn_final_江南",
        "price_realtime_jb_final_江北",
    ],
    lag2_cols=[
        "load_actual_total_汇总",
        "gas_actual_江北",
        "gas_actual_江南",
        "pv_actual_江北",
        "pv_actual_江南",
        "wind_actual_江北",
        "wind_actual_江南",
        "receive_actual_huadong_华东",
    ],
)

MARKET_CONFIGS: Dict[str, MarketConfig] = {
    "neimeng": NEIMENG,
    "chongqing": CHONGQING,
    "jiangsu": JIANGSU,
}
