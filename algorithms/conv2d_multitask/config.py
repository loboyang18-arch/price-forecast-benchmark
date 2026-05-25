"""Conv2D-MultiTask — 市场配置。

每市场定义三组特征通道（与 lightgbm_baseline 语义一致）：
  lag0_cols：D 日 boundary/预测/计划值（不 shift）
  lag1_cols：D-1 日历史价格/出清
  lag2_cols：D-2 日实际运行值

注意：test_start / test_end 由 config/markets/<market_id>.yaml 统一定义，
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
    floor_price: float = 50.0


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
        "price_hbd",
        "price_hbx",
        "price_sudun_500kv1m_energy",
        "price_sudun_500kv1m_cong",
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
    ],
    lag1_cols=[
        "market_clearing_price",
        "realtime_clearing_price",
        "reliability_clearing_price",
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
        "price_dayahead_jb_node_江北",
        "price_realtime_jn_final_江南",
        "price_realtime_jb_final_江北",
    ],
    lag2_cols=[
        "load_actual_total_汇总",
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


# ── Conv2D 架构参数（市场无关） ───────────────────────────────
LOOKBACK_DAYS = 7        # 回看 7 天
SLOTS_PER_HOUR = 4       # 15min 粒度
SLOTS_PER_DAY = 96
CONTEXT_BEFORE = 1       # 1h 模式：前后各 1 小时上下文
CONTEXT_AFTER = 1
SLOTS_BEFORE = 7         # 15min 模式：前 7 步 + 后 4 步
SLOTS_AFTER = 4

# 时间编码通道数（sin/cos hour + sin/cos dow）
N_TIME_ENC = 4
