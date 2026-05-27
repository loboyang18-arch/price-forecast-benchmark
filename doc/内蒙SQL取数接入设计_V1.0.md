# 内蒙 SQL 取数数据源接入设计 V1.0

> 版本：V1.0（2026-05-27 修订 R2，含 6 项决策点拍板 + 5 lag-bucket 重构）
> 状态：**已审阅，准备落地**
> 关联工作清单：`doc/后续工作清单_对齐公司指南V1.0.md`

---

## 0. 修订说明

本文档是与用户对齐 6 个决策点后的最终版。相对初稿（R0）的实质性修订：

| 决策点 | 初稿 (R0) | 最终 (R2) |
|---|---|---|
| Q1 旧 dws v1 | 保留并标 deprecated | **直接删除**，不保留 |
| Q2 弃风/弃光列 | 合并为 `is_curtailed` | 进一步：**只留 4 类原始** `renewable_energy_surplus_level`，连 `is_curtailed` 都不要（可由前者派生） |
| Q3 节点价 2M | 保留作 alt_target | **全部去掉**（与 1M 同值，4 列剔除） |
| Q4 LightGBM 多 lag shift | `BOUNDARY_DM1: ["1d","2d","3d"]` | **作废**：5 lag-bucket 设计不再使用"同一列多 shift 副本"，LightGBM 与 Conv2D **共用同一套** `window_lag` |
| Q5 全量回归 | 仅 1h | **1h + 15min 全跑** |
| Q6 commit 策略 | 分 5 个 | **一次性 commit** |
| Conv2D 特征拼装 | stream 三相位（boundary=0/history=1/actual=2） | **5 lag-bucket**：按可用性 lag 分 0/1d/2d/3d/4d 五桶，每列按其桶取相应 7 天窗口，每列 1 通道（无膨胀） |
| 列命名 | 业务语义短名（`load_forecast_d1`） | **按 SQL `point_identifier` 转 snake_case**（`dispatched_load_forecast_d1`） |

---

## 1. 背景与目标

### 1.1 现状

本工程内蒙市场目前使用兄弟工程产出的中间数据集
`${WORKSPACE_ROOT}/neimeng_prj/output/dws_15min_features_ext_data4_hongjing.csv`
（21 列特征 + ts，覆盖 2025-01-13 ~ 2026-04-17 约 460 天）。该 dws 默认按"D-1 末已知 = lag0"假设构造特征，**未严格对齐生产部署时的真实数据可用性**——例如节点价实测实际入库延迟最长 4 天，旧 dws 当作 1d 可用，存在未来信息泄漏。

### 1.2 新数据源

`runs/data/neimeng/sqllab_电价模型取数daily_20260527T034109.csv`：

- **100 000 行 × 47 列**（44 个特征列 + 3 个时间索引列）
- 时间覆盖 **2023-07-22 ~ 2026-05-28**（约 1042 天，比旧 dws 多 +125%）
- 严格 15min 步长（99 999/99 999 步长一致）
- **取数时刻定义为 D-1 09:30**（业务约束：日前预测必须在 D-1 上午 09:30 同步完成）
- 来自 `kronos_prod.dwd_logic_point_detail`，每列对应一个 `point_identifier`（见 §4 命名表）

### 1.3 核心目标

1. **真实因果对齐**：所有特征的窗口位置以 D-1 09:30 取数瞬间文件中各列的实际可用性为准。
2. **算法侧统一**：LightGBM / Conv2D / ResConv2D **共用同一套 `window_lag` 配置**，不再有"表格模型多 shift / 深度模型 stream 相位"的分裂。
3. **零侵入数据接入**：复用现有 `pfbench.data.builder + feature_registry + lag_resolver` 框架，只改 yaml + 增加 5 个预处理钩子；feature_registry 加一个字段 `window_lag`、加一个类别 `BOUNDARY_DM1` 和 `CLEARING_RT_NODAL`。
4. **回归对照**：接入后必须重跑内蒙四算法（LightGBM-Baseline / LightGBM-TwoStage / Conv2D-MultiTask / ResConv2D）的 1h+15min 实验，并写入 RESULTS.md。

---

## 2. 5 lag-bucket 设计（核心设计）

### 2.1 设计理念

**每列只有一个数据可用性 lag**（取数时刻 D-1 09:30 该列最末非空日距 D 日的天数），按此分 5 桶：

| Bucket | 可用性 lag | Conv2D 7 天窗口位置 | LightGBM `lag_periods` |
|---|---|---|---|
| `lag0`  | 0   | `[D-6, D]`     | 0（不 shift） |
| `lag1d` | 1d  | `[D-7, D-1]`   | 1d shift（即 96 步） |
| `lag2d` | 2d  | `[D-8, D-2]`   | 2d shift |
| `lag3d` | 3d  | `[D-9, D-3]`   | 3d shift |
| `lag4d` | 4d  | `[D-10, D-4]`  | 4d shift |

### 2.2 关键性质

- **每列 1 通道，无膨胀**：不再做"同列多 shift 副本"，与初稿的 `["1d","2d","3d"]` 多 lag 写法决裂
- **Conv2D 与 LightGBM 共用 `window_lag`**：LightGBM 端把 `window_lag` 直接当成单一 shift periods，深度模型端按此偏移 7 天窗口起点
- **零信息泄漏**：5 桶相位都是后向偏移，不存在 `D + k` 这种未来时点
- **5 桶都是连续 7 天有真实数据**：因为 `window_lag` 取的就是"最末非空日"，窗口右端是该列最后一个真实日

### 2.3 与原 Conv2D `STREAM_DAY_OFFSET` 的关系

| 原设计（3 stream 相位） | 新设计（5 lag-bucket） |
|---|---|
| `boundary` stream → offset=0 | `lag0` bucket → offset=0 |
| `history` stream → offset=1 | `lag1d` bucket → offset=1 |
| `actual` stream → offset=2 | `lag2d` bucket → offset=2 |
| — | `lag3d` bucket → offset=3（新增） |
| — | `lag4d` bucket → offset=4（新增） |

原 3 相位不够用，新数据需要 5 相位（节点价实测 lag=4d、日前出清 lag=3d 都凑不进旧三档）。

### 2.4 三 stream 网络结构是否保留

**保留**：Conv2D / ResConv2D 的网络仍按 boundary / history / actual 三分支（model.py 零改动）。"分支归属"按业务语义决定：

| Feature Group | 业务语义 | window_lag | Conv2D stream 归属 |
|---|---|---|---|
| BOUNDARY              | D 日已发布的事前预测/计划/标签 | 0   | boundary |
| BOUNDARY_DM1          | D-1 当天数据（备用、平衡裕度、实时价早间） | 1d  | history |
| CLEARING_DA           | 日前出清统调价 | 3d  | history |
| CLEARING_RT           | 实时类（参考价、申报电价） | 2d  | history |
| CLEARING_RT_NODAL     | 红井 1M 节点价实测（含 target） | 4d  | history |
| ACTUAL                | 边界类各 actual | 2d  | actual |

也就是说 stream 仍然存在（控制网络分支），但**每个 stream 内部不同 group 可以有不同 window_lag**，窗口起点由 group 的 `window_lag` 决定，不再由 stream 统一决定。

---

## 3. 数据源描述

### 3.1 原始 SQL

数据由 ClickHouse 查询 `kronos_prod.dwd_logic_point_detail` 表生成，按 `(raw_upload_time, point_identifier)` 取 `argMax(val, ts)` 透视成宽表。详细 SQL 见 §A.1（附录）。

### 3.2 与旧 dws v1 的对比

| 维度 | 旧 dws v1 | 新 SQL v2 |
|---|---|---|
| 时间覆盖 | 460 天 (2025-01-13 ~ 2026-04-17) | 1042 天 (2023-07-22 ~ 2026-05-28) |
| 列数（特征） | 21 | 37（剔除 7 列冗余/空列后） |
| Lag 假设 | 全部当 D-1 末已可见 | 严格按取数时刻可用性，5 档 lag |
| 苏敦节点价 | 有 3 列 | 无（新数据源不包含） |

**苏敦节点价的处理**：由于 Q1 决策 v1 不保留，苏敦三列直接放弃（`alt_targets` 中删除）。

---

## 4. 列命名表（37 列）

**命名规则**：按 SQL `point_identifier` 的 PascalCase 转 snake_case；`D-1`/`D-2` 转 `d1`/`d2` 避开 yaml 减法解析。

### 4.1 BOUNDARY (window_lag=0, 16 列)

| point_identifier | 列名 | 备注 |
|---|---|---|
| `BiddingSpaceForecastD-1` | `bidding_space_forecast_d1` | |
| `DispatchedLoadForecastD-1` | `dispatched_load_forecast_d1` | |
| `DispatchedLoadForecastD-2` | `dispatched_load_forecast_d2` | |
| `EastwardDeliveryPlanForecastD-1` | `eastward_delivery_plan_forecast_d1` | |
| `EastwardDeliveryPlanForecastD-2` | `eastward_delivery_plan_forecast_d2` | |
| `NonMarketOutputPlanD-1` | `non_market_output_plan_d1` | |
| `NonMarketOutputPlanD-2` | `non_market_output_plan_d2` | |
| `GridRenewableForecastD-1` | `grid_renewable_forecast_d1` | |
| `GridRenewableForecastD-2` | `grid_renewable_forecast_d2` | |
| `GridPvForecastD-1` | `grid_pv_forecast_d1` | |
| `GridPvForecastD-2` | `grid_pv_forecast_d2` | |
| `GridWindPowerForecastD-1` | `grid_wind_power_forecast_d1` | |
| `GridWindPowerForecastD-2` | `grid_wind_power_forecast_d2` | |
| `HydroOutputPlanD-1` | `hydro_output_plan_d1` | |
| `HydroOutputPlanD-2` | `hydro_output_plan_d2` | |
| `RenewableEnergySurplusLevel` | `renewable_energy_surplus_level` | 4 类，LabelEncoder → int8 |

### 4.2 BOUNDARY_DM1 (window_lag=1d, 4 列)

| point_identifier | 列名 | 备注 |
|---|---|---|
| `UpwardReserveCapacity` | `upward_reserve_capacity` | |
| `DownwardReserveCapacity` | `downward_reserve_capacity` | |
| `ProvincialPowerBalanceMargin` | `provincial_power_balance_margin` | |
| `GridSpotRtClearingPrice` | `grid_spot_rt_clearing_price` | **特殊**：D-1 仅 0~7 点真实，其余填 0 |

### 4.3 ACTUAL (window_lag=2d, 7 列)

| point_identifier | 列名 |
|---|---|
| `DispatchedLoadActual` | `dispatched_load_actual` |
| `EastwardDeliveryPlanActual` | `eastward_delivery_plan_actual` |
| `NonMarketOutputPlanActual` | `non_market_output_plan_actual` |
| `GridRenewableActual` | `grid_renewable_actual` |
| `GridPvActual` | `grid_pv_actual` |
| `GridWindPowerActual` | `grid_wind_power_actual` |
| `HydroOutputPlanActual` | `hydro_output_plan_actual` |

### 4.4 CLEARING_RT (window_lag=2d, 4 列)

| point_identifier | 列名 | 备注 |
|---|---|---|
| `RTNodePriceRef_NmHongJingSta220kV1M` | `rt_node_price_ref_nm_hongjing_sta_220kv_1m` | 节点价参考 |
| `SpotMarketAverageDeclaredPrice_Coal` | `spot_market_avg_declared_price_coal` | 日级，ffill 到 96 点 |
| `SpotMarketAverageDeclaredPrice_WindPower` | `spot_market_avg_declared_price_wind_power` | 日级 ffill |
| `SpotMarketAverageDeclaredPrice_Pv` | `spot_market_avg_declared_price_pv` | 日级 ffill |

### 4.5 CLEARING_DA (window_lag=3d, 3 列)

| point_identifier | 列名 |
|---|---|
| `GridUnifiedClearingPrice` | `grid_unified_clearing_price` |
| `HubaodongUnifiedClearingPrice` | `hubaodong_unified_clearing_price` |
| `HubaoxiUnifiedClearingPrice` | `hubaoxi_unified_clearing_price` |

### 4.6 CLEARING_RT_NODAL (window_lag=4d, 3 列)

| point_identifier | 列名 | 备注 |
|---|---|---|
| `RTNodePrice_NmHongJingSta220kV1M` | `rt_node_price_nm_hongjing_sta_220kv_1m` | **target** + 特征 |
| `RTNodeEnergyPrice_NmHongJingSta220kV1M` | `rt_node_energy_price_nm_hongjing_sta_220kv_1m` | |
| `RTNodeCongestionPrice_NmHongJingSta220kV1M` | `rt_node_congestion_price_nm_hongjing_sta_220kv_1m` | |

### 4.7 剔除清单（7 列）

| point_identifier / SQL 派生 | 原因 |
|---|---|
| `ActualBiddingSpace` | 全空（100% NaN） |
| `RTNodePrice_NmHongJingSta220kV2M` | Q3：与 1M 同值 |
| `RTNodeEnergyPrice_NmHongJingSta220kV2M` | Q3 |
| `RTNodeCongestionPrice_NmHongJingSta220kV2M` | Q3 |
| `RTNodePriceRef_NmHongJingSta220kV2M` | Q3 |
| `是否弃风`（SQL 派生） | Q2：信息冗余于 `renewable_energy_surplus_level` |
| `是否弃光`（SQL 派生） | Q2：同上 |

### 4.8 汇总

```
BOUNDARY          16 列  window_lag=0    →  Conv2D boundary stream
BOUNDARY_DM1       4 列  window_lag=1d   →  Conv2D history stream
CLEARING_DA        3 列  window_lag=3d   →  Conv2D history stream
CLEARING_RT        4 列  window_lag=2d   →  Conv2D history stream
CLEARING_RT_NODAL  3 列  window_lag=4d   →  Conv2D history stream
ACTUAL             7 列  window_lag=2d   →  Conv2D actual stream
─────────────────────
合计              37 列（含 target）

target: rt_node_price_nm_hongjing_sta_220kv_1m
```

---

## 5. feature_registry 改造

### 5.1 `FEATURE_GROUPS` 扩展

```python
FEATURE_GROUPS = (
    "BOUNDARY",            # 已有
    "BOUNDARY_CLEARED",    # 已有
    "BOUNDARY_DM1",        # ← 新增（D-1 当天数据）
    "WEATHER",             # 已有
    "CLEARING_DA",         # 已有
    "CLEARING_RT",         # 已有
    "CLEARING_RT_NODAL",   # ← 新增（节点价实测 lag=4d，与一般 RT 区分）
    "ACTUAL",              # 已有
    "CALENDAR",            # 已有
    "DERIVED",             # 已有
)
```

### 5.2 `FeatureGroup` / `ResolvedGroup` 增加 `window_lag` 字段

```python
@dataclass
class FeatureGroup:
    name: str
    enabled: bool
    cols: List[str]
    lag_labels: List[str]       # 保留以兼容旧 yaml；新 yaml 用 window_lag
    window_lag: str = "0"       # ← 新增：可用性 lag 标签，如 "0" / "1d" / "2d" / "3d" / "4d"

@dataclass
class ResolvedGroup:
    name: str
    cols: List[str]
    lag_periods: List[int]      # 保留以兼容
    window_lag_days: int        # ← 新增：可用性 lag 整数化（按天）
```

### 5.3 `lag_resolver` 增加工具函数

```python
def lag_label_to_days(label: str) -> int:
    """把 '0' / '1d' / '2d' / '3d' / '4d' 解析为整数天数。"""
    if label in ("0", "0d", ""):
        return 0
    m = re.match(r"^(\d+)d$", label.strip())
    if not m:
        raise ValueError(f"window_lag 仅支持 '0' / 'Nd'；got {label!r}")
    return int(m.group(1))
```

### 5.4 yaml 解析逻辑

`load_feature_registry()` 中：

```python
for g_name in FEATURE_GROUPS:
    spec = raw_groups.get(g_name, {})
    enabled = bool(spec.get("enabled", False))
    cols = list(spec.get("cols") or [])
    lag_labels = list(spec.get("lag_labels") or default_lags.get(g_name) or [])
    window_lag = spec.get("window_lag", "0")        # ← 新增读取
    groups[g_name] = FeatureGroup(
        name=g_name, enabled=enabled, cols=cols,
        lag_labels=lag_labels, window_lag=window_lag,
    )
```

`resolve_columns()` 中把 `window_lag` 转 days 写入 `ResolvedGroup`。

---

## 6. neimeng.yaml 完整改造

```yaml
market_id: neimeng
name: 内蒙古
region: neimeng
task: da
freq: hourly
timezone: Asia/Shanghai
test_start: "2026-01-27"
test_end: "2026-04-17"

prediction_globs:
  - "${WORKSPACE_ROOT}/neimeng_prj/output/experiments/**/test_predictions_hourly.csv"

data:
  version: v2
  source: "${PROJECT_ROOT}/runs/data/neimeng/sqllab_电价模型取数daily_20260527T034109.csv"
  source_ts_col: 时间标签
  freq: 15min
  preprocess:
    - rename_neimeng_columns               # 中→英列名映射 + 剔除空列 + 剔除 2M + 剔除冗余弃风弃光
    - encode_renewable_surplus_level       # 4 类中文标签 → int8 (0/1/2/3)
    - fill_spot_rt_after_cutoff            # D-1 07:00 之后填 0
    - ffill_daily_bid_avg                  # 3 列日级申报电价 → 当日 96 点 ffill
  time_range: ["2023-07-22 08:00:00", "2026-05-28 23:45:00"]

storage:
  station_name: hongjing_220kv1m
  settlement_price_col: rt_node_price_nm_hongjing_sta_220kv_1m
  battery:
    p_max_mw: 200
    cap_mwh: 400
    max_charge_mwh: 600
    eta_rt: 0.910
    aux_mwh: 13.03
    dp_ramp_mw: 66.67
    l_min: 4
    cap_comp_per_mwh: 280
    comp_mode: in_objective
    carry_soc: false

notes: |
  v2 数据源：内蒙古现货 SQL 取数库 daily 表（kronos_prod.dwd_logic_point_detail）。
  取数时刻 = D-1 09:30，所有 window_lag 严格按"文件中该列最末非空日距 D 日天数"设定。
  v1 旧 dws 已直接弃用（含苏敦节点价的 alt_targets 选项一并删除）。
  设计详见 doc/内蒙SQL取数接入设计_V1.0.md。

features:
  target_default: rt_node_price_nm_hongjing_sta_220kv_1m
  alt_targets:
    - rt_node_price_nm_hongjing_sta_220kv_1m          # target（默认）
    - rt_node_energy_price_nm_hongjing_sta_220kv_1m
    - rt_node_congestion_price_nm_hongjing_sta_220kv_1m
    - grid_unified_clearing_price
    - hubaodong_unified_clearing_price
    - hubaoxi_unified_clearing_price

  # LightGBM 端：window_lag 直接作为单 shift periods（无多副本）；以下 default 保留兼容字段
  default_lag_labels:
    BOUNDARY:           []           # 即不 shift
    BOUNDARY_DM1:       ["1d"]
    CLEARING_DA:        ["3d"]
    CLEARING_RT:        ["2d"]
    CLEARING_RT_NODAL:  ["4d"]
    ACTUAL:             ["2d"]

  groups:
    BOUNDARY:
      enabled: true
      window_lag: "0"
      cols:
        - bidding_space_forecast_d1
        - dispatched_load_forecast_d1
        - dispatched_load_forecast_d2
        - eastward_delivery_plan_forecast_d1
        - eastward_delivery_plan_forecast_d2
        - non_market_output_plan_d1
        - non_market_output_plan_d2
        - grid_renewable_forecast_d1
        - grid_renewable_forecast_d2
        - grid_pv_forecast_d1
        - grid_pv_forecast_d2
        - grid_wind_power_forecast_d1
        - grid_wind_power_forecast_d2
        - hydro_output_plan_d1
        - hydro_output_plan_d2
        - renewable_energy_surplus_level

    BOUNDARY_DM1:
      enabled: true
      window_lag: "1d"
      cols:
        - upward_reserve_capacity
        - downward_reserve_capacity
        - provincial_power_balance_margin
        - grid_spot_rt_clearing_price

    BOUNDARY_CLEARED:
      enabled: false
      cols: []

    WEATHER:
      enabled: false
      cols: []

    CLEARING_DA:
      enabled: true
      window_lag: "3d"
      cols:
        - grid_unified_clearing_price
        - hubaodong_unified_clearing_price
        - hubaoxi_unified_clearing_price

    CLEARING_RT:
      enabled: true
      window_lag: "2d"
      cols:
        - rt_node_price_ref_nm_hongjing_sta_220kv_1m
        - spot_market_avg_declared_price_coal
        - spot_market_avg_declared_price_wind_power
        - spot_market_avg_declared_price_pv

    CLEARING_RT_NODAL:
      enabled: true
      window_lag: "4d"
      cols:
        - rt_node_price_nm_hongjing_sta_220kv_1m
        - rt_node_energy_price_nm_hongjing_sta_220kv_1m
        - rt_node_congestion_price_nm_hongjing_sta_220kv_1m

    ACTUAL:
      enabled: true
      window_lag: "2d"
      cols:
        - dispatched_load_actual
        - eastward_delivery_plan_actual
        - non_market_output_plan_actual
        - grid_renewable_actual
        - grid_pv_actual
        - grid_wind_power_actual
        - hydro_output_plan_actual
```

---

## 7. Builder 改造：4 个新预处理钩子

放入 `pfbench/data/neimeng_sqllab_fill.py`（新文件），在 `pfbench/data/builder.py` 的 `_PREPROCESS_REGISTRY` 中注册。

### 7.1 `rename_neimeng_columns`

中→英列名映射 + 剔除空列、2M、冗余弃风弃光。一次完成"原始 44 列特征 → 37 列入库"。

```python
COLUMN_MAP = {
    "实际竞价空间": None,            # None 表示 drop
    "竞价空间预测D-1": "bidding_space_forecast_d1",
    "统调负荷实测": "dispatched_load_actual",
    "统调负荷预测D-1": "dispatched_load_forecast_d1",
    "统调负荷预测D-2": "dispatched_load_forecast_d2",
    "东送计划实测": "eastward_delivery_plan_actual",
    "东送计划预测D-1": "eastward_delivery_plan_forecast_d1",
    "东送计划预测D-2": "eastward_delivery_plan_forecast_d2",
    "非市场出力计划实测": "non_market_output_plan_actual",
    "非市场出力计划D-1": "non_market_output_plan_d1",
    "非市场出力计划D-2": "non_market_output_plan_d2",
    "全网新能源实测": "grid_renewable_actual",
    "全网新能源预测D-1": "grid_renewable_forecast_d1",
    "全网新能源预测D-2": "grid_renewable_forecast_d2",
    "全网光伏实测": "grid_pv_actual",
    "全网光伏预测D-1": "grid_pv_forecast_d1",
    "全网光伏预测D-2": "grid_pv_forecast_d2",
    "全网风电实测": "grid_wind_power_actual",
    "全网风电预测D-1": "grid_wind_power_forecast_d1",
    "全网风电预测D-2": "grid_wind_power_forecast_d2",
    "水电出力计划实测": "hydro_output_plan_actual",
    "水电出力计划D-1": "hydro_output_plan_d1",
    "水电出力计划D-2": "hydro_output_plan_d2",
    "全网现货实时出清电价": "grid_spot_rt_clearing_price",
    "全网统一出清电价": "grid_unified_clearing_price",
    "呼包东统一出清电价": "hubaodong_unified_clearing_price",
    "呼包西统一出清电价": "hubaoxi_unified_clearing_price",
    "实时节点电价_红井站1M":   "rt_node_price_nm_hongjing_sta_220kv_1m",
    "实时节点电能价格_红井站1M": "rt_node_energy_price_nm_hongjing_sta_220kv_1m",
    "实时节点阻塞价格_红井站1M": "rt_node_congestion_price_nm_hongjing_sta_220kv_1m",
    "实时节点电价_红井站2M":   None,   # Q3 剔除
    "实时节点电能价格_红井站2M": None,
    "实时节点阻塞价格_红井站2M": None,
    "实时节点电价参考_红井站1M": "rt_node_price_ref_nm_hongjing_sta_220kv_1m",
    "实时节点电价参考_红井站2M": None,
    "正备用容量": "upward_reserve_capacity",
    "负备用容量": "downward_reserve_capacity",
    "省内电力平衡裕度": "provincial_power_balance_margin",
    "现货市场平均申报电价_燃煤": "spot_market_avg_declared_price_coal",
    "现货市场平均申报电价_风电": "spot_market_avg_declared_price_wind_power",
    "现货市场平均申报电价_光伏": "spot_market_avg_declared_price_pv",
    "可再生能源富余程度": "renewable_energy_surplus_level",
    "是否弃风": None,         # Q2 合并，冗余于 renewable_energy_surplus_level
    "是否弃光": None,
    # 时间列 "交易日" / "时段" 在 ts 索引化后冗余，一并 drop
    "交易日": None,
    "时段":   None,
}

def rename_neimeng_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c, v in COLUMN_MAP.items() if v is None and c in df.columns]
    rename_map = {c: v for c, v in COLUMN_MAP.items() if v is not None and c in df.columns}
    return df.drop(columns=drop_cols, errors="ignore").rename(columns=rename_map)
```

> ⚠️ 注意：builder 的 `_read_source` 用 `parse_dates=[ts_col]` 后已经把 ts 列改名为 `"ts"` 并 set_index，所以 `时间标签` 不会出现在 preprocess 阶段的 columns 里。

### 7.2 `encode_renewable_surplus_level`

```python
CURTAIL_MAP = {
    "无弃风、弃光": 0,
    "弃风": 1, "只弃风": 1,
    "弃光": 2, "只弃光": 2,
    "弃风弃光": 3,
}

def encode_renewable_surplus_level(df: pd.DataFrame) -> pd.DataFrame:
    col = "renewable_energy_surplus_level"
    if col not in df.columns:
        return df
    df = df.copy()
    df[col] = df[col].map(CURTAIL_MAP).fillna(0).astype("int8")
    return df
```

### 7.3 `fill_spot_rt_after_cutoff`

```python
def fill_spot_rt_after_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    """grid_spot_rt_clearing_price 每天仅保留 slot 0~28（00:00~07:00），其余填 0。

    取数瞬间（D-1 09:30）业务上只能拿到 D-1 当天 0~7 点的实时出清；与生产推断对齐，
    训练时也按相同规则裁剪，避免训练-推断分布漂移。
    """
    col = "grid_spot_rt_clearing_price"
    if col not in df.columns:
        return df
    out = df.copy()
    daily_slot = (out.index.hour * 4 + out.index.minute // 15).to_numpy()
    mask_keep = daily_slot <= 28
    out.loc[~mask_keep, col] = 0.0
    out[col] = out[col].fillna(0.0)
    return out
```

### 7.4 `ffill_daily_bid_avg`

```python
def ffill_daily_bid_avg(df: pd.DataFrame) -> pd.DataFrame:
    """3 列日级申报电价 → 当日 96 点 ffill；跨日不传递。"""
    cols = [
        "spot_market_avg_declared_price_coal",
        "spot_market_avg_declared_price_wind_power",
        "spot_market_avg_declared_price_pv",
    ]
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out.groupby(out.index.date)[c].ffill()
    return out
```

### 7.5 Builder 注册

`pfbench/data/builder.py` 的 `_PREPROCESS_REGISTRY`：

```python
from .neimeng_sqllab_fill import (
    encode_renewable_surplus_level,
    ffill_daily_bid_avg,
    fill_spot_rt_after_cutoff,
    rename_neimeng_columns,
)

_PREPROCESS_REGISTRY = {
    # ... 原有保留 ...
    "rename_neimeng_columns": rename_neimeng_columns,
    "encode_renewable_surplus_level": encode_renewable_surplus_level,
    "fill_spot_rt_after_cutoff": fill_spot_rt_after_cutoff,
    "ffill_daily_bid_avg": ffill_daily_bid_avg,
}
```

---

## 8. Conv2D / ResConv2D `data.py` 改造

### 8.1 删除三 stream 硬编码 offset

```python
# 原
STREAM_DAY_OFFSET = {"boundary": 0, "history": 1, "actual": 2}

# 新（删除！由 ResolvedGroup.window_lag_days 取代）
```

### 8.2 改造 `_resolve_stream_cols`：保留 stream 分组但记录每个 col 的 window_lag_days

```python
def _resolve_stream_cols(spec: ResolvedSpec) -> Dict[str, List[Tuple[str, int]]]:
    """返回 {stream: [(col, window_lag_days), ...]}。"""
    stream_groups = {
        "boundary": ("BOUNDARY", "BOUNDARY_CLEARED", "WEATHER"),
        "history":  ("BOUNDARY_DM1", "CLEARING_DA", "CLEARING_RT", "CLEARING_RT_NODAL"),
        "actual":   ("ACTUAL",),
    }
    out: Dict[str, List[Tuple[str, int]]] = {"boundary": [], "history": [], "actual": []}
    for stream, group_names in stream_groups.items():
        for g_name in group_names:
            g = spec.groups.get(g_name)
            if g is None:
                continue
            for c in g.cols:
                out[stream].append((c, g.window_lag_days))
    return out
```

### 8.3 改造 `build_daily_arrays`：每列独立按其 window_lag_days 取 7 天窗口

原逻辑是"每个 stream 整体按 `STREAM_DAY_OFFSET[stream]` 后退 N 天"。新逻辑改成"**每列按其自己的 `window_lag_days` 后退 N 天**"。

由于不同列可能有不同 lag（如 `history` stream 内同时存在 lag1d/2d/3d/4d），不能再用整张表统一切片，需要**逐列**生成"按 lag 偏移后的 (96, 1) 时间序列"，然后水平拼接成 stream 张量。

具体实现：

```python
def build_daily_arrays(df_15min, spec, freq="1h"):
    # 1. resolve_stream_cols → 拿到每个 stream 的 [(col, lag_days), ...]
    stream_cols = _resolve_stream_cols(spec)

    # 2. 对每列做按 lag_days 的 shift（shift 单位为 96 步=1 天）
    df = df_15min.sort_index()
    df_shifted = df.copy()
    for stream, items in stream_cols.items():
        for col, lag_days in items:
            if lag_days > 0 and col in df_shifted.columns:
                df_shifted[col] = df_shifted[col].shift(periods=lag_days * SLOTS_PER_DAY)

    # 3. 按日切 96 步，分别构建每个 stream 的 (96, C_stream) ndarray dict
    #    后续 Dataset 端取连续 7 天 [D-6, D]（不再按 stream 偏移日期）
    # ... 余下逻辑同原版，但 STREAM_DAY_OFFSET 统一取 0
```

等价于"先按每列的 lag shift 整张表，然后 Conv2D 统一取 [D-6, D] 这一个相位"。这样 5 lag-bucket 的相位通过 shift 表达，Dataset 取窗口逻辑完全不变。

### 8.4 `Conv2dDataset.__getitem__` 简化

原版有 `off_b / off_h / off_a` 三 stream 各自的日期偏移；新版**全部置 0**，因为偏移已经在 `build_daily_arrays` 的 shift 阶段完成。

```python
off_b = off_h = off_a = 0   # 统一从 D 日往回数 7 天
```

### 8.5 model.py 零改动

网络结构（3 分支输入、ResBlock、双头输出）不变。只是输入张量中"history 通道"的含义从"按 stream 偏移"换成"按列 lag 偏移"，对模型而言透明。

### 8.6 ResConv2D 同步改动

`algorithms/resconv2d/data.py` reuse 了 `algorithms/conv2d_multitask/data.py` 的工具函数，自动跟随。

---

## 9. LightGBM 端改造（保持简单）

`algorithms/lightgbm_baseline/config.py` 中 `price_lag_hours` 反推逻辑：现在用 `window_lag` 单值，而非 `lag_periods` 列表。

```python
# 旧（多 lag 反推）
da = resolved.groups.get("CLEARING_DA")
if da is not None and da.lag_periods:
    price_lag_hours = [int(p * step_minutes / 60) for p in da.lag_periods]

# 新（单 lag）
da = resolved.groups.get("CLEARING_DA")
if da is not None and da.window_lag_days > 0:
    price_lag_hours = [da.window_lag_days * 24]
```

或更激进的做法：LightGBM 端不再生成"shift 副本特征"，直接对每列做一次 `shift(window_lag_days * SLOTS_PER_DAY)` 作为单列特征（与 Conv2D 完全等价）。这条留作可选优化。

---

## 10. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 旧 v1 实验产物 (`runs/predictions/neimeng/<algo>/`) 与新数据列名冲突 | 新实验落 `_v2` 后缀目录或直接覆盖；RESULTS.md 区分版本 |
| v2 数据下旧 v1 baseline MAE 严重劣化（因 v1 隐含未来信息泄漏） | 用 RESULTS.md "更新日志" 注明：v1 数字仅作历史参考，不可与 v2 横比 |
| `renewable_energy_surplus_level` 实际类别可能多于 4 类 | LabelEncoder 默认 fillna(0) 即映射未知值为"无弃用"；后续监测 distinct 值 |
| `grid_spot_rt_clearing_price` D-1 07:00 后填 0 引入"伪零段" | 模型可能学到"slot>28 即 0"的捷径；监测特征重要性时关注 |
| 节点价 2M 一并删除可能造成将来 1M 异常时无备份信号 | 在 raw csv 中本身保留；如需可以反向恢复 |

---

## 11. 验收计划

### 阶段 1：数据接入（1 人天）

1. 新建 `pfbench/data/neimeng_sqllab_fill.py` + 注册 4 个 hook
2. `feature_registry.FEATURE_GROUPS` 加 `BOUNDARY_DM1` / `CLEARING_RT_NODAL`，`FeatureGroup` 加 `window_lag`
3. `lag_resolver` 加 `lag_label_to_days`
4. 改 `config/markets/neimeng.yaml`
5. 运行 `python scripts/build_dataset.py --market neimeng --version v2 --force`
6. 验收：产出 `runs/data/neimeng/v2/data.parquet`（~100k 行 × 37 列 + ts）；抽样核对 5 列：
   - `dispatched_load_forecast_d1` 5-28 行有值
   - `upward_reserve_capacity` 5-28 行 NaN、5-27 行有值
   - `grid_spot_rt_clearing_price` 5-27 行 slot 29~95 全 0
   - `grid_unified_clearing_price` 最末非空日 = 5-25
   - `rt_node_price_nm_hongjing_sta_220kv_1m` 最末非空日 = 5-24

### 阶段 2：feature_registry 单元验证（0.3 人天）

```python
from pfbench.feature_registry import resolve_columns, FeatureSpec
spec = resolve_columns("neimeng", FeatureSpec(), freq="1h")
print(spec.to_dict())
# 期望：6 个 enabled group（BOUNDARY/BOUNDARY_DM1/CLEARING_DA/CLEARING_RT/CLEARING_RT_NODAL/ACTUAL）
# 每个 group 含 window_lag_days
```

### 阶段 3：Conv2D / ResConv2D `data.py` 改造（1 人天）

按 §8 改造。烟测：

```bash
python algorithms/resconv2d/run.py --market neimeng --freq 1h --depth base --epochs 15 --output-suffix smoke_v2
```

期望：训练能跑完不报错；MAE 在 110~140 量级（与 v1 烟测 121.62 ±15%）。

### 阶段 4：全量回归（2 人天）

| 算法 | 命令 |
|---|---|
| LightGBM-Baseline 1h | `python algorithms/lightgbm_baseline/run.py --market neimeng --freq 1h` |
| LightGBM-Baseline 15min | `--freq 15min` |
| LightGBM-TwoStage 1h | `python algorithms/lightgbm_twostage/run.py --market neimeng --freq 1h` |
| LightGBM-TwoStage 15min | `--freq 15min` |
| Conv2D-MultiTask 1h | `python algorithms/conv2d_multitask/run.py --market neimeng --freq 1h` |
| Conv2D-MultiTask 15min | `--freq 15min` |
| ResConv2D 1h aggressive | `python algorithms/resconv2d/run.py --market neimeng --freq 1h --depth aggressive --epochs 80` |
| ResConv2D 15min aggressive | `--freq 15min --epochs 80` |
| 储能评估 | `python scripts/run_storage_eval.py --market neimeng --algorithm all --freq 15min` 和 `--freq hourly` |

### 阶段 5：RESULTS.md 更新（0.5 人天）

- §2 横向比较表加 "数据版本" 列（v1 / v2），原 v1 数字保留但标注"已废弃口径"
- §3 各算法详情节加 "v2 重跑后" 子节
- §7 更新日志加一条
- §8 储能评估 v2 表

---

## 12. 落地里程碑

| 阶段 | 工作量 |
|---|---|
| 1. 数据接入 | 1 人天 |
| 2. feature_registry 单元 | 0.3 人天 |
| 3. Conv2D / ResConv2D 改造 + 烟测 | 1 人天 |
| 4. 全量回归（4 算法 × 2 粒度 + 储能） | 2 人天 |
| 5. RESULTS.md 更新 | 0.5 人天 |
| **合计** | **4.8 人天** |

---

## A. 附录

### A.1 原始 SQL

```sql
WITH base_data AS (
    SELECT
        raw_upload_time,
        toDate(raw_upload_time) AS stat_date,
        toHour(raw_upload_time) * 4 + intDiv(toMinute(raw_upload_time), 15) AS point_index,
        point_identifier,
        toFloat64OrNull(point_value) AS val,
        toString(point_value) AS str_val,
        ts
    FROM kronos_prod.dwd_logic_point_detail
    WHERE point_value IS NOT NULL
      AND raw_upload_time >= today() - INTERVAL 1040 DAY
      AND raw_upload_time < today() + INTERVAL 4 DAY
      AND point_identifier IN (
          'ActualBiddingSpace', 'BiddingSpaceForecastD-1',
          'DispatchedLoadActual', 'DispatchedLoadForecastD-1', 'DispatchedLoadForecastD-2',
          'EastwardDeliveryPlanActual', 'EastwardDeliveryPlanForecastD-1', 'EastwardDeliveryPlanForecastD-2',
          'NonMarketOutputPlanActual', 'NonMarketOutputPlanD-1', 'NonMarketOutputPlanD-2',
          'GridRenewableActual', 'GridRenewableForecastD-1', 'GridRenewableForecastD-2',
          'GridPvActual', 'GridPvForecastD-1', 'GridPvForecastD-2',
          'GridWindPowerActual', 'GridWindPowerForecastD-1', 'GridWindPowerForecastD-2',
          'HydroOutputPlanActual', 'HydroOutputPlanD-1', 'HydroOutputPlanD-2',
          'GridSpotRtClearingPrice', 'GridUnifiedClearingPrice',
          'HubaodongUnifiedClearingPrice', 'HubaoxiUnifiedClearingPrice',
          'RTNodePrice_NmHongJingSta220kV1M', 'RTNodeEnergyPrice_NmHongJingSta220kV1M', 'RTNodeCongestionPrice_NmHongJingSta220kV1M',
          'RTNodePrice_NmHongJingSta220kV2M', 'RTNodeEnergyPrice_NmHongJingSta220kV2M', 'RTNodeCongestionPrice_NmHongJingSta220kV2M',
          'RTNodePriceRef_NmHongJingSta220kV1M', 'RTNodePriceRef_NmHongJingSta220kV2M',
          'UpwardReserveCapacity', 'DownwardReserveCapacity', 'ProvincialPowerBalanceMargin',
          'RenewableEnergySurplusLevel',
          'SpotMarketAverageDeclaredPrice_Coal', 'SpotMarketAverageDeclaredPrice_WindPower', 'SpotMarketAverageDeclaredPrice_Pv'
      )
)
SELECT
    raw_upload_time AS `时间标签`,
    stat_date       AS `交易日`,
    point_index     AS `时段`,
    -- 各列 argMaxIf(val, ts, point_identifier = '...') AS 中文列名
    -- RenewableEnergySurplusLevel 用 str_val 取字符串
    -- 是否弃风/弃光为 multiIf 派生（本次接入剔除）
    ...
FROM base_data
GROUP BY `时间标签`, `交易日`, `时段`
ORDER BY `时间标签` DESC;
```

### A.2 决策点最终拍板

| # | 决策点 | 最终选择 |
|---|---|---|
| Q1 | 旧 dws v1 处理 | 直接弃用，不保留 |
| Q2 | 弃风/弃光列 | 只保留 4 类原始 `renewable_energy_surplus_level`，剔除两个 0/1 派生列 |
| Q3 | 节点价 2M | 全部去掉（4 列） |
| Q4 | LightGBM 多 lag shift | 作废，每列只一个 `window_lag` |
| Q5 | 全量回归粒度 | 1h + 15min 都跑 |
| Q6 | commit 策略 | 一次性 commit |
| 额外 | Conv2D 特征拼装 | 5 lag-bucket（按可用性 lag 分 0/1d/2d/3d/4d 五桶取窗口） |
| 额外 | 列命名规则 | 按 SQL `point_identifier` 转 snake_case |
| 额外 | target 列 | `rt_node_price_nm_hongjing_sta_220kv_1m` |
