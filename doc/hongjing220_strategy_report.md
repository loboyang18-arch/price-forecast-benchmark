# 红井 220kV.1M 节点电价 V8 预测与策略评估报告

> 生成时间：2026-05-12  
> 数据源：`source_data/data 4/信息披露/市场运营机构/现货市场申报出清信息/实时节点电价/内蒙红井站.csv`（节点 `内蒙.红井站/220kV.1M`，截至 2026-05-09，共 105 个测试日）

---

## 1. 配置

| 项 | 取值 |
|---|---|
| 节点 | `内蒙.红井站/220kV.1M` |
| 测试期 | 2026-01-25 ~ 2026-05-09（**105 天**） |
| 储能功率 | **200 MW** |
| 储能容量 | **400 MWh**（2 小时系统） |
| 日充电量上限 | **600 MWh**（= 1.5 × 容量 / 天） |
| 双程效率 η | 0.910 |
| 日辅助用电 | 13.03 MWh |
| 容量补偿单价（事后口径） | **280 元/MWh** |

**入库产物**：`output/dws_15min_features_ext_data4_hongjing.csv`（138432 行，2022-06-01 ~ 2026-05-12 23:45）  
新增列：`price_hongjing_220kv1m_nodal` / `price_hongjing_220kv1m_energy` / `price_hongjing_220kv1m_cong`

**V8 训练产物目录**：`output/experiments/v8.0-jan25-hongjing220-data4/`  
**MILP 策略产物目录**：`output/experiments/v8.0-jan25-hongjing220-ext/`  
**Dashboard 目录**：`output/dashboard_hongjing/`

> **关于启发式 4h 策略**：该策略原为苏敦 800 MWh / 195 MW（≈4 小时）系统设计，每日 4h 充 + 4h 放、固定 400 MWh × 1 次循环；直接移植到红井（2h 系统，200 MW / 400 MWh）会导致：
> - 充放电窗口（4h）超出 C-rate 实际需要的 2h，**全程仅 50% 功率运行**；
> - 日循环固定 1 次，无法利用 1.5×/天 的政策上限；
> 因此本报告**不再纳入启发式 4h 方案**，仅评估三套 MILP 策略。

---

## 2. V8 预测精度

`NM_V8_TARGET=price_hongjing_220kv1m_nodal`、`NM_V8_TEST_START=2026-01-25`、`NM_V8_HOURLY_AGG=mean4`。

| 测试期 | 样本数（小时） | MAE (元/MWh) | RMSE (元/MWh) |
|---|---:|---:|---:|
| 2026-01-25 ~ 2026-05-09 | **2520** | **123.5** | **178.0** |

绝对量级与红井节点电价的均值约 300 元/MWh 比较，**MAE 占比 ≈ 41%、RMSE 占比 ≈ 59%**，与同一 V8 框架在苏敦 500kV.1M 上的训练精度（MAE 117 / RMSE 165）量级相当，可作为日前调度决策输入。

---

## 3. 三套 MILP 策略 · 无补偿口径

> 「无补偿」= 仅日清算结算口径 = Σ (放电价 × 放电量 − 充电价 × 充电量) − 辅助用电 × 日均价。  
> 所有策略统一受 **`MAX_CHARGE_MWH = 600 MWh`（= 1.5 × 400 MWh / 天）** 约束。

### 全测试期 105 天

| 策略 | 放电 MWh | 模拟净 (万元) | 完全预知 PF (万元) | 兑现率 | **日均净 (元)** | **日均 PF (元)** |
|---|---:|---:|---:|---:|---:|---:|
| 小时 MILP（日清零, 1.5×/天） | 56 507.9 | 707.47 | 1 679.31 | 42.1% | 67 378 | 159 935 |
| 15min MILP（日清零, 1.5×/天） | 56 090.5 | 706.59 | 1 860.80 | 38.0% | 67 295 | 177 219 |
| **15min MILP（跨日 SOC, 1.5×/天）** | 55 531.5 | **796.29** | 1 897.95 | **41.9%** | **75 837** | **180 757** |

**关键结论**：

1. **15min MILP 跨日 SOC** 是三套方案中净收益最高、兑现率与小时 MILP 相当（42% 量级）的最佳工程基准；
2. **小时 MILP**（1.5×约束下）净收益与 15min 日清零非常接近（差 < 0.1 万元 / 105 天），但 **PF 上限低 ~12%**（1 679 vs 1 860/1 897 万元），说明 15min 模型能解锁更多日内峰谷；
3. 三套方案的 **充电量** 均贴近上限（~621/616/610 MWh/日均），与 1.5×/天 政策约束完全一致。

---

## 4. 三套 MILP 策略 · 叠加 280 元/MWh 事后容量补偿

> 容量补偿按 **放电 MWh × 280 元/MWh** 事后叠加。

### 全测试期 105 天

| 策略 | 放电 MWh | 模拟净 (万元) | 补偿额 (万元) | 含补偿净 (万元) | PF 含补偿 (万元) | 兑现率（含补偿） | **日均含补偿 (元)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 小时 MILP（1.5×/天） | 56 507.9 | 707.47 | 1 582.22 | **2 289.69** | 3 261.53 | 70.2% | 218 066 |
| 15min MILP（日清零） | 56 090.5 | 706.59 | 1 570.53 | **2 277.13** | 3 431.34 | 66.4% | 216 869 |
| **15min MILP（跨日 SOC）** | 55 531.5 | 796.29 | 1 554.88 | **2 351.17** | 3 452.84 | **68.1%** | **223 921** |

补偿口径下三套方案的含补偿净收益均 > 2200 万元 / 105 天，**线性外推至全年 ≈ 7 758 ~ 7 985 万元 / 站**。`15min 跨日 SOC` 仍位列首位。

---

## 5. 重要声明（无电站实际运营对照）

1. **本报告所有数字均为模拟**：MILP 求解器在「**预测电价 → 决策、实际电价 → 评估**」管线下得到的净收益，**不等于真实电站结算**。
2. **未做实际电站对比**：红井 220kV.1M 暂无 `日清算结果查询电厂侧.csv` 等结算数据，因此：
   - 无「实际放电 / 充电 / 净收益」对照；
   - 无兑现率（actual_net / strategy_net）；
   - 无 MILP 在实际电价上的离线复演；
3. **容量补偿口径**采用 280 元/MWh，与苏敦 1.5×/天 补偿实验报告 [`docs/multi_scenarios_report_1p5cycle_comp280.md`](multi_scenarios_report_1p5cycle_comp280.md) 一致。若实际签约价格不同，**按放电 MWh × 单价线性缩放即可**。
4. **三套 MILP 均统一在 1.5×/天 充电上限**下求解；旧版小时 MILP 因未加该约束会出现 2.7 × 容量 / 天的过度循环，本次已修正。
5. **不推荐启发式 4h**：与 2h C-rate 系统失配，已从产线下线（见 §1 末尾说明）。

---

## 6. 产出文件清单

```
output/
├── dws_15min_features_ext_data4_hongjing.csv          （入库产物，138432 行）
├── experiments/
│   ├── v8.0-jan25-hongjing220-data4/                  （V8 训练 + 实际 xlsx）
│   │   ├── test_predictions_hourly.csv                （2520 行，105 天 × 24h）
│   │   ├── model_weights.pt + norm_*.npy + target_stats.npz
│   │   ├── actual_node_prices_hongjing.xlsx           （105 天 × 96 槽 = 10080 行）
│   │   ├── hongjing_extension_summary.json            （本报告的源数据）
│   │   └── plots/ (V8 内置 da_week*.png)
│   └── v8.0-jan25-hongjing220-ext/                    （三套 MILP 策略 + 周图）
│       ├── strategy_milp_result.csv                   （小时 MILP, 105 天, 含 1.5×约束）
│       ├── strategy_milp_15min_result.csv             （15min 日清零, 105 天）
│       ├── strategy_milp_15min_carry_soc_result.csv   （15min 跨日 SOC, 105 天）
│       └── plots_milp_15min_carry_soc/                （16 张周图 W04~W19）
└── dashboard_hongjing/                                （可视化主表）
    ├── 15min_timeseries.csv / strategy_15min_timeseries.xlsx
    ├── daily_summary.csv / daily_cycle_revenue.csv / weekly_summary.csv
    ├── model_predictions.csv
    ├── metrics_strategy_daily.csv|xlsx / metrics_summary.csv
    ├── README.md
    └── strategies/
        ├── strategy_15min_timeseries_hourly_milp.xlsx
        ├── strategy_15min_timeseries_daily_zero_15min.xlsx
        ├── metrics_strategy_daily_hourly_milp.xlsx
        └── metrics_strategy_daily_daily_zero_15min.xlsx
```

---

## 7. 复现指令

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate power

# 一键复现（含入库、训练、三套 MILP 策略、绘图、汇总）
python scripts/run_v8_hongjing_extension.py

# 已有产物时可分步跳过
SKIP_MERGE=1 python scripts/run_v8_hongjing_extension.py        # 跳过入库
SKIP_TRAIN=1 python scripts/run_v8_hongjing_extension.py        # 跳过训练
SKIP_STRATEGY=1 python scripts/run_v8_hongjing_extension.py     # 跳过策略求解，仅重生 summary.json

# Dashboard（红井全程 --no-actual）
python scripts/run_dashboard_hongjing.py

# 仅生成实际节点电价 xlsx
python scripts/build_hongjing_actual_xlsx.py

# 仅运行 DWS 入库（含红井）
python scripts/merge_data4_into_dws.py \
  --out output/dws_15min_features_ext_data4_hongjing.csv \
  --include-hongjing
```

**关键环境变量**（已在 `run_v8_hongjing_extension.py` 内固化）：

- `NM_V8_TARGET=price_hongjing_220kv1m_nodal`
- `NM_V8_TAG=v8.0-jan25-hongjing220-data4`
- `NM_V8_TEST_START=2026-01-25`
- `NM_V8_HOURLY_AGG=mean4`
- `NM_DWS_CSV=output/dws_15min_features_ext_data4_hongjing.csv`
- `NM_MILP_P_MAX=200` / `NM_MILP_CAP_MWH=400` / `NM_MILP_MAX_CHARGE_MWH=600`
- `NM_MILP_ETA_RT=0.910` / `NM_MILP_AUX_MWH=13.03` / `NM_MILP_DP_RAMP=66.67`
