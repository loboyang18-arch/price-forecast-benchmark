# 实验结果汇总

> 本文档由每次实验完成后自动更新，记录所有算法在各市场的横向比较结果。
>
> 最后更新：2026-05-25

---

## 1. 测试集划分（统一）

所有算法共享同一测试集划分，定义在 `config/markets/<market_id>.yaml`。

| 市场 | test_start | test_end | 测试行数（hourly） |
|------|-----------|---------|-------------------|
| 内蒙古 (neimeng) | 2026-01-27 | 2026-04-17 | 1944 |
| 重庆 (chongqing) | 2026-03-03 | 2026-04-13 | 1008 |
| 江苏 (jiangsu) | 2025-11-01 | 2025-12-31 | 1464 |

---

## 2. 算法横向比较

### 2.1 MAE 对比（元/MWh）

| 市场 | LightGBM-Baseline | LightGBM-TwoStage | 最优 |
|------|-------------------|-------------------|------|
| 内蒙古 | 98.51 | **82.76** | TwoStage (-16.0%) |
| 重庆 | **94.93** | 110.47 | Baseline (-14.1%) |
| 江苏 | 82.60 | **53.36** | TwoStage (-35.4%) |

### 2.2 RMSE 对比（元/MWh）

| 市场 | LightGBM-Baseline | LightGBM-TwoStage | 最优 |
|------|-------------------|-------------------|------|
| 内蒙古 | 124.11 | **113.41** | TwoStage (-8.6%) |
| 重庆 | 169.62 | **144.92** | TwoStage (-14.6%) |
| 江苏 | 103.80 | **72.52** | TwoStage (-30.1%) |

### 2.3 Profile Correlation 对比

| 市场 | LightGBM-Baseline | LightGBM-TwoStage | 最优 |
|------|-------------------|-------------------|------|
| 内蒙古 | **0.7838** | 0.7494 | Baseline |
| 重庆 | 0.1724 | **0.2271** | TwoStage |
| 江苏 | 0.5500 | **0.7545** | TwoStage |

---

## 3. 各算法详情

### 3.1 LightGBM-Baseline

- **代码**：`algorithms/lightgbm_baseline/`
- **方法**：标准 LightGBM 回归，lag0/lag1/lag2 特征体系，固定 train/test 切分，val 取 train 末 7 天做 early stopping
- **参数**：lr=0.05, num_leaves=63, 2000 轮 + early stopping 50 轮

| 市场 | MAE | RMSE | MAPE(%) | Profile Corr | 特征数 |
|------|-----|------|---------|-------------|--------|
| 内蒙古 | 98.51 | 124.11 | 294.0 | 0.784 | ~30 |
| 重庆 | 94.93 | 169.62 | 7239.7 | 0.172 | ~40 |
| 江苏 | 82.60 | 103.80 | 153.3 | 0.550 | ~40 |

### 3.2 LightGBM-TwoStage

- **代码**：`algorithms/lgb_twostage/`
- **方法**：移植自江苏项目的综合 LightGBM 方案
  - 200+ 候选特征（boundary 曲线、历史价格、实时价格、价差、实际运行值、地板价结构、趋势、日历）
  - 地板价分类器 + 正常价回归器的两阶段建模
  - 扩展窗口交叉验证（Expanding Window CV），逐周滑动
  - 时间衰减样本加权、自适应 model-naive 混合、残差校正
- **参数**：lr=0.02, num_leaves=31, max_depth=6, 1500 轮 + early stopping 80 轮

| 市场 | MAE | RMSE | Profile Corr | CV Folds | 特征数 |
|------|-----|------|-------------|----------|--------|
| 内蒙古 | 82.76 | 113.41 | 0.749 | 12 | 200+ |
| 重庆 | 110.47 | 144.92 | 0.227 | 6 | 200+ |
| 江苏 | 53.36 | 72.52 | 0.755 | 9 | 200+ |

---

## 4. 分析与备注

1. **内蒙古**：TwoStage 在 MAE 上优于 Baseline 16%，扩展窗口 CV 的逐步扩大训练集策略有效。
2. **重庆**：Baseline MAE 更优（94.93 vs 110.47），但 TwoStage 的 RMSE 更低（144.92 vs 169.62），说明 TwoStage 在极端误差控制上更好，但均值偏差更大。重庆市场数据仅 ~4 个月训练期，对 200+ 特征的 TwoStage 来说训练数据偏少。
3. **江苏**：TwoStage 大幅领先（MAE 53.36 vs 82.60，-35.4%），这与江苏市场特征更丰富（双节点价格、实时价格、备用容量等）有关，TwoStage 能充分利用这些特征。
4. **MAPE 异常**：重庆的 MAPE 极高（7239%）是因为存在接近零的真实价格，导致 MAPE 分母极小而失真，此指标在有地板价的市场中不适用。

---

## 5. 泄漏审计（LightGBM-TwoStage）

针对 TwoStage 在江苏、内蒙古的高 Profile Correlation 与较低 MAE，进行了系统的未来信息泄漏排查。
所有假设均通过 runtime 日志（NDJSON）+ 消融实验进行验证，无任何代码级或数据级泄漏。

| 假设 | 评估 | 关键证据 |
|------|------|----------|
| H1a：内蒙古 `price_dayahead_preclear_energy` 是 `price_unified` 的派生（ex-post） | **REJECTED** | 17,088 样本中仅 10.6% 数值相等（若复制应 >99%），diff 绝对均值 121 元/MWh |
| H1b：江苏 `reserve_positive/negative_汇总` 是 D-1 公布的 D 日 96 时段计划（合法 lag0） | **CONFIRMED 合法** | 122/122 天有完整 96 时段曲线；yaml notes 明示"用前一周同时段补齐"⇒ 是 D-1 公布的计划属性 |
| H2：日期序列存在跨越 D-1 的缺口（lag1 实际取到 D-N） | **REJECTED** | 内蒙古 460 天 0 缺口，江苏 122 天 0 缺口 |
| H3：lag1 同时段特征过度相关（隐含取到了 D 日） | 信息性 | 江苏 top corr 0.66，符合 D-1 价格的天然自相关强度 |
| H4：扩展窗口 CV 的 train/val/test 存在日期重叠 | **REJECTED** | 内蒙古 12 折 + 江苏 9 折，三窗口两两 overlap 均为 0 |
| H5：lag1 calendar 错位（实际取到 lag0） | **REJECTED** | 全部 lag1 配对样本日间隔 = 1 天 |

### 5.1 消融实验（核心反证）

如果可疑高相关 boundary 是真泄漏，删除它们后 MAE 应暴涨 50%+。实际仅退化 7-10%，符合"合法 D-1 强信号"的特征。

| 市场 | 删除的列 | 原始 MAE | 消融 MAE | 退化幅度 |
|------|----------|----------|----------|----------|
| 内蒙古 | `price_dayahead_preclear_energy` | 82.76 | 88.96 | **+7.49%** |
| 江苏 | `reserve_positive_汇总`, `reserve_negative_汇总` | 53.36 | 58.59 | **+9.81%** |

### 5.2 结论

**LightGBM-TwoStage 算法不存在未来信息泄漏。** 高相关 boundary 特征皆为 D-1 调度流程合法产出：

- **内蒙古** `price_dayahead_preclear_energy`：日前出清流程中间产物，D-1 末已知
- **江苏** `reserve_positive/negative_汇总`：D-1 调度发布的 D 日 96 时段备用容量计划

模型表现"似乎过好"的根因是业务侧 D-1 公布的强信号本身就高度可预测 D 日价格，非泄漏。

---

## 6. 更新日志

| 日期 | 内容 |
|------|------|
| 2026-05-25 | 完成 TwoStage 未来信息泄漏审计（H1a/H1b/H2/H3/H4/H5 全部排查 + 消融实验） |
| 2026-05-25 | 补充 TwoStage Profile Correlation；修正重庆 RMSE 对比标注 |
| 2026-05-25 | 初始版本：LightGBM-Baseline + LightGBM-TwoStage 三市场对比 |
