# 算法登记

本文档记录纳入 benchmark 验证的算法，按建模范式分为三类（朴素基准 / ML / DL）。每个算法不绑定具体预测任务（日前/实时），后续均可在多市场、多任务上测试。

## 〇、Naive — 朴素基准（零线参考）

| 算法 | 策略 | 本工程实现 |
|------|------|-----------|
| **naive_lag_1d** | `y_pred(t) = y_actual(t − 1 day)` | `algorithms/naive_baseline/` |
| **naive_lag_7d** | `y_pred(t) = y_actual(t − 7 days)` | `algorithms/naive_baseline/` |
| **naive_rolling_7d_mean** | `y_pred(t) = mean(y_actual(t − 1d ... t − 7d))` | `algorithms/naive_baseline/` |

零训练，仅做时序 lag/平均，作为算法对比的零线。任何 ML 模型若 MAE 不能稳定低于 `naive_rolling_7d_mean`，则未产生 ML 价值（参考公司模型研发指南 §3.3 / §8.1）。

**三市场朴素基准结果（1h / 15min）：**

| 市场 | 粒度 | lag_1d | lag_7d | rolling_7d_mean ← 最强 |
|------|------|--------|--------|------------------------|
| 内蒙 | 1h | 187.10 | 211.06 | **170.00** |
| 内蒙 | 15min | 196.00 | 219.75 | **178.60** |
| 重庆 | 1h | 101.61 | 125.45 | **100.81** |
| 重庆 | 15min | 103.06 | 127.11 | **105.69** |
| 江苏 | 1h | 75.47 | 79.63 | **68.64** |
| 江苏 | 15min | 77.20 | 81.38 | **71.39** |

> **关键发现**：江苏 1h `rolling_7d_mean` MAE 68.64 反而**超越** LightGBM-Baseline (83.91) 和 Conv2D (90.18)，仅次于 LightGBM-TwoStage (52.39)。这意味着这两个 ML 算法在江苏未产生超过零线的价值。详见 `RESULTS.md` §2.4 / §4.5。

## 一、ML — 基于人工特征的机器学习方法

| 算法 | 说明 | 本工程实现 | 源工程 |
|------|------|-----------|--------|
| **LightGBM-Baseline** | 标准 LightGBM 回归 baseline。lag0/lag1/lag2 特征 + 日历 + 滚动统计。固定 train/test 切分，单次评估。 | `algorithms/lightgbm_baseline/` | jiangsu_prj |
| **LightGBM-TwoStage** | 完整方法体系（移植自报告验证方案）：200+ 候选特征、Two-Stage 地板价分类器 + 回归器、多分位数回归、时间衰减加权、model-naive 自适应混合、残差校正、Expanding Window CV。 | `algorithms/lightgbm_twostage/` | jiangsu_prj |

**共同特点：**
- 特征由领域知识手工构造（非原始时序输入）
- 模型为单时段独立预测（hour 作为特征，单模型覆盖 24 小时）
- 三市场统一接口；2026-05-26 起所有特征由 `pfbench.feature_registry` 解析自 `config/markets/<market>.yaml`，无 hardcode

**两个算法都支持 `--freq {1h,15min}`**，输出落入 `<algo>/` 和 `<algo>_15min/` 两个目录。

**LightGBM-Baseline 三市场结果（接入 Feature Registry 后，固定 train/test，1h vs 15min）：**

> 内蒙 target = 红井 220kV 节点电价（`price_hongjing_220kv1m_nodal`），重庆/江苏 target 见 `RESULTS.md` §1。

| 市场 | 粒度 | 特征数 | MAE | RMSE | Profile Corr | best_iter |
|------|------|--------|-----|------|--------------|-----------|
| 内蒙 | 1h | 44 | 134.19 | 165.49 | 0.6468 | 21 |
| 内蒙 | 15min | 47 | 138.18 | 178.82 | 0.5976 | 27 |
| 重庆 | 1h | 43 | 95.47 | 169.96 | 0.0597 | 1 |
| 重庆 | 15min | 46 | 95.39 | 175.69 | 0.1734 | 1 |
| 江苏 | 1h | 49 | 83.91 | 105.49 | 0.5532 | 70 |
| 江苏 | 15min | 52 | 91.69 | 114.87 | 0.4820 | 115 |

**LightGBM-TwoStage 三市场结果（单次训练-预测，2026-05-26 起；1h vs 15min）：**

| 市场 | 粒度 | 特征数 | train_d / test_d | MAE | RMSE | Profile Corr | naive_yesterday MAE | 相对 naive |
|------|------|--------|-------------------|-----|------|--------------|---------------------|-----------|
| 内蒙 | 1h | 245 | 372 / 81 | 107.96 | 155.60 | 0.5938 | 187.10 | +42.3% |
| 内蒙 | 15min | 247 | 372 / 81 | 120.10 | 179.73 | 0.5522 | 196.00 | +38.7% |
| 重庆 | 1h | 190 | 115 / 42 | 92.46 | 168.08 | 0.0432 | 101.61 | +9.0% |
| 重庆 | 15min | 192 | 115 / 42 | 89.89 | 170.17 | 0.1509 | 103.06 | +12.8% |
| 江苏 | 1h | 241 | 54 / 61 | 74.08 | 91.75 | 0.6592 | 75.47 | +1.8% |
| 江苏 | 15min | 243 | 54 / 61 | 81.55 | 105.27 | 0.5884 | 77.20 | **−5.6%** |

> 2026-05-26 弃用 Expanding Window CV，统一改为单次训练-预测。详细变化与 CV 版对比见 `RESULTS.md` §3.2 / §4.1 / §4.5。

**两算法横向比较（含零线参考；MAE，单位 元/MWh）：**

| 市场 | 粒度 | naive_rolling_7d | Baseline (Δ vs naive) | TwoStage (Δ vs naive) | 最优 |
|------|------|------------------:|------------------------|------------------------|------|
| 内蒙 | 1h | 170.00 | 134.19 (−21.1%) | **107.96** (−36.5%) | TwoStage |
| 内蒙 | 15min | 178.60 | 138.18 (−22.6%) | **120.10** (−32.8%) | TwoStage |
| 重庆 | 1h | 100.81 | 95.47 (−5.3%) | **92.46** (−8.3%) | TwoStage |
| 重庆 | 15min | 105.69 | 95.39 (−9.7%) | **89.89** (−15.0%) | TwoStage |
| 江苏 | 1h | 68.64 | 83.91 (**+22.2% 劣于零线**) | 74.08 (**+7.9% 劣于零线**) | naive_rolling_7d |
| 江苏 | 15min | 71.39 | 91.69 (**+28.4% 劣于零线**) | 81.55 (**+14.2% 劣于零线**) | naive_rolling_7d |

> 15min 与 1h 的对比分析见项目根目录 `RESULTS.md` §2bis / §4.2。粒度切换不改变三市场的最优算法名次。

## 二、DL — 基于原始特征的深度学习方法

| 算法 | 说明 | 本工程实现 | 源工程 | 源文件 |
|------|------|-----------|--------|--------|
| **Conv2D-MultiTask** | Conv2D 多任务网络。输入 (C, H_SLOTS, 7) 张量，3 层 Conv2D + BN + GELU + MaxPool 提取时空模式，回归头（L1）+ 方向分类头（CE，λ=0.3）联合学习。 | `algorithms/conv2d_multitask/` | neimeng_prj | `src/model_v8_multitask.py` |
| **Transformer-Joint24h** | Pure Transformer 全日联合预测。每小时原始特征 (C×4×7) 展平后 Linear 投影，经 Transformer Encoder 建模 24 小时间依赖关系，联合输出全天价格预测。 | 待移植 | neimeng_prj | `src/model_v10_joint.py` |
| **Transformer-Quantile** | 基于 Transformer-Joint24h 的分位数版本。输出每小时 5 个分位数（P10/P30/P50/P70/P90），使用 Pinball Loss 训练，可用于下游鲁棒优化。 | 待移植 | neimeng_prj | `src/model_v10_quantile.py` |
| **ResConv2D** | 10 层残差 Conv2D 网络（ResBlock + GELU）。双头架构：价格回归头 + 涨跌方向头。跨市场同构设计（内蒙/重庆共享网络结构）。 | 待移植 | neimeng_prj / chongqing_prj | `src/model_v25_resconv.py` |

**共同特点：**
- 输入为原始特征矩阵（多通道×时段×回看天数），不做手工特征工程
- 全日联合输出（24h 或 96×15min），利用时段间相关性
- PyTorch 实现，GPU 训练

**Conv2D-MultiTask 三市场结果（接入 Feature Registry 后重跑，80 epochs，RTX 4090）：**

| 市场 | 粒度 | C / H_SLOTS | 训练样本 | MAE | RMSE | Profile Corr | Dir Acc |
|------|------|-------------|----------|-----|------|--------------|---------|
| 内蒙 | 1h | 25 / 12 | 8736 | 113.12 | 162.75 | 0.7191 | 0.536 |
| 内蒙 | 15min | 25 / 12 | 34944 | 126.03 | 186.78 | 0.6594 | 0.451 |
| 重庆 | 1h | 24 / 12 | 2568 | **89.17** | 164.30 | 0.2015 | 0.498 |
| 重庆 | 15min | 24 / 12 | 10272 | **90.46** | 171.12 | 0.1661 | 0.374 |
| 江苏 | 1h | 30 / 12 | 1104 | 90.18 | 114.45 | 0.7114 | 0.635 |
| 江苏 | 15min | 30 / 12 | 4416 | 86.95 | 113.61 | 0.6902 | 0.472 |

**四类（含零线）横向比较（MAE，单位 元/MWh；TwoStage 已改 single-pass）：**

| 市场 | 粒度 | naive_rolling_7d | Baseline | TwoStage | Conv2D | 最优 |
|------|------|------------------:|---------:|---------:|-------:|------|
| 内蒙 | 1h | 170.00 | 134.19 | **107.96** | 113.12 | TwoStage |
| 内蒙 | 15min | 178.60 | 138.18 | **120.10** | 126.03 | TwoStage |
| 重庆 | 1h | 100.81 | 95.47 | 92.46 | **89.17** | Conv2D |
| 重庆 | 15min | 105.69 | 95.39 | **89.89** | 90.46 | TwoStage (差异 0.6) |
| 江苏 | 1h | **68.64** | 83.91 | 74.08 | 90.18 | naive_rolling_7d |
| 江苏 | 15min | **71.39** | 91.69 | 81.55 | 86.95 | naive_rolling_7d |

> **零线对照（详 `RESULTS.md` §4.5）**：江苏 1h/15min 三个 ML 算法全部劣于 7 日均值零线（−7.9% ~ −31.4%）。统一口径后江苏 ML 普遍未跨过零线，应优先做特征/数据审计而不是调参。
> **Profile Corr 全市场最优均为 Conv2D**（详 `RESULTS.md` §2.3）。

**Conv2D-MultiTask 早停模式（`--early-stop --patience 10`，2026-05-26 引入）：**

| 市场 | 粒度 | 原 MAE | 早停 MAE | ΔMAE | best_ep / stop_ep |
|------|------|--------|----------|------|-------------------|
| 内蒙 | 1h | 113.12 | **105.31** | **−7.81** | 1 / 11 |
| 内蒙 | 15min | 126.03 | **119.83** | **−6.19** | 0 / 10 |
| 重庆 | 1h | 89.17 | 92.82 | +3.65 | 0 / 10 |
| 重庆 | 15min | 90.46 | 92.80 | +2.34 | 0 / 10 |
| 江苏 | 1h | 90.18 | **88.16** | **−2.02** | 24 / 34 |
| 江苏 | 15min | 86.95 | **81.74** | **−5.21** | 21 / 31 |

> - 4/6 改善（内蒙 ×2、江苏 ×2），平均 ΔMAE = −2.54；内蒙节点价场景 MAE 降 6.9% / 4.9%，profile_corr 同步从 0.72/0.66 提到 0.75/0.71。
> - 重庆 2 个粒度反退（+2.3/+3.6），best_ep=0 反映 warmup 起点 val 反而最低、patience 触发过早；后续优化方向（sliding-window val / `min_epochs_before_es`）见 `RESULTS.md` §4.4。
> - 原结果保留在 `runs/predictions/<market>/conv2d_multitask[_15min]/`，早停结果在 `_es` 后缀目录；复现命令 `python algorithms/conv2d_multitask/run.py --market all --freq {1h,15min} --early-stop --patience 10`。

## 三、暂不纳入验证

以下算法/实验因不成熟或属于研究性质，暂不进入 benchmark 流程：

| 类别 | 说明 |
|------|------|
| Moirai / Chronos 系列 | 时序基础模型 zero-shot 及微调（v11/v12），尚不成熟 |
| SPO+ / DFL | 决策导向微调（基于 MILP 收益优化），属于决策层而非预测层 |
| 联合训练（Joint CQ-NM） | 跨市场联合训练实验，需进一步验证有效性 |
| 早期迭代（v1~v7） | 已被后续版本淘汰 |

## 四、对应关系速查

下表列出算法在各源工程中的历史版本号，便于追溯实验产出。

| 算法 | 内蒙 (neimeng_prj) | 重庆 (chongqing_prj) | 江苏 (jiangsu_prj) | 本工程 |
|------|---------------------|----------------------|---------------------|--------|
| naive_baseline (lag/mean) | — | — | — | `algorithms/naive_baseline/` |
| LightGBM-Baseline | — | — | realtime_v3 / v4 / v6 | `algorithms/lightgbm_baseline/` |
| LightGBM-TwoStage | — | — | dayahead_v7_residual | `algorithms/lightgbm_twostage/` |
| Conv2D-MultiTask | v8.0 系列 | — | — | `algorithms/conv2d_multitask/` |
| Transformer-Joint24h | v10.0-joint | — | — | 待移植 |
| Transformer-Quantile | v10.0-quantile | — | — | 待移植 |
| ResConv2D | v25 系列 | v25_deep_nm_only_sudun | — | 待移植 |
