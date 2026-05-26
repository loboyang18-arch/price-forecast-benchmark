# 算法登记

本文档记录纳入 benchmark 验证的算法，按建模范式分为两类。每个算法不绑定具体预测任务（日前/实时），后续均可在多市场、多任务上测试。

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

| 市场 | 粒度 | 特征数 | MAE | RMSE | Profile Corr | best_iter |
|------|------|--------|-----|------|--------------|-----------|
| 内蒙 | 1h | 44 | 97.54 | 123.48 | 0.7833 | 45 |
| 内蒙 | 15min | 47 | 112.67 | 140.23 | 0.7635 | 27 |
| 重庆 | 1h | 43 | 95.47 | 169.96 | 0.0597 | 1 |
| 重庆 | 15min | 46 | 95.39 | 175.69 | 0.1734 | 1 |
| 江苏 | 1h | 49 | 83.91 | 105.49 | 0.5532 | 70 |
| 江苏 | 15min | 52 | 91.69 | 114.87 | 0.4820 | 115 |

**LightGBM-TwoStage 三市场结果（Expanding Window CV，1h vs 15min）：**

| 市场 | 粒度 | 特征数 | CV 折数 | MAE | RMSE | Profile Corr | Naive MAE | 相对 Naive |
|------|------|--------|---------|-----|------|--------------|-----------|------------|
| 内蒙 | 1h | 246 | 12 | 83.12 | 114.53 | 0.7463 | 157.93 | +47.4% |
| 内蒙 | 15min | 248 | 12 | 90.90 | 124.02 | 0.7189 | 161.43 | +43.7% |
| 重庆 | 1h | 191 | 6 | 113.13 | 151.52 | 0.2254 | 101.61 | −11.3% |
| 重庆 | 15min | 193 | 6 | 112.02 | 153.08 | 0.2663 | 103.06 | −8.7% |
| 江苏 | 1h | 242 | 9 | 52.39 | 72.18 | 0.7466 | 74.13 | +29.3% |
| 江苏 | 15min | 244 | 9 | 55.01 | 76.43 | 0.7267 | 75.86 | +27.5% |

**两算法横向比较（MAE，单位 元/MWh）：**

| 市场 | 粒度 | Baseline | TwoStage | 最优 |
|------|------|----------|----------|------|
| 内蒙 | 1h | 97.54 | **83.12** | TwoStage (−14.8%) |
| 内蒙 | 15min | 112.67 | **90.90** | TwoStage (−19.3%) |
| 重庆 | 1h | **95.47** | 113.13 | Baseline (−15.6%) |
| 重庆 | 15min | **95.39** | 112.02 | Baseline (−14.8%) |
| 江苏 | 1h | 83.91 | **52.39** | TwoStage (−37.6%) |
| 江苏 | 15min | 91.69 | **55.01** | TwoStage (−40.0%) |

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
| 内蒙 | 1h | 25 / 12 | 8736 | 99.16 | 136.60 | 0.7694 | 0.526 |
| 内蒙 | 15min | 25 / 12 | 34944 | 102.46 | 139.65 | 0.7626 | 0.856 |
| 重庆 | 1h | 24 / 12 | 2568 | **89.17** | 164.30 | 0.2015 | 0.498 |
| 重庆 | 15min | 24 / 12 | 10272 | **90.46** | 171.12 | 0.1661 | 0.374 |
| 江苏 | 1h | 30 / 12 | 1104 | 90.18 | 114.45 | 0.7114 | 0.635 |
| 江苏 | 15min | 30 / 12 | 4416 | 86.95 | 113.61 | 0.6902 | 0.472 |

**三算法横向比较（MAE，单位 元/MWh）：**

| 市场 | 粒度 | Baseline | TwoStage | Conv2D | 最优 |
|------|------|----------|----------|--------|------|
| 内蒙 | 1h | 97.54 | **83.12** | 99.16 | TwoStage |
| 内蒙 | 15min | 112.67 | **90.90** | 102.46 | TwoStage |
| 重庆 | 1h | 95.47 | 113.13 | **89.17** | Conv2D (−6.6% vs Baseline) |
| 重庆 | 15min | 95.39 | 112.02 | **90.46** | Conv2D (−5.2% vs Baseline) |
| 江苏 | 1h | 83.91 | **52.39** | 90.18 | TwoStage |
| 江苏 | 15min | 91.69 | **55.01** | 86.95 | TwoStage |

> **重庆是 Conv2D 唯一显著胜出的市场**：训练集仅 ~115 天，对 TwoStage 的 190+ 特征体系数据不足；Conv2D 端到端学时空模式对小样本更鲁棒。其他两市场详细对比与训练观察见 `RESULTS.md` §3.3 / §4.3。

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
| LightGBM-Baseline | — | — | realtime_v3 / v4 / v6 | `algorithms/lightgbm_baseline/` |
| LightGBM-TwoStage | — | — | dayahead_v7_residual | `algorithms/lightgbm_twostage/` |
| Conv2D-MultiTask | v8.0 系列 | — | — | `algorithms/conv2d_multitask/` |
| Transformer-Joint24h | v10.0-joint | — | — | 待移植 |
| Transformer-Quantile | v10.0-quantile | — | — | 待移植 |
| ResConv2D | v25 系列 | v25_deep_nm_only_sudun | — | 待移植 |
