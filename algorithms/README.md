# 算法登记

本文档记录纳入 benchmark 验证的算法，按建模范式分为两类。每个算法不绑定具体预测任务（日前/实时），后续均可在多市场、多任务上测试。

## 一、ML — 基于人工特征的机器学习方法

| 算法 | 说明 | 本工程实现 | 源工程 |
|------|------|-----------|--------|
| **LightGBM-Baseline** | 标准 LightGBM 回归 baseline。lag0/lag1/lag2 特征 + 日历 + 滚动统计。固定 train/test 切分，单次评估。 | `algorithms/lightgbm_baseline/` | jiangsu_prj |
| **LightGBM-TwoStage** | 完整方法体系（移植自报告验证方案）：200+ 候选特征、Two-Stage 地板价分类器 + 回归器、多分位数回归、时间衰减加权、model-naive 自适应混合、残差校正、Expanding Window CV。 | `algorithms/lgb_twostage/` | jiangsu_prj |

**共同特点：**
- 特征由领域知识手工构造（非原始时序输入）
- 模型为单时段独立预测（hour 作为特征，单模型覆盖 24 小时）
- 三市场统一接口，通过 MarketConfig 适配不同市场

**LightGBM-TwoStage 三市场最新结果（Expanding Window CV）：**

| 市场 | 特征数 | CV折数 | Model MAE | Naive MAE | 提升 |
|------|--------|--------|-----------|-----------|------|
| 江苏 | 172 | 7 | 56.87 | 83.42 | +31.8% |
| 内蒙 | 208 | 51 | 114.59 | 176.50 | +35.1% |
| 重庆 | 171 | 13 | 43.62 | 51.20 | +14.8% |

## 二、DL — 基于原始特征的深度学习方法

| 算法 | 说明 | 源工程 | 源文件 |
|------|------|--------|--------|
| **Conv2D-MultiTask** | Conv2D 多任务网络。输入原始特征矩阵 (C×4×7)，通过 3 层 Conv2D + BatchNorm + MaxPool 提取时空模式，多节点/多目标联合学习。 | neimeng_prj | `src/model_v8_multitask.py` |
| **Transformer-Joint24h** | Pure Transformer 全日联合预测。每小时原始特征 (C×4×7) 展平后 Linear 投影，经 Transformer Encoder 建模 24 小时间依赖关系，联合输出全天价格预测。 | neimeng_prj | `src/model_v10_joint.py` |
| **Transformer-Quantile** | 基于 Transformer-Joint24h 的分位数版本。输出每小时 5 个分位数（P10/P30/P50/P70/P90），使用 Pinball Loss 训练，可用于下游鲁棒优化。 | neimeng_prj | `src/model_v10_quantile.py` |
| **ResConv2D** | 10 层残差 Conv2D 网络（ResBlock + GELU）。双头架构：价格回归头 + 涨跌方向头。跨市场同构设计（内蒙/重庆共享网络结构）。 | neimeng_prj / chongqing_prj | `src/model_v25_resconv.py` |

**共同特点：**
- 输入为原始特征矩阵（多通道×时段×回看天数），不做手工特征工程
- 全日联合输出（24h 或 96×15min），利用时段间相关性
- PyTorch 实现，GPU 训练

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
| LightGBM-TwoStage | — | — | dayahead_v7_residual | `algorithms/lgb_twostage/` |
| Conv2D-MultiTask | v8.0 系列 | — | — | 待移植 |
| Transformer-Joint24h | v10.0-joint | — | — | 待移植 |
| Transformer-Quantile | v10.0-quantile | — | — | 待移植 |
| ResConv2D | v25 系列 | v25_deep_nm_only_sudun | — | 待移植 |
