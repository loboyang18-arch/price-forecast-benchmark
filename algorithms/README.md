# 算法登记

本文档记录纳入 benchmark 验证的算法，按建模范式分为两类。每个算法不绑定具体预测任务（日前/实时），后续均可在多市场、多任务上测试。

## 一、ML — 基于人工特征的机器学习方法

| 算法 | 说明 | 源工程 | 源文件 |
|------|------|--------|--------|
| **LightGBM** | 标准 LightGBM 回归。人工构造特征：boundary 曲线、日历（dow/month/is_weekend/hh_index）、历史价格多阶滞后。支持多决策时点（A0/A1/A2/B）和分位数预测（quantile_k80）。 | jiangsu_prj | `scripts/train_realtime.py` |
| **LightGBM-Residual** | 两阶段残差建模：第一阶段用 LightGBM 分类器识别底价时段，第二阶段对非底价时段做 LightGBM 回归，最终通过自适应 model-naive 混合（grid-search alpha）后处理。 | jiangsu_prj | `scripts/train_dayahead.py` |

**共同特点：**
- 特征由领域知识手工构造（非原始时序输入）
- 模型为单时段独立预测（hh_index 作为特征，单模型覆盖 96 个时段）
- Expanding window CV 按 trade_date 切分，防止数据泄漏

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

| 算法 | 内蒙 (neimeng_prj) | 重庆 (chongqing_prj) | 江苏 (jiangsu_prj) |
|------|---------------------|----------------------|---------------------|
| LightGBM | — | — | realtime_v3 / v4 / v6 |
| LightGBM-Residual | — | — | dayahead_v7_residual |
| Conv2D-MultiTask | v8.0 系列 | — | — |
| Transformer-Joint24h | v10.0-joint | — | — |
| Transformer-Quantile | v10.0-quantile | — | — |
| ResConv2D | v25 系列 | v25_deep_nm_only_sudun | — |
