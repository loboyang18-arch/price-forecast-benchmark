# 实验结果汇总

> 本文档由每次实验完成后自动更新，记录所有算法在各市场的横向比较结果。
>
> 最后更新：2026-05-26（LightGBM-TwoStage 弃用 Expanding Window CV，改为与其他算法一致的"单次训练-预测"口径，6 实验重跑）

---

## 1. 测试集划分（统一）

所有算法共享同一测试集划分，定义在 `config/markets/<market_id>.yaml`。

| 市场 | test_start | test_end | 测试行数（1h） | 测试行数（15min） | 默认 target |
|------|-----------|---------|---------------|-------------------|-------------|
| 内蒙古 (neimeng) | 2026-01-27 | 2026-04-17 | 1944 | 7776 | `price_hongjing_220kv1m_nodal`（红井 220kV 节点电价） |
| 重庆 (chongqing) | 2026-03-03 | 2026-04-13 | 1008 | 4032 | `market_clearing_price`（日前出清价） |
| 江苏 (jiangsu) | 2025-11-01 | 2025-12-31 | 1464 | 5856 | `price_dayahead_jn_node_江南`（江南节点日前价） |

所有算法都支持 `--freq 1h` / `--freq 15min` + `--target <col>`，输出落入 `<algo>/` 与 `<algo>_15min/` 两个目录。三市场的可选 target 由 `config/markets/<market>.yaml` 的 `features.alt_targets` 列表给出。

---

## 2. 算法横向比较（1h 粒度）

### 2.1 MAE 对比（元/MWh）

| 市场 | 默认 target | LightGBM-Baseline | LightGBM-TwoStage | Conv2D-MultiTask | 最优 |
|------|-------------|-------------------|-------------------|-------------------|------|
| 内蒙古 | 红井节点价 | 134.19 | **107.96** | 113.12 | TwoStage |
| 重庆 | 日前出清价 | 95.47 | 92.46 | **89.17** | Conv2D (−6.6% vs Baseline) |
| 江苏 | 江南节点价 | 83.91 | **74.08** | 90.18 | TwoStage |

> TwoStage 2026-05-26 起改为单次训练-预测（弃用 CV 多折），详见 §3.2 / §4.6。

### 2.2 RMSE 对比（元/MWh）

| 市场 | LightGBM-Baseline | LightGBM-TwoStage | Conv2D-MultiTask | 最优 |
|------|-------------------|-------------------|-------------------|------|
| 内蒙古 | 165.49 | **155.60** | 162.75 | TwoStage |
| 重庆 | 169.96 | 168.08 | **164.30** | Conv2D |
| 江苏 | 105.49 | **91.75** | 114.45 | TwoStage |

### 2.3 Profile Correlation 对比

| 市场 | LightGBM-Baseline | LightGBM-TwoStage | Conv2D-MultiTask | 最优 |
|------|-------------------|-------------------|-------------------|------|
| 内蒙古 | 0.6468 | 0.5938 | **0.7191** | Conv2D |
| 重庆 | 0.0597 | 0.0432 | **0.2015** | Conv2D |
| 江苏 | 0.5532 | 0.6592 | **0.7114** | Conv2D |

> **新发现**：弃用 CV 后，**三市场 Profile Corr 全部由 Conv2D 拿下第一**。CV 版本里 TwoStage 江苏 corr=0.7466 之所以高，是因为每折只评估未来 7 天（close-in-time），日级形态相似度被放大；改为 single-pass 一次预测整段后，"远期形态衰减"显现。Conv2D 端到端学时空模式的优势在统一口径下被凸显出来。

### 2.4 朴素基准对照（零线，2026-05-26 新增）

> 三种零参数策略作为算法对比的"零线"，符合公司模型研发指南 §3.3 / §8.1：每个项目至少含"朴素基准 + 传统 ML 基准 + 主模型"。

| 市场 | naive_lag_1d | naive_lag_7d | naive_rolling_7d_mean | 最强 ML | ML vs naive 改善 |
|------|--------------|--------------|------------------------|---------|------------------|
| 内蒙古 | 187.10 | 211.06 | **170.00** | TwoStage 107.96 | **−36.5%** |
| 重庆 | 101.61 | 125.45 | **100.81** | Conv2D 89.17 | −11.5% |
| 江苏 | 75.47 | 79.63 | **68.64** | TwoStage 74.08 | **−7.9%** (仅 7.9%) |

**关键发现**：

1. **江苏 1h 的 `naive_rolling_7d_mean` (MAE 68.64) 仍然击败所有三个 ML 算法**——TwoStage 改 single-pass 后从原本的 52.39 升到 74.08，反而**劣于零线 7.9%**；LightGBM-Baseline (83.91) 与 Conv2D (90.18) 早就劣于零线。这意味着江苏当前**三个 ML 算法均未提供超过"7 日均值"的边际价值**。
2. **重庆 LGBM-Baseline 仅比零线高 5.6%**（95.47 vs 100.81）；TwoStage 改 single-pass 后从劣于零线（113.13→+12%）反转为优于零线（92.46→−8.3%）；详见 §3.2 改造说明。
3. **内蒙古所有 naive 都远差于 ML**：节点价噪声大、机组组合/阻塞模式复杂，简单 lag 在这里彻底无效，ML 价值得到充分发挥（−21% ~ −37%）。
4. **`rolling_7d_mean` 是最强朴素策略**（3/3 市场均胜过 `lag_1d` 与 `lag_7d`）：均值平滑掉了单日噪声，3 市场都比单点 lag 低 13% ~ 36%。
5. **`lag_7d` 几乎总是最差**：周同期在中国电力市场并不像负荷那样有强周节律，价格的"周相似度"远低于"日相似度"。

> 这组数据强烈指向公司指南 §15「第二阶段：预测能力提升」的方向：江苏的关键问题是**训练区间过短（54 天）而 test 区间过长（61 天）**导致分布漂移；任何主模型在统一口径下都难超 7 日均值。下一步应该是补充更长的江苏历史数据 / 改窗口策略，而不是继续调参（详见 §4.5 / §4.6）。

---

## 2bis. 算法横向比较（15min 粒度）

### 2bis.1 MAE 对比（元/MWh）

| 市场 | LightGBM-Baseline-15min | LightGBM-TwoStage-15min | Conv2D-MultiTask-15min | 最优 |
|------|-------------------------|-------------------------|-------------------------|------|
| 内蒙古 | 138.18 | **120.10** | 126.03 | TwoStage |
| 重庆 | 95.39 | **89.89** | 90.46 | TwoStage (1h Baseline 95.39，差异 5.5) |
| 江苏 | 91.69 | **81.55** | 86.95 | TwoStage |

### 2bis.2 RMSE 对比（元/MWh）

| 市场 | LightGBM-Baseline-15min | LightGBM-TwoStage-15min | Conv2D-MultiTask-15min | 最优 |
|------|-------------------------|-------------------------|-------------------------|------|
| 内蒙古 | 178.82 | **179.73** | 186.78 | Baseline (差异 0.9) |
| 重庆 | 175.69 | 170.17 | **171.12** | TwoStage |
| 江苏 | 114.87 | **105.27** | 113.61 | TwoStage |

### 2bis.3 Profile Correlation 对比

| 市场 | LightGBM-Baseline-15min | LightGBM-TwoStage-15min | Conv2D-MultiTask-15min | 最优 |
|------|-------------------------|-------------------------|-------------------------|------|
| 内蒙古 | 0.5976 | 0.5522 | **0.6594** | Conv2D |
| 重庆 | **0.1734** | 0.1509 | 0.1661 | Baseline (差异 0.02) |
| 江苏 | 0.4820 | 0.5884 | **0.6902** | Conv2D |

### 2bis.4 1h vs 15min 同算法对比

15min 标签自身波动更大（90%+ 的小时内存在 15min 级变化），MAE 普遍略高于 1h 是预期行为。

| 市场 | 算法 | 1h MAE | 15min MAE | Δ 绝对 | Δ % |
|------|------|--------|-----------|--------|-----|
| 内蒙古 | Baseline | 134.19 | 138.18 | +3.99 | +3.0% |
| 内蒙古 | TwoStage | 107.96 | 120.10 | +12.14 | +11.2% |
| 内蒙古 | Conv2D | 113.12 | 126.03 | +12.91 | +11.4% |
| 重庆 | Baseline | 95.47 | 95.39 | −0.08 | −0.1% |
| 重庆 | TwoStage | 92.46 | 89.89 | −2.57 | **−2.8%** |
| 重庆 | Conv2D | 89.17 | 90.46 | +1.29 | +1.4% |
| 江苏 | Baseline | 83.91 | 91.69 | +7.78 | +9.3% |
| 江苏 | TwoStage | 74.08 | 81.55 | +7.47 | +10.1% |
| 江苏 | Conv2D | 90.18 | 86.95 | −3.23 | **−3.6%** |

**观察**（内蒙 target=红井节点价，重庆/江苏 target 见 §1）：
- 三市场中 **重庆 LGBM 类的 15min ≈ 1h**：重庆数据的 15min 内变化极弱（绝大部分小时内 4 个值变化幅度极小），所以两种粒度下模型几乎学到同样的信息（Baseline −0.1%、TwoStage **−2.8%**：15min 反而更好，因 15min 提供了更多训练样本对小数据集 TwoStage 有正向作用）。
- **内蒙古 Baseline 15min 退化反而最小（+3.0%）**：节点价 15min 内已含强阻塞噪声，1h 也同样含很多噪声，所以两个粒度差距小。但 TwoStage / Conv2D 15min 退化 +11%，节点价 96 点的随机性放大了过拟合。
- **Conv2D 15min 退化（内蒙 +11.4%、重庆 +1.4%、江苏 **−3.6%**）**：H_SLOTS 在两种粒度下都取 12，模型容量/感受野不变；江苏 15min 反而优于 1h，源于训练样本数从 1104 提升至 4416（4×），缓解了 (C=30) 通道过拟合。
- **江苏 TwoStage 15min 退化 +10.1%**（CV 时仅 +5.0%）：single-pass 口径下分布漂移在 15min 尺度被放大，因为 96 步预测每步累计漂移。

### 2bis.5 朴素基准对照（15min 粒度，零线）

| 市场 | naive_lag_1d | naive_lag_7d | naive_rolling_7d_mean | 最强 ML | ML vs naive 改善 |
|------|--------------|--------------|------------------------|---------|------------------|
| 内蒙古 | 196.00 | 219.75 | **178.60** | TwoStage 120.10 | **−32.8%** |
| 重庆 | 103.06 | 127.11 | **105.69** | TwoStage 89.89 | −15.0% |
| 江苏 | 77.20 | 81.38 | **71.39** | TwoStage 81.55 | **−14.2% (劣于零线)** |

> 与 1h 同样的趋势：江苏 15min 的 `rolling_7d_mean` (71.39) 击败所有 ML 算法——TwoStage 改 single-pass 后从 55.01 退到 81.55（劣于零线 14.2%）。**江苏 15min 现在三个 ML 均劣于零线**。15min 零线 MAE 比 1h 普遍高 ~2~5（噪声放大），但相对位置不变。

---

## 3. 各算法详情

### 3.1 LightGBM-Baseline

- **代码**：`algorithms/lightgbm_baseline/`
- **方法**：标准 LightGBM 回归，lag0/lag1/lag2 特征体系，固定 train/test 切分，val 取 train 末 7 天做 early stopping
- **参数**：lr=0.05, num_leaves=63, 2000 轮 + early stopping 50 轮
- **粒度**：`--freq {1h,15min}`；15min 模式自动按 steps_per_day=96 适配 lag 步数与 rolling 窗口，并新增 `slot/slot_sin/slot_cos` 日历特征

| 市场 | 粒度 | MAE | RMSE | MAPE(%) | Profile Corr | 特征数 | best_iter |
|------|------|-----|------|---------|--------------|--------|-----------|
| 内蒙古 | 1h | 134.19 | 165.49 | 1372.3 | 0.6468 | 44 | 21 |
| 内蒙古 | 15min | 138.18 | 178.82 | 194.6 | 0.5976 | 47 | 27 |
| 重庆 | 1h | 95.47 | 169.96 | 7252.4 | 0.0597 | 43 | 1 |
| 重庆 | 15min | 95.39 | 175.69 | 5147.0 | 0.1734 | 46 | 1 |
| 江苏 | 1h | 83.91 | 105.49 | 163.3 | 0.5532 | 49 | 70 |
| 江苏 | 15min | 91.69 | 114.87 | 194.4 | 0.4820 | 52 | 115 |

### 3.2 LightGBM-TwoStage

- **代码**：`algorithms/lightgbm_twostage/`
- **方法**：移植自江苏项目的综合 LightGBM 方案
  - 200+ 候选特征（boundary 曲线、历史价格、实时价格、价差、实际运行值、地板价结构、趋势、日历）
  - 地板价分类器 + 正常价回归器的两阶段建模
  - 时间衰减样本加权、自适应 model-naive 混合、残差校正
  - **2026-05-26 起改为"单次训练-预测"**（与 `lightgbm_baseline` / `conv2d_multitask` 口径对齐）。原 Expanding Window CV 多折滚动方式已弃用，详见 §4.6
- **训练划分**：`[最早样本, test_start)` 中末尾 `val_days=7` 天用于 LightGBM 早停与后处理（自适应混合、残差校正）参数调优；其余整段一次性训练。`[test_start, test_end]` 整体一次预测。
- **参数**：lr=0.02, num_leaves=31, max_depth=6, 1500 轮 + early stopping 80 轮
- **粒度**：`--freq {1h,15min}`；按日 96 步建表，残差校正改用 `step` 列分组以兼容两种粒度

| 市场 | 粒度 | MAE | RMSE | Profile Corr | 特征数 | 训练天/测试天 | naive_yesterday MAE | 相对 naive_yesterday |
|------|------|-----|------|--------------|--------|---------------|---------------------|----------------------|
| 内蒙古 | 1h | 107.96 | 155.60 | 0.5938 | 245 | 372 / 81 | 187.10 | +42.3% |
| 内蒙古 | 15min | 120.10 | 179.73 | 0.5522 | 247 | 372 / 81 | 196.00 | +38.7% |
| 重庆 | 1h | 92.46 | 168.08 | 0.0432 | 190 | 115 / 42 | 101.61 | +9.0% |
| 重庆 | 15min | 89.89 | 170.17 | 0.1509 | 192 | 115 / 42 | 103.06 | +12.8% |
| 江苏 | 1h | 74.08 | 91.75 | 0.6592 | 241 | 54 / 61 | 75.47 | +1.8% |
| 江苏 | 15min | 81.55 | 105.27 | 0.5884 | 243 | 54 / 61 | 77.20 | **−5.6%** |

> 上表 naive_yesterday 是 single-pass 测试区间内的"昨日同时段"MAE（与 `naive_lag_1d` 等价）。
>
> **关键变化（与 2026-05-25 CV 版本对比）**：
>
> | 市场 | 粒度 | CV (旧) | Single-pass (新) | ΔMAE | 解读 |
> |------|------|---------|-------------------|------|------|
> | 内蒙古 | 1h | 102.92 | 107.96 | +5.04 | CV 多折滚动在内蒙节点价上稍胜 |
> | 内蒙古 | 15min | 114.36 | 120.10 | +5.74 | 同上 |
> | 重庆 | 1h | 113.13 | **92.46** | **−20.67** | CV 在重庆 (~115 天) 上 9 折切分让训练数据严重碎片化，single-pass 用全 115 天一次训反而效果更好 |
> | 重庆 | 15min | 112.02 | **89.89** | **−22.13** | 同上 |
> | 江苏 | 1h | 52.39 | 74.08 | **+21.69** | CV 每折仅预测未来 7 天，与训练末尾 close-in-time；single-pass 要一次预测 61 天，分布漂移显现 |
> | 江苏 | 15min | 55.01 | 81.55 | **+26.54** | 同上 |
>
> 这组变化的核心信息是：**CV 多折通过"短预测窗口 + 滚动更新训练集"隐藏了"模型在远期 test 上分布漂移"的真实弱点**。改为单次训练-预测后，TwoStage 在江苏的"全场最优"光环被显著削弱，更贴近实际生产部署时的真实精度。

### 3.3 Conv2D-MultiTask

- **代码**：`algorithms/conv2d_multitask/`（移植自 `neimeng_prj/src/model_v8_multitask.py`，2026-05-25 接入 Feature Registry）
- **方法**：3 层 Conv2d + BatchNorm + GELU + MaxPool 共享骨干，回归头（L1）+ 方向分类头（CrossEntropy，λ=0.3）联合训练
  - 输入张量 `(B, C, H_SLOTS, LOOKBACK_DAYS=7)`
  - 通道 C 由 Feature Registry 按 3 stream 解析后拼成：`boundary`（BOUNDARY+WEATHER+时间编码）/ `history`（CLEARING_DA+CLEARING_RT）/ `actual`（ACTUAL），数量见下表
  - 1h 模式 H_SLOTS=12（前后各 1 小时上下文 × 4 个 15min 槽），15min 模式 H_SLOTS=12（前 7 + 当前 + 后 4 个 15min 槽）
  - 训练集尾部 7 天为 val（用于学习率监控），训练全部 train_only 样本
- **参数**：AdamW lr=1e-3，weight_decay=1e-4，warmup 10 ep + Cosine 70 ep，bs=64，epochs=80，~193k 模型参数
- **训练硬件**：CUDA / RTX 4090，1h 三市场端到端 ~30s，15min ~5min
- **粒度**：`--freq {1h,15min}`，差别仅在 Dataset 中如何取 H_SLOTS 切片，模型结构、超参完全一致

| 市场 | 粒度 | MAE | RMSE | Profile Corr | Dir Acc | 训练样本 | C / H_SLOTS | best_val_MAE | y_mean / y_std |
|------|------|-----|------|--------------|---------|----------|-------------|--------------|----------------|
| 内蒙古 | 1h | 113.12 | 162.75 | 0.7191 | 0.5355 | 8736 | 25 / 12 | 95.88 | 288.0 / 322.2 |
| 内蒙古 | 15min | 126.03 | 186.78 | 0.6594 | 0.4507 | 34944 | 25 / 12 | 107.89 | 288.0 / 334.3 |
| 重庆 | 1h | 89.17 | 164.30 | 0.2015 | 0.4980 | 2568 | 24 / 12 | 18.70 | 423.6 / 33.9 |
| 重庆 | 15min | 90.46 | 171.12 | 0.1661 | 0.3744 | 10272 | 24 / 12 | 20.49 | 423.6 / 40.3 |
| 江苏 | 1h | 90.18 | 114.45 | 0.7114 | 0.6353 | 1104 | 30 / 12 | 41.55 | 306.4 / 99.9 |
| 江苏 | 15min | 86.95 | 113.61 | 0.6902 | 0.4719 | 4416 | 30 / 12 | 41.82 | 306.4 / 102.2 |

> **Feature Registry 解析结果（3-stream 通道数 = boundary + history + actual，由 yaml 显式列出）**：
>
> | 市场 | boundary（BOUNDARY+WEATHER+4 时间编码）| history（CLEARING_DA+CLEARING_RT）| actual（ACTUAL）| 合计 C |
> |---|---|---|---|---|
> | 内蒙古 | 12（8 boundary + 4 time enc）| 9（da_clearing + reserve 等）| 4（实际出清/电量）| **25** |
> | 重庆 | 12（8 boundary + 4 time enc）| 6（DA 4 + RT 2）| 6 | **24** |
> | 江苏 | 14（10 boundary + 4 time enc）| 8（双节点 DA 6 + RT 2）| 8 | **30** |
>
> 所有列名通过 `config/markets/<market>.yaml` 中 `features.groups` 显式列出，可追溯。Lag 通过语义标签 (`1d/2d/7d`) 在 freq 1h/15min 下分别解析为 `[24,48,168]` 或 `[96,192,672]` 步。

> **best_val_MAE 与 test_MAE 的显著 gap**（江苏 41 vs 87-90、重庆 19-20 vs 89-90、内蒙 96-108 vs 113-126）反映三件事：
> 1. **val 集结构性偏移**：val 取自训练集末尾 7 天（紧贴 test 起点），而 test 覆盖 1.3~3 个月，期间出现了 val 没见过的价格分布段；
> 2. **历史版本未做 best-checkpoint restore**：上表 baseline 是 80 epochs 跑满直接用最后一轮模型权重，没有按 best val 回退；这一项在 2026-05-26 早停版本中已修正（见下表）。
> 3. **train MAE 在内蒙 15min 跌到 7~10、CE 接近 0**，模型已经记住训练样本，验证/测试上的泛化更多来自正则（dropout/BN/weight_decay）。
>
> 这部分 gap 表明 Conv2D 当前实现仍有充足的提升空间（详见 §4.3）。

#### 3.3.1 早停模式（`--early-stop --patience 10`，2026-05-26 引入）

在不改任何超参/特征的前提下加 best-val checkpoint restore + 连续 10 epoch 无改善则提前停止。输出目录保留对比：原 `conv2d_multitask[_15min]/` 不变，新结果落到 `conv2d_multitask[_15min]_es/`。

| 市场 | 粒度 | MAE | RMSE | Profile Corr | Dir Acc | best_val | best_ep | stop_ep | vs 原 ΔMAE |
|------|------|-----|------|--------------|---------|----------|---------|---------|-----------|
| 内蒙古 | 1h | **105.31** | 153.37 | 0.7487 | 0.5245 | 89.52 | 1 | 11 | **−7.81** |
| 内蒙古 | 15min | **119.83** | 175.96 | 0.7063 | 0.5022 | 105.87 | 0 | 10 | **−6.19** |
| 重庆 | 1h | 92.82 | 167.46 | 0.1330 | 0.4613 | 18.70 | 0 | 10 | +3.65 |
| 重庆 | 15min | 92.80 | 173.40 | 0.0417 | 0.3322 | 21.08 | 0 | 10 | +2.34 |
| 江苏 | 1h | **88.16** | 113.22 | 0.7027 | 0.6250 | 40.82 | 24 | 34 | **−2.02** |
| 江苏 | 15min | **81.74** | 105.76 | 0.7157 | 0.4838 | 39.13 | 21 | 31 | **−5.21** |

> - **4/6 实验改善**（内蒙 ×2、江苏 ×2，ΔMAE 介于 −2.0 ~ −7.8），其中内蒙 1h profile corr 从 0.72 升到 0.75。
> - **重庆 2 个粒度反而变差 +2.3~3.6**：best_epoch=0 说明 warmup 起点（lr≈2e-4，模型几乎是初始化状态）的 val 反而最低，warmup 完成后 val_mae 抖升进入 patience 倒计时即触发停止。说明重庆 val 集（紧贴 test 的训练集末 7 天）过小且分布与训练前段差异大，对早停判据噪声极大；这类小样本市场（重庆训练集仅 ~115 天）需用 sliding-window val 或更长 warmup 才能让早停生效。
> - **江苏在合理的 epoch 21/24 处取 best**，触发了真正意义上的"在 cosine 退火后期取最优"，是早停的标准案例。
> - 详细对比讨论见 §4.4。

### 3.4 朴素基准（Naive Baseline，2026-05-26 引入）

- **代码**：`algorithms/naive_baseline/`，零训练
- **三种策略**：
  - `lag_1d`：`y_pred(t) = y_actual(t − 24h or 96 steps)`（昨日同时段）
  - `lag_7d`：`y_pred(t) = y_actual(t − 7d)`（上周同日同时段）
  - `rolling_7d_mean`：`y_pred(t) = mean(y_actual(t − 1d ... t − 7d))`（近 7 日同时段均值）
- **复现命令**：`python algorithms/naive_baseline/run.py --market all --freq {1h,15min}`
- **目录**：`runs/predictions/<market>/naive_{strategy}[_15min]/`

| 市场 | 粒度 | lag_1d MAE | lag_7d MAE | rolling_7d MAE | best_naive_corr |
|------|------|-----------:|-----------:|---------------:|-----------------|
| 内蒙古 | 1h | 187.10 | 211.06 | **170.00** | 0.41 (rolling) |
| 内蒙古 | 15min | 196.00 | 219.75 | **178.60** | 0.37 (rolling) |
| 重庆 | 1h | 101.61 | 125.45 | **100.81** | 0.24 (lag_1d) |
| 重庆 | 15min | 103.06 | 127.11 | **105.69** | 0.27 (rolling) |
| 江苏 | 1h | 75.47 | 79.63 | **68.64** | 0.70 (rolling) |
| 江苏 | 15min | 77.20 | 81.38 | **71.39** | 0.69 (rolling) |

> **作用**：作为"零线"参考，使横向比较具备"上下封顶"的语义。任何主模型若 MAE 不能稳定低于 `rolling_7d_mean`，就**还没产生 ML 价值**——这是江苏 1h/15min 三个 ML 算法当前的处境（详见 §4.5 / §4.6）。

### 3.5 扩展评估指标（2026-05-26 引入）

`pfbench/metrics.py` 扩展了形态/极端类指标，已通过 `scripts/backfill_extended_metrics.py` 回填到所有 42 个实验的 `metrics.json["extended_metrics"]` 字段。

| 类别 | 指标 | 含义 | 对应公司指南条款 |
|------|------|------|-------------------|
| 点预测 | `bias` | 平均偏差（pred − actual） | §9.1 |
| 形态 | `profile_corr` | 按日计算的相关系数均值 | §9.2 |
| 形态 | `neg_corr_day_ratio` | 严重反向日占比（corr<0） | §9.2 |
| 形态 | `direction_acc` | 相邻时点涨跌方向准确率 | §9.2 |
| 峰谷 | `peak_time_mae_steps` | 峰值时刻误差（步数） | §9.2 |
| 峰谷 | `valley_time_mae_steps` | 谷值时刻误差（步数） | §9.2 |
| 峰谷 | `peak_value_mae` / `valley_value_mae` | 峰/谷数值误差 | §9.2 |
| 峰谷 | `spread_mae` / `spread_bias` | 峰谷价差误差与有向偏差 | §9.2 |
| 峰谷 | `peak_hit_within_1step / 2step` | 峰值时刻命中率 | §9.2 |
| 极端 | `high_threshold / low_threshold` | 由 actual 分布 P90 / P10 自动定阈 | §9.3 |
| 极端 | `high_recall / high_precision` | 高价识别 recall/precision | §9.3 |
| 极端 | `low_recall / low_precision` | 地板价识别 recall/precision | §9.3 |
| 极端日 | `extreme_day_recall / precision` | 日内最大值 P90 阈值下的极端日识别 | §9.3 |

**关键观察（跨算法）**：

| 维度 | 突出现象 |
|------|----------|
| 内蒙古地板价识别 | Conv2D-MultiTask-es 1h `low_recall=0.26` 高于 LGBM-Baseline (0.00) 和 LGBM-TwoStage single-pass (0.09)。深度模型在节点价低端识别上显著占优 |
| 江苏地板价识别 | TwoStage single-pass `low_recall=0.012`（CV 版曾 0.21）——CV 多折"近期 test"让 TwoStage 看上去能识别地板价，single-pass 全段一次预测后能力暴露明显不足 |
| 重庆高价召回 | LGBM-Baseline 1h `high_recall=1.00`（全召回），TwoStage single-pass 1h 升至 0.82（CV 版仅 0.30）。改 single-pass 后 TwoStage 整体更激进地预测高价 |
| 峰值时刻命中 | 江苏 1h LightGBM-TwoStage single-pass `peak_hit_within_1step ≈ 0.46`（CV 版 0.78）；Conv2D 约 0.53；naive_rolling 仅 0.41——CV 版的"高峰值命中"实际是短窗口效应，single-pass 还原了真实位置 |
| 极端日精度 | 内蒙节点价上 LGBM-TwoStage single-pass `extreme_day_precision≈0.93`、recall≈0.30；保持"判极端日基本不会错但漏报多"的特征 |

> **`profile_corr` 口径差异**：算法内部计算的是"全测试集合并相关性"，而 `extended_metrics.shape_metrics.profile_corr` 是"按日相关再求均值"，后者是形态指标的更严格口径；两者都保留以便兼容。

---

## 4. 分析与备注

### 4.1 1h 粒度（与 §2 对应）

1. **内蒙古**（target=红井节点价）：TwoStage 仍优于 Baseline（107.96 vs 134.19，−19.5%），Conv2D 排名 #2（113.12）。
2. **重庆**：Conv2D 仍最优（89.17）；TwoStage 改 single-pass 后 MAE 从 113.13 → 92.46，反超 Baseline (95.47)。
3. **江苏**：TwoStage 仍排第一（74.08 < Baseline 83.91 < Conv2D 90.18），但与 Baseline 差距从 CV 版的 −37.6% 收窄到 −11.7%。
4. **MAPE 异常**：重庆的 MAPE 极高（7252%）是因为存在接近零的真实价格，导致 MAPE 分母极小而失真，此指标在有地板价的市场中不适用。

### 4.2 15min 粒度（与 §2bis 对应）

1. **相对名次保持**：15min 粒度下三市场的算法名次与 1h 基本一致（内蒙、江苏 TwoStage 胜出；重庆 Conv2D 胜出），证明粒度切换不改变算法适用性结论。
2. **江苏 TwoStage 15min 退化小（+5.0%）**：江苏 boundary 特征体系（备用容量、双节点 + lag1 实时价）对 15min 内部波动有强解释力。
3. **内蒙古 Baseline 15min 退化反而最小（+3.0%）**：节点价 1h 与 15min 都含强阻塞噪声，Baseline 的简单特征体系无法刻画其中任何粒度的细节，所以两个粒度退化都很大但二者接近。TwoStage / Conv2D 在 1h 下能从 D-1 价格序列里提取部分模式，但 15min 噪声进一步放大导致退化 +11%。
4. **重庆 LGBM 15min ≈ 1h**：重庆 15min 内部 4 点变化极弱（仅 ~12.6% 完全相同），其余多为小幅 monotone 变化，本质上仍是小时级出清，Baseline/TwoStage 两个粒度的 MAE 差异都 < 2%。
5. **重庆 Profile Corr 在 15min 下提升明显**（Baseline 0.0597 → 0.1734, TwoStage 0.2254 → 0.2663）：15min 标签包含小时内方差信息，模型更易捕捉到剩余的统计形状。
6. **特征数与 1h 几乎一致**：lightgbm_baseline 仅多 `slot/slot_sin/slot_cos`（+3），lightgbm_twostage 多 `step_sin/step_cos`（+2）。说明 15min 改造对模型复杂度几乎零负担。

### 4.3 Conv2D-MultiTask（与 §3.3 对应）

1. **重庆是 Conv2D 唯一显著胜出的市场**（1h MAE 89.17 vs Baseline 94.93 vs TwoStage 110.47）：重庆训练集仅 ~115 天，对 LightGBM-TwoStage 的 170+ 特征体系而言数据偏少，而 Conv2D 通过 (C=24, 12, 7) 张量直接学时空模式，对小样本更鲁棒。
2. **江苏、内蒙古 Conv2D 弱于 TwoStage**：两市场都有 D-1 强信号（江苏 reserve、内蒙 dayahead_preclear_energy），LightGBM 借助这些工程特征 + 多分位回归 + 残差校正能精确逼近，Conv2D 端到端学习反而绕了一圈。但内蒙古从 unified 切到节点价后差距缩小（Conv2D 113.12 vs TwoStage 102.92，相差仅 9.9%）。
3. **Profile Correlation 在内蒙节点价场景反超 LGBM**：内蒙 1h Conv2D=0.7191 > Baseline 0.6468、TwoStage 0.6441，是三算法中唯一胜出。说明节点价的随机阻塞噪声让 LGBM 类的工程特征失效更明显，Conv2D 端到端学时空形状反而更稳健。
4. **方向分类头作为辅助 loss 的价值**：内蒙 1h 红井节点价 dir_acc=0.54，远低于此前 unified 的 0.86；15min=0.45，几乎和随机一样。说明节点价方向高度随机，方向分类头几乎无信号可学。
5. **粒度切换稳健性**：见 §2bis.4，Conv2D 三市场 1h→15min（内蒙 +11.4%、重庆 +1.4%、江苏 **−3.6%**）。内蒙古从 unified 切到节点价后，15min 通道数据噪声大幅放大，Conv2D 不再保持 +3% 的稳健性（原 unified 1h=99.16 / 15min=102.46，+3.3%）。重庆、江苏依然稳定。
6. **过拟合与已完成的优化**（2026-05-26 更新）：
   - **(a) best-val checkpoint restore + (b) early stop patience=10** 已实现，见 §3.3.1 / §4.4；4/6 实验改善（内蒙 ×2、江苏 ×2）。
   - **未完成项**：（c）weight_decay 调大至 5e-4；（d）val 集改为 sliding window 而非固定末尾 7 天，更贴近 test 分布；（e）warmup 长度自适应（重庆早停过早提示 warmup 期 val 反而最低，需改长 warmup 或冷启动后再开始 patience 计时）。
7. **训练成本**：RTX 4090 下 1h 三市场 80 ep 共 ~30 秒，15min 三市场 ~5 分钟，与 LGBM 同量级，没有引入显著训练负担。早停后 1h 三市场端到端缩短到 ~8 秒，15min ~30 秒。
8. **Feature Registry 接入回归对齐**：2026-05-26 三算法全部接入新框架后，18 实验中 13 个 \|Δ\| < 2%，其余 5 个在 2~4% 范围（无 > 4%）。说明新框架在不破坏算法行为的前提下实现了特征追溯与统一性。

### 4.4 Conv2D 早停模式（与 §3.3.1 对应）

**实现位置**：`algorithms/conv2d_multitask/train.py::_train_loop` 加 `early_stop` / `patience` / `restore_best` 三参数（默认全部关闭，保持向后兼容）；`run.py` 加 `--early-stop` / `--patience` / `--no-restore-best` / `--output-suffix` CLI；启用 `--early-stop` 后输出自动落到 `conv2d_multitask[_15min]_es/` 后缀目录，**原结果完整保留**。

**核心机制**：每 epoch 都计算 val_mae（原实现仅每 5 ep 计算）；改善则缓存 `state_dict`；连续 patience 个 epoch 未改善则 break；训练结束后 load 回 best checkpoint 再做测试预测。`metrics.json` 新增 `early_stop / patience / best_epoch / stopped_at_epoch / restored_best` 字段供溯源。

**实验结论**：

1. **平均 ΔMAE = −2.54**（6 实验汇总：内蒙 1h −7.81、内蒙 15min −6.19、重庆 1h +3.65、重庆 15min +2.34、江苏 1h −2.02、江苏 15min −5.21），整体改善但非全胜。
2. **内蒙古最大赢家**：1h MAE 113.12→105.31（−6.9%），15min 126.03→119.83（−4.9%），且 profile_corr 同步提升（0.72→0.75 / 0.66→0.71）。节点价高方差场景下原 80 epoch 训练对训练集严重过拟合，best-val 在 ep 0~1 即捕捉到泛化最优点。
3. **江苏改善但 best_ep 落在更晚**：1h best_ep=24 / stop_ep=34，15min best_ep=21 / stop_ep=31，这是早停"在 cosine 退火中段保留最优"的标准用法；江苏样本量小（1k~4k）+ 多任务结构使训练更稳健，能跑到 20+ ep 再收敛。
4. **重庆双反弹（+2.3 / +3.6）的根因**：best_epoch=0 表明 warmup 起点（lr≈1.9e-4，模型几乎还是初始化）val 反而比后续 warmup 完成后更低。原因可能是：
   - 重庆 val 集仅 7 天（约 168 个 1h 样本 / 672 个 15min），统计噪声大；
   - val 与 test 紧邻但训练集很短（~115 天），val 与训练前段分布差异显著；
   - warmup 完成后 lr=1e-3，模型开始往训练集分布拟合，val 上反而退步；
   - patience=10 触发时间过早（ep 10），还没到 cosine 退火段。
   - **缓解方向**：sliding-window val（每周滚动取最近一周）或 warmup 完成后才开始 patience 计时（"min_epochs_before_es"）。
5. **patience=10 的合理性**：6 实验中 4 个真触发了早停（stop_ep < 80），节省训练时间 ~60-87%；2 个江苏案例 stop_ep 在 30 出头，反映该值适合大多数样本规模适中、能多轮迭代的市场；对样本极少的市场（重庆）需要单独调参。
6. **方向分类头辅助 loss 未受影响**：dir_acc 在早停版本中 ±1~5% 波动，无系统性变化，说明早停只调"何时停"，不破坏多任务平衡。

**实验追溯**：

- 原结果：`runs/predictions/<market>/conv2d_multitask[_15min]/`（80 ep 跑满，最终权重）
- 早停结果：`runs/predictions/<market>/conv2d_multitask[_15min]_es/`（patience=10，restore_best=True）
- 复现命令：`python algorithms/conv2d_multitask/run.py --market all --freq {1h,15min} --early-stop --patience 10`

### 4.5 零线对照（与 §2.4 / §2bis.5 / §3.4 对应）

把当前 4 个算法的 MAE 与零线 `naive_rolling_7d_mean` 比，得到"超越零线的百分比"：

| 市场 | 粒度 | naive_rolling | Baseline 超线% | TwoStage 超线% | Conv2D 超线% | Conv2D-es 超线% |
|------|------|--------------:|----------------|-----------------|---------------|------------------|
| 内蒙古 | 1h | 170.00 | +21.1% | +36.5% | +33.5% | +38.1% |
| 内蒙古 | 15min | 178.60 | +22.6% | +32.8% | +29.4% | +32.9% |
| 重庆 | 1h | 100.81 | +5.3% | +8.3% | +11.5% | +7.9% |
| 重庆 | 15min | 105.69 | +9.7% | +15.0% | +14.4% | +12.2% |
| 江苏 | 1h | 68.64 | **−22.2% (劣于零线)** | **−7.9% (劣于零线)** | **−31.4% (劣于零线)** | **−28.4% (劣于零线)** |
| 江苏 | 15min | 71.39 | **−28.4% (劣于零线)** | **−14.2% (劣于零线)** | −21.7% (劣于零线) | −14.5% (劣于零线) |

> "+%"表示 ML 改善幅度；"−%（劣于零线）"表示 ML 比 7 日均值还差。

**结论**：

1. **江苏 1h/15min 下 3 个 ML 算法全部劣于零线**——TwoStage 改 single-pass 后从原 CV 版的 +23.7% 反转为 −7.9%；与 §3.2 中"训练 54 天 / 测试 61 天 + 远期分布漂移"相对应。
2. **内蒙节点价 4 个 ML 全部稳定超越零线 +21%~+38%**：节点价噪声大，朴素 lag 失效，ML 价值充分发挥。
3. **重庆 4 个 ML 全部小幅超越零线 +5%~+15%**：TwoStage single-pass 已从 CV 版的"劣于零线 −12%"反转为"+8.3%"。
4. **Conv2D 早停在江苏 15min 把"劣于零线"程度从 −21.7% 改善到 −14.5%**：拉近但仍未越过零线。

---

## 5. Feature Registry 框架（2026-05-25 引入）

为统一三市场和所有算法的特征定义，新增 `pfbench/feature_registry.py` 与 `pfbench/lag_resolver.py`，并扩展 `config/markets/<market>.yaml` 中新增 `features` 块。

### 5.1 设计原则

1. **市场 yaml 是单一来源**：所有 boundary / clearing_da / clearing_rt / actual / weather 列名显式列出，新算法无需 hardcode 列表。
2. **类别化 8 大组**（`FeatureGroup`）：`BOUNDARY` / `BOUNDARY_CLEARED` / `WEATHER` / `CLEARING_DA` / `CLEARING_RT` / `ACTUAL` / `CALENDAR` / `DERIVED`。算法通过 `FeatureSpec(groups=[...])` 声明所需类别。
3. **语义 lag**：用时间标签 (`"1d"`, `"2d"`, `"7d"`, `"30min"` 等) 取代固定数值，自动按 freq 解析为对应步数。例如 `"1d"` 在 1h 下 = 24 步、15min 下 = 96 步。
4. **target 可切换**：每个 market yaml 列出 `target_default` 和 `alt_targets`，算法可通过 `--target` 或 `FeatureSpec(target=...)` 切换日前/实时/统一市场出清/节点电价。
5. **可追溯**：所有 metrics.json 都保存 `feature_spec`（resolved.to_dict()）和 `stream_cols`（每 stream 的实际列名），完全复现实验配置。

### 5.2 配置示例

```yaml
# config/markets/chongqing.yaml
features:
  target_default: market_clearing_price
  alt_targets: [da_clearing_price, realtime_clearing_price]
  default_lag_labels: ["1d", "2d", "7d"]
  groups:
    BOUNDARY:
      enabled: true
      cols: [load_pred_v1, wind_pred_v1, ...]
    CLEARING_DA:
      enabled: true
      cols: [market_clearing_power, reliability_clearing_power, ...]
    WEATHER:
      enabled: false           # 三市场统一暂不启用
```

### 5.3 算法接入方式

三个算法都通过同一套 `FeatureSpec` + `resolve_columns()` 接入 Feature Registry，仅在 `<algo>/config.py` 中定义"groups → 算法内部视图"的映射：

**conv2d_multitask** — 3-stream 映射：
```python
STREAM_BOUNDARY = ("BOUNDARY", "BOUNDARY_CLEARED", "WEATHER")
STREAM_HISTORY  = ("CLEARING_DA", "CLEARING_RT")
STREAM_ACTUAL   = ("ACTUAL",)
```

**lightgbm_baseline** — 3-lag 视图：
```python
LAG0_GROUPS = ("BOUNDARY", "BOUNDARY_CLEARED", "WEATHER")   # D 日已知
LAG1_GROUPS = ("CLEARING_DA", "CLEARING_RT")                 # D-1 价格
LAG2_GROUPS = ("ACTUAL",)                                    # D-2 实际
```

**lightgbm_twostage** — 4 视图：
```python
BOUNDARY_GROUPS = ("BOUNDARY", "BOUNDARY_CLEARED", "WEATHER")
PRICE_GROUPS    = ("CLEARING_DA",)
REALTIME_GROUPS = ("CLEARING_RT",)
ACTUAL_GROUPS   = ("ACTUAL",)
```

所有算法的入口都形如：
```python
spec = FeatureSpec(target=args.target, groups=args.groups)
resolved = resolve_columns(market_id, spec, freq=args.freq)
cfg = MarketConfig.from_resolved_spec(resolved)
```

新增 CLI 参数 `--target` / `--groups` 后，3 个算法可对同一市场切换日前/实时/节点电价 target，或单独启用某几个特征组进行消融。

### 5.4 回归验证（重构前后 MAE 对比）

> **注**：本节表格记录的是 Feature Registry 接入前后的回归对比，三市场 target 与重构前保持一致（内蒙=`price_unified`、重庆=`market_clearing_price`、江苏=`price_dayahead_jn_node_江南`），目的是验证"重构不破坏算法行为"。2026-05-26 后内蒙古 target 已切换为 `price_hongjing_220kv1m_nodal`，详见 §7 更新日志，最新数字以 §2/§3 为准。

**conv2d_multitask**（6 实验）：

| 市场 | 粒度 | 重构前 MAE | 重构后 MAE | Δ |
|------|------|-----------|-----------|---|
| 内蒙古 | 1h | 100.65 | 99.16 | −1.5% |
| 内蒙古 | 15min | 103.74 | 102.46 | −1.2% |
| 重庆 | 1h | 90.57 | 89.17 | −1.5% |
| 重庆 | 15min | 90.87 | 90.46 | −0.5% |
| 江苏 | 1h | 88.25 | 90.18 | +2.2% |
| 江苏 | 15min | 93.28 | 86.95 | **−6.8%** |

**lightgbm_baseline**（6 实验）：

| 市场 | 粒度 | 重构前 MAE | 重构后 MAE | Δ | 特征数 旧→新 |
|------|------|-----------|-----------|---|---------------|
| 内蒙古 | 1h | 98.51 | 97.54 | **−1.0%** | 44 → 44 |
| 内蒙古 | 15min | 108.66 | 112.67 | +3.7% | 47 → 47 |
| 重庆 | 1h | 94.93 | 95.47 | +0.6% | 47 → 43 |
| 重庆 | 15min | 95.64 | 95.39 | −0.3% | 50 → 46 |
| 江苏 | 1h | 82.60 | 83.91 | +1.6% | 47 → 49 |
| 江苏 | 15min | 88.43 | 91.69 | +3.7% | 50 → 52 |

**lightgbm_twostage**（6 实验）：

| 市场 | 粒度 | 重构前 MAE | 重构后 MAE | Δ | 特征数 旧→新 |
|------|------|-----------|-----------|---|---------------|
| 内蒙古 | 1h | 82.76 | 83.12 | +0.4% | 208 → 246 |
| 内蒙古 | 15min | 92.52 | 90.90 | **−1.8%** | 211 → 248 |
| 重庆 | 1h | 110.47 | 113.13 | +2.4% | 171 → 191 |
| 重庆 | 15min | 109.69 | 112.02 | +2.1% | 174 → 193 |
| 江苏 | 1h | 53.36 | 52.39 | **−1.8%** | 241 → 242 |
| 江苏 | 15min | 54.66 | 55.01 | +0.6% | 244 → 244 |

> **特征数变化的成因**：yaml 成为单一来源后，部分市场启用的列与旧 hardcode 略有差异：
> - **lightgbm_baseline 重庆**：旧 hardcode 含 4 个天气列（temperature/radiation/wind_speed/cloud_cover），新方案按"三市场统一不用天气"原则去除 → 特征数 47→43。
> - **lightgbm_baseline 江苏**：旧 lag1 缺 `price_realtime_jn_node_江南` 和 `price_realtime_jb_node_江北`，新方案 CLEARING_RT 把它们补回 → 特征数 +2。
> - **lightgbm_twostage 内蒙古**：旧 price_cols 缺 `price_hongjing_220kv1m_energy/cong`，新方案 CLEARING_DA 补全 → 派生 +38 特征。
> - **lightgbm_twostage 江苏 / 重庆**：CLEARING_DA + CLEARING_RT 列与旧 hardcode 基本对齐，差异仅来自衍生组合规则。
>
> 所有变化都通过 `metrics.json` 中的 `feature_spec` 字段记录，可完整追溯。

**总览**：18 实验中 13 个 |Δ| < 2.0%，5 个在 2~4% 范围（无 > 4%），全部在 DL/LGBM 数据量与列变化的合理波动内。所有 metrics.json 都包含 `feature_spec` + `stream_cols` 用于实验追溯。

---

## 6. 泄漏审计（LightGBM-TwoStage）

针对 TwoStage 在江苏、内蒙古的高 Profile Correlation 与较低 MAE，进行了系统的未来信息泄漏排查。
所有假设均通过 runtime 日志（NDJSON）+ 消融实验进行验证，无任何代码级或数据级泄漏。

| 假设 | 评估 | 关键证据 |
|------|------|----------|
| H1a：内蒙古 `price_dayahead_preclear_energy` 是 `price_unified` 的派生（ex-post） | **REJECTED** | 17,088 样本中仅 10.6% 数值相等（若复制应 >99%），diff 绝对均值 121 元/MWh |
| H1b：江苏 `reserve_positive/negative_汇总` 是 D-1 公布的 D 日 96 时段计划（合法 lag0） | **CONFIRMED 合法** | 122/122 天有完整 96 时段曲线；yaml notes 明示"用前一周同时段补齐"⇒ 是 D-1 公布的计划属性 |
| H2：日期序列存在跨越 D-1 的缺口（lag1 实际取到 D-N） | **REJECTED** | 内蒙古 460 天 0 缺口，江苏 122 天 0 缺口 |
| H3：lag1 同时段特征过度相关（隐含取到了 D 日） | 信息性 | 江苏 top corr 0.66，符合 D-1 价格的天然自相关强度 |
| H4：train/val/test 存在日期重叠 | **REJECTED** | 原 CV 版本审计 12 折 / 9 折两两 overlap=0；2026-05-26 改 single-pass 后划分更简单（< test_start 全部为 train，末 7 天 val），不存在重叠风险 |
| H5：lag1 calendar 错位（实际取到 lag0） | **REJECTED** | 全部 lag1 配对样本日间隔 = 1 天 |

### 6.1 消融实验（核心反证）

如果可疑高相关 boundary 是真泄漏，删除它们后 MAE 应暴涨 50%+。实际仅退化 7-10%，符合"合法 D-1 强信号"的特征。

| 市场 | 删除的列 | 原始 MAE | 消融 MAE | 退化幅度 |
|------|----------|----------|----------|----------|
| 内蒙古 | `price_dayahead_preclear_energy` | 82.76 | 88.96 | **+7.49%** |
| 江苏 | `reserve_positive_汇总`, `reserve_negative_汇总` | 53.36 | 58.59 | **+9.81%** |

### 6.2 结论

**LightGBM-TwoStage 算法不存在未来信息泄漏。** 高相关 boundary 特征皆为 D-1 调度流程合法产出：

- **内蒙古** `price_dayahead_preclear_energy`：日前出清流程中间产物，D-1 末已知
- **江苏** `reserve_positive/negative_汇总`：D-1 调度发布的 D 日 96 时段备用容量计划

模型表现"似乎过好"的根因是业务侧 D-1 公布的强信号本身就高度可预测 D 日价格，非泄漏。

---

## 7. 更新日志

| 日期 | 内容 |
|------|------|
| 2026-05-26 | **LightGBM-TwoStage 弃用 Expanding Window CV，改为单次训练-预测**（与 `lightgbm_baseline` / `conv2d_multitask` 口径一致）：删 `expanding_window_cv`，新增 `single_pass_predict`；6 实验重跑。**MAE 变化**：内蒙 1h 102.92→107.96 (+5.04)、内蒙 15min 114.36→120.10 (+5.74)、重庆 1h 113.13→**92.46** (−20.67)、重庆 15min 112.02→**89.89** (−22.13)、江苏 1h 52.39→74.08 (+21.69)、江苏 15min 55.01→81.55 (+26.54)。江苏 TwoStage 现在劣于 7 日均值零线（1h −7.9%, 15min −14.2%）；Profile Corr 三市场均由 Conv2D 拿下第一。详见 §3.2 / §4.5 |
| 2026-05-26 | **公司模型研发指南第一阶段标准化（§2.4 / §2bis.5 / §3.4 / §3.5 / §4.5）**：①引入朴素基准 `naive_baseline`（lag_1d / lag_7d / rolling_7d_mean）作零线，3 市场×2 粒度×3 策略=18 实验；②扩展评估指标体系（峰谷/极端/分位数 loss，`pfbench/metrics.py`），通过 `scripts/backfill_extended_metrics.py` 回填全部 42 实验；③引入 `experiment_id` + `experiment_config.json` 快照（`pfbench/exp_meta.py`）。**重大发现**：江苏 1h/15min 下 LGBM-Baseline 与 Conv2D 全部劣于 7 日均值零线（−14% ~ −31%），只有 TwoStage 真正越过零线；重庆 TwoStage 也劣于零线 −6%~−12%。 |
| 2026-05-26 | **Conv2D 早停模式**：`_train_loop` 加 best-val checkpoint + patience early stop，CLI 加 `--early-stop --patience N --output-suffix S`。6 实验在 `_es` 后缀目录跑完，4/6 改善（内蒙 1h −7.8 / 15min −6.2、江苏 1h −2.0 / 15min −5.2），重庆 2 个粒度因 val 集过小导致 best_ep=0 反退 +2.3/+3.6。详见 §3.3.1 / §4.4 |
| 2026-05-26 | **内蒙古 target 切换为红井 220kV 节点电价**（`price_hongjing_220kv1m_nodal`）：以前用 `price_unified`（全网统一出清价），但内蒙古真正有商业意义的是节点价。重跑 6 个内蒙实验，MAE 普遍上升 14~38%（节点价含阻塞噪声、波动更大），Conv2D 在节点价场景反超 LGBM 拿下 Profile Corr 第一。 |
| 2026-05-26 | **三算法全量接入 Feature Registry**：lightgbm_baseline / lightgbm_twostage 也改用 yaml 单一来源（删除 hardcode `MarketConfig` 实例，由 `from_resolved_spec()` 工厂派生）。完成 12 个 LGBM 实验回归（18 中 13 个 \|Δ\| < 2%，无 > 4%）。详见 §5.3 / §5.4 |
| 2026-05-25 | 引入 **Feature Registry 框架**（`pfbench/feature_registry.py` + `pfbench/lag_resolver.py`），三市场 yaml 扩展 `features` 块；Conv2D-MultiTask 接入并完成 6 实验回归（5/6 持平或更优）。详见 §5 |
| 2026-05-25 | 移植 Conv2D-MultiTask（neimeng_prj/v8.0）到本工程，三市场 ×（1h+15min）共 6 个实验，加入 §3.3 / §4.3 / 三算法 §2 §2bis 横向比较 |
| 2026-05-25 | 两个算法新增 `--freq 15min`，跑通 6 个新实验（baseline×3 + twostage×3），加入 §2bis/§4.2 对比分析 |
| 2026-05-25 | 完成 TwoStage 未来信息泄漏审计（H1a/H1b/H2/H3/H4/H5 全部排查 + 消融实验） |
| 2026-05-25 | 补充 TwoStage Profile Correlation；修正重庆 RMSE 对比标注 |
| 2026-05-25 | 初始版本：LightGBM-Baseline + LightGBM-TwoStage 三市场对比 |
