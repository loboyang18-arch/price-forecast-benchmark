# 操作记录

## 2026-05-22 — 统一冻结数据集构建

### 1. 设计原则

数据集只负责**清洗和入库**，保证数据安全和完整。不设 target_col，不做 train/val/test 切分，不生成 hourly 聚合。这些全部由下游算法自行决定。

### 2. 产物结构

```
runs/data/
├── neimeng/v1/
│   ├── data.parquet      # 全量 15min，21 列，缺测已填补
│   └── meta.yaml         # 源文件 hash、列名、时间范围、预处理记录
└── chongqing/v1/
    ├── data.parquet      # 全量 15min，80 列
    └── meta.yaml
```

### 3. 数据集配置

| 市场 | source | 列数 | 时间范围 | 行数 | 预处理 |
|------|--------|------|---------|------|--------|
| chongqing | `chongqing_prj/sql_data/chongqing_market_join.csv` | 80 | 2025-11-01 ~ 2026-04-13 | 15,744 | — |
| jiangsu | `jiangsu_prj/warehouse/feature_ready/V0/power_market_feature_ready_wide.parquet` | 36 | 2025-09-01 ~ 2025-12-31 | 11,712 | `shift_ts_to_period_start` + `drop_qflag_columns` + `fill_reserve_from_prev_week` |
| neimeng | `neimeng_prj/output/dws_15min_features_ext_data4_hongjing.csv` | 21 | 2025-01-13 ~ 2026-04-17 | 44,160 | `fill_sudun_prices` + `fill_hongjing_from_unified` + `fill_unified_from_sudun` |

### 4. 缺测填补

三个预处理钩子均针对整日缺失、用同源文件中其他价格列代理：

- `fill_sudun_prices`：苏敦节点价三列在 2025-09-29 整日缺失 → 用 `price_unified` 代理
- `fill_hongjing_from_unified`：红井节点价三列在 2025-09-29 整日缺失 → 用 `price_unified` 代理
- `fill_unified_from_sudun`：统一出清价在 2026-04-09 整日缺失 → 用苏敦节点价代理

**江苏预处理（`jiangsu_fill.py`）：**
- `shift_ts_to_period_start`：时间戳从 period-end（00:15~次日00:00）左移 15min 统一为 period-start（00:00~23:45），与重庆/内蒙一致
- `drop_qflag_columns`：去掉 34 个 `_qflag` 质量标记列，仅保留业务特征
- `fill_reserve_from_prev_week`：正负备用在 2025-11-16/11-23（周日）整日缺失 → 用前一周同时段填补

填补后所有数值列在数据范围内无缺失。

### 5. 校验结果

```
chongqing/v1:  15744 行 × 80 列  irregular_steps=0  numeric_na=0
jiangsu/v1:    11712 行 × 36 列  irregular_steps=0  numeric_na=0
neimeng/v1:    44160 行 × 21 列  irregular_steps=0  numeric_na=0
```

时间轴连续、无跳跃、无重复。

### 6. 使用方式

```python
from pfbench.data import load_market_data

df, meta = load_market_data("neimeng")
# df: 44160 行 × 21 列, index=ts (15min)
# 算法自行选择 target_col、切分、聚合
target = df["price_sudun_500kv1m_nodal"]   # 或 price_hongjing_*, price_unified
train = df.loc[:"2026-01-26"]
test  = df.loc["2026-01-27":]
hourly = df.resample("1h").mean()
```

### 7. 工程文件清单

| 文件 | 说明 |
|------|------|
| `config/markets/neimeng.yaml` | 内蒙市场配置（合并原 sudun/hongjing/unified 三个） |
| `config/markets/chongqing.yaml` | 重庆市场配置 |
| `pfbench/data/builder.py` | 构建器：读源 → 裁剪 → 预处理 → 对齐 → 写 parquet |
| `pfbench/data/loader.py` | 加载器：读 data.parquet + meta.yaml |
| `pfbench/data/checks.py` | 校验：时间完整性、缺测统计 |
| `pfbench/data/sudun_fill.py` | 内蒙缺测填补函数（苏敦/红井/统一出清价） |
| `pfbench/data/jiangsu_fill.py` | 江苏预处理函数（去 qflag + 备用填补） |
| `config/markets/jiangsu.yaml` | 江苏市场配置 |
| `scripts/build_dataset.py` | CLI 入口 |

### 8. Cursor 规则

从 `neimeng_prj` 和 `chongqing_prj` 移植：
- `no-silent-waits.mdc` — 长任务须可见进展
- `no-sync-shell-waits.mdc` — 所有 Shell 调用后台化
- `imported/.../karpathy-guidelines.mdc` — 编码行为准则
- `benchmark-conventions.mdc` — 本仓库目录与评价约定
- `power-conda-env.mdc` — 使用 power conda 环境
