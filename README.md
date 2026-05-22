# price_forecast_benchmark

跨市场电价预测**能力测试平台**：统一指标对比各模型在不同省份/节点上的表现，并作为**规范测试流程**与**新算法研发**的主工程。

## 定位

| 能力 | 说明 |
|------|------|
| **外部评价** | 读取 `neimeng_prj`、`chongqing_prj`、`jiangsu_prj` 等兄弟工程 `output/` 下的预测 CSV，不改动其训练代码 |
| **本仓研发** | 在 `algorithms/` 开发基线/新算法，预测写入 `runs/predictions/<market_id>/` 后同样走评价 |
| **规范流程** | `config/pipelines/` + `scripts/run_pipeline.py`（逐步落地）串联数据→推理→评价 |

## 目录结构

```
price_forecast_benchmark/
├── config/
│   ├── markets/              # 市场注册（评价窗口、预测发现路径、数据集配置）
│   └── pipelines/            # 测试流程配置（规划中）
├── pfbench/                    # 评价 + 数据层核心库
│   └── data/                   # 统一冻结数据集 builder/loader/checks
├── algorithms/                 # 本仓算法与基线（研发区）
├── pipelines/                  # 流程说明与约定（文档）
├── external/                   # 消费兄弟工程的约定说明
├── runs/
│   ├── data/                   # 统一冻结数据集（parquet + meta.yaml）
│   ├── benchmark/              # 排行榜、report.md（评价输出）
│   └── predictions/            # 本仓算法产出的预测 CSV
├── scripts/
│   ├── build_dataset.py        # 从兄弟工程冻结数据集 CLI
│   ├── run_benchmark.py        # 批量评价 CLI
│   └── run_pipeline.py         # 规范流程 CLI（占位）
├── examples/                   # 数据格式示例
└── .cursor/rules/              # Cursor Agent 行为规则（自 neimeng_prj 移植）
```

## 快速开始

```bash
cd /root/workspace/price_forecast_benchmark
pip install -r requirements.txt

# 列出市场及可发现的预测数量（external / local）
python scripts/run_benchmark.py --list-markets

# 仅评价兄弟工程中的实验
export WORKSPACE_ROOT=/root/workspace
python scripts/run_benchmark.py --markets neimeng_sudun --sources external \
  --run-name neimeng_sudun

# 多市场对比
python scripts/run_benchmark.py --markets neimeng_sudun,chongqing,jiangsu \
  -o runs/benchmark/all_markets

# 评价本仓 runs/predictions/ 下的预测
python scripts/run_benchmark.py --markets neimeng_sudun --sources local

# 单文件
python scripts/run_benchmark.py \
  --csv /root/workspace/neimeng_prj/output/experiments/.../test_predictions_hourly.csv \
  --market-id neimeng_sudun --run-name one_model
```

输出（默认 `runs/benchmark/<run-name>/`）：

- `cross_market_leaderboard.csv` — 含 `source` 列（`external` / `local`）
- `report.md` — 分市场 Markdown 表
- `errors.txt` — 无法解析的文件（若有）

## 统一冻结数据集

数据集只做**清洗和入库**，不设 target_col，不做 train/val/test 切分。算法自行决定预测目标、切分方式和时间粒度。

```bash
# 构建
python scripts/build_dataset.py --market all

# 在算法里读
from pfbench.data import load_market_data
df, meta = load_market_data("neimeng")       # 44160 行 × 21 列
df, meta = load_market_data("chongqing")     # 15744 行 × 80 列
df, meta = load_market_data("jiangsu")       # 11712 行 × 36 列

# 算法自行选择 target、切分、聚合
target = df["price_sudun_500kv1m_nodal"]
train, test = df.loc[:"2026-01-26"], df.loc["2026-01-27":]
hourly = df.resample("1h").mean()
```

| market_id | 主源 | 列数 | 时间范围 |
|-----------|------|------|---------|
| `chongqing` | `chongqing_prj/sql_data/chongqing_market_join.csv` | 80 | 2025-11-01 ~ 2026-04-13 |
| `jiangsu` | `jiangsu_prj/warehouse/feature_ready/V0/...parquet` | 36 | 2025-09-01 ~ 2025-12-31 |
| `neimeng` | `neimeng_prj/output/dws_15min_features_ext_data4_hongjing.csv` | 21 | 2025-01-13 ~ 2026-04-17 |

详见 [CHANGELOG.md](CHANGELOG.md)。

## 注册新市场

1. 复制 `config/markets/_template.yaml` → `config/markets/your_market.yaml`
2. 填写 `prediction_globs`（兄弟工程）和/或依赖默认的 `runs/predictions/<market_id>/`
3. 设置 `test_start` / `test_end` 统一评价窗口
4. 若要纳入冻结数据集，补 `data:` 段，再跑 `python scripts/build_dataset.py --market your_market`

## 本仓算法产出预测

将 CSV 放到：

```
runs/predictions/<market_id>/<算法或实验名>/predictions.csv
```

格式见 `examples/predictions/README.md`。然后：

```bash
python scripts/run_benchmark.py --markets <market_id> --sources local
```

## 指标说明

| 指标 | 含义 |
|------|------|
| MAE / RMSE | 点预测误差 |
| profile_corr | 按日实际 vs 预测曲线 Pearson 相关（均值） |
| direction_acc | 相邻时段涨跌方向一致比例 |
| neg_corr_day_ratio | 日相关为负的天数占比 |

若环境已安装 `price_forecast_eval`（power conda），单文件评价会额外写入 `extended` 字段。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `WORKSPACE_ROOT` | `/root/workspace` | 兄弟工程根目录 |
| `${PROJECT_ROOT}` | 本仓库根目录 | 可用于 YAML 中的本仓路径 |

## 与储能 MILP 收益评价

本工程专注 **预测精度与形态**。内蒙等市场的 **MILP 净收益** 仍在各省份工程内评估；收益结果可作为外部列合并，后续可扩展 `revenue` 类指标。

## 路径迁移说明

早期版本将评价输出放在 `output/`；现已统一为 `runs/benchmark/`。旧目录下的 smoke 结果可手动迁入或重新跑评价。

## Cursor 规则

自 `neimeng_prj` 移植的 Agent 规则位于 `.cursor/rules/`：

- `no-silent-waits.mdc` — 长任务须可见进展
- `imported/.../karpathy-guidelines.mdc` — 编码行为准则
- `benchmark-conventions.mdc` — 本仓库目录与评价约定
