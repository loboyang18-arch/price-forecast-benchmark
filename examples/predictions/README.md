# 预测文件格式

评价脚本接受 CSV，需包含：

| 列 | 说明 |
|----|------|
| `ts` | 时间戳（或 DatetimeIndex） |
| `actual` | 实际电价（也可用 `y_true` / `target`） |
| `pred` | 模型预测（也可用 `predicted` / `forecast`） |

## 放置位置

| 来源 | 路径 | 配置 |
|------|------|------|
| 兄弟工程 | `*_prj/output/...` | `config/markets/*.yaml` 的 `prediction_globs` |
| 本仓算法 | `runs/predictions/<market_id>/<实验名>/*.csv` | 默认自动发现，或 `local_prediction_globs` |

外部接入说明见 `external/README.md`；本仓研发见 `algorithms/README.md`。
