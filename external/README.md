# 消费兄弟工程预测

本目录说明如何从各省份/节点工程接入预测结果，**无需**在本仓库复刻其训练逻辑。

## 配置

在 `config/markets/<market_id>.yaml` 中设置 `prediction_globs`，指向例如：

- `${WORKSPACE_ROOT}/neimeng_prj/output/experiments/**/test_predictions_hourly.csv`
- `${WORKSPACE_ROOT}/chongqing_prj/output/**/test_predictions_hourly.csv`

## 环境

```bash
export WORKSPACE_ROOT=/root/workspace   # 含 neimeng_prj、chongqing_prj 等的目录
```

## 评价

```bash
python scripts/run_benchmark.py --markets neimeng_sudun,chongqing --sources external
```

## CSV 格式

需含 `ts`、`actual`（或 `y_true`）、`pred`（或 `predicted` 等别名），详见 `examples/predictions/README.md`。

## 解耦原则

- 不 import 兄弟工程的训练模块。
- 评价窗口由本工程 `test_start` / `test_end` 统一裁剪，便于跨市场对比。
