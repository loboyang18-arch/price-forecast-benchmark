# 本仓算法研发

在此目录实现、对比新算法与基线。**不修改**各省份兄弟工程的训练代码。

## 约定

1. 每个算法一个子目录，例如 `algorithms/baselines/persistence/`。
2. 推理脚本将测试集预测写入标准 CSV（见 `examples/predictions/README.md`）。
3. 输出路径：

   ```
   runs/predictions/<market_id>/<算法目录名>/test_predictions_hourly.csv
   ```

4. 评价：

   ```bash
   python scripts/run_benchmark.py --markets <market_id> --sources local --run-name <实验名>
   ```

## 建议结构

```
algorithms/
├── baselines/          # 持久化、季节性 naive 等
├── shared/             # 特征、数据加载（跨算法复用）
└── <your_model>/       # 新算法
```

数据切片、训练/测试划分应与 `config/markets/*.yaml` 中的 `test_start` / `test_end` 对齐，保证跨市场可比。
