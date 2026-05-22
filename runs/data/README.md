# 统一冻结数据集

由 [scripts/build_dataset.py](../../scripts/build_dataset.py) 从兄弟工程的真源 CSV 导出，本仓**所有算法只读这套数据**，不再依赖 `neimeng_prj` / `chongqing_prj` 的训练代码。

## 目录约定

```
runs/data/
  manifest.json
  <market_id>/
    <version>/                  # 例如 v1
      meta.yaml                 # 切分日期、源 sha256、列规范
      train_15min.parquet
      val_15min.parquet         # 可选
      test_15min.parquet
      train_hourly.parquet      # 4×15min 槽均值聚合
      val_hourly.parquet        # 可选
      test_hourly.parquet
```

## 当前已注册市场（v1）

| market_id | 目标列 | 源文件 | 备注 |
|-----------|--------|--------|------|
| chongqing | `market_clearing_price` | `chongqing_prj/sql_data/chongqing_market_join.csv` | 81 列原样保留 |
| neimeng_sudun | `price_sudun_500kv1m_nodal` | `neimeng_prj/output/dws_15min_features_ext_data4.csv` | 套 fill_sudun_prices |
| neimeng_hongjing | `price_hongjing_220kv1m_nodal` | `neimeng_prj/output/dws_15min_features_ext_data4_hongjing.csv` | 套 fill_sudun_prices |
| neimeng_unified | `price_unified` | `neimeng_prj/output/dws_15min_features_ext_data4.csv` | 与 sudun 同源、不同目标 |

切分边界由各 `config/markets/<market_id>.yaml:data.splits` 钉死，构建时落入 `meta.yaml`。

## 使用

构建（默认 v1）：

```bash
python scripts/build_dataset.py --market all          # 全部
python scripts/build_dataset.py --market chongqing --force
python scripts/build_dataset.py --market neimeng_sudun --check-only
python scripts/build_dataset.py --list
```

在算法中读取：

```python
from pfbench.data import load_market_data

df_15, meta = load_market_data("neimeng_sudun", "test", "15min")
df_h, _    = load_market_data("neimeng_sudun", "test", "hourly")
# df 列含 actual（来自 meta["target_col"]）+ 全部源列
```

## 版本管理

- parquet **不入库**（大文件）；`meta.yaml` 与 `manifest.json` **入库**
- 同一 `<market>/<version>/` 一旦写入应视为不可变；源数据更新时**新增 v2**，不覆盖 v1
- bump 时机：源 CSV 的 `sha256` 变化 / 切分边界变化 / 预处理逻辑变化

## 真源校验记录

- 重庆：`sql_data:market_clearing_price` 按小时均值 vs `output/dws_hourly_features.csv:da_clearing_price` max-diff = 0.0
- 内蒙：原版 `dws_15min_features.csv` 与 `ext_data4` 在重叠期 `price_sudun_500kv1m_nodal` max-diff = 0.0；选 ext_data4 以获得更长时间窗
- `日清算结果查询电厂侧.xlsx` 是充放电执行后实际清算（节点价 + 电量），**不用作训练标签**（节点价语义不同，仅作 MILP 收益评价用）
