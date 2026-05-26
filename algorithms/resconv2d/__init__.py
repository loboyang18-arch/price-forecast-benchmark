"""ResConv2D 算法（vendored from neimeng_prj/src/model_v25_{resconv,deep_resconv}.py）。

双头 ResConv2D：
- price_head: 价格 L1 回归
- delta_head: 邻时段 Δ价 L1 回归（首时段 anchor = 前一日末时段）

aggressive depth: res64×2 + res128×6 + head_hidden=512 ≈ 2.5M 参数。
对应内蒙工程 `v25-deep-sudun500-mean4`（test MAE≈121.5 / 105 天 苏敦节点价）。
"""
