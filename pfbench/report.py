"""生成 Markdown 对比报告。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_markdown_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# 跨市场电价预测评价",
        "",
        "按市场分组，MAE 升序（越小越好）。",
        "",
    ]
    if summary.empty:
        lines.append("_无有效评价结果。_")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    for mid, g in summary.groupby("market_id", sort=False):
        name = g["market_name"].iloc[0] if "market_name" in g.columns else mid
        lines.append(f"## {name} (`{mid}`)")
        lines.append("")
        has_src = "source" in g.columns
        hdr = "| 模型 | 来源 | MAE | RMSE | 曲线相关 | 方向准确率 | 样本数 |" if has_src else "| 模型 | MAE | RMSE | 曲线相关 | 方向准确率 | 样本数 |"
        sep = "|------|------|-----|------|----------|------------|--------|" if has_src else "|------|-----|------|----------|------------|--------|"
        lines.append(hdr)
        lines.append(sep)
        for _, r in g.sort_values("mae").iterrows():
            corr = r.get("profile_corr")
            dacc = r.get("direction_acc")
            if has_src:
                lines.append(
                    f"| {r['model_key']} | {r['source']} | {r['mae']:.2f} | {r['rmse']:.2f} | "
                    f"{corr if corr is not None else '—'} | "
                    f"{dacc if dacc is not None else '—'} | {int(r['valid_point_count'])} |"
                )
            else:
                lines.append(
                    f"| {r['model_key']} | {r['mae']:.2f} | {r['rmse']:.2f} | "
                    f"{corr if corr is not None else '—'} | "
                    f"{dacc if dacc is not None else '—'} | {int(r['valid_point_count'])} |"
                )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
