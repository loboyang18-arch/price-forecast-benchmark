"""统一绘图模块 — 按周绘制预测 vs 真实曲线。

所有算法共用此模块生成标准化图表。

输入 DataFrame 约定（统一格式）：
  - ts: datetime 类型，小时级时间戳
  - actual: 真实值
  - predicted: 模型预测值

生成图表：
  1. 按周分页的逐日对比图（每周一张图，包含 7 天子图）
  2. 全测试集拼接时序图
  3. 误差分布 / 分时段分析图
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FONT_CANDIDATES = [
    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]

_cn_font_loaded = False
for _fp in _FONT_CANDIDATES:
    try:
        if __import__("os").path.exists(_fp):
            fm.fontManager.addfont(_fp)
            _cn_name = fm.FontProperties(fname=_fp).get_name()
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [_cn_name, "DejaVu Sans"]
            _cn_font_loaded = True
            break
    except Exception:
        pass

if not _cn_font_loaded:
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]

matplotlib.rcParams["axes.unicode_minus"] = False

COLORS = {
    "actual": "#1f77b4",
    "predicted": "#d62728",
    "naive": "#2ca02c",
    "fill": "#d62728",
}


def plot_weekly_predictions(
    df: pd.DataFrame,
    output_dir: Path,
    market_id: str,
    algorithm: str,
    target_col: str = "",
    naive_col: Optional[str] = None,
    freq: str = "1h",
) -> List[Path]:
    """按周绘制预测 vs 真实，每周一张图包含逐日子图。

    Args:
        df: 必须包含 ts, actual, predicted 列
        output_dir: 图表输出目录
        market_id: 市场 ID
        algorithm: 算法名称
        target_col: 目标列名（显示在标题中）
        naive_col: 如有 naive 列名，也会绘制
        freq: "1h" 或 "15min"，决定 x 轴刻度与 marker 密度

    Returns:
        生成的图表文件路径列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    df["date"] = df["ts"].dt.date
    df["week"] = df["ts"].dt.isocalendar().week.astype(int)
    df["year"] = df["ts"].dt.year

    overall_mae = np.mean(np.abs(df["actual"] - df["predicted"]))
    overall_rmse = np.sqrt(np.mean((df["actual"] - df["predicted"]) ** 2))

    weeks = df.groupby(["year", "week"]).ngroups
    week_groups = list(df.groupby(["year", "week"], sort=True))

    saved: List[Path] = []
    is_15min = freq == "15min"
    marker_size = 2 if is_15min else 3

    for (yr, wk), wdf in week_groups:
        dates_in_week = sorted(wdf["date"].unique())
        n_days = len(dates_in_week)

        fig, axes = plt.subplots(
            n_days, 1, figsize=(14, 3.2 * n_days),
            squeeze=False, sharex=False,
        )
        fig.suptitle(
            f"{market_id.upper()} — {algorithm}  |  {yr}年第{wk}周"
            + (f"  |  target: {target_col}" if target_col else ""),
            fontsize=13, fontweight="bold", y=1.01,
        )

        for idx, d in enumerate(dates_in_week):
            ax = axes[idx, 0]
            day = wdf[wdf["date"] == d].sort_values("ts")
            if is_15min:
                x_vals = day["ts"].dt.hour.values + day["ts"].dt.minute.values / 60.0
            else:
                x_vals = day["ts"].dt.hour.values
            y_true = day["actual"].values
            y_pred = day["predicted"].values
            day_mae = np.mean(np.abs(y_true - y_pred))

            ax.plot(x_vals, y_true, color=COLORS["actual"], linewidth=1.5,
                    marker="o", markersize=marker_size, label="真实", zorder=3)
            ax.plot(x_vals, y_pred, color=COLORS["predicted"], linewidth=1.5,
                    linestyle="--", marker="s", markersize=marker_size, label="预测", zorder=3)

            if naive_col and naive_col in day.columns:
                y_naive = day[naive_col].values
                ax.plot(x_vals, y_naive, color=COLORS["naive"], linewidth=1.0,
                        linestyle=":", alpha=0.7, label="Naive")

            granularity_label = "(15min)" if is_15min else "(hourly)"
            ax.set_title(f"{d} {granularity_label}  (MAE={day_mae:.1f})",
                         fontsize=10, loc="left")
            ax.set_xlabel("小时")
            ax.set_ylabel("价格")
            ax.set_xticks(range(0, 24, 2))
            ax.legend(fontsize=8, loc="upper right", ncol=3)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, 23.5)

        fig.tight_layout()
        fname = f"week_{yr}w{wk:02d}.png"
        fpath = output_dir / fname
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(fpath)

    fig_all = _plot_full_series(df, market_id, algorithm, target_col,
                                overall_mae, overall_rmse, naive_col)
    all_path = output_dir / "full_series.png"
    fig_all.savefig(all_path, dpi=150, bbox_inches="tight")
    plt.close(fig_all)
    saved.append(all_path)

    fig_err = _plot_error_analysis(df, market_id, algorithm)
    err_path = output_dir / "error_analysis.png"
    fig_err.savefig(err_path, dpi=150, bbox_inches="tight")
    plt.close(fig_err)
    saved.append(err_path)

    logger.info(
        "%s/%s: 生成 %d 张图 (含 %d 周图 + 1 全序列 + 1 误差分析)",
        market_id, algorithm, len(saved), len(week_groups),
    )
    return saved


def _plot_full_series(
    df: pd.DataFrame,
    market_id: str, algorithm: str, target_col: str,
    mae: float, rmse: float,
    naive_col: Optional[str] = None,
) -> plt.Figure:
    """全测试集拼接时序图。"""
    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(df["ts"], df["actual"], color=COLORS["actual"],
            linewidth=1.0, label="真实", alpha=0.9)
    ax.plot(df["ts"], df["predicted"], color=COLORS["predicted"],
            linewidth=1.0, linestyle="--", label="预测", alpha=0.9)

    if naive_col and naive_col in df.columns:
        valid = df[naive_col].notna()
        ax.plot(df.loc[valid, "ts"], df.loc[valid, naive_col],
                color=COLORS["naive"], linewidth=0.8, linestyle=":",
                label="Naive(昨日同时段)", alpha=0.7)

    dates = sorted(df["date"].unique())
    for d in dates:
        ts_start = pd.Timestamp(d)
        if ts_start.hour == 0:
            ax.axvline(x=ts_start, color="gray", linewidth=0.3, alpha=0.4)

    monday_dates = [d for d in dates if pd.Timestamp(d).dayofweek == 0]
    for md in monday_dates:
        ax.axvline(x=pd.Timestamp(md), color="#333", linewidth=0.8, alpha=0.5)

    title = (f"{market_id.upper()} — {algorithm}  |  "
             f"MAE={mae:.2f}  RMSE={rmse:.2f}")
    if target_col:
        title += f"  |  {target_col}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("价格")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def _plot_error_analysis(
    df: pd.DataFrame, market_id: str, algorithm: str,
) -> plt.Figure:
    """误差分析：分时段 MAE + 误差分布直方图。"""
    df = df.copy()
    df["error"] = df["predicted"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    df["hour_val"] = df["ts"].dt.hour

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"{market_id.upper()} — {algorithm} 误差分析",
                 fontsize=12, fontweight="bold")

    hourly_mae = df.groupby("hour_val")["abs_error"].mean()
    axes[0].bar(hourly_mae.index, hourly_mae.values, color="#4c72b0", alpha=0.8)
    axes[0].set_xlabel("小时")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("分时段 MAE")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].hist(df["error"], bins=50, color="#4c72b0", alpha=0.7, edgecolor="white")
    axes[1].axvline(x=0, color="red", linewidth=1, linestyle="--")
    axes[1].set_xlabel("误差 (pred - actual)")
    axes[1].set_ylabel("频次")
    axes[1].set_title("误差分布")
    axes[1].grid(True, alpha=0.3, axis="y")

    daily_mae = df.groupby("date")["abs_error"].mean()
    axes[2].plot(range(len(daily_mae)), daily_mae.values,
                 color="#4c72b0", marker="o", markersize=3)
    axes[2].set_xlabel("测试日序号")
    axes[2].set_ylabel("日均 MAE")
    axes[2].set_title("逐日 MAE 趋势")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
