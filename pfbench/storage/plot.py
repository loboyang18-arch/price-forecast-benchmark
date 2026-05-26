# Aligned with NM strategy_milp_15min L791-910
"""储能策略周图绘制。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

FS = 8.5
COLORS = {
    "charge_bg": "#BBDEFB",
    "discharge_bg": "#FFCDD2",
    "actual": "#1565C0",
    "pred": "#E53935",
    "soc": "#2E7D32",
    "charge_bar": "#1565C0",
    "discharge_bar": "#C62828",
}


def _setup_chinese_font() -> None:
    try:
        import matplotlib.font_manager as fm

        font_path = "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _week_label(date_str: str) -> str:
    ts = pd.Timestamp(date_str)
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def plot_weekly(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    show_pred_price: bool = True,
    label: str = "MILP-15min",
) -> list[Path]:
    """每周一张 PNG，每天一个 subplot。"""
    _setup_chinese_font()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    plot_df["week"] = plot_df["date"].apply(_week_label)
    t_axis = np.arange(96) * 15 / 60
    saved: list[Path] = []

    for wk, wdf in plot_df.groupby("week", sort=False):
        days = wdf["date"].tolist()
        n = len(days)
        fig, axes = plt.subplots(n, 1, figsize=(15, 3.4 * n), constrained_layout=True)
        if n == 1:
            axes = [axes]
        title_suffix = "" if show_pred_price else " (actual only)"
        fig.suptitle(
            f"{label} charge/discharge vs price{title_suffix}  {wk}",
            fontsize=11,
            fontweight="bold",
        )
        has_soc = "_soc" in wdf.columns

        for ax, (_, r) in zip(axes, wdf.iterrows()):
            actual_96 = np.array(r["_actual"])
            pred_96 = np.array(r["_pred"]) if show_pred_price else None
            c = np.array(r["_c"])
            d = np.array(r["_d"])
            soc_96 = np.array(r["_soc"]) if has_soc else None

            for t in range(96):
                x0 = t * 15 / 60
                x1 = (t + 1) * 15 / 60
                if c[t] > 0.5:
                    ax.axvspan(x0, x1, color=COLORS["charge_bg"], alpha=0.75, zorder=0)
                if d[t] > 0.5:
                    ax.axvspan(x0, x1, color=COLORS["discharge_bg"], alpha=0.75, zorder=0)

            ax.plot(
                t_axis + 15 / 120,
                actual_96,
                color=COLORS["actual"],
                lw=1.5,
                label="actual(15m)",
                alpha=0.9,
                zorder=3,
            )
            if show_pred_price and pred_96 is not None:
                native = bool(r.get("pred_native_15m", False))
                if native:
                    ax.plot(
                        t_axis + 15 / 120,
                        pred_96,
                        color=COLORS["pred"],
                        lw=1.3,
                        ls="--",
                        label="pred(15m)",
                        zorder=3,
                        alpha=0.88,
                    )
                else:
                    ax.step(
                        np.arange(24) + 0.5,
                        pred_96[::4],
                        color=COLORS["pred"],
                        lw=1.4,
                        ls="--",
                        label="pred(hourly)",
                        where="mid",
                        zorder=3,
                        alpha=0.85,
                    )

            ax2 = ax.twinx()
            bar_w = 14 / 60
            ax2.bar(
                t_axis + 15 / 120,
                c,
                width=bar_w,
                color=COLORS["charge_bar"],
                alpha=0.3,
                label="charge(MW)",
                zorder=2,
            )
            ax2.bar(
                t_axis + 15 / 120,
                -d,
                width=bar_w,
                color=COLORS["discharge_bar"],
                alpha=0.3,
                label="discharge(MW)",
                zorder=2,
            )
            if soc_96 is not None and soc_96.max() > 1:
                ax2.plot(
                    t_axis + 15 / 120,
                    soc_96,
                    color=COLORS["soc"],
                    lw=1.2,
                    ls=":",
                    alpha=0.85,
                    label="SOC(MWh)",
                    zorder=4,
                )
            ax2.set_ylim(-280, 280)
            ax2.set_ylabel("Power(MW) / SOC", fontsize=FS - 1, color="#888888")
            ax2.tick_params(labelsize=FS - 1, colors="#888888")
            ax2.axhline(0, color="#aaaaaa", lw=0.5, ls=":")

            soc_end_str = f"  SOC_end:{r.get('soc_end', 0):.0f}MWh" if has_soc else ""
            net_show = r.get("net_with_comp", r.get("net", 0))
            ax.set_title(
                f"{r['date']}   chg: {r['charge_window']} {r['charge_mwh']:.0f}MWh  "
                f"dis: {r['discharge_window']} {r['discharge_mwh']:.0f}MWh  "
                f"net: {net_show / 1e4:+.2f} wan{soc_end_str}",
                fontsize=FS,
                loc="left",
                pad=3,
            )
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 2))
            ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], fontsize=FS - 1)
            ax.set_ylabel("Price (CNY/MWh)", fontsize=FS)
            ax.tick_params(labelsize=FS - 1)
            ax.grid(axis="y", ls=":", alpha=0.4)
            ax.set_facecolor("#FAFAFA")

            if r["date"] == days[0]:
                h1, l1 = ax.get_legend_handles_labels()
                patch_c = mpatches.Patch(color=COLORS["charge_bg"], label="charge slot")
                patch_d = mpatches.Patch(color=COLORS["discharge_bg"], label="discharge slot")
                ncol_leg = 4 if show_pred_price else 3
                ax.legend(
                    handles=h1 + [patch_c, patch_d],
                    labels=l1 + ["charge slot", "discharge slot"],
                    loc="upper right",
                    fontsize=FS - 1,
                    framealpha=0.85,
                    ncol=ncol_leg,
                )

        out_path = out_dir / f"{wk}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved
