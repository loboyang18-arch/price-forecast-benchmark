# Aligned with NM strategy_milp_15min L720-787
"""储能策略控制台与 Markdown 报告。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _week_label(date_str: str) -> str:
    ts = pd.Timestamp(date_str)
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _fmt(v: float) -> str:
    return f"{v / 1e4:+.2f}万"


def print_report(
    df: pd.DataFrame,
    label: str = "MILP-15min",
    *,
    carry_soc: bool = False,
    use_comp: bool = True,
) -> None:
    """打印周分组与全段汇总表。"""
    net_col = "net_with_comp" if use_comp else "net_arbitrage"
    pf_col = "pf_net_with_comp" if use_comp else "pf_net_arbitrage"

    print(f"\n{'=' * 82}")
    print(f" {label}{'  [carry SOC]' if carry_soc else ''}")
    print(f"{'=' * 82}")

    soc_col = "  SOC_end" if carry_soc else ""
    hdr = (
        f"{'date':^12} {'charge':^12} {'discharge':^12} "
        f"{'chg MWh':>8} {'dis MWh':>8}"
        f"{soc_col}"
        f"  {'net(wan)':>10} {'PF(wan)':>10} {'realize':>7}"
    )
    print(hdr)
    print("-" * 82)

    wdf_all = df.copy()
    wdf_all["week"] = wdf_all["date"].apply(_week_label)

    for wk, wdf in wdf_all.groupby("week", sort=False):
        for _, r in wdf.iterrows():
            net_v = r.get(net_col, r.get("net", 0))
            pf_v = r.get(pf_col, r.get("pf_net", 0))
            ratio = net_v / pf_v if abs(pf_v) > 1 else float("nan")
            ratio_str = f"{ratio:>7.1%}" if not np.isnan(ratio) else f"{'-':>7}"
            soc_str = f"  {r.get('soc_end', 0):>6.0f}" if carry_soc else ""
            print(
                f"{r['date']:^12} {r['charge_window']:^12} {r['discharge_window']:^12} "
                f"{r['charge_mwh']:>8.0f} {r['discharge_mwh']:>8.0f}"
                f"{soc_str}"
                f"  {_fmt(net_v):>10} {_fmt(pf_v):>10} {ratio_str}"
            )

        s = wdf[[
            "charge_mwh",
            "discharge_mwh",
            net_col,
            pf_col,
        ]].sum()
        wk_ratio = s[net_col] / s[pf_col] if abs(s[pf_col]) > 1 else float("nan")
        print(
            f"  >> week {wk}: net={_fmt(s[net_col])}  PF={_fmt(s[pf_col])}  "
            f"realize={wk_ratio:.1%}"
        )
        print()

    print("=" * 82)
    n = len(df)
    tot_net = float(df[net_col].sum())
    tot_pf = float(df[pf_col].sum())
    ratio = tot_net / tot_pf if abs(tot_pf) > 1 else float("nan")
    print(f"period {n} days")
    print(f"  total net: {_fmt(tot_net)}  PF: {_fmt(tot_pf)}  realize: {ratio:.1%}")
    ann_strat = tot_net / n * 365 if n else 0
    ann_pf = tot_pf / n * 365 if n else 0
    print(f"  annualized: strat {ann_strat / 1e8:.3f} yi  PF {ann_pf / 1e8:.3f} yi")


def write_markdown_summary(
    df: pd.DataFrame,
    metrics: dict,
    out_path: Path,
    *,
    label: str = "MILP-15min",
    market_id: str = "",
    algorithm: str = "",
) -> None:
    """写出 Markdown 摘要。"""
    dm = metrics.get("decision_metrics", metrics)
    arb = dm.get("arbitrage", {})
    comp = dm.get("with_comp", {})

    lines = [
        f"# Storage evaluation: {market_id} / {algorithm}",
        "",
        f"- Label: {label}",
        f"- Days: {len(df)}",
        f"- comp_mode: {dm.get('comp_mode', 'in_objective')}",
        f"- carry_soc: {dm.get('carry_soc', False)}",
        "",
        "## Decision metrics (arbitrage)",
        "",
        "| metric | value |",
        "|--------|-------|",
    ]
    for k, v in arb.items():
        lines.append(f"| {k} | {v} |")
    lines.extend([
        "",
        "## Decision metrics (with compensation)",
        "",
        "| metric | value |",
        "|--------|-------|",
    ])
    for k, v in comp.items():
        lines.append(f"| {k} | {v} |")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
