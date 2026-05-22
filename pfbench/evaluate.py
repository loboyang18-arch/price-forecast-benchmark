"""单文件 / 单市场 / 跨市场批量评价。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io import load_prediction_csv
from .markets import MarketConfig, discover_predictions, load_market
from .paths import BENCHMARK_RUNS_DIR
from .metrics import evaluate_frame, try_extended_eval


def _model_key_from_path(path: Path, market_id: str, source: str = "external") -> str:
    parts = path.parts
    if source == "local" and "predictions" in parts:
        i = parts.index("predictions")
        rel = list(parts[i + 2 :])
        if rel and rel[-1].lower().endswith(".csv"):
            rel = rel[:-1]
        if rel:
            return f"{market_id}/{'/'.join(rel)}"
    if "experiments" in parts:
        i = parts.index("experiments")
        if i + 1 < len(parts):
            return f"{market_id}/{parts[i + 1]}"
    return f"{market_id}/{path.stem}"


def evaluate_predictions(
    csv_path: Path,
    *,
    market_id: str = "custom",
    test_start: str | None = None,
    test_end: str | None = None,
    task: str = "da",
    use_extended: bool = True,
    source: str = "external",
) -> dict[str, Any]:
    df = load_prediction_csv(csv_path, test_start=test_start, test_end=test_end)
    core = evaluate_frame(df)
    result: dict[str, Any] = {
        "market_id": market_id,
        "source": source,
        "prediction_file": str(csv_path.resolve()),
        "model_key": _model_key_from_path(csv_path, market_id, source=source),
        **core,
    }
    if use_extended:
        ext = try_extended_eval(df.reset_index().rename(columns={"index": "ts"}), task=task)
        if ext:
            result["extended"] = ext
    return result


def evaluate_market(
    market: MarketConfig,
    workspace: Path | None = None,
    *,
    extra_globs: list[str] | None = None,
    use_extended: bool = True,
    max_files: int | None = None,
    sources: str = "all",
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    discovered = discover_predictions(
        market, workspace, extra_globs=extra_globs, sources=sources,
    )
    if max_files:
        discovered = discovered[:max_files]

    rows = []
    errors: list[tuple[str, str]] = []
    for p, src in discovered:
        rel = str(p)
        try:
            ev = evaluate_predictions(
                p,
                market_id=market.market_id,
                test_start=market.test_start,
                test_end=market.test_end,
                task=market.task,
                use_extended=use_extended,
                source=src,
            )
            pm = ev["point_metrics"]
            sm = ev["shape_metrics"]
            rows.append({
                "market_id": market.market_id,
                "market_name": market.name,
                "region": market.region,
                "source": src,
                "model_key": ev["model_key"],
                "prediction_file": rel,
                "mae": pm["mae"],
                "rmse": pm["rmse"],
                "mape_pct": pm["mape_pct"],
                "bias": pm["bias"],
                "valid_point_count": pm["valid_point_count"],
                "profile_corr": sm.get("profile_corr"),
                "direction_acc": sm.get("direction_acc"),
                "neg_corr_day_ratio": sm.get("neg_corr_day_ratio"),
                "n_days": sm.get("n_days"),
            })
        except Exception as e:
            errors.append((rel, str(e)))
    return pd.DataFrame(rows), errors


def default_benchmark_output(name: str = "latest") -> Path:
    return BENCHMARK_RUNS_DIR / name


def evaluate_markets(
    market_ids: list[str],
    workspace: Path | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, list[tuple[str, str]]]]:
    all_rows = []
    all_errors: dict[str, list[tuple[str, str]]] = {}
    for mid in market_ids:
        m = load_market(mid, workspace)
        df, errs = evaluate_market(m, workspace, **kwargs)
        if len(df):
            all_rows.append(df)
        if errs:
            all_errors[mid] = errs
    if not all_rows:
        return pd.DataFrame(), all_errors
    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["market_id", "mae"], na_position="last")
    return out, all_errors


def save_results(
    summary: pd.DataFrame,
    out_dir: Path,
    errors: dict[str, list[tuple[str, str]]] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cross_market_leaderboard.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if errors:
        err_lines = []
        for mid, pairs in errors.items():
            for path, msg in pairs:
                err_lines.append(f"{mid}\t{path}\t{msg}")
        (out_dir / "errors.txt").write_text("\n".join(err_lines), encoding="utf-8")
    meta = {
        "n_models": len(summary),
        "markets": summary["market_id"].unique().tolist() if len(summary) else [],
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return csv_path
