#!/usr/bin/env python3
"""一次性脚本：扫描 runs/predictions/<market>/<algo>/ 下所有 metrics.json，
读取对应 test_predictions*.csv，计算扩展指标（peak/valley/extreme 等）并写回
metrics.json 的 `extended_metrics` 键。

幂等：重复运行只会更新 extended_metrics 块，不动其他字段。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfbench.exp_meta import make_experiment_id, save_config_snapshot
from pfbench.metrics import evaluate_predictions_csv

PRED_ROOT = ROOT / "runs" / "predictions"


def _find_pred_csv(algo_dir: Path) -> Path | None:
    for name in ("test_predictions_hourly.csv", "test_predictions_15min.csv"):
        p = algo_dir / name
        if p.exists():
            return p
    candidates = list(algo_dir.glob("test_predictions*.csv"))
    return candidates[0] if candidates else None


def main() -> None:
    if not PRED_ROOT.exists():
        print(f"no predictions root at {PRED_ROOT}")
        return

    rows = []
    for market_dir in sorted(PRED_ROOT.iterdir()):
        if not market_dir.is_dir():
            continue
        for algo_dir in sorted(market_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            mj = algo_dir / "metrics.json"
            csv = _find_pred_csv(algo_dir)
            if not mj.exists() or csv is None:
                continue
            try:
                ext = evaluate_predictions_csv(csv)
            except Exception as e:
                print(f"  FAIL {market_dir.name}/{algo_dir.name}: {e}")
                continue
            metrics = json.loads(mj.read_text())
            metrics["extended_metrics"] = ext
            if "experiment_id" not in metrics:
                target = metrics.get("target") or metrics.get("target_col")
                exp_id = make_experiment_id(
                    market_dir.name, algo_dir.name, target=target,
                )
                metrics["experiment_id"] = exp_id
                if not (algo_dir / "experiment_config.json").exists():
                    save_config_snapshot(
                        algo_dir, exp_id,
                        algorithm=metrics.get("algorithm", algo_dir.name),
                        market=market_dir.name,
                        target=target,
                        freq=metrics.get("freq", "1h"),
                        extra={"backfilled": True},
                    )
            mj.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
            rows.append((market_dir.name, algo_dir.name,
                         ext["point_metrics"]["mae"],
                         ext["shape_metrics"]["profile_corr"],
                         ext["peak_valley"]["peak_time_mae_steps"],
                         ext["extreme"]["high_recall"],
                         ext["extreme"]["low_recall"]))

    print(f"\n回填完成：{len(rows)} 个实验\n")
    print(f"{'market':10s} {'algo':28s} {'MAE':>8s} {'corr_d':>7s} "
          f"{'pk_dt':>7s} {'hi_rec':>7s} {'lo_rec':>7s}")
    print("-" * 80)
    for r in rows:
        m, a, mae, corr, pkdt, hi, lo = r
        print(f"{m:10s} {a:28s} {mae:8.2f} "
              f"{corr if corr is not None else '-':>7} "
              f"{pkdt if pkdt is not None else '-':>7} "
              f"{hi if hi is not None else '-':>7} "
              f"{lo if lo is not None else '-':>7}")


if __name__ == "__main__":
    main()
