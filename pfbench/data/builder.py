"""统一冻结数据集构建器。

读取 ``config/markets/<market_id>.yaml`` 的 ``data:`` 段：

    data:
      version: v1
      source: ${WORKSPACE_ROOT}/...csv
      source_ts_col: ts
      freq: 15min
      preprocess: [fill_sudun_prices]
      time_range: ["2025-01-13 00:00:00", "2026-04-17 23:45:00"]

→ 写入 ``runs/data/<market_id>/<version>/`` 下：

    data.parquet    # 全量 15min，所有列，缺测已填补
    meta.yaml       # 源文件 hash、列名、时间范围、预处理记录
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..markets import DEFAULT_WORKSPACE
from ..paths import DATA_DIR, MARKETS_DIR, ROOT
from .sudun_fill import fill_hongjing_from_unified, fill_sudun_price_columns, fill_unified_from_sudun

logger = logging.getLogger(__name__)

_PREPROCESS_REGISTRY = {
    "fill_sudun_prices": fill_sudun_price_columns,
    "fill_hongjing_from_unified": fill_hongjing_from_unified,
    "fill_unified_from_sudun": fill_unified_from_sudun,
}


class BuildError(RuntimeError):
    pass


def _expand(s: str, workspace: Path) -> str:
    s = s.replace("${WORKSPACE_ROOT}", str(workspace))
    s = s.replace("${PROJECT_ROOT}", str(ROOT))
    return re.sub(
        r"\$\{(\w+)\}",
        lambda m: os.environ.get(m.group(1), ""),
        s,
    )


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _load_data_config(market_id: str, workspace: Path) -> dict:
    path = MARKETS_DIR / f"{market_id}.yaml"
    if not path.is_file():
        raise BuildError(f"未找到市场配置: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "data" not in raw:
        raise BuildError(f"{path} 缺少 data: 段")
    cfg = raw["data"]
    cfg["_market_id"] = market_id
    cfg["_market_name"] = raw.get("name", market_id)
    cfg["_region"] = raw.get("region", "unknown")
    cfg["_task"] = raw.get("task", "da")
    cfg["_yaml_path"] = str(path)
    return cfg


def _read_source(cfg: dict, workspace: Path) -> tuple[pd.DataFrame, Path]:
    src = _expand(cfg["source"], workspace)
    src_path = Path(src)
    if not src_path.is_file():
        raise BuildError(f"源文件不存在: {src_path}")
    ts_col = cfg.get("source_ts_col", "ts")
    logger.info("读取源文件: %s", src_path)
    df = pd.read_csv(src_path, parse_dates=[ts_col])
    if ts_col != "ts":
        df = df.rename(columns={ts_col: "ts"})
    if "ts" not in df.columns:
        raise BuildError(f"源文件无 ts 列 ({src_path})")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df = df.drop_duplicates(subset=["ts"], keep="last")
    df = df.set_index("ts").sort_index()
    return df, src_path


def _apply_preprocess(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    steps = cfg.get("preprocess") or []
    if isinstance(steps, str):
        steps = [steps]
    for name in steps:
        hook = _PREPROCESS_REGISTRY.get(name)
        if hook is None:
            raise BuildError(f"未注册的预处理钩子: {name}")
        logger.info("preprocess: %s", name)
        df = hook(df)
    return df


def _align_to_grid(df: pd.DataFrame) -> pd.DataFrame:
    """对齐到 15min 网格；起止由现有数据决定。"""
    start = df.index.min().floor("15min")
    end = df.index.max().ceil("15min")
    grid = pd.date_range(start, end, freq="15min")
    return df.reindex(grid)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index().rename(columns={"index": "ts"})
    if "ts" not in out.columns:
        out = out.rename(columns={out.columns[0]: "ts"})
    out.to_parquet(path, index=False)


def list_dataset_versions(market_id: str) -> list[str]:
    market_dir = DATA_DIR / market_id
    if not market_dir.is_dir():
        return []
    return sorted(p.name for p in market_dir.iterdir() if p.is_dir())


def build_market(
    market_id: str,
    *,
    version: str = "v1",
    workspace: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """构建一个市场的冻结数据集。

    只做清洗和入库：读源 → 裁剪到 time_range → 预处理 → 对齐 15min 网格 → 写 data.parquet + meta.yaml。
    """
    ws = workspace or DEFAULT_WORKSPACE
    cfg = _load_data_config(market_id, ws)

    time_range = cfg.get("time_range")
    if not time_range or len(time_range) != 2:
        raise BuildError(f"{market_id}: data.time_range 必须为 [start, end]")
    range_start = pd.Timestamp(time_range[0])
    range_end = pd.Timestamp(time_range[1])

    out_dir = DATA_DIR / market_id / version
    if out_dir.exists():
        if not force:
            raise BuildError(f"{market_id}/{version} 已存在，若需覆盖请加 --force")
        shutil.rmtree(out_dir)

    df_raw, src_path = _read_source(cfg, ws)
    n_raw = len(df_raw)

    clip_start = range_start - pd.Timedelta(days=1)
    clip_end = range_end + pd.Timedelta(days=1)
    df_raw = df_raw.loc[(df_raw.index >= clip_start) & (df_raw.index <= clip_end)]
    if len(df_raw) < n_raw:
        logger.info(
            "裁剪到 %s ~ %s: %d -> %d 行",
            clip_start.date(), clip_end.date(), n_raw, len(df_raw),
        )

    df = _apply_preprocess(df_raw, cfg)
    df = _align_to_grid(df)

    df = df.loc[(df.index >= range_start) & (df.index <= range_end)]
    logger.info("最终数据: %d 行, %d 列, %s ~ %s",
                len(df), df.shape[1], df.index.min(), df.index.max())

    diffs = df.index.to_series().diff().dropna()
    expected_step = pd.Timedelta(minutes=15)
    irregular = int((diffs != expected_step).sum())
    total_na = int(df.select_dtypes(include=["number"]).isna().sum().sum())

    meta: dict[str, Any] = {
        "market_id": market_id,
        "market_name": cfg["_market_name"],
        "region": cfg["_region"],
        "version": version,
        "freq": "15min",
        "time_range": [str(range_start), str(range_end)],
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "irregular_steps": irregular,
        "total_numeric_na": total_na,
        "preprocess": cfg.get("preprocess") or [],
        "source": {
            "path": str(src_path),
            "sha256": _sha256_file(src_path),
            "raw_rows": n_raw,
        },
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "build_command": f"python scripts/build_dataset.py --market {market_id} --version {version}",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(df, out_dir / "data.parquet")
    (out_dir / "meta.yaml").write_text(
        yaml.safe_dump(meta, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("已写入: %s", out_dir)
    _update_manifest(market_id, version, meta)
    return meta


def _update_manifest(market_id: str, version: str, meta: dict) -> None:
    manifest_path = DATA_DIR / "manifest.json"
    manifest: dict[str, Any] = {"datasets": {}}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    manifest.setdefault("datasets", {})
    manifest["datasets"][market_id] = {
        "version": version,
        "time_range": meta["time_range"],
        "n_rows": meta["n_rows"],
        "n_columns": meta["n_columns"],
        "source_path": meta["source"]["path"],
        "source_sha256": meta["source"]["sha256"],
        "built_at": meta["built_at"],
    }
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
