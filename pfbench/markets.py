"""市场配置加载与预测文件发现（外部工程 + 本仓）。"""
from __future__ import annotations

import glob as glob_mod
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .paths import LOCAL_PREDICTIONS_DIR, MARKETS_DIR, ROOT

DEFAULT_WORKSPACE = Path(os.environ.get("WORKSPACE_ROOT", "/root/workspace"))


@dataclass
class MarketConfig:
    market_id: str
    name: str
    region: str
    task: str = "da"
    freq: str = "hourly"
    timezone: str = "Asia/Shanghai"
    test_start: str | None = None
    test_end: str | None = None
    # 兄弟工程 output/ 下的预测
    prediction_globs: list[str] = field(default_factory=list)
    prediction_files: list[str] = field(default_factory=list)
    # 本仓 runs/predictions/{market_id}/ 下的预测（算法研发产出）
    local_prediction_globs: list[str] = field(default_factory=list)
    local_prediction_files: list[str] = field(default_factory=list)
    notes: str = ""

    def resolved_globs(self, workspace: Path | None = None) -> list[str]:
        ws = str(workspace or DEFAULT_WORKSPACE)
        out = []
        for g in self.prediction_globs:
            out.append(g.replace("${WORKSPACE_ROOT}", ws).replace("${PROJECT_ROOT}", str(ROOT)))
        return out

    def resolved_local_globs(self) -> list[str]:
        default = str(LOCAL_PREDICTIONS_DIR / self.market_id / "**" / "*.csv")
        if not self.local_prediction_globs:
            return [default]
        out = []
        for g in self.local_prediction_globs:
            out.append(g.replace("${PROJECT_ROOT}", str(ROOT)).replace("${WORKSPACE_ROOT}", str(DEFAULT_WORKSPACE)))
        return out

    def resolved_files(self, workspace: Path | None = None) -> list[Path]:
        ws = workspace or DEFAULT_WORKSPACE
        files: list[Path] = []
        for raw in self.prediction_files:
            p = Path(raw.replace("${WORKSPACE_ROOT}", str(ws)).replace("${PROJECT_ROOT}", str(ROOT)))
            if p.is_file():
                files.append(p.resolve())
        return files

    def resolved_local_files(self) -> list[Path]:
        files: list[Path] = []
        for raw in self.local_prediction_files:
            p = Path(raw.replace("${PROJECT_ROOT}", str(ROOT)))
            if p.is_file():
                files.append(p.resolve())
        return files


def _expand_env(s: str, workspace: Path) -> str:
    s = s.replace("${WORKSPACE_ROOT}", str(workspace))
    s = s.replace("${PROJECT_ROOT}", str(ROOT))
    return re.sub(
        r"\$\{(\w+)\}",
        lambda m: os.environ.get(m.group(1), ""),
        s,
    )


def _expand_value(v, workspace: Path):
    if isinstance(v, str):
        return _expand_env(v, workspace)
    if isinstance(v, list):
        return [_expand_value(x, workspace) for x in v]
    return v


def list_markets() -> list[str]:
    return sorted(
        p.stem for p in MARKETS_DIR.glob("*.yaml")
        if not p.stem.startswith("_")
    )


def load_market(market_id: str, workspace: Path | None = None) -> MarketConfig:
    ws = workspace or DEFAULT_WORKSPACE
    path = MARKETS_DIR / f"{market_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"未找到市场配置: {market_id} ({path})")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = {k: _expand_value(v, ws) for k, v in raw.items()}
    return MarketConfig(
        market_id=raw["market_id"],
        name=raw.get("name", market_id),
        region=raw.get("region", "unknown"),
        task=raw.get("task", "da"),
        freq=raw.get("freq", "hourly"),
        timezone=raw.get("timezone", "Asia/Shanghai"),
        test_start=raw.get("test_start"),
        test_end=raw.get("test_end"),
        prediction_globs=raw.get("prediction_globs") or [],
        prediction_files=raw.get("prediction_files") or [],
        local_prediction_globs=raw.get("local_prediction_globs") or [],
        local_prediction_files=raw.get("local_prediction_files") or [],
        notes=raw.get("notes", ""),
    )


def _glob_csv(patterns: list[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        for s in glob_mod.glob(pattern, recursive=True):
            p = Path(s)
            if p.is_file() and p.suffix.lower() == ".csv":
                if "_archive" in p.parts:
                    continue
                found.add(p.resolve())
    return sorted(found, key=lambda x: str(x))


def discover_external_predictions(
    market: MarketConfig,
    workspace: Path | None = None,
    *,
    extra_globs: list[str] | None = None,
) -> list[Path]:
    """扫描兄弟工程中的预测 CSV。"""
    ws = workspace or DEFAULT_WORKSPACE
    found: set[Path] = set(market.resolved_files(ws))
    globs = market.resolved_globs(ws)
    if extra_globs:
        globs.extend(_expand_value(extra_globs, ws))
    for p in _glob_csv(globs):
        found.add(p)
    return sorted(found, key=lambda x: str(x))


def discover_local_predictions(market: MarketConfig) -> list[Path]:
    """扫描本仓 runs/predictions/{market_id}/ 下的预测 CSV。"""
    found: set[Path] = set(market.resolved_local_files())
    for p in _glob_csv(market.resolved_local_globs()):
        found.add(p)
    return sorted(found, key=lambda x: str(x))


def discover_predictions(
    market: MarketConfig,
    workspace: Path | None = None,
    *,
    extra_globs: list[str] | None = None,
    sources: str = "all",
) -> list[tuple[Path, str]]:
    """
    发现预测文件。

    sources: all | external | local
    返回 (path, source) 列表，source 为 external 或 local。
    """
    out: list[tuple[Path, str]] = []
    if sources in ("all", "external"):
        for p in discover_external_predictions(market, workspace, extra_globs=extra_globs):
            out.append((p, "external"))
    if sources in ("all", "local"):
        for p in discover_local_predictions(market):
            out.append((p, "local"))
    return out
