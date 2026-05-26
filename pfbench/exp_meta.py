"""实验元数据：生成 experiment_id 并保存 config 快照。

遵守公司模型研发管理指南 §12 / §3.2：每个实验须可追溯（配置 + 数据 + 代码版本）。
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _short_target(target: Optional[str]) -> str:
    if not target:
        return "default"
    t = target.lower()
    for prefix in ("price_", "market_"):
        if t.startswith(prefix):
            t = t[len(prefix):]
    # 取关键 token 拼接
    tokens = [tok for tok in t.replace("clearing", "clr").split("_") if tok][:3]
    return "-".join(tokens) if tokens else "default"


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def make_experiment_id(
    market: str,
    algo_dir: str,
    target: Optional[str] = None,
    when: Optional[datetime] = None,
) -> str:
    """生成形如 ``EXP_<market>_<algo>_<target短名>_<YYYYMMDD-HHMM>`` 的 ID。"""
    ts = (when or datetime.now()).strftime("%Y%m%d-%H%M")
    parts = ["EXP", market, algo_dir, _short_target(target), ts]
    return "_".join(parts)


def save_config_snapshot(
    out_dir: Path,
    experiment_id: str,
    *,
    algorithm: str,
    market: str,
    target: Optional[str],
    freq: str,
    extra: Optional[Dict[str, Any]] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    """保存 experiment_config.json 到 out_dir，返回路径。

    包含：experiment_id、算法、市场、target、freq、调用时间戳、git commit、
    以及调用方提供的 extra（超参、特征版本等）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if repo_root is None:
        repo_root = out_dir.resolve()
        for _ in range(8):
            if (repo_root / ".git").exists():
                break
            repo_root = repo_root.parent
    cfg: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "algorithm": algorithm,
        "market": market,
        "target": target,
        "freq": freq,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(repo_root),
    }
    if extra:
        cfg["extra"] = extra
    p = out_dir / "experiment_config.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, default=str),
                 encoding="utf-8")
    return p
