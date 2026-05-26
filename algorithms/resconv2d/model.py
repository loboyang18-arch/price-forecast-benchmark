"""ResConv2D 双头网络结构（vendored from neimeng_prj/src/model_v25_{resconv,deep_resconv}.py）。

输入: (B, C_in, H_SLOTS, LOOKBACK_DAYS=7)
输出: (price (B,), delta (B,))

提供两个深度档位：
  - shallow / base: 浅网（res64×1 + res128×2 + head 128）≈ 0.6M 参数
  - aggressive:     深网（res64×2 + res128×6 + head 512）≈ 2.5M 参数（v25 最优）

特征图沿时间维 H_SLOTS 池化两次（MaxPool(2,1)），最后两次 conv kernel=3 padding=0。
所以 h_out = H_SLOTS // 4 − 4 必须 ≥ 1 ⇒ H_SLOTS ≥ 20。
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + x)


def resolve_depth_config(profile: str = "aggressive") -> Dict[str, object]:
    """根据 profile 名解析深度配置。

    profile:
      - "aggressive": res64×2 + res128×6 + head 512（v25 最优）
      - "base":       res64×1 + res128×2 + head 128
      - "shallow":    同 base
    """
    p = profile.strip().lower()
    if p == "aggressive":
        return {"depth_profile": "aggressive", "n_res64": 2, "n_res128": 6, "head_hidden": 512}
    if p in ("base", "shallow"):
        return {"depth_profile": p, "n_res64": 1, "n_res128": 2, "head_hidden": 128}
    raise ValueError(f"unknown depth profile: {profile!r}")


def _fc_in(h_slots: int, lookback: int) -> int:
    h_out = h_slots // 2 // 2 - 2 - 2
    w_out = lookback - 2 - 2
    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            f"ResConv2D 输出维度非法: h_out={h_out}, w_out={w_out}, "
            f"h_slots={h_slots}, lookback={lookback}. "
            f"H_SLOTS 至少为 20（v25 原版 1h 时为 24）"
        )
    return 64 * h_out * w_out


def _make_head(fc_in: int, head_hidden: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(fc_in, head_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(head_hidden, 1),
    )


class _SharedTrunk(nn.Module):
    def __init__(self, n_res64: int, n_res128: int):
        super().__init__()
        self.res64 = nn.ModuleList([_ResBlock(64) for _ in range(n_res64)])
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.trans = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.res128 = nn.ModuleList([_ResBlock(128) for _ in range(n_res128)])
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.final1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=0),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=0),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.res64:
            x = block(x)
        x = self.pool1(x)
        x = self.trans(x)
        for block in self.res128:
            x = block(x)
        x = self.pool2(x)
        x = self.final1(x)
        return self.final2(x)


class DualHeadResConv2D(nn.Module):
    """ResConv2D 双头：价格回归 + Δ价回归。

    aggressive 配置参数量 ≈ 2.5M，对应内蒙工程 v25-deep-sudun500-mean4。
    """

    _dual_head = True

    def __init__(
        self,
        c_in: int,
        h_slots: int,
        lookback: int = 7,
        dropout: float = 0.44,
        depth_profile: str = "aggressive",
    ):
        super().__init__()
        cfg = resolve_depth_config(depth_profile)
        self.depth_profile = str(cfg["depth_profile"])
        self.n_res64 = int(cfg["n_res64"])
        self.n_res128 = int(cfg["n_res128"])
        self.head_hidden = int(cfg["head_hidden"])
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.trunk = _SharedTrunk(self.n_res64, self.n_res128)
        fc = _fc_in(h_slots, lookback)
        self.price_head = _make_head(fc, self.head_hidden, dropout)
        self.delta_head = _make_head(fc, self.head_hidden, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(self.stem(x))
        price = self.price_head(feat).squeeze(-1)
        delta = self.delta_head(feat).squeeze(-1)
        return price, delta
