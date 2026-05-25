"""Conv2D-MultiTask 网络结构。

(B, C, H_SLOTS, LOOKBACK_DAYS) → Conv2d×3 → 共享特征
  ├→ 回归头 → (B,)   价格预测
  └→ 方向头 → (B,3)  涨/平/跌分类
"""
from __future__ import annotations

import torch.nn as nn

from .config import LOOKBACK_DAYS

DIR_CLASSES = 3


class Conv2dMultiTaskNet(nn.Module):
    def __init__(self, c_in: int, h_slots: int):
        super().__init__()
        k_h = min(3, h_slots)
        self.block1 = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=(k_h, 3), padding=(k_h // 2, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        h_out = h_slots // 2 // 2 - 2
        w_out = LOOKBACK_DAYS - 2
        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"Conv2D 输出维度非法: h_out={h_out}, w_out={w_out}, "
                f"请检查 h_slots={h_slots}, LOOKBACK_DAYS={LOOKBACK_DAYS}"
            )
        fc_in = 64 * h_out * w_out
        self.flatten = nn.Flatten()
        self.reg_head = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self.dir_head = nn.Sequential(
            nn.Linear(fc_in, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, DIR_CLASSES),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        feat = self.flatten(x)
        price = self.reg_head(feat).squeeze(-1)
        direction = self.dir_head(feat)
        return price, direction
