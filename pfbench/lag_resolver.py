"""Lag 标签语义化解析。

提供 yaml/算法层用的「时间语义 lag」表达，运行时按当前预测粒度
（``freq``）解析为具体的 period 步数。

支持的语法（简单形式）：
    "15min" / "30min" / "1h" / "2h" / "4h" / "1d" / "2d" / "7d" / "1w" ...
    "0" 或 "0min" / "0h" / "0d" 表示不 shift

不支持组合形式（如 "1d12h"）。

不支持等价别名（如 ``"24h" ≡ "1d"`` 同时出现）—— 一个 lag 用一种写法即可，
``lag_to_minutes`` 不区分 ``24h`` 与 ``1d``（它们都返回 1440 分钟），
但 yaml 中建议**只用一种**以保持可读性。

不整除时直接 raise（例如 ``"30min"`` 在 ``freq="1h"`` 下不能整除 → 报错）。
"""
from __future__ import annotations

import re
from typing import Iterable, List

_UNIT_MAP = {
    "min": 1,
    "h":   60,
    "d":   60 * 24,
    "w":   60 * 24 * 7,
}

_LABEL_RE = re.compile(r"^\s*(\d+)\s*(min|h|d|w)\s*$")


def lag_to_minutes(label) -> int:
    """把 lag 标签解析为分钟数。

    支持的输入：
      - 字符串 ``"1d"`` / ``"30min"`` / ``"1h"`` 等
      - 整数 ``0``（仅允许 0，表示不 shift）

    Examples:
        >>> lag_to_minutes("1d")
        1440
        >>> lag_to_minutes("30min")
        30
        >>> lag_to_minutes("1w")
        10080
        >>> lag_to_minutes(0)
        0
    """
    if isinstance(label, (int, float)):
        if int(label) == 0:
            return 0
        raise ValueError(
            f"数字 lag 仅允许 0；其他请用字符串 '1h'/'1d' 等表达。got: {label!r}"
        )
    if not isinstance(label, str):
        raise TypeError(f"lag 标签必须是 str 或 0，got: {type(label).__name__}: {label!r}")

    s = label.strip()
    if s in ("0", "0min", "0h", "0d", "0w"):
        return 0

    m = _LABEL_RE.match(s)
    if not m:
        raise ValueError(
            f"无法解析 lag 标签 {label!r}；合法格式形如 '15min' / '1h' / '1d' / '1w'"
        )
    n, unit = int(m.group(1)), m.group(2)
    if n <= 0:
        raise ValueError(f"lag 标签 {label!r} 数值必须为正整数（或 0）")
    return n * _UNIT_MAP[unit]


def lag_to_periods(label, freq: str) -> int:
    """根据当前预测粒度把 lag 标签解析为 period 步数。

    Args:
        label: lag 标签（见 ``lag_to_minutes``）
        freq: 当前粒度，``"1h"`` 或 ``"15min"``

    Returns:
        正整数（或 0）。

    Raises:
        ValueError: 如果 lag 标签的分钟数无法整除 freq 的分钟数。

    Examples:
        >>> lag_to_periods("1d", "1h")
        24
        >>> lag_to_periods("1d", "15min")
        96
        >>> lag_to_periods("30min", "15min")
        2
        >>> lag_to_periods("30min", "1h")
        Traceback (most recent call last):
            ...
        ValueError: lag '30min' (30 min) 在 freq='1h' (60 min) 下非整除...
    """
    if freq not in ("1h", "15min"):
        raise ValueError(f"不支持的 freq={freq!r}，目前仅支持 '1h' / '15min'")
    lag_min = lag_to_minutes(label)
    freq_min = lag_to_minutes(freq)
    if lag_min == 0:
        return 0
    if lag_min % freq_min != 0:
        raise ValueError(
            f"lag {label!r} ({lag_min} min) 在 freq={freq!r} ({freq_min} min) 下非整除"
        )
    return lag_min // freq_min


def lag_labels_to_periods(labels: Iterable, freq: str) -> List[int]:
    """批量把 labels 列表解析为 periods。空列表返回 ``[0]``（默认不 shift）。"""
    labels = list(labels)
    if not labels:
        return [0]
    return [lag_to_periods(x, freq) for x in labels]


def lag_label_to_days(label) -> int:
    """把 ``window_lag`` 标签解析为整数天数。

    用于 feature_registry 的 ``window_lag`` 字段（即数据可用性 lag，
    单位天）。仅支持以"日"为粒度的语义：``"0"`` / ``"1d"`` / ``"2d"`` / ...

    Args:
        label: lag 标签，``"0"`` / ``"Nd"`` 或整数 0。

    Returns:
        非负整数天数。

    Raises:
        ValueError: 输入既不是 ``"0"`` 也不是 ``"Nd"`` 格式。

    Examples:
        >>> lag_label_to_days("0")
        0
        >>> lag_label_to_days("1d")
        1
        >>> lag_label_to_days("4d")
        4
        >>> lag_label_to_days(0)
        0
    """
    if isinstance(label, (int, float)):
        if int(label) == 0:
            return 0
        raise ValueError(
            f"window_lag 仅允许整数 0；其他粒度请用 '1d' / '2d' / ... 字符串。got: {label!r}"
        )
    if not isinstance(label, str):
        raise TypeError(
            f"window_lag 必须是 str 或 0；got: {type(label).__name__}: {label!r}"
        )
    s = label.strip()
    if s in ("0", "0d", ""):
        return 0
    m = re.match(r"^(\d+)d$", s)
    if not m:
        raise ValueError(
            f"window_lag 仅支持 '0' / 'Nd' 格式（粒度=天）；got {label!r}"
        )
    n = int(m.group(1))
    if n < 0:
        raise ValueError(f"window_lag 天数必须为非负整数；got {label!r}")
    return n
