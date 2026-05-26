"""点预测与形态类指标（不依赖具体模型）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def point_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / (np.abs(actual) + 1e-6)) * 100)
    bias = float(np.mean(err))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape_pct": round(mape, 2),
        "bias": round(bias, 4),
        "valid_point_count": int(len(actual)),
    }


def shape_metrics(df: pd.DataFrame) -> dict:
    """按日聚合：曲线相关、涨跌方向准确率。"""
    daily = df.copy()
    daily["date"] = daily.index.normalize()
    corrs = []
    dir_hits = []
    for _, g in daily.groupby("date"):
        a = g["actual"].values
        p = g["pred"].values
        if len(a) < 3 or np.std(a) < 1e-6 or np.std(p) < 1e-6:
            continue
        corrs.append(float(np.corrcoef(a, p)[0, 1]))
        da = np.diff(a)
        dp = np.diff(p)
        if len(da):
            dir_hits.append(float(np.mean(np.sign(da) == np.sign(dp))))

    if not corrs:
        return {
            "profile_corr": None,
            "neg_corr_day_ratio": None,
            "direction_acc": None,
            "n_days": 0,
        }
    corrs_arr = np.array(corrs)
    return {
        "profile_corr": round(float(np.nanmean(corrs_arr)), 4),
        "neg_corr_day_ratio": round(float(np.mean(corrs_arr < 0)), 4),
        "direction_acc": round(float(np.mean(dir_hits)), 4) if dir_hits else None,
        "n_days": len(corrs),
    }


def peak_valley_metrics(df: pd.DataFrame) -> dict:
    """按日聚合：峰/谷时刻误差与峰谷价差误差。

    输入 df 需有 DatetimeIndex、actual、pred 列。
    返回：
      - peak_time_mae_steps: 峰值时刻索引误差（步数；与 freq 无关）
      - valley_time_mae_steps: 谷值时刻索引误差
      - peak_value_mae: 峰值数值误差
      - valley_value_mae: 谷值数值误差
      - spread_mae: 峰谷价差误差（|true_spread - pred_spread|）
      - spread_bias: 预测峰谷价差 - 实际峰谷价差（正=高估波动）
      - peak_hit_within_1step / 2step: 峰值时刻命中率（误差 ≤1/2 步）
    """
    work = df.copy()
    work["date"] = work.index.normalize()
    pk_t, vl_t = [], []
    pk_v, vl_v = [], []
    spread_err = []
    spread_signed = []
    pk_hit1 = pk_hit2 = 0
    n_days = 0
    for _, g in work.groupby("date"):
        a = g["actual"].values
        p = g["pred"].values
        if len(a) < 3 or np.any(np.isnan(a)) or np.any(np.isnan(p)):
            continue
        n_days += 1
        ai_pk, pi_pk = int(np.argmax(a)), int(np.argmax(p))
        ai_vl, pi_vl = int(np.argmin(a)), int(np.argmin(p))
        pk_t.append(abs(ai_pk - pi_pk))
        vl_t.append(abs(ai_vl - pi_vl))
        pk_v.append(abs(float(a[ai_pk]) - float(p[pi_pk])))
        vl_v.append(abs(float(a[ai_vl]) - float(p[pi_vl])))
        true_spread = float(a.max() - a.min())
        pred_spread = float(p.max() - p.min())
        spread_err.append(abs(true_spread - pred_spread))
        spread_signed.append(pred_spread - true_spread)
        d = abs(ai_pk - pi_pk)
        if d <= 1:
            pk_hit1 += 1
        if d <= 2:
            pk_hit2 += 1
    if n_days == 0:
        return {
            "peak_time_mae_steps": None,
            "valley_time_mae_steps": None,
            "peak_value_mae": None,
            "valley_value_mae": None,
            "spread_mae": None,
            "spread_bias": None,
            "peak_hit_within_1step": None,
            "peak_hit_within_2step": None,
            "n_days_pv": 0,
        }
    return {
        "peak_time_mae_steps": round(float(np.mean(pk_t)), 4),
        "valley_time_mae_steps": round(float(np.mean(vl_t)), 4),
        "peak_value_mae": round(float(np.mean(pk_v)), 4),
        "valley_value_mae": round(float(np.mean(vl_v)), 4),
        "spread_mae": round(float(np.mean(spread_err)), 4),
        "spread_bias": round(float(np.mean(spread_signed)), 4),
        "peak_hit_within_1step": round(pk_hit1 / n_days, 4),
        "peak_hit_within_2step": round(pk_hit2 / n_days, 4),
        "n_days_pv": n_days,
    }


def extreme_metrics(
    df: pd.DataFrame,
    high_q: float = 0.90,
    low_q: float = 0.10,
) -> dict:
    """高价 / 地板价识别指标（§9.3）。

    阈值取自整段 test 的 actual 分布的 high_q / low_q 分位数。
    返回 high/low 的 recall（真极端中模型识出的比例）和
    precision（模型判极端中真为极端的比例），以及阈值本身用于追溯。
    """
    a = df["actual"].astype(float).values
    p = df["pred"].astype(float).values
    m = ~(np.isnan(a) | np.isnan(p))
    a, p = a[m], p[m]
    if len(a) < 20:
        return {
            "high_threshold": None,
            "low_threshold": None,
            "high_recall": None,
            "high_precision": None,
            "low_recall": None,
            "low_precision": None,
            "n_points_ext": int(len(a)),
        }
    th_hi = float(np.quantile(a, high_q))
    th_lo = float(np.quantile(a, low_q))
    a_hi = a >= th_hi
    p_hi = p >= th_hi
    a_lo = a <= th_lo
    p_lo = p <= th_lo

    def _safe(num: int, den: int) -> float | None:
        return round(num / den, 4) if den > 0 else None

    return {
        "high_threshold": round(th_hi, 4),
        "low_threshold": round(th_lo, 4),
        "high_recall":    _safe(int((a_hi & p_hi).sum()), int(a_hi.sum())),
        "high_precision": _safe(int((a_hi & p_hi).sum()), int(p_hi.sum())),
        "low_recall":     _safe(int((a_lo & p_lo).sum()), int(a_lo.sum())),
        "low_precision":  _safe(int((a_lo & p_lo).sum()), int(p_lo.sum())),
        "n_points_ext":   int(len(a)),
    }


def extreme_day_metrics(df: pd.DataFrame, day_q: float = 0.90) -> dict:
    """极端日识别准确率：定义"极端日"为日内最大值 ≥ 全体日内最大值序列的 day_q 分位数。"""
    work = df.copy()
    work["date"] = work.index.normalize()
    rows = []
    for d, g in work.groupby("date"):
        a = g["actual"].values
        p = g["pred"].values
        if len(a) == 0 or np.any(np.isnan(a)) or np.any(np.isnan(p)):
            continue
        rows.append((float(a.max()), float(p.max())))
    if len(rows) < 10:
        return {"extreme_day_recall": None, "extreme_day_precision": None,
                "extreme_day_threshold": None, "n_days_ext": len(rows)}
    a_max = np.array([r[0] for r in rows])
    p_max = np.array([r[1] for r in rows])
    th = float(np.quantile(a_max, day_q))
    a_ext = a_max >= th
    p_ext = p_max >= th
    tp = int((a_ext & p_ext).sum())
    return {
        "extreme_day_threshold": round(th, 4),
        "extreme_day_recall":    round(tp / max(int(a_ext.sum()), 1), 4),
        "extreme_day_precision": round(tp / max(int(p_ext.sum()), 1), 4),
        "n_days_ext": len(rows),
    }


def quantile_loss(
    actual: np.ndarray, q_preds: dict[float, np.ndarray]
) -> dict:
    """Pinball loss。q_preds: {0.1: arr, 0.5: arr, 0.9: arr, ...}"""
    out = {}
    a = actual.astype(float)
    for q, pred in q_preds.items():
        pred = pred.astype(float)
        err = a - pred
        loss = np.maximum(q * err, (q - 1) * err)
        out[f"qloss_q{int(q * 100)}"] = round(float(np.mean(loss)), 4)
    return out


def evaluate_frame(
    df: pd.DataFrame,
    high_q: float = 0.90,
    low_q: float = 0.10,
    day_q: float = 0.90,
) -> dict:
    """统一评估入口；df 需含 DatetimeIndex + actual + pred 列。"""
    a = df["actual"].astype(float).values
    p = df["pred"].astype(float).values
    return {
        "point_metrics":   point_metrics(a, p),
        "shape_metrics":   shape_metrics(df),
        "peak_valley":     peak_valley_metrics(df),
        "extreme":         extreme_metrics(df, high_q=high_q, low_q=low_q),
        "extreme_day":    extreme_day_metrics(df, day_q=day_q),
    }


def evaluate_predictions_csv(
    csv_path,
    ts_col: str = "ts",
    actual_col: str = "actual",
    pred_col: str = "predicted",
    high_q: float = 0.90,
    low_q: float = 0.10,
    day_q: float = 0.90,
) -> dict:
    """从算法保存的 predictions CSV 计算全套指标。

    现有算法 CSV 列名为 ts / actual / predicted；evaluate_frame 期望
    ts 作为 DatetimeIndex、列名 actual / pred，这里统一处理。
    """
    df = pd.read_csv(csv_path, parse_dates=[ts_col])
    df = df.set_index(ts_col).rename(columns={pred_col: "pred"})
    df = df[[actual_col, "pred"]].dropna()
    return evaluate_frame(df, high_q=high_q, low_q=low_q, day_q=day_q)


def try_extended_eval(df: pd.DataFrame, task: str = "da") -> dict | None:
    """若已安装 price_forecast_eval，返回扩展指标。"""
    try:
        from price_forecast_eval import evaluate_model_predictions, from_result_columns
    except ImportError:
        return None
    try:
        frame = df.reset_index()
        if "ts" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "ts"})
        ef = from_result_columns(
            frame, actual_col="actual", pred_col="pred", ts_index=False,
        )
        return evaluate_model_predictions(ef, task_type=task, include_extended=True)
    except Exception:
        return None
