"""LightGBM-TwoStage — 模型训练与预测。

移植自 jiangsu_prj/scripts/train_dayahead.py，包含：
  - 标准 LightGBM 回归
  - Two-Stage：地板价分类器 + 正常价回归器
  - 多分位数回归 + 智能组合
  - 时间衰减样本加权
  - 特征选择（feature importance top-K）
  - 后处理：model-naive 自适应混合、残差校正
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from sklearn.metrics import root_mean_squared_error as _rmse_func
    _HAS_RMSE = True
except ImportError:
    _HAS_RMSE = False

logger = logging.getLogger(__name__)

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

DEFAULT_PARAMS = {
    "n_estimators": 1500,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 50,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


# ── Metrics ──────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
        denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
        return float(np.nanmean(np.abs((y_true - y_pred) / denom)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if _HAS_RMSE:
        return float(_rmse_func(y_true, y_pred))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
    }


# ── Core training ────────────────────────────────────────────

def fit_lgbm(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    params: Optional[Dict] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> LGBMRegressor:
    p = {**DEFAULT_PARAMS, **(params or {})}
    model = LGBMRegressor(**p)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[early_stopping(stopping_rounds=80), log_evaluation(period=100)],
    )
    return model


def fit_quantile_lgbm(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    alpha: float,
    params: Optional[Dict] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> LGBMRegressor:
    p = {**DEFAULT_PARAMS, **(params or {})}
    p.pop("objective", None)
    p["objective"] = "quantile"
    p["alpha"] = alpha
    model = LGBMRegressor(**p)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric="quantile",
        callbacks=[early_stopping(stopping_rounds=80), log_evaluation(period=200)],
    )
    return model


def fit_floor_classifier(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    floor_price: float,
    params: Optional[Dict] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Optional[LGBMClassifier]:
    y_cls_train = (y_train <= floor_price).astype(int)
    y_cls_val = (y_val <= floor_price).astype(int)

    if y_cls_train.nunique() < 2 if hasattr(y_cls_train, 'nunique') else len(set(y_cls_train)) < 2:
        logger.warning("训练集中地板价样本不足（仅有单类），跳过分类器")
        return None

    pos_rate = y_cls_train.mean()
    scale = max(1.0, (1 - pos_rate) / max(pos_rate, 0.01))

    cls_params = {
        "n_estimators": 600,
        "learning_rate": (params or DEFAULT_PARAMS).get("learning_rate", 0.03),
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "scale_pos_weight": scale,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = LGBMClassifier(**cls_params)
    model.fit(
        X_train, y_cls_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_cls_val)],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=200)],
    )
    return model


# ── Quantile combination ─────────────────────────────────────

def quantile_combine(
    q_preds: Dict[float, np.ndarray],
    naive_pred: Optional[np.ndarray],
    floor_q10_threshold: float = 80.0,
    floor_blend_weight: float = 0.6,
    uncertainty_threshold: float = 200.0,
    naive_weight: float = 0.3,
) -> np.ndarray:
    """多分位数合成单点预测：q50 为基础，低 q10 区间向地板价混合，
    高不确定性区间向 naive 混合。"""
    q10 = q_preds[0.10]
    q25 = q_preds[0.25]
    q50 = q_preds[0.50]
    q90 = q_preds[0.90]

    pred = q50.copy()
    floor_mask = q10 <= floor_q10_threshold
    pred[floor_mask] = (
        floor_blend_weight * q10[floor_mask]
        + (1 - floor_blend_weight) * q25[floor_mask]
    )

    if naive_pred is not None:
        spread = q90 - q10
        uncertain_mask = spread > uncertainty_threshold
        valid = uncertain_mask & ~floor_mask & ~np.isnan(naive_pred)
        pred[valid] = (1 - naive_weight) * q50[valid] + naive_weight * naive_pred[valid]

    return pred


def tune_quantile_params(
    q_preds: Dict[float, np.ndarray],
    naive_pred: Optional[np.ndarray],
    y_true: np.ndarray,
) -> Dict:
    best_mae = float("inf")
    best_params: Dict = {
        "floor_q10_threshold": 80.0,
        "floor_blend_weight": 0.6,
        "uncertainty_threshold": 200.0,
        "naive_weight": 0.0,
    }

    for fq10 in [50.0, 80.0, 100.0, 130.0, 160.0]:
        for fbw in [0.4, 0.6, 0.8, 1.0]:
            for unc_thr in [150, 200, 250, 300, 400]:
                for nw in [0.0, 0.2, 0.4]:
                    pred = quantile_combine(
                        q_preds, naive_pred,
                        floor_q10_threshold=fq10,
                        floor_blend_weight=fbw,
                        uncertainty_threshold=float(unc_thr),
                        naive_weight=nw,
                    )
                    mae = float(mean_absolute_error(y_true, pred))
                    if mae < best_mae:
                        best_mae = mae
                        best_params = {
                            "floor_q10_threshold": fq10,
                            "floor_blend_weight": fbw,
                            "uncertainty_threshold": float(unc_thr),
                            "naive_weight": nw,
                        }
    return best_params


# ── Two-stage predict ────────────────────────────────────────

def find_best_threshold(
    floor_prob: np.ndarray,
    reg_pred: np.ndarray,
    y_true: np.ndarray,
    floor_price: float,
    floor_pred_value: float,
) -> Tuple[float, float]:
    """在验证集上搜索最优概率阈值，要求 precision >= 0.60 且 MAE 改善。"""
    reg_mae = float(mean_absolute_error(y_true, reg_pred))
    best_t, best_mae = 1.1, reg_mae

    for t in np.arange(0.40, 0.85, 0.05):
        pred_floor = floor_prob >= t
        if pred_floor.sum() == 0:
            continue
        actual_floor = y_true <= floor_price
        tp = (pred_floor & actual_floor).sum()
        fp = (pred_floor & ~actual_floor).sum()
        precision = tp / max(tp + fp, 1)
        if precision < 0.60:
            continue
        combined = np.where(pred_floor, floor_pred_value, reg_pred)
        mae = float(mean_absolute_error(y_true, combined))
        if mae < best_mae:
            best_mae = mae
            best_t = float(t)

    return best_t, best_mae


def two_stage_predict(
    clf: LGBMClassifier, reg: LGBMRegressor,
    X: pd.DataFrame, threshold: float,
    floor_pred_value: float,
) -> np.ndarray:
    floor_prob = clf.predict_proba(X)[:, 1]
    reg_pred = reg.predict(X)
    return np.where(floor_prob >= threshold, floor_pred_value, reg_pred)


# ── Time decay ───────────────────────────────────────────────

def compute_time_decay_weights(
    trade_dates: pd.Series, half_life_days: int = 30,
) -> np.ndarray:
    date_nums = pd.to_datetime(trade_dates).astype(np.int64) // 10**9
    max_t = date_nums.max()
    diff_days = (date_nums - max_t) / 86400.0
    return np.power(2.0, diff_days / half_life_days)


# ── Adaptive model-naive blending ────────────────────────────

def tune_adaptive_naive_blend(
    pred: np.ndarray, naive: np.ndarray, y_true: np.ndarray,
) -> Tuple[float, float]:
    valid_mask = ~np.isnan(naive)
    if valid_mask.sum() == 0:
        return 0.0, float(mean_absolute_error(y_true, pred))

    best_alpha = 0.0
    best_mae = float(mean_absolute_error(y_true, pred))

    for alpha in np.arange(0.00, 0.55, 0.05):
        blended = pred.copy()
        blended[valid_mask] = (1 - alpha) * pred[valid_mask] + alpha * naive[valid_mask]
        mae = float(mean_absolute_error(y_true, blended))
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)

    return best_alpha, best_mae


def apply_adaptive_naive_blend(
    pred: np.ndarray, naive: np.ndarray, alpha: float,
) -> np.ndarray:
    if alpha <= 0.0:
        return pred
    result = pred.copy()
    valid = ~np.isnan(naive)
    result[valid] = (1 - alpha) * pred[valid] + alpha * naive[valid]
    return result


# ── Residual correction ──────────────────────────────────────

def compute_residual_bias(
    pred_val: np.ndarray, y_val: np.ndarray, hh_val: np.ndarray,
) -> Dict[int, float]:
    residuals = y_val - pred_val
    bias: Dict[int, float] = {}
    for hh in np.unique(hh_val):
        mask = hh_val == hh
        if mask.sum() > 0:
            bias[int(hh)] = float(np.mean(residuals[mask]))
    return bias


def tune_residual_gamma(
    pred_val: np.ndarray, y_val: np.ndarray, hh_val: np.ndarray,
) -> Tuple[float, Dict[int, float]]:
    bias_full = compute_residual_bias(pred_val, y_val, hh_val)
    best_gamma = 0.0
    best_mae = float(mean_absolute_error(y_val, pred_val))

    for gamma in np.arange(0.0, 1.05, 0.1):
        corrected = pred_val.copy()
        for hh, b in bias_full.items():
            mask = hh_val == hh
            corrected[mask] += gamma * b
        mae = float(mean_absolute_error(y_val, corrected))
        if mae < best_mae:
            best_mae = mae
            best_gamma = float(gamma)

    return best_gamma, bias_full


def apply_residual_correction(
    pred: np.ndarray, hh_arr: np.ndarray,
    bias: Dict[int, float], gamma: float,
) -> np.ndarray:
    if gamma <= 0.0:
        return pred
    result = pred.copy()
    for hh, b in bias.items():
        mask = hh_arr == hh
        result[mask] += gamma * b
    return result


# ── Dynamic floor value ──────────────────────────────────────

def compute_dynamic_floor_value(y_train: np.ndarray, floor_price: float) -> float:
    floor_vals = y_train[y_train <= floor_price]
    if len(floor_vals) > 0:
        return float(np.median(floor_vals))
    return floor_price * 0.6


# ── Feature selection ────────────────────────────────────────

def select_top_k_features(
    feat: pd.DataFrame, x_cols: List[str],
    min_train_days: int, val_days: int,
    top_k: int, params: Optional[Dict] = None,
) -> List[str]:
    """预训练一个轻量模型，按 importance 保留 top-K 特征。"""
    dates = sorted(feat["trade_date"].unique())
    train_dates = dates[:min_train_days]
    val_dates = dates[min_train_days:min_train_days + val_days]
    train_mask = feat["trade_date"].isin(train_dates)
    val_mask = feat["trade_date"].isin(val_dates)

    X_pre = feat.loc[train_mask, x_cols]
    y_pre = feat.loc[train_mask, "y"].values
    X_val = feat.loc[val_mask, x_cols]
    y_val = feat.loc[val_mask, "y"].values

    pre_params = {**DEFAULT_PARAMS, **(params or {}), "n_estimators": 300}
    model = fit_lgbm(X_pre, y_pre, X_val, y_val, pre_params)
    imp = model.feature_importances_
    sorted_idx = np.argsort(imp)[::-1]
    selected = [x_cols[i] for i in sorted_idx[:top_k] if imp[i] > 0]
    logger.info("特征选择: %d/%d → %d", len(x_cols), len(x_cols), len(selected))
    return selected
