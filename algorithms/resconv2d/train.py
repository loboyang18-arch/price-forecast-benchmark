"""ResConv2D 训练 / 评估循环。

接口：``run_experiment(df_15min, market_id, test_start, test_end, freq, ...) -> dict``
返回 metrics + predictions DataFrame，与 conv2d_multitask 接口对齐。

损失：L1(price) + λ · L1(delta)（λ 默认 0.2，v25 原版最优）。
优化器：AdamW(1e-3, wd=1e-4) + LinearLR warmup(10ep) + CosineAnnealingLR → 1e-6；
        grad_clip 1.0。
默认 epochs=100，dropout=0.44；末轮权重为准（v25_deep_sudun500 同口径），
可通过 --early-stop 切到 val-best 早停。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pfbench.feature_registry import FeatureSpec, resolve_columns

from .config import (
    CONTEXT_AFTER,
    CONTEXT_BEFORE,
    LOOKBACK_DAYS,
    SLOTS_AFTER,
    SLOTS_BEFORE,
    SLOTS_PER_HOUR,
)
from .data import (
    ResConv2dDataset,
    build_daily_arrays,
    build_delta_targets,
    compute_norm,
)
from .model import DualHeadResConv2D

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_WD = 1e-4
DEFAULT_DROPOUT = 0.44
DEFAULT_DELTA_LAMBDA = 0.2
DEFAULT_DEPTH = "aggressive"
WARMUP_EPOCHS = 10


def _seed(s: int = 42) -> None:
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _eval_mae(model, loader, y_mean: float, y_std: float) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for batch in loader:
            grid, tgt = batch[0], batch[1]
            price, _ = model(grid.to(DEVICE))
            ps.append(price.cpu().numpy())
            ts.append(tgt.numpy())
    if not ps:
        return float("nan")
    p = np.concatenate(ps) * y_std + y_mean
    t = np.concatenate(ts) * y_std + y_mean
    return float(np.mean(np.abs(p - t)))


def _predict(model, ds: ResConv2dDataset, y_mean: float, y_std: float
             ) -> Tuple[Dict, List]:
    if len(ds) == 0:
        return {}, []
    loader = DataLoader(ds, batch_size=min(512, len(ds)), shuffle=False)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            grid = batch[0]
            price, _ = model(grid.to(DEVICE))
            all_preds.append(price.cpu().numpy())
    preds_flat = np.concatenate(all_preds) * y_std + y_mean

    day_preds: Dict = {}
    n_out = ds.steps_per_day
    for i, (d, idx) in enumerate(ds.meta):
        if d not in day_preds:
            day_preds[d] = np.full(n_out, np.nan)
        day_preds[d][idx] = preds_flat[i]
    return day_preds, sorted(day_preds.keys())


def _train_loop(
    model, train_ds, val_ds, test_ds,
    y_mean: float, y_std: float,
    epochs: int, batch_size: int, lr: float, weight_decay: float,
    delta_lambda: float,
    early_stop: bool = False, patience: int = 15, restore_best: bool = True,
) -> Tuple[float, dict]:
    """训练循环。

    early_stop=False（默认，v25 原版）：跑满 epochs，末轮权重做测试。
    early_stop=True：每 epoch 评估 val_mae，连续 patience 个 epoch 无改善则提前停止。
    """
    tl = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    train_eval_l = DataLoader(train_ds, min(512, len(train_ds)), shuffle=False)
    val_l = (DataLoader(val_ds, min(512, max(len(val_ds), 1)), shuffle=False)
             if val_ds and len(val_ds) > 0 else None)
    test_l = (DataLoader(test_ds, min(512, max(len(test_ds), 1)), shuffle=False)
              if test_ds and len(test_ds) > 0 else None)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs - WARMUP_EPOCHS), eta_min=1e-6,
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS],
    )

    best_val_mae = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0
    stopped_at = epochs - 1
    history: List[dict] = []
    log_interval = 5 if epochs > 50 else 2

    for ep in range(epochs):
        model.train()
        ep_lp, ep_ld, nb = 0.0, 0.0, 0
        for grid, tgt, dtgt in tl:
            grid = grid.to(DEVICE)
            tgt = tgt.to(DEVICE)
            dtgt = dtgt.to(DEVICE)
            opt.zero_grad()
            price, delta = model(grid)
            lp = F.l1_loss(price, tgt)
            ld = F.l1_loss(delta, dtgt)
            loss = lp + delta_lambda * ld
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_lp += lp.item()
            ep_ld += ld.item()
            nb += 1
        sched.step()

        do_eval = early_stop or (ep % log_interval == 0) or (ep == epochs - 1)
        if do_eval and val_l is not None:
            val_mae = _eval_mae(model, val_l, y_mean, y_std)
            improved = val_mae < best_val_mae - 1e-6
            if improved:
                best_val_mae = val_mae
                best_epoch = ep
                no_improve = 0
                if early_stop and restore_best:
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
            else:
                no_improve += 1
        else:
            val_mae = float("nan")
            improved = False

        if ep % log_interval == 0 or ep == epochs - 1:
            train_mae = _eval_mae(model, train_eval_l, y_mean, y_std)
            test_mae = _eval_mae(model, test_l, y_mean, y_std) if test_l else float("nan")
            cur_lr = opt.param_groups[0]["lr"]
            logger.info(
                "  ep%3d  Lp=%.4f Ld=%.4f"
                " | train=%.1f val=%.1f test=%.1f best=%.1f@%d"
                " | lr=%.1e  no_impr=%d",
                ep, ep_lp / max(nb, 1), ep_ld / max(nb, 1),
                train_mae, val_mae, test_mae,
                best_val_mae, best_epoch,
                cur_lr, no_improve,
            )
            history.append({
                "epoch": ep,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "test_mae": test_mae,
                "lr": cur_lr,
                "improved": improved,
                "no_improve": no_improve,
            })

        if early_stop and no_improve >= patience:
            logger.info(
                "  early stop @ epoch %d (best=%.2f @ epoch %d, patience=%d)",
                ep, best_val_mae, best_epoch, patience,
            )
            stopped_at = ep
            break

    if early_stop and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        logger.info("  restored model weights from best epoch %d (val_mae=%.2f)",
                    best_epoch, best_val_mae)

    return best_val_mae, {
        "history": history,
        "stopped_at_epoch": stopped_at,
        "best_epoch": best_epoch,
        "restored_best": bool(early_stop and restore_best and best_state is not None),
    }


def run_experiment(
    df_15min: pd.DataFrame,
    market_id: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    freq: str = "1h",
    spec: FeatureSpec = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WD,
    dropout: float = DEFAULT_DROPOUT,
    delta_lambda: float = DEFAULT_DELTA_LAMBDA,
    depth_profile: str = DEFAULT_DEPTH,
    val_days: int = 7,
    seed: int = 42,
    early_stop: bool = False,
    patience: int = 15,
    restore_best: bool = True,
) -> dict:
    """跑单市场单粒度的训练 + 测试。

    Args:
        df_15min: 输入 15min 长表（DataFrame，DatetimeIndex）
        market_id: 市场 ID（用于查 feature_registry）
        spec: FeatureSpec（若 None，使用 yaml 默认 target + 默认启用类别）
    """
    _seed(seed)
    if spec is None:
        spec = FeatureSpec()
    resolved = resolve_columns(market_id, spec, freq=freq)
    logger.info(
        "=" * 60 + "\nResConv2D  market=%s  target=%s  freq=%s  test=%s ~ %s\n"
        "  depth=%s  dropout=%.2f  λΔ=%.2f  epochs=%d  early_stop=%s\n"
        "  feature groups: %s",
        market_id, resolved.target, freq, test_start.date(), test_end.date(),
        depth_profile, dropout, delta_lambda, epochs, early_stop,
        {n: len(g.cols) for n, g in resolved.groups.items()},
    )

    (valid_dates, day_boundary, day_history, day_actual, day_targets,
     c_total, steps_per_day, stream_cols) = build_daily_arrays(df_15min, resolved, freq=freq)
    if not valid_dates:
        raise RuntimeError(f"{market_id}: 无有效日期")

    day_delta = build_delta_targets(day_targets)

    test_d0 = test_start.date()
    test_d1 = test_end.date()
    train_days = [d for d in valid_dates if d < test_d0]
    test_days = [d for d in valid_dates if test_d0 <= d <= test_d1]

    if len(train_days) <= val_days:
        raise RuntimeError(f"{market_id}: 训练集过小 ({len(train_days)} 天)")
    val_split_days = train_days[-val_days:]
    train_only = train_days[:-val_days]

    if freq == "15min":
        h_slots = SLOTS_BEFORE + 1 + SLOTS_AFTER
    else:
        h_slots = (CONTEXT_BEFORE + 1 + CONTEXT_AFTER) * SLOTS_PER_HOUR

    norm_mean, norm_std = compute_norm(day_boundary, day_history, day_actual, train_only)
    tgt_stack = np.stack([day_targets[d] for d in train_only if d in day_targets])
    y_mean = float(tgt_stack.mean())
    y_std = float(tgt_stack.std()) + 1e-8

    delta_stack = np.concatenate(
        [day_delta[d] for d in train_only if d in day_delta],
    )
    delta_y_mean = float(delta_stack.mean())
    delta_y_std = max(float(delta_stack.std()), 1e-6)

    ds_kwargs = dict(
        day_boundary=day_boundary, day_history=day_history, day_actual=day_actual,
        day_targets=day_targets, day_delta_targets=day_delta,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
        delta_y_mean=delta_y_mean, delta_y_std=delta_y_std,
        c_total=c_total, steps_per_day=steps_per_day, freq=freq,
    )
    train_ds = ResConv2dDataset(sample_dates=train_only, **ds_kwargs)
    val_ds = ResConv2dDataset(sample_dates=val_split_days, **ds_kwargs)
    test_ds = ResConv2dDataset(sample_dates=test_days, **ds_kwargs)

    logger.info(
        "  样本数 train=%d  val=%d  test=%d  c_total=%d  h_slots=%d"
        "  y_mean=%.1f  y_std=%.1f  δ_mean=%.2f  δ_std=%.2f",
        len(train_ds), len(val_ds), len(test_ds), c_total, h_slots,
        y_mean, y_std, delta_y_mean, delta_y_std,
    )
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("训练或测试样本为空")

    _seed(seed)
    model = DualHeadResConv2D(
        c_in=c_total, h_slots=h_slots, lookback=LOOKBACK_DAYS,
        dropout=dropout, depth_profile=depth_profile,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "  device=%s  model=DualHeadResConv2D[%s]  params=%d"
        "  (res64=%d, res128=%d, head=%d)",
        DEVICE, model.depth_profile, n_params,
        model.n_res64, model.n_res128, model.head_hidden,
    )

    best_val_mae, train_log = _train_loop(
        model, train_ds, val_ds, test_ds, y_mean, y_std,
        epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        delta_lambda=delta_lambda,
        early_stop=early_stop, patience=patience, restore_best=restore_best,
    )

    day_preds, pred_dates = _predict(model, test_ds, y_mean, y_std)
    rows = []
    for d in pred_dates:
        if d not in day_targets:
            continue
        actual = day_targets[d]
        pred = day_preds[d]
        if freq == "15min":
            for s in range(steps_per_day):
                if not np.isnan(pred[s]):
                    rows.append({
                        "ts": pd.Timestamp(d) + pd.Timedelta(minutes=15 * s),
                        "actual": float(actual[s]), "predicted": float(pred[s]),
                    })
        else:
            for h in range(steps_per_day):
                if not np.isnan(pred[h]):
                    rows.append({
                        "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                        "actual": float(actual[h]), "predicted": float(pred[h]),
                    })
    pred_df = pd.DataFrame(rows)
    if "ts" in pred_df.columns:
        pred_df = pred_df.sort_values("ts").reset_index(drop=True)

    if len(pred_df) == 0:
        mae = rmse = float("nan")
        profile_corr = float("nan")
    else:
        a = pred_df["actual"].values
        p = pred_df["predicted"].values
        m = ~(np.isnan(a) | np.isnan(p))
        a, p = a[m], p[m]
        mae = float(np.mean(np.abs(a - p)))
        rmse = float(np.sqrt(np.mean((a - p) ** 2)))
        if a.std() > 1e-6 and p.std() > 1e-6:
            profile_corr = float(np.corrcoef(a, p)[0, 1])
        else:
            profile_corr = float("nan")

    metrics = {
        "market": market_id,
        "algorithm": "resconv2d",
        "freq": freq,
        "target": resolved.target,
        "test_start": str(test_d0),
        "test_end": str(test_d1),
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "n_test_samples": len(test_ds),
        "n_train_days": len(train_only),
        "n_val_days": len(val_split_days),
        "n_test_days": len(test_days),
        "c_total": c_total,
        "h_slots": h_slots,
        "lookback_days": LOOKBACK_DAYS,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "delta_lambda": delta_lambda,
        "depth_profile": model.depth_profile,
        "n_res64": model.n_res64,
        "n_res128": model.n_res128,
        "head_hidden": model.head_hidden,
        "model_params": n_params,
        "y_mean": y_mean,
        "y_std": y_std,
        "delta_y_mean": delta_y_mean,
        "delta_y_std": delta_y_std,
        "best_val_mae": round(best_val_mae, 3) if best_val_mae != float("inf") else None,
        "test_mae": round(mae, 3),
        "test_rmse": round(rmse, 3),
        "test_profile_corr": round(profile_corr, 4) if not np.isnan(profile_corr) else None,
        "feature_spec": resolved.to_dict(),
        "stream_cols": stream_cols,
        "early_stop": early_stop,
        "patience": patience if early_stop else None,
        "restore_best": restore_best if early_stop else None,
        "stopped_at_epoch": train_log.get("stopped_at_epoch"),
        "best_epoch": train_log.get("best_epoch"),
        "restored_best": train_log.get("restored_best", False),
    }
    logger.info(
        "ResConv2D | %s [%s]  test_MAE=%.2f  RMSE=%.2f  profile_corr=%.3f  params=%.1fM",
        market_id, freq, mae, rmse,
        profile_corr if not np.isnan(profile_corr) else -1.0,
        n_params / 1e6,
    )
    return {
        "metrics": metrics,
        "predictions": pred_df,
        "model": model,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "delta_y_mean": delta_y_mean,
        "delta_y_std": delta_y_std,
        "train_log": train_log,
    }
