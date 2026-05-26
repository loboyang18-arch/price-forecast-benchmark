"""Conv2D-MultiTask 训练 / 评估循环。

接口：``run_experiment(df_15min, cfg, test_start, test_end, freq, ...) -> dict``
返回包含 metrics + predictions DataFrame，与 lightgbm_baseline / lightgbm_twostage 接口对齐。
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

from pfbench.feature_registry import FeatureSpec, ResolvedSpec, resolve_columns

from .config import (
    CONTEXT_AFTER,
    CONTEXT_BEFORE,
    LOOKBACK_DAYS,
    SLOTS_AFTER,
    SLOTS_BEFORE,
    SLOTS_PER_HOUR,
)
from .data import Conv2dDataset, build_daily_arrays, compute_norm
from .model import Conv2dMultiTaskNet

logger = logging.getLogger(__name__)

# ── 设备 ────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 训练超参 ──────────────────────────────────────
DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
WARMUP_EPOCHS = 10
LAMBDA_DIR = 0.3


def _seed(s: int = 42) -> None:
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _eval_mae(model, loader, y_mean: float, y_std: float) -> float:
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


def _eval_dir_acc(model, loader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            grid, _, dir_label = batch[0], batch[1], batch[2]
            _, dir_logits = model(grid.to(DEVICE))
            pred_cls = dir_logits.argmax(dim=1)
            correct += (pred_cls.cpu() == dir_label).sum().item()
            total += dir_label.numel()
    return correct / max(total, 1)


def _predict(model, ds: Conv2dDataset, y_mean: float, y_std: float
             ) -> Tuple[Dict, List]:
    """对 dataset 整体做一次前向，按 (date, idx) 还原 day_preds。"""
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
    epochs: int, batch_size: int, lr: float,
    early_stop: bool = False, patience: int = 10, restore_best: bool = True,
) -> Tuple[float, dict]:
    """训练循环。

    early_stop=False（默认）：跑满 epochs，用最后一轮权重做测试（原行为）。
    early_stop=True：每个 epoch 都评估 val_mae；val 改善则保存 state_dict；
        连续 patience 个 epoch 无改善则提前停止；若 restore_best=True，
        训练结束后把模型权重恢复到 best-val checkpoint。
    """
    tl = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    train_eval_l = DataLoader(train_ds, min(512, len(train_ds)), shuffle=False)
    val_l = DataLoader(val_ds, min(512, max(len(val_ds), 1)), shuffle=False) if val_ds else None
    test_l = DataLoader(test_ds, min(512, max(len(test_ds), 1)), shuffle=False) if test_ds else None

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs - WARMUP_EPOCHS), eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched],
        milestones=[WARMUP_EPOCHS])

    best_val_mae = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0
    stopped_at = epochs - 1
    history: List[dict] = []
    log_interval = 5 if epochs > 50 else 2

    for ep in range(epochs):
        model.train()
        ep_l1, ep_ce, ep_dir_ok, ep_dir_n, nb = 0.0, 0.0, 0, 0, 0
        for grid, tgt, dir_label in tl:
            grid = grid.to(DEVICE)
            tgt = tgt.to(DEVICE)
            dir_label = dir_label.to(DEVICE)
            opt.zero_grad()
            price, dir_logits = model(grid)
            l1 = F.l1_loss(price, tgt)
            ce = F.cross_entropy(dir_logits, dir_label)
            loss = l1 + LAMBDA_DIR * ce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_l1 += l1.item()
            ep_ce += ce.item()
            ep_dir_ok += (dir_logits.argmax(1) == dir_label).sum().item()
            ep_dir_n += dir_label.numel()
            nb += 1
        sched.step()

        # 早停模式下每 epoch 都计算 val_mae 用于改善判断；
        # 否则按 log_interval 计算（原行为）。
        do_eval = early_stop or (ep % log_interval == 0) or (ep == epochs - 1)
        if do_eval and val_l is not None:
            val_mae = _eval_mae(model, val_l, y_mean, y_std)
            improved = val_mae < best_val_mae
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
            val_dir = _eval_dir_acc(model, val_l) if val_l else float("nan")
            cur_lr = opt.param_groups[0]["lr"]
            logger.info(
                "  ep%3d  L1=%.4f CE=%.3f dir_acc=%.3f"
                " | train=%.1f val=%.1f test=%.1f best=%.1f@%d"
                " | v_dir=%.3f  lr=%.1e  no_impr=%d",
                ep, ep_l1 / max(nb, 1), ep_ce / max(nb, 1),
                ep_dir_ok / max(ep_dir_n, 1),
                train_mae, val_mae, test_mae,
                best_val_mae, best_epoch,
                val_dir, cur_lr, no_improve,
            )
            history.append({
                "epoch": ep,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "test_mae": test_mae,
                "val_dir_acc": val_dir,
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
    val_days: int = 7,
    seed: int = 42,
    early_stop: bool = False,
    patience: int = 10,
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
        "=" * 60 + "\nConv2D-MultiTask  market=%s  target=%s  freq=%s  test=%s ~ %s\n"
        "  feature groups: %s",
        market_id, resolved.target, freq, test_start.date(), test_end.date(),
        {n: len(g.cols) for n, g in resolved.groups.items()},
    )

    (valid_dates, day_boundary, day_history, day_actual, day_targets,
     c_total, steps_per_day, stream_cols) = build_daily_arrays(df_15min, resolved, freq=freq)
    if not valid_dates:
        raise RuntimeError(f"{market_id}: 无有效日期")

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

    ds_kwargs = dict(
        day_boundary=day_boundary, day_history=day_history, day_actual=day_actual,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std, y_mean=y_mean, y_std=y_std,
        c_total=c_total, steps_per_day=steps_per_day, freq=freq,
    )
    train_ds = Conv2dDataset(sample_dates=train_only, **ds_kwargs)
    val_ds = Conv2dDataset(sample_dates=val_split_days, **ds_kwargs)
    test_ds = Conv2dDataset(sample_dates=test_days, **ds_kwargs)

    logger.info(
        "  样本数 train=%d  val=%d  test=%d  c_total=%d  h_slots=%d  y_mean=%.1f  y_std=%.1f",
        len(train_ds), len(val_ds), len(test_ds), c_total, h_slots, y_mean, y_std,
    )
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("训练或测试样本为空")

    _seed(seed)
    model = Conv2dMultiTaskNet(c_in=c_total, h_slots=h_slots).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  device=%s  params=%d", DEVICE, n_params)

    best_val_mae, train_log = _train_loop(
        model, train_ds, val_ds, test_ds, y_mean, y_std,
        epochs=epochs, batch_size=batch_size, lr=lr,
        early_stop=early_stop, patience=patience, restore_best=restore_best,
    )

    # ── 测试集预测 ────────────────────────────────────
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

    test_dir_acc = _eval_dir_acc(
        model, DataLoader(test_ds, min(512, max(len(test_ds), 1)), shuffle=False)
    )

    metrics = {
        "market": market_id,
        "algorithm": "conv2d_multitask",
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
        "lambda_dir": LAMBDA_DIR,
        "model_params": n_params,
        "y_mean": y_mean,
        "y_std": y_std,
        "best_val_mae": round(best_val_mae, 3),
        "test_mae": round(mae, 3),
        "test_rmse": round(rmse, 3),
        "test_profile_corr": round(profile_corr, 4) if not np.isnan(profile_corr) else None,
        "test_dir_acc": round(test_dir_acc, 4),
        "feature_spec": resolved.to_dict(),         # 用于追溯
        "stream_cols": stream_cols,
        "early_stop": early_stop,
        "patience": patience if early_stop else None,
        "restore_best": restore_best if early_stop else None,
        "stopped_at_epoch": train_log.get("stopped_at_epoch"),
        "best_epoch": train_log.get("best_epoch"),
        "restored_best": train_log.get("restored_best", False),
    }
    logger.info(
        "Conv2D-MultiTask | %s [%s]  test_MAE=%.2f  RMSE=%.2f  profile_corr=%.3f  dir_acc=%.3f",
        market_id, freq, mae, rmse, profile_corr if not np.isnan(profile_corr) else -1.0,
        test_dir_acc,
    )
    return {
        "metrics": metrics,
        "predictions": pred_df,
        "model": model,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "train_log": train_log,
    }
