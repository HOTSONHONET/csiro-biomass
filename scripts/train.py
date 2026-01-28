# scripts/train.py
# ---------------
# CSIRO Biomass Regression Training (Albumentations + tqdm + MLflow)
#
# Features:
# - Auto-create folds if --splits-dir missing fold CSVs (uses make_folds from data_split.py)
# - Supports:
#     (A) Dinov3MultiReg (classic 5-target regression)
#     (B) DinoV3Structured (predicts pred5 + ratio heads; uses structured_loss)
#     (C) DinoV3Hybrid (left/right token fusion + separate heads + ratio heads; uses hybrid_loss)
# - Logs to MLflow:
#     * fold sizes, args, huber deltas, lambda weights
#     * train/val fold CSVs as artifacts
#     * per-epoch train/val loss + score
#     * best checkpoint as artifact
#
# Assumptions:
# - split CSVs are WIDE with columns:
#   image_path + targets: Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFile
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from warnings import filterwarnings

from utils import Config, set_seed

# Models
from models.dinov3_multi_reg import Dinov3MultiReg, Dinov3Config
from models.dinov3_multi_reg_structured import DinoV3Structured, DinoV3StructuredConfig
from models.dinov3_mamba_2_tiles import DinoV3HybridConfig, DinoV3Hybrid

# Folds
from data_split import make_folds

filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------
# Dataset
# -----------------------------
class CSIRODataset(Dataset):
    """
    Returns:
      img_t: (3, H, W) where H=2S, W=4S
      y:     (5,) in order:
             [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
    """
    def __init__(self, df, img_dir, img_size, is_train=True, shadow_p=0.5):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.S = int(img_size)
        self.is_train = is_train

        self.target5_cols = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

        self.H = self.S * 2
        self.W = self.S * 4

        if is_train:
            self.aug = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomShadow(
                        shadow_roi=(0, 0, 1, 1),
                        num_shadows_lower=1,
                        num_shadows_upper=2,
                        shadow_dimension=5,
                        p=shadow_p,
                    ),
                    A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 2.0), p=0.2),
                    # IMPORTANT: keep Resize AFTER geometric transforms
                    A.Resize(self.H, self.W),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self.aug = A.Compose(
                [
                    A.Resize(self.H, self.W),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        y = torch.tensor(
            row[self.target5_cols].to_numpy(dtype="float32", na_value=0.0),
            dtype=torch.float32,
        )  # (5,)

        rel_path = str(row["image_path"])
        img_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.img_dir, rel_path)

        with open(img_path, "rb") as f:
            img = np.array(Image.open(f).convert("RGB"))

        img_t = self.aug(image=img)["image"]  # (3,H,W)

        # safety: ensure consistent shape for stacking
        if img_t.shape[-2:] != (self.H, self.W):
            raise RuntimeError(f"Bad shape: {img_t.shape}, expected (3,{self.H},{self.W})")

        return img_t, y


# -----------------------------
# Loss + Metric
# -----------------------------
def weighted_rmse_loss(preds_5: torch.Tensor, targets_5: torch.Tensor) -> torch.Tensor:
    """
    preds_5, targets_5: (B,5) in order
      [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
    """
    weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], device=preds_5.device, dtype=torch.float32)
    se = (preds_5 - targets_5) ** 2
    return (weights * se).sum(dim=1).mean()


def competition_metric(preds_5: torch.Tensor, targets_5: torch.Tensor) -> float:
    """Weighted R^2 over 5 targets."""
    w = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], device=preds_5.device, dtype=torch.float32).view(1, 5)
    w_sum = w.sum()
    N = targets_5.shape[0]

    y_true = targets_5
    y_pred = preds_5

    y_mean = (w * y_true).sum() / (w_sum * N)
    ss_res = (w * (y_true - y_pred) ** 2).sum()
    ss_tot = (w * (y_true - y_mean) ** 2).sum() + 1e-9

    return float((1.0 - ss_res / ss_tot).item())


def structured_loss(
    out: dict,
    gt5: torch.Tensor,
    huber_dead_delta: float,
    huber_clover_delta: float,
    lambda_ratio: float = 0.3,
    lambda_cons: float = 0.05,
) -> torch.Tensor:
    """
    out: dict from DinoV3Structured forward
      must contain:
        out["pred5"]             (B,5)
        out["dead_ratio_pred"]   (B,)  in [0,1] typically
        out["clover_ratio_pred"] (B,)  in [0,1] typically
        out["gdm_pred"]          (B,)  optional, used in constraint here
    """
    green_gt = gt5[:, 0]
    dead_gt  = gt5[:, 1]
    clov_gt  = gt5[:, 2]
    gdm_gt   = gt5[:, 3]
    total_gt = gt5[:, 4]

    eps = 1e-6
    dead_ratio_gt = dead_gt / (total_gt + eps)     # Dead/Total
    clov_ratio_gt = clov_gt / (gdm_gt + eps)       # Clover/GDM

    pred5 = out["pred5"]
    loss_main = weighted_rmse_loss(pred5, gt5)

    # Huber on ratios
    loss_dead_ratio = F.smooth_l1_loss(out["dead_ratio_pred"], dead_ratio_gt, beta=huber_dead_delta)
    loss_clov_ratio = F.smooth_l1_loss(out["clover_ratio_pred"], clov_ratio_gt, beta=huber_clover_delta)

    # Simple physical-ish constraint (optional): penalize negative predictions if any
    gdm_pred = out.get("gdm_pred", None)
    if gdm_pred is None:
        loss_cons = torch.zeros((), device=gt5.device)
    else:
        loss_cons = F.relu(-gdm_pred).mean()

    loss = loss_main + lambda_ratio * (loss_dead_ratio + loss_clov_ratio) + lambda_cons * loss_cons
    return loss


def hybrid_loss(
    out: dict,
    gt5: torch.Tensor,
    huber_dead_delta: float,
    huber_clover_delta: float,
    lambda_ratio: float = 0.3,
    lambda_cons: float = 0.05,
    lambda_align: float = 0.02,
) -> torch.Tensor:
    """
    For DinoV3Hybrid:
      - main weighted RMSE on out["pred5"]
      - ratio supervision (dead_ratio, clover_ratio)
      - tiny non-neg penalty
      - optional alignment if you expose direct heads later (safe if missing)
    """
    dead_gt  = gt5[:, 1]
    clov_gt  = gt5[:, 2]
    gdm_gt   = gt5[:, 3]
    total_gt = gt5[:, 4]

    eps = 1e-6
    dead_ratio_gt = dead_gt / (total_gt + eps)
    clov_ratio_gt = clov_gt / (gdm_gt + eps)

    pred5 = out["pred5"]
    loss_main = weighted_rmse_loss(pred5, gt5)

    loss_dead_ratio = F.smooth_l1_loss(out["dead_ratio_pred"], dead_ratio_gt, beta=huber_dead_delta)
    loss_clov_ratio = F.smooth_l1_loss(out["clover_ratio_pred"], clov_ratio_gt, beta=huber_clover_delta)

    # small non-neg penalty (usually ~0 if your model uses softplus/recompose)
    loss_cons = (
        F.relu(-out["green_pred"]).mean() +
        F.relu(-out["gdm_pred"]).mean() +
        F.relu(-out["total_pred"]).mean() +
        F.relu(-out["dead_pred"]).mean() +
        F.relu(-out["clover_pred"]).mean()
    )

    # optional alignment if keys exist
    loss_align = torch.zeros((), device=gt5.device)
    if "dead_direct" in out and "dead_pred" in out:
        loss_align = loss_align + F.smooth_l1_loss(out["dead_direct"], out["dead_pred"].detach(), beta=5.0)
    if "clover_direct" in out and "clover_pred" in out:
        loss_align = loss_align + F.smooth_l1_loss(out["clover_direct"], out["clover_pred"].detach(), beta=5.0)

    return loss_main + lambda_ratio * (loss_dead_ratio + loss_clov_ratio) + lambda_cons * loss_cons + lambda_align * loss_align


# -----------------------------
# Train / Val loops
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    model_name: str,
    huber_dead_delta: float,
    huber_clover_delta: float,
    lambda_ratio: float,
    lambda_cons: float,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_score = 0.0
    count = 0

    use_amp = scaler is not None and scaler.is_enabled()
    pbar = tqdm(loader, desc="Train", leave=False, dynamic_ncols=True)

    for img_full_t, y in pbar:
        img_full_t = img_full_t.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            if model_name == "Dinov3MultiReg":
                preds5 = model(img_full_t)  # (B,5)
                loss = weighted_rmse_loss(preds5, y)
                score = competition_metric(preds5.detach(), y)

            elif model_name == "DinoV3Structured":
                out = model(img_full_t)     # dict
                loss = structured_loss(
                    out, y,
                    huber_dead_delta=huber_dead_delta,
                    huber_clover_delta=huber_clover_delta,
                    lambda_ratio=lambda_ratio,
                    lambda_cons=lambda_cons,
                )
                preds5 = out["pred5"]
                score = competition_metric(preds5.detach(), y)

            elif model_name == "DinoV3Hybrid":
                # Split full image (B,3,H,4S) -> left/right (B,3,H,2S)
                _, _, _, W = img_full_t.shape
                mid = W // 2
                left = img_full_t[:, :, :, :mid]
                right = img_full_t[:, :, :, mid:]

                out = model(left, right)
                loss = hybrid_loss(
                    out, y,
                    huber_dead_delta=huber_dead_delta,
                    huber_clover_delta=huber_clover_delta,
                    lambda_ratio=lambda_ratio,
                    lambda_cons=lambda_cons,
                )
                preds5 = out["pred5"]
                score = competition_metric(preds5.detach(), y)

            else:
                raise ValueError(f"Unknown model_name: {model_name}")

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_score += score * bs
        count += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}", score=f"{score:.4f}")

    return total_loss / max(1, count), total_score / max(1, count)


@torch.no_grad()
def val_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    huber_dead_delta: float,
    huber_clover_delta: float,
    lambda_ratio: float,
    lambda_cons: float,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_score = 0.0
    count = 0

    pbar = tqdm(loader, desc="Val", leave=False, dynamic_ncols=True)
    for img_full_t, y in pbar:
        img_full_t = img_full_t.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if model_name == "Dinov3MultiReg":
            preds5 = model(img_full_t)
            loss = weighted_rmse_loss(preds5, y)
            score = competition_metric(preds5, y)

        elif model_name == "DinoV3Structured":
            out = model(img_full_t)
            loss = structured_loss(
                out, y,
                huber_dead_delta=huber_dead_delta,
                huber_clover_delta=huber_clover_delta,
                lambda_ratio=lambda_ratio,
                lambda_cons=lambda_cons,
            )
            preds5 = out["pred5"]
            score = competition_metric(preds5, y)

        elif model_name == "DinoV3Hybrid":
            _, _, _, W = img_full_t.shape
            mid = W // 2
            left = img_full_t[:, :, :, :mid]
            right = img_full_t[:, :, :, mid:]

            out = model(left, right)
            loss = hybrid_loss(
                out, y,
                huber_dead_delta=huber_dead_delta,
                huber_clover_delta=huber_clover_delta,
                lambda_ratio=lambda_ratio,
                lambda_cons=lambda_cons,
            )
            preds5 = out["pred5"]
            score = competition_metric(preds5, y)

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_score += score * bs
        count += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}", score=f"{score:.4f}")

    return total_loss / max(1, count), total_score / max(1, count)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


# -----------------------------
# Splits utilities
# -----------------------------
def fold_files_exist(splits_dir: Path, n_splits: int) -> bool:
    for f in range(n_splits):
        if not (splits_dir / f"train_fold{f}.csv").exists():
            return False
        if not (splits_dir / f"val_fold{f}.csv").exists():
            return False
    return True


def ensure_splits_exist(
    splits_dir: Path,
    train_csv_path: Path,
    n_splits: int,
    seed: int,
):
    splits_dir.mkdir(parents=True, exist_ok=True)

    if fold_files_exist(splits_dir, n_splits):
        print(f"[Splits] Found existing fold CSVs in: {splits_dir}")
        return

    print(f"[Splits] Fold CSVs not found in: {splits_dir}")
    print(f"[Splits] Creating folds from: {train_csv_path}")
    make_folds(
        train_csv=str(train_csv_path),
        out_dir=str(splits_dir),
        n_splits=n_splits,
        seed=seed,
    )
    print(f"[Splits] Done. Wrote folds to: {splits_dir}")


def load_fold(splits_dir: Path, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = splits_dir / f"train_fold{fold}.csv"
    val_path = splits_dir / f"val_fold{fold}.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing split CSVs for fold {fold} in {splits_dir}")
    return pd.read_csv(train_path), pd.read_csv(val_path)


# -----------------------------
# Huber delta helpers
# -----------------------------
def _robust_std(x: np.ndarray) -> float:
    """
    Robust scale estimate: 1.4826 * median(|x - median(x)|)
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def auto_huber_deltas_from_train_df(train_df: pd.DataFrame) -> tuple[float, float]:
    """
    We need a reasonable delta for ratio Huber before training.
    Use robust std of the ratio * 1.5 (tunable).
    Clamp to a sensible range.
    """
    eps = 1e-6
    dead = train_df["Dry_Dead_g"].to_numpy(dtype=np.float32)
    total = train_df["Dry_Total_g"].to_numpy(dtype=np.float32)
    gdm = train_df["GDM_g"].to_numpy(dtype=np.float32)
    clov = train_df["Dry_Clover_g"].to_numpy(dtype=np.float32)

    dead_ratio = dead / (total + eps)
    clov_ratio = clov / (gdm + eps)

    dead_scale = _robust_std(dead_ratio)
    clov_scale = _robust_std(clov_ratio)

    dead_delta = 1.5 * dead_scale
    clov_delta = 1.5 * clov_scale

    # clamp: ratios typically [0,1], deltas too tiny can make loss too L1-ish/noisy
    dead_delta = float(np.clip(dead_delta, 0.02, 0.25))
    clov_delta = float(np.clip(clov_delta, 0.02, 0.25))
    return dead_delta, clov_delta


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="CSIRO Biomass Regression Training (auto folds + structured + hybrid)")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--img-dir", default=".", help="Base directory for image_path (relative paths).")
    parser.add_argument("--splits-dir", default="exps/splits", help="Directory to read/write fold CSVs.")
    parser.add_argument("--train-csv", default="train.csv", help="Competition train.csv (long format).")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds to create/use.")
    parser.add_argument("--no-folds", action="store_true", help="Run only fold 0.")
    parser.add_argument("--epochs", type=int, default=getattr(Config, "EPOCHS", 30))
    parser.add_argument("--early-stopping-patience", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=getattr(Config, "BATCH", 8))
    parser.add_argument("--lr", type=float, default=getattr(Config, "LR", 3e-4))
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "onecycle"], default="none")
    parser.add_argument("--warmup-epochs", type=int, default=0)

    parser.add_argument("--img-size", type=int, default=getattr(Config, "IMG_SIZE", 512))
    parser.add_argument("--num-workers", type=int, default=getattr(Config, "NUM_WORKERS", 4))
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--shadow-p", type=float, default=0.5, help="RandomShadow probability (train only).")

    parser.add_argument("--model-id", default="facebook/dinov3-vith16plus-pretrain-lvd1689m")

    parser.add_argument(
        "--model-name",
        type=str,
        choices=["Dinov3MultiReg", "DinoV3Structured", "DinoV3Hybrid"],
        default="Dinov3MultiReg",
    )

    # Structured/Hybrid ratio loss knobs
    parser.add_argument("--huber-dead-delta", type=float, default=-1.0, help="Huber delta for dead_ratio (<=0 => auto).")
    parser.add_argument("--huber-clover-delta", type=float, default=-1.0, help="Huber delta for clover_ratio (<=0 => auto).")
    parser.add_argument("--lambda-ratio", type=float, default=0.3)
    parser.add_argument("--lambda-cons", type=float, default=0.05)

    # MLflow
    parser.add_argument("--experiment-name", default=getattr(Config, "EXPERIMENT_NAME", "csiro_biomass"))
    parser.add_argument("--mlflow-uri", default=None)

    # Outputs
    parser.add_argument("--output-dir", default=getattr(Config, "OUTPUT_DIR", "outputs"))
    parser.add_argument("--seed", type=int, default=getattr(Config, "SEED", 42))
    parser.add_argument("--select-fold", type=int, default=None)

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    img_dir = (repo_root / args.img_dir).resolve()
    splits_dir = (repo_root / args.splits_dir).resolve()
    train_csv_path = (repo_root / args.train_csv).resolve()

    # Ensure splits exist (create if missing)
    ensure_splits_exist(
        splits_dir=splits_dir,
        train_csv_path=train_csv_path,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    # Output run folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out_dir = (repo_root / args.output_dir / timestamp).resolve()
    run_out_dir.mkdir(parents=True, exist_ok=True)

    with open(run_out_dir / "cli_args.json", "w", encoding="utf-8") as f:
        json.dump({"argv": sys.argv, "parsed": vars(args)}, f, indent=2, sort_keys=True)

    # Device + seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # MLflow
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(f"{args.experiment_name}_{timestamp}")

    folds_to_run = [0] if args.no_folds else list(range(args.n_splits))
    if args.select_fold is not None:
        folds_to_run = [args.select_fold]

    for fold in folds_to_run:
        train_df, val_df = load_fold(splits_dir, fold)

        print(f"\n========== FOLD {fold}/{len(folds_to_run)-1} ==========")
        print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")

        # Decide huber deltas (ratio supervision for Structured/Hybrid)
        if args.huber_dead_delta <= 0 or args.huber_clover_delta <= 0:
            auto_dead, auto_clov = auto_huber_deltas_from_train_df(train_df)
        else:
            auto_dead, auto_clov = None, None

        huber_dead_delta = float(auto_dead if args.huber_dead_delta <= 0 else args.huber_dead_delta)
        huber_clover_delta = float(auto_clov if args.huber_clover_delta <= 0 else args.huber_clover_delta)

        train_ds = CSIRODataset(
            train_df,
            img_dir=str(img_dir),
            img_size=args.img_size,
            is_train=True,
            shadow_p=args.shadow_p,
        )
        val_ds = CSIRODataset(
            val_df,
            img_dir=str(img_dir),
            img_size=args.img_size,
            is_train=False,
            shadow_p=0.0,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=False,
        )

        # Build model
        if args.model_name == "Dinov3MultiReg":
            cfg = Dinov3Config(model_id=args.model_id, patch_size=16)
            model = Dinov3MultiReg(cfg).to(device)

        elif args.model_name == "DinoV3Structured":
            cfg = DinoV3StructuredConfig(model_id=args.model_id)
            model = DinoV3Structured(cfg).to(device)

        elif args.model_name == "DinoV3Hybrid":
            cfg = DinoV3HybridConfig(model_id=args.model_id)
            model = DinoV3Hybrid(cfg).to(device)

        else:
            raise ValueError(f"Unknown model name: {args.model_name}")

        # optimizer only trainable params (encoder may be frozen inside model)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

        use_amp = device.type == "cuda"
        scaler = GradScaler(enabled=use_amp) if use_amp else None

        scheduler = None
        warmup_epochs = max(0, int(args.warmup_epochs))

        if args.lr_scheduler == "cosine":
            if warmup_epochs >= args.epochs:
                raise ValueError("--warmup-epochs must be < total epochs for cosine schedule.")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - warmup_epochs)
            )
        elif args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr,
                epochs=args.epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,
            )

        best_score = -1e18
        epochs_since_best = 0
        best_ckpt_path = run_out_dir / f"{args.model_name}_fold{fold}.pt"
        history = []

        # Save fold CSVs for this run (and log to MLflow)
        fold_train_csv = run_out_dir / f"train_fold{fold}.csv"
        fold_val_csv = run_out_dir / f"val_fold{fold}.csv"
        train_df.to_csv(fold_train_csv, index=False)
        val_df.to_csv(fold_val_csv, index=False)

        with mlflow.start_run(run_name=f"fold_{fold}"):
            mlflow.log_artifact(str(run_out_dir / "cli_args.json"))
            mlflow.log_artifact(str(fold_train_csv), artifact_path=f"fold_{fold}_data")
            mlflow.log_artifact(str(fold_val_csv), artifact_path=f"fold_{fold}_data")

            mlflow.log_params(
                {
                    "fold": fold,
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(val_df)),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "lr_scheduler": args.lr_scheduler,
                    "warmup_epochs": args.warmup_epochs,
                    "early_stopping_patience": args.early_stopping_patience,
                    "img_size_S": args.img_size,
                    "full_resize": f"{args.img_size*2}x{args.img_size*4}",
                    "shadow_p": args.shadow_p,
                    "model_id": args.model_id,
                    "seed": args.seed,
                    "device": device.type,
                    "splits_dir": str(splits_dir),
                    "train_csv_for_splits": str(train_csv_path),
                    # ratio knobs (logged even if not used by MultiReg)
                    "huber_dead_delta": huber_dead_delta,
                    "huber_clover_delta": huber_clover_delta,
                    "lambda_ratio": args.lambda_ratio,
                    "lambda_cons": args.lambda_cons,
                    "model_name": args.model_name,
                }
            )

            epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"Fold {fold}", dynamic_ncols=True)

            for epoch in epoch_bar:
                # cosine warmup
                if args.lr_scheduler == "cosine" and warmup_epochs > 0 and epoch <= warmup_epochs:
                    warmup_lr = args.lr * (epoch / warmup_epochs)
                    set_optimizer_lr(optimizer, warmup_lr)

                train_loss, train_score = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    model_name=args.model_name,
                    huber_dead_delta=huber_dead_delta,
                    huber_clover_delta=huber_clover_delta,
                    lambda_ratio=args.lambda_ratio,
                    lambda_cons=args.lambda_cons,
                )
                val_loss, val_score = val_one_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    model_name=args.model_name,
                    huber_dead_delta=huber_dead_delta,
                    huber_clover_delta=huber_clover_delta,
                    lambda_ratio=args.lambda_ratio,
                    lambda_cons=args.lambda_cons,
                )

                # scheduler steps
                if args.lr_scheduler == "onecycle" and scheduler is not None:
                    scheduler.step()
                elif args.lr_scheduler == "cosine" and scheduler is not None and epoch > warmup_epochs:
                    scheduler.step()

                lr_now = optimizer.param_groups[0]["lr"]
                epoch_bar.set_postfix(
                    tr_loss=f"{train_loss:.4f}",
                    tr_score=f"{train_score:.4f}",
                    va_loss=f"{val_loss:.4f}",
                    va_score=f"{val_score:.4f}",
                    lr=f"{lr_now:.2e}",
                )

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_score": train_score,
                        "val_loss": val_loss,
                        "val_score": val_score,
                        "lr": lr_now,
                    },
                    step=epoch,
                )

                print(
                    f"[Fold {fold:02d}] "
                    f"Epoch {epoch:03d}/{args.epochs:03d} | "
                    f"lr={lr_now:.2e} | "
                    f"train_loss={train_loss:.4f}, train_score={train_score:.4f} | "
                    f"val_loss={val_loss:.4f}, val_score={val_score:.4f} | "
                    f"best_val_score={best_score:.4f}"
                )

                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_score": train_score,
                        "val_loss": val_loss,
                        "val_score": val_score,
                        "lr": lr_now,
                    }
                )

                # best checkpoint on val_score (higher is better)
                if val_score > best_score:
                    best_score = val_score
                    epochs_since_best = 0
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "epoch": epoch,
                            "best_val_score": float(best_score),
                            "model_name": args.model_name,
                            "model_id": args.model_id,
                            "img_size_S": args.img_size,
                            "huber_dead_delta": huber_dead_delta,
                            "huber_clover_delta": huber_clover_delta,
                            "lambda_ratio": args.lambda_ratio,
                            "lambda_cons": args.lambda_cons,
                        },
                        best_ckpt_path,
                    )
                    print(f"[Fold {fold:02d}] 🎉 New best val_score={val_score:.4f} at epoch {epoch}. Saving checkpoint.")
                    mlflow.log_artifact(str(best_ckpt_path), artifact_path=f"fold_{fold}_best")
                else:
                    epochs_since_best += 1
                    if args.early_stopping_patience > 0 and epochs_since_best >= args.early_stopping_patience:
                        print(f"[Fold {fold:02d}] 🛑 Early stopping at epoch {epoch}. Best val_score={best_score:.4f}")
                        break

            mlflow.log_metric("best_val_score", float(best_score))

            hist_path = run_out_dir / f"history_fold{fold}.csv"
            pd.DataFrame(history).to_csv(hist_path, index=False)
            mlflow.log_artifact(str(hist_path), artifact_path=f"fold_{fold}_history")

    print(f"Done ✅ Saved outputs to: {run_out_dir}")


if __name__ == "__main__":
    main()
