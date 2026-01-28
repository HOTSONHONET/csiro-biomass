#!/usr/bin/env python3
import argparse
import os
import math
from pathlib import Path
from datetime import datetime
from warnings import filterwarnings

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

from utils import Config, set_seed
from models.dinov3_multi_reg import Dinov3Config, Dinov3MultiReg
from models.dinov3_multi_reg_structured import DinoV3StructuredConfig, DinoV3Structured
from tqdm import tqdm

filterwarnings("ignore")

# ----------------------------
# Configuration
# ----------------------------
KEY_COL = "sample_id"  # filenames usually match sample_id (e.g., ID4464212.jpg)
TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

# (optional) what you want printed first in the panel
PANEL_META_COLS = [
    "image_path",
    "sample_id",
    "Sampling_Date",
    "Month",
    "Year",
    "Day",
    "State",
    "Species",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
    "State_code",
    "Species_code",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ----------------------------
# Metric (optional)
# ----------------------------
def competition_metric(preds_5: torch.Tensor, targets_5: torch.Tensor) -> float:
    """Weighted R^2 over 5 targets (your current definition)."""
    w = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], device=preds_5.device, dtype=torch.float32).view(1, 5)
    w_sum = w.sum()
    N = targets_5.shape[0]

    y_true = targets_5
    y_pred = preds_5

    y_mean = (w * y_true).sum() / (w_sum * N)
    ss_res = (w * (y_true - y_pred) ** 2).sum()
    ss_tot = (w * (y_true - y_mean) ** 2).sum() + 1e-9
    return float((1.0 - ss_res / ss_tot).item())


# ----------------------------
# Dataset
# ----------------------------
class FullImageInference(Dataset):
    """
    Returns:
      key_id: str
      img_t: (3, H, W) where H=2S, W=4S
      y:     (5,) GT targets
    """

    def __init__(self, df: pd.DataFrame, img_dir: str, img_size: int, is_train: bool = False, shadow_p: float = 0.5):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.S = int(img_size)

        self.target5_cols = TARGETS
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

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # key id
        key_id = str(row[KEY_COL])

        img_path = str(row["image_path"])

        with open(img_path, "rb") as f:
            img = np.array(Image.open(f).convert("RGB"))

        img_t = self.aug(image=img)["image"]

        if img_t.shape[-2:] != (self.H, self.W):
            raise RuntimeError(f"Bad shape: {img_t.shape}, expected (3,{self.H},{self.W})")

        return key_id, img_t


# ----------------------------
# Formatting / panel helpers
# ----------------------------
def is_nan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


def format_value(val) -> str:
    if is_nan(val):
        return "NA"
    if isinstance(val, (np.floating, float)):
        if float(val).is_integer():
            return str(int(val))
        return f"{float(val):.6f}".rstrip("0").rstrip(".")
    return str(val)


def wrap_text(lines, max_chars: int = 50):
    wrapped = []
    for line in lines:
        s = str(line)
        while len(s) > max_chars:
            wrapped.append(s[:max_chars])
            s = s[max_chars:]
        wrapped.append(s)
    return wrapped


def load_font(font_size: int, font_path: str | None = None):
    # Try user font path first
    if font_path and Path(font_path).exists():
        return ImageFont.truetype(font_path, size=font_size)

    # Common linux fonts
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=font_size)

    # Last resort
    return ImageFont.load_default()


def make_side_panel_image(
    orig_img: Image.Image,
    text_lines: list[str],
    pad: int = 10,
    font_size: int = 24,
    font_path: str | None = None,
    max_chars: int = 55,
) -> Image.Image:
    """
    Output size: (2W + pad) x H
    Left: original
    Right: black panel with text
    """
    orig_img = orig_img.convert("RGB")
    W, H = orig_img.size
    out_w = 2 * W + pad
    out_h = H

    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    canvas.paste(orig_img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = load_font(font_size, font_path)

    x0 = W + pad + 10
    y = 10

    wrapped = wrap_text(text_lines, max_chars=max_chars)

    # line spacing
    bbox = font.getbbox("Ag")
    line_h = bbox[3] - bbox[1]
    step = max(10, int(line_h * 1.25))

    for line in wrapped:
        if y + step > out_h - 10:
            break
        draw.text((x0, y), line, fill=(255, 255, 255), font=font)
        y += step

    return canvas


def list_images(img_dir: str) -> list[Path]:
    paths = []
    for p in Path(img_dir).iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return sorted(paths)


def unwrap_state_dict(ckpt):
    """
    Supports:
      - raw state_dict
      - {"model_state": ...}  (your current)
      - {"state_dict": ...}
      - {"model": ...}
      - {"model_state_dict": ...}
    """
    if isinstance(ckpt, dict):
        for k in ["model_state", "state_dict", "model", "model_state_dict", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def strip_module_prefix(sd: dict):
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


# ----------------------------
# Main
# ----------------------------
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser("Inference + Diagnostics + Panel Collages")

    parser.add_argument("--model-name", type=str, required=True, choices=["Dinov3MultiReg", "DinoV3Structured"])
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--weights-path", type=str, required=True)

    parser.add_argument("--img-size", type=int, default=getattr(Config, "IMG_SIZE", 512))
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--img-dir", type=str, required=True)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    # panel options
    parser.add_argument("--font-size", type=int, default=24)
    parser.add_argument("--font-path", type=str, default=None)
    parser.add_argument("--max-panels", type=int, default=-1, help="limit number of panels saved (-1 = all)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Load df
    df = pd.read_csv(args.train_csv)

    # Ensure key is string everywhere
    if KEY_COL not in df.columns:
        raise ValueError(f"Expected KEY_COL='{KEY_COL}' in CSV. Columns: {list(df.columns)}")
    df[KEY_COL] = df[KEY_COL].astype(str)

    # Build model
    if args.model_name == "Dinov3MultiReg":
        cfg = Dinov3Config(model_id=args.model_id, patch_size=16)
        model = Dinov3MultiReg(cfg)
    else:
        cfg = DinoV3StructuredConfig(model_id=args.model_id)
        model = DinoV3Structured(cfg)

    model = model.to(device).eval()
    print("[INFO] Loaded model | ", args.model_name)

    # Load weights
    ckpt = torch.load(args.weights_path, map_location="cpu")
    state = unwrap_state_dict(ckpt)
    state = strip_module_prefix(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing (first 20):", missing[:20])
        if unexpected:
            print("  unexpected (first 20):", unexpected[:20])

    # Dataset / loader
    ds = FullImageInference(df=df, img_dir=args.img_dir, img_size=args.img_size, is_train=False)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print("[INFO] Loaded dataloader")

    # Inference
    all_preds = []
    pbar = tqdm(dl, desc="Inference", leave=True, dynamic_ncols=True)

    for key_ids, img_full_t in pbar:
        img_full_t = img_full_t.to(device, non_blocking=True)

        out = model(img_full_t)
        if args.model_name == "DinoV3Structured":
            # expected dict output
            out = out["pred5"]

        preds = out.detach().float().cpu().numpy()  # (B,5)

        for key_id, pred in zip(key_ids, preds):
            key_id = str(key_id)
            all_preds.append(
                {
                    KEY_COL: key_id,
                    "Dry_Green_g_pred": float(max(0.0, pred[0])),
                    "Dry_Dead_g_pred": float(max(0.0, pred[1])),
                    "Dry_Clover_g_pred": float(max(0.0, pred[2])),
                    "GDM_g_pred": float(max(0.0, pred[3])),
                    "Dry_Total_g_pred": float(max(0.0, pred[4])),
                }
            )

    preds_df = pd.DataFrame(all_preds)
    preds_df[KEY_COL] = preds_df[KEY_COL].astype(str)

    # Merge preds with GT
    merged_df = df.merge(preds_df, on=KEY_COL, how="inner")

    # Save merged predictions CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("inference/").resolve() / f"{args.model_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_dir / "train_with_preds.csv", index=False)

    # ----------------------------
    # Plots
    # ----------------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14, 18), constrained_layout=True)

    for i, t in enumerate(TARGETS):
        gt = merged_df[t].astype(float).to_numpy()
        pred = merged_df[f"{t}_pred"].astype(float).to_numpy()

        # Left: histogram
        ax_hist = axes[i, 0]
        sns.histplot(gt, bins=40, stat="density", kde=True, alpha=0.45, label="GT", ax=ax_hist)
        sns.histplot(pred, bins=40, stat="density", kde=True, alpha=0.45, label="Pred", ax=ax_hist)
        ax_hist.set_title(f"{t} — Distribution")
        ax_hist.set_xlabel(t)
        ax_hist.set_ylabel("Density")
        ax_hist.legend()

        # Right: scatter
        ax_scatter = axes[i, 1]
        r2 = r2_score(gt, pred)
        sns.scatterplot(x=gt, y=pred, s=25, alpha=0.6, ax=ax_scatter)

        mn = float(min(gt.min(), pred.min()))
        mx = float(max(gt.max(), pred.max()))
        ax_scatter.plot([mn, mx], [mn, mx], "r--", linewidth=1)

        ax_scatter.set_title(f"{t} — Pred vs GT (R² = {r2:.4f})")
        ax_scatter.set_xlabel("GT")
        ax_scatter.set_ylabel("Pred")

    fig.savefig(out_dir / "compare_hist_scatter.png", dpi=200)
    plt.close(fig)

    # ----------------------------
    # Collages
    # ----------------------------
    image_dir = out_dir / "panels"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Build a quick lookup dict from merged_df by KEY_COL
    merged_lookup = {str(r[KEY_COL]): r for _, r in merged_df.iterrows()}

    img_paths = list_images(args.img_dir)
    if args.max_panels and args.max_panels > 0:
        img_paths = img_paths[: args.max_panels]

    for img_path in tqdm(img_paths, desc="Generating panels", leave=True, dynamic_ncols=True):
        img_id = img_path.stem  # e.g. "ID4464212"
        row = merged_lookup.get(str(img_id))
        if row is None:
            # try exact match with extension removed vs full sample_id
            continue

        image = Image.open(img_path).convert("RGB")

        # Prepare text block
        text_lines = []

        # Metadata section (only those present)
        for col in PANEL_META_COLS:
            if col in row:
                text_lines.append(f"{col}: {format_value(row[col])}")

        text_lines.append("")
        text_lines.append("----- Predictions -----")
        for t in TARGETS:
            gt_val = row.get(t, "NA")
            pr_val = row.get(f"{t}_pred", "NA")
            # abs error if available
            try:
                ae = abs(float(gt_val) - float(pr_val))
                ae_s = format_value(ae)
            except Exception:
                ae_s = "NA"
            text_lines.append(f"{t} | GT: {format_value(gt_val)} | Pred: {format_value(pr_val)} | AbsErr: {ae_s}")

        panel = make_side_panel_image(
            image,
            text_lines,
            pad=10,
            font_size=args.font_size,
            font_path=args.font_path,
            max_chars=60,
        )
        panel_out = image_dir / f"{img_path.stem}_panel.png"
        panel.save(panel_out)

    print(f"\nSaved outputs to: {out_dir}")
    print(f"- Merged CSV: {out_dir/'train_with_preds.csv'}")
    print(f"- Plot: {out_dir/'compare_hist_scatter.png'}")
    print(f"- Panels dir: {image_dir}")


if __name__ == "__main__":
    main()
