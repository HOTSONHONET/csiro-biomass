import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import math
import seaborn as sns
from sklearn.metrics import r2_score

TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
W = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float32)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# -----------------------------
# Panel / Collage utilities
# -----------------------------
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

def wrap_text(lines, max_chars: int = 60):
    wrapped = []
    for line in lines:
        s = str(line)
        while len(s) > max_chars:
            wrapped.append(s[:max_chars])
            s = s[max_chars:]
        wrapped.append(s)
    return wrapped

def load_font(font_size: int, font_path: str | None = None):
    if font_path and Path(font_path).exists():
        return ImageFont.truetype(font_path, size=font_size)

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=font_size)

    return ImageFont.load_default()

def make_side_panel_image(
    orig_img: Image.Image,
    text_lines: list[str],
    pad: int = 12,
    font_size: int = 30,
    font_path: str | None = None,
    max_chars: int = 62,
    panel_w: int | None = None,
) -> Image.Image:
    """
    Output: (W + panel_w + pad) x H
    Left: original
    Right: black panel with readable text
    """
    orig_img = orig_img.convert("RGB")
    W, H = orig_img.size

    # Key change: panel width scales with image width (readability!)
    if panel_w is None:
        panel_w = max(900, int(0.75 * W))  # big panel by default

    out_w = W + panel_w + pad
    out_h = H

    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    canvas.paste(orig_img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = load_font(font_size, font_path)

    x0 = W + pad + 20
    y = 20

    wrapped = wrap_text(text_lines, max_chars=max_chars)

    bbox = font.getbbox("Ag")
    line_h = bbox[3] - bbox[1]
    step = max(10, int(line_h * 1.35))

    for line in wrapped:
        if y + step > out_h - 20:
            break
        draw.text((x0, y), line, fill=(255, 255, 255), font=font)
        y += step

    return canvas

def save_collage_readable(
    img_path: str,
    lines: list[str],
    out_path: Path,
    font_size: int = 30,
    max_chars: int = 62,
    panel_w: int | None = None,
    font_path: str | None = None,
):
    img = Image.open(img_path).convert("RGB")
    panel = make_side_panel_image(
        img,
        lines,
        font_size=font_size,
        max_chars=max_chars,
        panel_w=panel_w,
        font_path=font_path,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)

def save_r2_scatter_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names,
    out_dir: Path,
    filename: str = "compare.png",
):
    """
    Saves one figure:
      Row per target:
        Left  -> GT vs Pred histogram overlay
        Right -> GT vs Pred scatter + y=x line, with R²
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
    assert y_true.shape[1] == len(target_names), f"Expected {len(target_names)} targets, got {y_true.shape[1]}"

    sns.set_theme(style="whitegrid")

    n = len(target_names)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(14, 3.5 * n), constrained_layout=True)

    # if n == 1, axes shape can be (2,) -> normalize to 2D
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, name in enumerate(target_names):
        gt = y_true[:, i]
        pred = y_pred[:, i]

        # drop NaNs/inf if any
        m = np.isfinite(gt) & np.isfinite(pred)
        gt = gt[m]
        pred = pred[m]

        # Left: Histogram
        ax_hist = axes[i, 0]
        sns.histplot(gt, bins=40, stat="density", kde=True, alpha=0.45, label="GT", ax=ax_hist)
        sns.histplot(pred, bins=40, stat="density", kde=True, alpha=0.45, label="Pred", ax=ax_hist)
        ax_hist.set_title(f"{name} — Distribution")
        ax_hist.set_xlabel(name)
        ax_hist.set_ylabel("Density")
        ax_hist.legend()

        # Right: Scatter + R²
        ax_scatter = axes[i, 1]
        r2 = r2_score(gt, pred) if len(gt) > 1 else float("nan")
        sns.scatterplot(x=gt, y=pred, s=20, alpha=0.6, ax=ax_scatter)

        mn = float(min(gt.min(), pred.min()))
        mx = float(max(gt.max(), pred.max()))
        ax_scatter.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)

        ax_scatter.set_title(f"{name} — Pred vs GT (R² = {r2:.4f})")
        ax_scatter.set_xlabel("GT")
        ax_scatter.set_ylabel("Pred")

    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)

# -----------------------------
# Metric
# -----------------------------
def weighted_r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)

    w = W.reshape(1, 5)
    w_sum = float(w.sum())
    N = y_true.shape[0]

    y_mean = (w * y_true).sum() / (w_sum * N + 1e-9)
    ss_res = (w * (y_true - y_pred) ** 2).sum()
    ss_tot = (w * (y_true - y_mean) ** 2).sum() + 1e-9
    return float(1.0 - ss_res / ss_tot)


# -----------------------------
# Core constraints
# -----------------------------
def _nonneg(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def recompute_from_parts(
    green: np.ndarray,
    dead: np.ndarray,
    clover: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enforce exact identities: gdm = green + clover, total = gdm + dead"""
    green = _nonneg(green)
    dead = _nonneg(dead)
    clover = _nonneg(clover)
    gdm = green + clover
    total = gdm + dead
    return green, dead, clover, gdm, total


def ratio_recompose(
    total: np.ndarray,
    gdm: np.ndarray,
    dead_ratio: np.ndarray,
    clover_ratio: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """dead = dr * total, clover = cr * gdm, green = gdm - clover, then re-enforce sums"""
    total = _nonneg(total)
    gdm = _nonneg(gdm)

    dead_ratio = np.clip(dead_ratio, 0.0, 1.0)
    clover_ratio = np.clip(clover_ratio, 0.0, 1.0)

    dead = dead_ratio * total
    clover = clover_ratio * gdm
    green = gdm - clover

    return recompute_from_parts(green, dead, clover)


def blend_dead(dead_pred: np.ndarray, total: np.ndarray, gdm: np.ndarray, alpha: float) -> np.ndarray:
    """alpha=1 -> dead = total - gdm ; alpha=0 -> dead_pred"""
    dead_derived = np.maximum(0.0, total - gdm)
    return alpha * dead_derived + (1.0 - alpha) * np.maximum(0.0, dead_pred)


# -----------------------------
# Calibration + clipping
# -----------------------------
def fit_linear_calibration(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit y_true ≈ a*y_pred + b per-target."""
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)

    a = np.ones(5, dtype=np.float32)
    b = np.zeros(5, dtype=np.float32)

    for j in range(5):
        x = y_pred[:, j]
        y = y_true[:, j]
        vx = float(np.var(x))
        if vx < 1e-9:
            a[j] = 1.0
            b[j] = float(np.mean(y) - np.mean(x))
        else:
            cov = float(np.mean((x - x.mean()) * (y - y.mean())))
            a[j] = cov / (vx + 1e-9)
            b[j] = float(y.mean() - a[j] * x.mean())
    return a, b


def apply_linear_calibration(y_pred: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return y_pred * a.reshape(1, 5) + b.reshape(1, 5)


def quantile_bounds_from_train(y_train: np.ndarray, q_low=0.5, q_high=99.5) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.percentile(y_train, q_low, axis=0).astype(np.float32)
    hi = np.percentile(y_train, q_high, axis=0).astype(np.float32)
    return lo, hi


def max_bounds_from_train(y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hard [0, max] bound per target."""
    lo = np.zeros((5,), dtype=np.float32)
    hi = np.max(y_train, axis=0).astype(np.float32)
    return lo, hi


def apply_clip(y_pred: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(y_pred, lo.reshape(1, 5), hi.reshape(1, 5))


# -----------------------------
# Config
# -----------------------------
@dataclass
class PostprocessConfig:
    use_ratio_recompose: bool = True   # only if ratio cols exist
    alpha_dead_derive: float = 0.7     # only used when ratios NOT used
    use_linear_calib: bool = True
    clip_mode: str = "quantile"        # "quantile" or "max"
    q_low: float = 0.5
    q_high: float = 99.5


# -----------------------------
# Apply postprocess (no height)
# -----------------------------
def apply_postprocess(
    pred_df: pd.DataFrame,
    cfg: PostprocessConfig,
    calib_ab: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    clip_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> pd.DataFrame:
    df = pred_df.copy()

    green = df["Dry_Green_g"].to_numpy(np.float32)
    dead  = df["Dry_Dead_g"].to_numpy(np.float32)
    clov  = df["Dry_Clover_g"].to_numpy(np.float32)
    gdm   = df["GDM_g"].to_numpy(np.float32)
    total = df["Dry_Total_g"].to_numpy(np.float32)

    # Step A: non-neg first
    green = _nonneg(green); dead = _nonneg(dead); clov = _nonneg(clov)
    gdm = _nonneg(gdm); total = _nonneg(total)

    # IMPORTANT FIX:
    # used_ratio is NOT hardcoded; it reflects the actual branch used.
    has_ratio_cols = ("dead_ratio_pred" in df.columns) and ("clover_ratio_pred" in df.columns)
    used_ratio = bool(cfg.use_ratio_recompose and has_ratio_cols)

    # Step B: ratio recomposition (IF enabled + available)
    if used_ratio:
        dr = df["dead_ratio_pred"].to_numpy(np.float32)
        cr = df["clover_ratio_pred"].to_numpy(np.float32)
        green, dead, clov, gdm, total = ratio_recompose(total=total, gdm=gdm, dead_ratio=dr, clover_ratio=cr)
    else:
        # Step C: no ratio -> blend dead with derived total-gdm
        dead = blend_dead(dead_pred=dead, total=total, gdm=gdm, alpha=cfg.alpha_dead_derive)
        green, dead, clov, gdm, total = recompute_from_parts(green, dead, clov)

    pred5 = np.stack([green, dead, clov, gdm, total], axis=1)

    # Step D: linear calibration
    if cfg.use_linear_calib and calib_ab is not None:
        a, b = calib_ab
        pred5 = apply_linear_calibration(pred5, a, b)
        pred5 = np.maximum(pred5, 0.0)
        # enforce sums again (use green/dead/clover as base)
        green, dead, clov = pred5[:, 0], pred5[:, 1], pred5[:, 2]
        green, dead, clov, gdm, total = recompute_from_parts(green, dead, clov)
        pred5 = np.stack([green, dead, clov, gdm, total], axis=1)

    # Step E: clipping (quantile or max)
    if clip_bounds is not None:
        lo, hi = clip_bounds
        pred5 = apply_clip(pred5, lo, hi)
        # enforce sums after clip
        green, dead, clov = pred5[:, 0], pred5[:, 1], pred5[:, 2]
        green, dead, clov, gdm, total = recompute_from_parts(green, dead, clov)
        pred5 = np.stack([green, dead, clov, gdm, total], axis=1)

    # write back
    df[TARGETS] = pred5

    # optional: keep a debug flag column so you can verify which path was used
    # (safe for OOF; for Kaggle submission you can drop it)
    df["_used_ratio"] = int(used_ratio)

    return df


# -----------------------------
# Tuning on OOF (no height)
# -----------------------------
def tune_on_oof(
    oof_pred_df: pd.DataFrame,
    oof_true_df: pd.DataFrame,
    train_targets_for_clip: np.ndarray,
) -> Dict:
    y_true = oof_true_df[TARGETS].to_numpy(np.float32)

    has_ratio = ("dead_ratio_pred" in oof_pred_df.columns) and ("clover_ratio_pred" in oof_pred_df.columns)
    use_ratio_opts = [True, False] if has_ratio else [False]

    alpha_grid = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]  # only matters when use_ratio=False

    clip_modes = ["quantile", "max"]
    quantile_pairs = [(0.1, 99.9), (0.5, 99.5), (1.0, 99.0)]

    best = {"score": -1e18, "cfg": None, "calib_ab": None, "clip_bounds": None}

    combos = []
    for use_ratio in use_ratio_opts:
        for clip_mode in clip_modes:
            if clip_mode == "quantile":
                for ql, qh in quantile_pairs:
                    if use_ratio:
                        combos.append((use_ratio, 0.7, clip_mode, ql, qh))
                    else:
                        for a in alpha_grid:
                            combos.append((use_ratio, a, clip_mode, ql, qh))
            else:
                if use_ratio:
                    combos.append((use_ratio, 0.7, clip_mode, None, None))
                else:
                    for a in alpha_grid:
                        combos.append((use_ratio, a, clip_mode, None, None))

    pbar = tqdm(combos, desc="Tuning", total=len(combos), dynamic_ncols=True)

    for use_ratio, alpha, clip_mode, ql, qh in pbar:
        if clip_mode == "quantile":
            clip_bounds = quantile_bounds_from_train(train_targets_for_clip, q_low=ql, q_high=qh)
        else:
            clip_bounds = max_bounds_from_train(train_targets_for_clip)

        cfg = PostprocessConfig(
            use_ratio_recompose=bool(use_ratio),
            alpha_dead_derive=float(alpha),
            use_linear_calib=True,
            clip_mode=clip_mode,
            q_low=float(ql) if ql is not None else 0.0,
            q_high=float(qh) if qh is not None else 100.0,
        )

        # 1) postprocess w/o calib
        tmp = apply_postprocess(oof_pred_df, cfg=cfg, calib_ab=None, clip_bounds=clip_bounds)
        y_pp = tmp[TARGETS].to_numpy(np.float32)

        # 2) fit calib on postprocessed OOF
        a, b = fit_linear_calibration(y_pp, y_true)

        # 3) postprocess with calib
        tmp2 = apply_postprocess(oof_pred_df, cfg=cfg, calib_ab=(a, b), clip_bounds=clip_bounds)
        y_final = tmp2[TARGETS].to_numpy(np.float32)

        score = weighted_r2(y_final, y_true)
        if score > best["score"]:
            print(f"🎉 Score improved from {best['score']} to {score}")
            best.update({"score": score, "cfg": cfg, "calib_ab": (a, b), "clip_bounds": clip_bounds})
            print("\nOOF SCORE:", score)
            print("CFG:", cfg)
            a, b = best["calib_ab"]
            print("CALIB a:", a)
            print("CALIB b:", b)
            

        pbar.set_postfix(best=f"{best['score']:.5f}", last=f"{score:.5f}", ratio=int(use_ratio), clip=clip_mode)

    return best


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("calibrate model prediction (no height)")
    parser.add_argument("--oof_pred", type=str, required=True, help="path to oof_pred.csv")
    parser.add_argument("--oof_true", type=str, required=True, help="path to oof_true.csv")
    parser.add_argument("--train_csv", type=str, default="train.csv", help="path to train.csv")
    parser.add_argument("--out_dir", type=str, required=True, help="output dir")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_pred = pd.read_csv(args.oof_pred)
    oof_true = pd.read_csv(args.oof_true)

    # Align by key
    key = "sample_id"
    oof = oof_true[[key] + TARGETS].merge(oof_pred, on=key, how="inner", suffixes=("_true", ""))

    oof_true_df = oof[[key] + [c + "_true" for c in TARGETS]].copy()
    oof_true_df.columns = [key] + TARGETS

    keep_extra = [c for c in ["dead_ratio_pred", "clover_ratio_pred"] if c in oof.columns]
    oof_pred_df = oof[[key] + TARGETS + keep_extra].copy()

    train_df = pd.read_csv(args.train_csv)
    train_targets = train_df[TARGETS].to_numpy(np.float32)

    # Printing base score
    base_score = weighted_r2(
        y_pred = oof_pred_df[TARGETS].to_numpy(np.float32),
        y_true = oof_true_df[TARGETS].to_numpy(np.float32)
    )
    print("Without calibration base score: ", base_score)

    best = tune_on_oof(
        oof_pred_df=oof_pred_df,
        oof_true_df=oof_true_df,
        train_targets_for_clip=train_targets,
    )

    print("\nBEST OOF SCORE:", best["score"])
    print("BEST CFG:", best["cfg"])
    a, b = best["calib_ab"]
    print("CALIB a:", a)
    print("CALIB b:", b)

    # Save best config
    lo, hi = best["clip_bounds"]
    save_obj = {
        "best_score": float(best["score"]),
        "cfg": asdict(best["cfg"]),
        "calib_a": a.tolist(),
        "calib_b": b.tolist(),
        "clip_lo": lo.tolist(),
        "clip_hi": hi.tolist(),
    }
    with open(out_dir / "best_calibration.json", "w") as f:
        json.dump(save_obj, f, indent=2)

    print(f"\nSaved: {(out_dir / 'best_calibration.json')}")


    # 1) Build calibrated OOF predictions (final postprocessed)
    cfg = best["cfg"]
    calib_ab = best["calib_ab"]
    clip_bounds = best["clip_bounds"]

    # Apply to OOF (same rows as oof_pred_df)
    oof_pp = apply_postprocess(oof_pred_df, cfg=cfg, calib_ab=calib_ab, clip_bounds=clip_bounds)

    # Merge with GT for analysis assets
    key = "sample_id"
    merged = oof_true_df[[key] + TARGETS].merge(
        oof_pp[[key] + TARGETS + ([c for c in ["image_path", "dead_ratio_pred", "clover_ratio_pred"] if c in oof_pp.columns])],
        on=key,
        how="inner",
        suffixes=("_true", ""),
    )

    # 2) Save calibrated OOF CSV (useful for debugging)
    (out_dir / "oof_postprocessed.csv").parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "oof_postprocessed.csv", index=False)


    # 3) Scatter plots (per target)
    y_true = merged[[t + "_true" for t in TARGETS]].to_numpy(np.float32)
    y_pred = merged[TARGETS].to_numpy(np.float32)
    save_r2_scatter_plots(
        y_true=y_true,
        y_pred=y_pred,
        target_names=TARGETS,
        out_dir=out_dir / "plots",
    )

    # 4) Collages (choose worst K by weighted abs error)
    w = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float32).reshape(1, 5)
    abs_err = np.abs(y_pred - y_true)
    merged["weighted_abs_err"] = (abs_err * w).sum(axis=1)

    merged = merged.sort_values("weighted_abs_err", ascending=False).reset_index(drop=True)
    merged['image_path'] = merged['sample_id'].apply(lambda sid: f"train/{sid}.jpg")
    print("columns: ", merged.columns)

    img_dir = "."

    img_out = out_dir / "images"
    pbar = tqdm(merged.itertuples(index=False), total=len(merged), desc="Saving collages", dynamic_ncols=True)

    # for row in pbar:
    #     row = row._asdict()
    #     rel_path = str(row.get("image_path", ""))

    #     if not rel_path:
    #         continue

    #     img_path = rel_path if os.path.isabs(rel_path) else os.path.join(img_dir, rel_path)
    #     sid = row.get("sample_id", Path(rel_path).stem)

    #     lines = []
    #     lines.append(f"image_path: {rel_path}")
    #     lines.append(f"sample_id: {sid}")
    #     lines.append("")
    #     lines.append("Postprocessed predictions:")

    #     for t in TARGETS:
    #         gt = float(row[f"{t}_true"])
    #         pr = float(row[t])
    #         ae = abs(pr - gt)
    #         lines.append(f"{t} | GT: {gt:.4f} | Pred: {pr:.4f} | AbsErr: {ae:.4f}")

    #     lines.append("")
    #     lines.append(f"weighted_abs_err: {float(row['weighted_abs_err']):.6f}")
    #     lines.append(f"cfg.use_ratio_recompose: {best['cfg'].use_ratio_recompose}")
    #     lines.append(f"alpha_dead_derive: {best['cfg'].alpha_dead_derive}")
    #     lines.append(f"clip_mode: {best['cfg'].clip_mode}")

    #     out_path = img_out / f"{sid}_panel.png"

    #     try:
    #         # panel_w=None means: auto scale with image width
    #         save_collage_readable(
    #             img_path=img_path,
    #             lines=lines,
    #             out_path=out_path,
    #             font_size=32,      # bump this if needed
    #             max_chars=68,      # bump if you want fewer wraps
    #             panel_w=None,      # auto; or set panel_w=1200
    #             font_path=None,
    #         )
    #     except Exception as e:
    #         with open(img_out / "_failed.txt", "a", encoding="utf-8") as f:
    #             f.write(f"{sid}\t{img_path}\t{repr(e)}\n")

    #     pbar.set_postfix(file=out_path.name)

    # print("\nSaved extras:")
    # print(" -", out_dir / "oof_postprocessed.csv")
    # print(" -", out_dir / "plots/")
    # print(" -", out_dir / "images/")
