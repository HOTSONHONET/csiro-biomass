#!/usr/bin/env python3
import argparse
import math
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ID_COL_CANDIDATES = [
    "image_name", "image", "filename", "file", "path", "filepath",
    "img_path", "image_path", "id", "ID", "image_id"
]


def is_nan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


def sanitize_value(val) -> str:
    """
    Convert label value to string and replace '.' with 'pt'
    2.56 -> 2pt56
    """
    if isinstance(val, float):
        if val.is_integer():
            s = str(int(val))
        else:
            s = str(val)
    else:
        s = str(val)

    s = s.strip()
    s = s.replace(".", "pt")
    s = s.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return s


def detect_id_col(df: pd.DataFrame, user_id_col: str | None) -> str:
    if user_id_col:
        if user_id_col not in df.columns:
            raise ValueError(f"--id_col '{user_id_col}' not found in CSV columns.")
        return user_id_col

    cols_lower = {c.lower(): c for c in df.columns}
    for cand in DEFAULT_ID_COL_CANDIDATES:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    for c in df.columns:
        sample = df[c].astype(str).head(20).tolist()
        if sum(("/" in s or "\\" in s) for s in sample) >= 5:
            return c

    raise ValueError(
        "Could not auto-detect image id/path column. "
        "Please pass --id_col (e.g. --id_col image_id or --id_col filename)."
    )


def find_image_file(img_dir: Path, token: str) -> Path | None:
    token = str(token).strip()
    if not token:
        return None

    p = Path(token)

    # 1) token with extension: try basename in img_dir
    if p.suffix:
        candidate = img_dir / p.name
        if candidate.exists():
            return candidate

    # 2) token as relative path inside img_dir
    candidate = img_dir / token
    if candidate.exists() and candidate.is_file():
        return candidate

    # 3) token as stem/id: token.*
    matches = list(img_dir.glob(f"{token}.*"))
    if matches:
        preferred_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"]
        matches_sorted = sorted(
            matches,
            key=lambda x: (
                preferred_exts.index(x.suffix.lower()) if x.suffix.lower() in preferred_exts else 999,
                x.name
            )
        )
        return matches_sorted[0]

    return None


def format_value(val) -> str:
    if is_nan(val):
        return "NA"
    # keep a clean float format (avoid scientific)
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return f"{val:.6f}".rstrip("0").rstrip(".")
    return str(val)


def wrap_text(lines, max_chars: int = 45):
    """
    Simple wrap based on character count (font-agnostic).
    """
    wrapped = []
    for line in lines:
        s = str(line)
        while len(s) > max_chars:
            wrapped.append(s[:max_chars])
            s = s[max_chars:]
        wrapped.append(s)
    return wrapped


def make_side_panel_image(orig_img: Image.Image, text_lines: list[str], pad: int = 10) -> Image.Image:
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

    # right panel is black already; draw text on it
    draw = ImageDraw.Draw(canvas)

    # default font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=40)
    except Exception:
        font = ImageFont.load_default()

    x0 = W + pad + 10
    y = 10

    # wrap to avoid running off
    wrapped = wrap_text(text_lines, max_chars=50)

    # line spacing
    line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
    step = int(line_h * 1.25)

    for line in wrapped:
        if y + step > out_h - 10:
            break
        draw.text((x0, y), line, fill=(255, 255, 255), font=font)
        y += step

    return canvas


def main():
    ap = argparse.ArgumentParser(
        description="Create per-label folders and copy images with renamed prefix by label value, plus a 1x2 composite debug image."
    )
    ap.add_argument("--train_csv", required=True, help="Path to train.csv")
    ap.add_argument("--img_dir", required=True, help="Directory containing images")
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Output root directory. Default: <img_dir>/by_label"
    )
    ap.add_argument(
        "--id_col",
        default=None,
        help="Column name in CSV that identifies the image (id/filename/path). Auto-detected if omitted."
    )
    ap.add_argument(
        "--label_cols",
        default=None,
        help="Comma-separated label columns to use. Default: all columns ending with '_g'."
    )
    ap.add_argument(
        "--skip_nan",
        action="store_true",
        help="If set, skip copying for a label when its value is NaN/empty."
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, prints actions without copying files."
    )
    ap.add_argument(
        "--make_panels",
        action="store_true",
        help="If set, creates all_images/<image_stem>_panel.png with original + label text."
    )

    args = ap.parse_args()

    train_csv = Path(args.train_csv)
    img_dir = Path(args.img_dir)

    if not train_csv.exists():
        raise FileNotFoundError(f"train_csv not found: {train_csv}")
    if not img_dir.exists():
        raise FileNotFoundError(f"img_dir not found: {img_dir}")

    df = pd.read_csv(train_csv)

    id_col = detect_id_col(df, args.id_col)

    if args.label_cols:
        label_cols = [c.strip() for c in args.label_cols.split(",") if c.strip()]
        missing = [c for c in label_cols if c not in df.columns]
        if missing:
            raise ValueError(f"These label cols are missing in CSV: {missing}")
    else:
        label_cols = [c for c in df.columns if c.endswith("_g")]
        if not label_cols:
            raise ValueError("No label columns ending with '_g' found. Provide --label_cols explicitly.")

    out_dir = Path(args.out_dir) if args.out_dir else (img_dir / "by_label")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create label folders
    for lab in label_cols:
        (out_dir / lab).mkdir(parents=True, exist_ok=True)

    # all_images folder
    all_images_dir = out_dir / "all_images"
    all_images_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_imgs = 0
    panel_saved = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        token = row[id_col]
        img_path = find_image_file(img_dir, str(token))

        if img_path is None or not img_path.exists():
            missing_imgs += 1
            continue

        # 1) per-label copy
        for lab in label_cols:
            val = row[lab]

            if args.skip_nan and (is_nan(val) or str(val).strip() == ""):
                continue

            val_str = sanitize_value(val) if not is_nan(val) else "NA"
            dst_dir = out_dir / lab
            dst_name = f"{val_str}_{img_path.name}"
            dst_path = dst_dir / dst_name

            if args.dry_run:
                pass
            else:
                shutil.copy2(img_path, dst_path)
                copied += 1

        # 2) composite panel image into all_images
        if args.make_panels:
            try:
                orig = Image.open(img_path).convert("RGB")
                # text = id + all label values
                text_lines = [f"{id_col}: {str(token)}"]
                labels = [
                    "sample_id",
                    "image_path",
                    "Sampling_Date",
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

                for lab in labels:
                    if lab in df.columns:
                        text_lines.append(f"{lab}: {format_value(row[lab])}")
                    else:
                        text_lines.append(f"{lab}: <missing_col>")

                panel = make_side_panel_image(orig, text_lines, pad=10)
                panel_out = all_images_dir / f"{img_path.stem}_panel.png"

                if not args.dry_run:
                    panel.save(panel_out)
                    panel_saved += 1
            except Exception:
                # ignore panel failures but continue copying
                pass

    print("\nDone.")
    print(f"Rows processed: {len(df)}")
    print(f"Copied files:  {copied}  (per-row x per-label)")
    print(f"Panels saved:  {panel_saved}  (into all_images/)")
    print(f"Missing imgs:  {missing_imgs}")
    print(f"Output dir:    {out_dir}")


if __name__ == "__main__":
    main()
