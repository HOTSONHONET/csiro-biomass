import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from models.dinov3_mamba_2_tiles import DinoV3Hybrid, DinoV3HybridConfig

TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]


# -----------------------------
# Dataset (val only)
# -----------------------------
class CSIRODataset(Dataset):
    def __init__(self, df, img_dir, img_size):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.S = int(img_size)

        self.H = self.S * 2
        self.W = self.S * 4

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
            row[TARGETS].to_numpy(dtype="float32", na_value=0.0),
            dtype=torch.float32,
        )

        rel_path = str(row["image_path"])
        img_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.img_dir, rel_path)

        with open(img_path, "rb") as f:
            img = np.array(Image.open(f).convert("RGB"))

        img_t = self.aug(image=img)["image"]  # (3, H, W)
        return img_t, y, rel_path


# -----------------------------
# Model loader
# -----------------------------
def load_hybrid_from_ckpt(ckpt_path: str, device: torch.device, model_id: str):
    cfg = DinoV3HybridConfig(model_id=model_id)
    model = DinoV3Hybrid(cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    model.eval()
    return model


@torch.no_grad()
def infer_one_fold(model, loader, device, fold: int):
    rows_pred = []
    rows_true = []

    pbar = tqdm(
        loader,
        desc=f"Fold {fold} inference",
        dynamic_ncols=True,
        leave=True,
    )

    for x, y, rel_path in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Split full image into left/right for DinoV3Hybrid
        _, _, _, W = x.shape
        mid = W // 2
        left = x[:, :, :, :mid]
        right = x[:, :, :, mid:]

        out = model(left, right)

        pred5 = out["pred5"].detach().float().cpu().numpy()  # (B,5)
        y5 = y.detach().float().cpu().numpy()

        # optional ratio columns
        dead_ratio = out.get("dead_ratio_pred", None)
        clov_ratio = out.get("clover_ratio_pred", None)
        if dead_ratio is not None:
            dead_ratio = dead_ratio.detach().float().cpu().numpy()
        if clov_ratio is not None:
            clov_ratio = clov_ratio.detach().float().cpu().numpy()

        # write rows
        for i in range(pred5.shape[0]):
            sample_id = Path(rel_path[i]).stem

            # true row
            tr = {"sample_id": sample_id, "image_path": rel_path[i]}
            for j, t in enumerate(TARGETS):
                tr[t] = float(y5[i, j])
            rows_true.append(tr)

            # pred row
            pr = {"sample_id": sample_id, "image_path": rel_path[i]}
            for j, t in enumerate(TARGETS):
                pr[t] = float(pred5[i, j])

            if dead_ratio is not None:
                pr["dead_ratio_pred"] = float(dead_ratio[i])
            if clov_ratio is not None:
                pr["clover_ratio_pred"] = float(clov_ratio[i])

            rows_pred.append(pr)

        pbar.set_postfix(
            rows=len(rows_pred),
            bs=pred5.shape[0],
        )

    return rows_pred, rows_true


# -----------------------------
# MAIN OOF LOOP
# -----------------------------
def build_oof_csvs(
    per_fold_model_mapping: dict,
    per_fold_val_mapping: dict,
    img_dir: str,
    model_id: str,
    img_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    out_dir: str = "./oof",
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_pred = []
    all_true = []

    folds = sorted(per_fold_model_mapping.keys())
    fold_pbar = tqdm(folds, desc="Folds", dynamic_ncols=True, leave=True)

    for fold in fold_pbar:
        ckpt_path = per_fold_model_mapping[fold]
        val_csv = per_fold_val_mapping[fold]

        df_val = pd.read_csv(val_csv)

        fold_pbar.set_postfix(
            fold=fold,
            val_rows=len(df_val),
        )

        ds = CSIRODataset(df_val, img_dir=img_dir, img_size=img_size)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        model = load_hybrid_from_ckpt(ckpt_path, device=device, model_id=model_id)

        pred_rows, true_rows = infer_one_fold(model, loader, device=device, fold=fold)

        # add fold column
        for r in pred_rows:
            r["fold"] = fold
        for r in true_rows:
            r["fold"] = fold

        all_pred.extend(pred_rows)
        all_true.extend(true_rows)

        # free memory between folds
        del model
        torch.cuda.empty_cache()

    oof_pred = pd.DataFrame(all_pred)
    oof_true = pd.DataFrame(all_true)

    oof_pred_path = os.path.join(out_dir, "oof_pred.csv")
    oof_true_path = os.path.join(out_dir, "oof_true.csv")

    oof_pred.to_csv(oof_pred_path, index=False)
    oof_true.to_csv(oof_true_path, index=False)

    print("\nSaved:")
    print(oof_pred_path)
    print(oof_true_path)

    return oof_pred, oof_true


if __name__ == "__main__":
    per_fold_model_mapping = {
        0: "/home/hotson/kaggle_work/csiro-biomass/mlruns/801380342288325569/2776e2c07fd544d79afcbfff8db8f429/artifacts/fold_0_best/DinoV3Hybrid_fold0.pt",
        1: "/home/hotson/kaggle_work/csiro-biomass/mlruns/801380342288325569/6a52b6a027b74fc0b65234a84fe50cc8/artifacts/fold_1_best/DinoV3Hybrid_fold1.pt",
        2: "/home/hotson/kaggle_work/csiro-biomass/mlruns/801380342288325569/b3a849cab7954bf5a98e62ab3d60964f/artifacts/fold_2_best/DinoV3Hybrid_fold2.pt",
        3: "/home/hotson/kaggle_work/csiro-biomass/mlruns/801380342288325569/9567cd0f28af4b8d9d407df1c30ddb1d/artifacts/fold_3_best/DinoV3Hybrid_fold3.pt",
        4: "/home/hotson/kaggle_work/csiro-biomass/mlruns/612035296614314672/7085174f4dc9463f9ee7c349453368e1/artifacts/fold_4_best/DinoV3Hybrid_fold4.pt",
    }
    per_fold_val_mapping = {
        0: "/home/hotson/kaggle_work/csiro-biomass/exps/splits/csiro_folds_5/val_fold0.csv",
        1: "/home/hotson/kaggle_work/csiro-biomass/exps/splits/csiro_folds_5/val_fold1.csv",
        2: "/home/hotson/kaggle_work/csiro-biomass/exps/splits/csiro_folds_5/val_fold2.csv",
        3: "/home/hotson/kaggle_work/csiro-biomass/exps/splits/csiro_folds_5/val_fold3.csv",
        4: "/home/hotson/kaggle_work/csiro-biomass/exps/splits/csiro_folds_5/val_fold4.csv",
    }

    oof_pred_df, oof_true_df = build_oof_csvs(
        per_fold_model_mapping=per_fold_model_mapping,
        per_fold_val_mapping=per_fold_val_mapping,
        img_dir=".",
        model_id="vit_huge_plus_patch16_dinov3.lvd1689m",
        img_size=512,
        batch_size=4,
        num_workers=4,
        out_dir="./oof/dinov3_mamba_hybrid_model",
    )
