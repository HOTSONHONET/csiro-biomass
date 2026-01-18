import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

def load_wide_train(train_csv: str) -> pd.DataFrame:
    df = pd.read_csv(train_csv)

    # image_id from filename stem
    df["image_id"] = df["image_path"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])

    # pivot to wide targets
    wide = (
        df.pivot_table(
            index=["image_id", "image_path", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"],
            columns="target_name",
            values="target",
            aggfunc="first",
        )
        .reset_index()
    )

    # ensure columns exist
    needed = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    for c in needed:
        if c not in wide.columns:
            raise ValueError(f"Missing target column after pivot: {c}")

    return wide


def make_compact_strata(wide: pd.DataFrame, n_bins_total=6, n_bins_green_ratio=4, seed=42) -> np.ndarray:
    eps = 1e-6

    total_raw = wide["Dry_Total_g"].astype(float).values
    green = wide["Dry_Green_g"].astype(float).values

    total = np.log1p(total_raw)
    green_ratio = green / (total_raw + eps)

    # pd.qcut returns either Series or ndarray depending on input type.
    # Force to numpy arrays with np.asarray(...).
    total_bins = np.asarray(pd.qcut(total, q=n_bins_total, labels=False, duplicates="drop")).astype(int)
    gr_bins    = np.asarray(pd.qcut(green_ratio, q=n_bins_green_ratio, labels=False, duplicates="drop")).astype(int)

    # Combine into a compact code (<= ~60 strata)
    strata = total_bins * 10 + gr_bins
    return strata


def make_folds(
    train_csv: str,
    out_dir: str,
    n_splits: int = 5,
    seed: int = 42,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wide = load_wide_train(train_csv)

    # group to avoid leakage
    wide["env_group"] = wide["State"].astype(str) + "_" + wide["Sampling_Date"].astype(str)

    # compact strata (recommended)
    wide["strata"] = make_compact_strata(wide, n_bins_total=6, n_bins_green_ratio=4, seed=seed)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    wide["fold"] = -1
    X_dummy = np.zeros(len(wide), dtype=np.int8)

    for fold_id, (_, val_idx) in enumerate(
        sgkf.split(X_dummy, wide["strata"].values, groups=wide["env_group"].values)
    ):
        wide.loc[wide.index[val_idx], "fold"] = fold_id

    # write fold csvs for your trainer
    for f in range(n_splits):
        tr = wide[wide["fold"] != f].copy()
        va = wide[wide["fold"] == f].copy()

        tr.to_csv(out_dir / f"train_fold{f}.csv", index=False)
        va.to_csv(out_dir / f"val_fold{f}.csv", index=False)

    wide.to_csv(out_dir / "all_folds_wide.csv", index=False)
    print("Wrote folds to:", out_dir)
    print("Fold sizes:", wide["fold"].value_counts().sort_index().to_dict())


# Example usage:
# make_folds("train.csv", "splits/csiro_folds", n_splits=5, seed=42)
