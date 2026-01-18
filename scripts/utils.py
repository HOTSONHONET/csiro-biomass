import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import random

def log_model_results(
    mlflow_instance,
    train_preds: np.ndarray,
    val_preds: np.ndarray,
    target_cols: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    exp_dir: str,
    fold: int,
):
    for idx, col in enumerate(target_cols):
        train_df[f"pred_{col}"] = train_preds[:, idx]
        val_df[f"pred_{col}"]   = val_preds[:, idx]

    train_df["split"] = "train"
    val_df["split"]   = "val"

    df_all = pd.concat([train_df, val_df], ignore_index=True)

    csv_path = os.path.join(exp_dir, f"fold_{fold+1}_predictions.csv")
    df_all.to_csv(csv_path, index=False)
    mlflow_instance.log_artifact(csv_path)

    fig_size = (15, 8)
    for col in target_cols:
        # TRAIN
        fig, ax = plt.subplots(figsize=fig_size)
        ax.hist(train_df[col].dropna(), bins=30, alpha=0.5, label=f"train target {col}")
        ax.hist(train_df[f"pred_{col}"], bins=30, alpha=0.5, label=f"train pred {col}")
        ax.set_title(f"Fold {fold+1} - TRAIN - {col}")
        ax.set_xlabel(col); ax.set_ylabel("count"); ax.legend()
        mlflow_instance.log_figure(fig, f"plots/fold{fold+1}_{col}_train_hist.png")
        plt.close(fig)

        # VAL
        fig, ax = plt.subplots(figsize=fig_size)
        ax.hist(val_df[col].dropna(), bins=30, alpha=0.5, label=f"val target {col}")
        ax.hist(val_df[f"pred_{col}"], bins=30, alpha=0.5, label=f"val pred {col}")
        ax.set_title(f"Fold {fold+1} - VAL - {col}")
        ax.set_xlabel(col); ax.set_ylabel("count"); ax.legend()
        mlflow_instance.log_figure(fig, f"plots/fold{fold+1}_{col}_val_hist.png")
        plt.close(fig)

class Config:
    IMG_SIZE = 1000
    PATCH = 16
    EMBED_DIM = 256
    HEADS = 4
    DEPTH = 1 # num of transformer blocks

    N_SPLITS = 5
    EPOCHS = 20
    BATCH = 4
    LR = 1e-4

    NUM_WORKERS = 16
    VAL_WORKERS = 16

    TRAIN_CSV = "dataset/train_df.csv"
    IMG_DIR = "."
    SEED = 42

    IN_CHANNELS = 3
    GRID = (2, 2)
    DROPOUT = 0.2

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    TARGET_COLS = [
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "GDM_g",
        "Dry_Total_g",
    ]
    NUM_CLASSES = len(TARGET_COLS)

    @classmethod
    def to_dict(cls):
        cfg = {
            name: value 
            for name, value in cls.__dict__.items() 
            if not name.startswith('__') 
            and not callable(value) 
            and name.upper() == name
        }

        # storing class names in MODEL_MAP
        model_map_classnames = {}
        for model_name, model_class in cfg['MODEL_MAP'].items():
            model_map_classnames[model_name] = model_class.__name__

        cfg['MODEL_MAP'] = model_map_classnames
        return cfg


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
