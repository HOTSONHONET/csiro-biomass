import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
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
