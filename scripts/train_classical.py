#!/usr/bin/env python3
"""
Train classical ML models on DINOv3 embeddings, rank by competition metric,
fine-tune top-3, and log everything to MLflow (including top-3 ensemble).

Typical usage (Kaggle-like):
python train_classical_from_dinov3.py \
  --train-csv /kaggle/input/csiro-biomass/train.csv \
  --img-root /kaggle/input/csiro-biomass \
  --image-col image_path \
  --id-col sample_id \
  --targets Dry_Green_g Dry_Clover_g Dry_Dead_g GDM_g Dry_Total_g \
  --model-id vit_huge_plus_patch16_dinov3.lvd1689m \
  --img-size 518 \
  --n-splits 5 \
  --mlflow-uri http://127.0.0.1:5000 \
  --experiment-name "dinov3_embeddings_classical" \
  --outdir ./outputs_classical \
  --device cuda
"""

import os
import gc
import json
import math
import time
import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from PIL import Image
from tqdm.auto import tqdm

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ---------------------------
# 1) Competition metric
# ---------------------------
# Competition target weights (commonly used in kernels)
TARGET_WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}  # :contentReference[oaicite:1]{index=1}

def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> float:
    """
    Global weighted R^2 over all (sample, target) pairs:
      ybar_w = sum(w_i * y_i) / sum(w_i)
      R2_w = 1 - sum(w_i*(y_i - yhat_i)^2) / sum(w_i*(y_i - ybar_w)^2)

    y_true, y_pred: shape (N, T)
    target_names: length T, order must match columns in y_true/y_pred
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # weights per target -> expanded to (N, T)
    w_t = np.array([TARGET_WEIGHTS[t] for t in target_names], dtype=np.float64)  # (T,)
    w = np.broadcast_to(w_t, y_true.shape)  # (N, T)

    # flatten everything into one long vector (global)
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    ww = w.reshape(-1)

    w_sum = ww.sum()
    if w_sum <= 0:
        return 0.0

    y_wmean = (ww * yt).sum() / w_sum
    ss_res = (ww * (yt - yp) ** 2).sum()
    ss_tot = (ww * (yt - y_wmean) ** 2).sum()

    if ss_tot <= 1e-12:
        return 0.0

    return float(1.0 - ss_res / ss_tot)


def competition_metric(y_true: np.ndarray, y_pred: np.ndarray, target_names=None) -> float:
    """
    IMPORTANT: Edit this to match the competition metric exactly.

    Default: mean R2 across targets (multi-output).
    If your metric is "weighted_r2_score", replace the logic here.

    Args:
      y_true, y_pred: shape (N, T)
    """
    return weighted_r2_score(y_true, y_pred, target_names)


# ---------------------------
# 2) DINOv3 embedding extractor (timm)
# ---------------------------
def build_dinov3_backbone(model_id: str, img_size: int, device: str):
    """
    Uses timm to load DINOv3-style ViT weights when available by model_id.
    Many DINOv3 weights are exposed via timm. If your exact ID is custom,
    you can adapt this to your loader.
    """
    import timm

    model = timm.create_model(model_id, pretrained=True, num_classes=0)  # num_classes=0 => features
    model.eval().to(device)

    # timm data config for transforms
    cfg = timm.data.resolve_model_data_config(model)
    # Override image size if user wants a fixed size
    cfg["input_size"] = (3, img_size, img_size)

    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform


@torch.inference_mode()
def extract_embeddings(
    df: pd.DataFrame,
    img_root: Path,
    image_col: str,
    model_id: str,
    img_size: int,
    device: str,
    cache_path: Path,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract embeddings for all rows in df. Caches to .npy to avoid recomputation.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        embs = np.load(cache_path)
        if embs.shape[0] == len(df):
            return embs
        print(f"[WARN] Cache exists but shape mismatch. Recomputing: {cache_path}")

    model, transform = build_dinov3_backbone(model_id=model_id, img_size=img_size, device=device)

    # Lazy-load images into batches
    paths = [img_root / p for p in df[image_col].astype(str).tolist()]

    all_embs = []
    batch_imgs = []
    for i, p in enumerate(tqdm(paths, desc="Extracting DINOv3 embeddings")):
        img = Image.open(p).convert("RGB")
        x = transform(img)  # tensor C,H,W
        batch_imgs.append(x)

        if len(batch_imgs) == batch_size or (i == len(paths) - 1):
            bx = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
            feats = model(bx)  # (B, D) or (B, tokens, D) depending on model
            if feats.ndim == 3:
                # If token-wise features returned, pool (CLS token if present else mean)
                feats = feats[:, 0, :]  # common: CLS token
            feats = feats.detach().float().cpu().numpy()
            all_embs.append(feats)
            batch_imgs = []

    embs = np.concatenate(all_embs, axis=0)
    np.save(cache_path, embs)
    return embs


# ---------------------------
# 3) Models zoo
# ---------------------------
def make_models(random_state: int = 42):
    """
    A reasonably strong spread for embedding-based regression.
    Some are wrapped with MultiOutputRegressor when needed.
    """
    models = {}

    # Linear baselines
    models["ridge"] = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", Ridge(random_state=random_state)),
    ])
    models["lasso"] = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", Lasso(random_state=random_state, max_iter=20000)),
    ])
    models["elasticnet"] = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", ElasticNet(random_state=random_state, max_iter=20000)),
    ])

    # Tree ensembles (often strong on embeddings)
    models["rf"] = RandomForestRegressor(
        n_estimators=500, random_state=random_state, n_jobs=-1
    )
    models["extratrees"] = ExtraTreesRegressor(
        n_estimators=1000, random_state=random_state, n_jobs=-1
    )
    models["hgb"] = MultiOutputRegressor(
        HistGradientBoostingRegressor(random_state=random_state)
    )

    models["gbr"] = MultiOutputRegressor(GradientBoostingRegressor(random_state=random_state))

    # kNN (surprisingly decent sometimes)
    models["knn"] = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", KNeighborsRegressor(n_neighbors=25, weights="distance")),
    ])

    # SVR needs MultiOutput wrapper for multi-target
    models["svr_rbf"] = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", MultiOutputRegressor(SVR(kernel="rbf", C=10.0, epsilon=0.1))),
    ])

    try:
        from xgboost import XGBRegressor
        models["xgb"] = MultiOutputRegressor(
                XGBRegressor(
                    n_estimators=6000,
                    learning_rate=0.02,
                    max_depth=6,

                    subsample=0.8,
                    colsample_bytree=0.6,

                    min_child_weight=1,   # make splits easier
                    reg_lambda=1.0,
                    reg_alpha=0.0,

                    tree_method="hist",   # "gpu_hist" if GPU
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=0,          # keeps output clean
                )
            )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor
        models["lgbm"] = MultiOutputRegressor(
            LGBMRegressor(
                objective="regression",
                n_estimators=6000,
                learning_rate=0.02,

                # make splits easier on small data
                min_child_samples=5,      # aka min_data_in_leaf
                min_split_gain=0.0,       # aka min_gain_to_split

                # control complexity
                num_leaves=31,
                max_depth=-1,

                # feature subsampling helps high-dim embeddings
                feature_fraction=0.6,     # colsample_bytree
                bagging_fraction=0.8,     # subsample
                bagging_freq=1,

                reg_lambda=1.0,
                reg_alpha=0.0,

                random_state=random_state,
                n_jobs=-1,   
                device_type="gpu",
                gpu_platform_id=0,
                gpu_device_id=0,
                verbosity=-1,
            )
        )
    except Exception:
        pass


    return models


def tuning_spaces():
    """
    RandomizedSearch spaces for top-3.
    Keep these compact to fit your 'one day left' constraint.
    """
    spaces = {}

    spaces["ridge"] = {
        "model__alpha": np.logspace(-4, 4, 30),
    }

    spaces["elasticnet"] = {
        "model__alpha": np.logspace(-4, 2, 30),
        "model__l1_ratio": np.linspace(0.05, 0.95, 20),
    }

    spaces["rf"] = {
        "n_estimators": [600, 1000, 1600],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, 0.8],
    }

    spaces["extratrees"] = {
        "n_estimators": [800, 1400, 2200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, 0.8],
    }

    spaces["hgb"] = {
        "learning_rate": [0.01, 0.03, 0.06],
        "max_depth": [None, 6, 10],
        "max_iter": [1000, 2000, 4000],
        "min_samples_leaf": [10, 20, 50],
        "l2_regularization": [0.0, 0.1, 1.0],
    }

    spaces["knn"] = {
        "model__n_neighbors": [5, 10, 20, 35, 50, 75],
        "model__weights": ["uniform", "distance"],
    }

    spaces["svr_rbf"] = {
        "model__estimator__C": [1.0, 3.0, 10.0, 30.0],
        "model__estimator__gamma": ["scale", "auto"],
        "model__estimator__epsilon": [0.01, 0.05, 0.1, 0.2],
    }

    # Optional models
    if True:
        try:
            from lightgbm import LGBMRegressor  # noqa
            spaces["lgbm"] = {
                "estimator__n_estimators": [2000, 4000, 7000],
                "estimator__learning_rate": [0.01, 0.03, 0.06],
                "estimator__num_leaves": [31, 63, 127],
                "estimator__subsample": [0.7, 0.85, 1.0],
                "estimator__colsample_bytree": [0.7, 0.85, 1.0],
            }
        except Exception:
            pass
            
        try:
            from xgboost import XGBRegressor
            spaces["xgb"] = {
                "estimator__n_estimators": [2000, 4000, 7000],
                "estimator__learning_rate": [0.01, 0.03, 0.06],
                "estimator__max_depth": [6, 8, 10],
                "estimator__min_child_weight": [1, 3, 5, 10],
                "estimator__subsample": [0.7, 0.85, 1.0],
                "estimator__colsample_bytree": [0.7, 0.85, 1.0],
                "estimator__reg_lambda": [0.5, 1.0, 2.0, 5.0],
            }
        except Exception:
            pass

    return spaces


# ---------------------------
# 4) CV training + MLflow logging
# ---------------------------
def oof_cv_score(model, X, Y, n_splits: int, seed: int, target_names):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros_like(Y, dtype=np.float32)

    fold_scores = []
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        Xtr, Xva = X[tr], X[va]
        Ytr, Yva = Y[tr], Y[va]

        m = model
        m.fit(Xtr, Ytr)
        pred = m.predict(Xva)
        oof[va] = pred

        sc = competition_metric(Yva, pred, target_names=target_names)
        fold_scores.append(sc)

    overall = competition_metric(Y, oof, target_names=target_names)
    return overall, fold_scores, oof


def fit_full_and_predict(model, X, Y, X_test):
    model.fit(X, Y)
    return model.predict(X_test)


def weighted_ensemble(preds_list, scores):
    """
    Weighted average ensemble where weights are proportional to (score - min + eps).
    Works for metrics where 'higher is better'.
    """
    scores = np.array(scores, dtype=np.float64)
    w = scores - scores.min()
    w = w + 1e-9
    w = w / w.sum()

    out = None
    for wi, pi in zip(w, preds_list):
        out = pi * wi if out is None else out + pi * wi
    return out, w.tolist()


# ---------------------------
# 5) Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=str, required=True)
    ap.add_argument("--test-csv", type=str, default=None, help="Optional: if you want test predictions.")
    ap.add_argument("--img-root", type=str, required=True)
    ap.add_argument("--image-col", type=str, required=True)
    ap.add_argument("--id-col", type=str, default=None)
    ap.add_argument("--targets", nargs="+", required=True)

    ap.add_argument("--model-id", type=str, required=True, help="timm model id, e.g. vit_huge_plus_patch16_dinov3...")
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--mlflow-uri", type=str, default=None)
    ap.add_argument("--experiment-name", type=str, default="dinov3_classical")
    ap.add_argument("--run-name", type=str, default=None)

    ap.add_argument("--outdir", type=str, default="./outputs_classical")
    ap.add_argument("--tune-iter", type=int, default=25, help="RandomizedSearch iterations for each top model")
    ap.add_argument("--tune-cv", type=int, default=3)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # MLflow setup
    if args.mlflow_uri:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    train_df = pd.read_csv(args.train_csv)
    img_root = Path(args.img_root)

    # Build targets
    Y = train_df[args.targets].values.astype(np.float32)
    target_names = list(args.targets)

    # Embeddings cache paths
    cache_dir = outdir / "cache_embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / f"train_{args.model_id}_sz{args.img_size}.npy"

    # Extract embeddings
    X = extract_embeddings(
        df=train_df,
        img_root=img_root,
        image_col=args.image_col,
        model_id=args.model_id,
        img_size=args.img_size,
        device=args.device,
        cache_path=train_cache,
        batch_size=args.batch_size,
    ).astype(np.float32)

    # Optional test embeddings
    test_df, X_test = None, None
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv)
        test_cache = cache_dir / f"test_{args.model_id}_sz{args.img_size}.npy"
        X_test = extract_embeddings(
            df=test_df,
            img_root=img_root,
            image_col=args.image_col,
            model_id=args.model_id,
            img_size=args.img_size,
            device=args.device,
            cache_path=test_cache,
            batch_size=args.batch_size,
        ).astype(np.float32)

    models = make_models(random_state=args.seed)
    spaces = tuning_spaces()

    # ---------------------------
    # A) Train & rank base models
    # ---------------------------
    report_rows = []
    oof_store = {}   # model_name -> oof preds
    fold_store = {}  # model_name -> fold scores

    with mlflow.start_run(run_name=args.run_name or f"base_models_{int(time.time())}"):
        mlflow.log_params({
            "model_id": args.model_id,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "n_splits": args.n_splits,
            "seed": args.seed,
            "targets": json.dumps(target_names),
        })

        for name, model in models.items():
            print(f"\n=== Training base model: {name} ===")
            with mlflow.start_run(run_name=f"base_{name}", nested=True):
                mlflow.log_param("model_name", name)

                score, fold_scores, oof = oof_cv_score(
                    model=model,
                    X=X,
                    Y=Y,
                    n_splits=args.n_splits,
                    seed=args.seed,
                    target_names=target_names,
                )

                mlflow.log_metric("cv_score", score)
                for i, fs in enumerate(fold_scores, start=1):
                    mlflow.log_metric(f"fold_{i}_score", fs)

                report_rows.append({
                    "model": name,
                    "stage": "base",
                    "cv_score": score,
                    "fold_scores": fold_scores,
                })
                oof_store[name] = oof
                fold_store[name] = fold_scores

                # Save OOF artifact
                np.save(outdir / f"oof_{name}_base.npy", oof)
                mlflow.log_artifact(str(outdir / f"oof_{name}_base.npy"))

                gc.collect()

        report_df = pd.DataFrame(report_rows).sort_values("cv_score", ascending=False).reset_index(drop=True)
        report_path = outdir / "report_base_models.csv"
        report_df.to_csv(report_path, index=False)
        mlflow.log_artifact(str(report_path))

        print("\n\n===== BASE MODELS RANKING =====")
        print(report_df[["model", "cv_score"]].head(20))

        # Pick top 3 for tuning
        top3 = report_df["model"].head(3).tolist()
        print(f"\nTop-3 for tuning: {top3}")

        # ---------------------------
        # B) Fine-tune top-3 models
        # ---------------------------
        tuned_results = []
        tuned_models = {}

        for name in top3:
            base_model = models[name]
            if name not in spaces:
                print(f"[WARN] No tuning space for {name}, skipping tuning.")
                continue

            print(f"\n=== TUNING: {name} ===")
            with mlflow.start_run(run_name=f"tune_{name}", nested=True):
                mlflow.log_param("model_name", name)
                mlflow.log_param("tune_iter", args.tune_iter)
                mlflow.log_param("tune_cv", args.tune_cv)

                # RandomizedSearchCV uses its own CV; we score using our competition metric via a custom scorer
                # We wrap by scoring on full multioutput after prediction.
                from sklearn.base import clone

                def scorer(estimator, Xva, Yva):
                    pred = estimator.predict(Xva)
                    return competition_metric(Yva, pred, target_names=target_names)

                search = RandomizedSearchCV(
                    estimator=clone(base_model),
                    param_distributions=spaces[name],
                    n_iter=args.tune_iter,
                    cv=args.tune_cv,
                    random_state=args.seed,
                    n_jobs=-1,
                    verbose=1,
                    scoring=scorer,
                    refit=True,
                )
                search.fit(X, Y)

                best = search.best_estimator_
                mlflow.log_metric("best_cv_score_internal", float(search.best_score_))
                mlflow.log_params({f"best__{k}": v for k, v in search.best_params_.items()})

                # Now evaluate best estimator with your main CV scheme (apples-to-apples with base ranking)
                score, fold_scores, oof = oof_cv_score(
                    model=best,
                    X=X,
                    Y=Y,
                    n_splits=args.n_splits,
                    seed=args.seed,
                    target_names=target_names,
                )
                mlflow.log_metric("cv_score", score)
                for i, fs in enumerate(fold_scores, start=1):
                    mlflow.log_metric(f"fold_{i}_score", fs)

                tuned_results.append({
                    "model": name,
                    "stage": "tuned",
                    "cv_score": score,
                    "fold_scores": fold_scores,
                    "best_params": search.best_params_,
                })
                tuned_models[name] = best
                oof_store[f"{name}__tuned"] = oof

                # Save artifact
                np.save(outdir / f"oof_{name}_tuned.npy", oof)
                mlflow.log_artifact(str(outdir / f"oof_{name}_tuned.npy"))

                # Persist best estimator via mlflow sklearn
                try:
                    import mlflow.sklearn
                    mlflow.sklearn.log_model(best, artifact_path=f"model_{name}_tuned")
                except Exception:
                    pass

        tuned_df = pd.DataFrame(tuned_results).sort_values("cv_score", ascending=False).reset_index(drop=True)
        tuned_path = outdir / "report_tuned_top3.csv"
        tuned_df.to_csv(tuned_path, index=False)
        mlflow.log_artifact(str(tuned_path))

        print("\n\n===== TUNED TOP-3 RANKING =====")
        if len(tuned_df) > 0:
            print(tuned_df[["model", "cv_score"]])
        else:
            print("[WARN] No tuned models produced. (Missing spaces or installs)")

        # ---------------------------
        # C) Ensemble (top-3 tuned if available else base)
        # ---------------------------
        # Choose 3 models for ensemble: prefer tuned models if they exist
        ensemble_names = []
        ensemble_oofs = []
        ensemble_scores = []

        # Start from tuned ranking if available
        if len(tuned_df) >= 3:
            chosen = tuned_df["model"].head(3).tolist()
            for name in chosen:
                ensemble_names.append(f"{name}__tuned")
                ensemble_oofs.append(oof_store[f"{name}__tuned"])
                ensemble_scores.append(float(tuned_df.loc[tuned_df["model"] == name, "cv_score"].values[0]))
        else:
            chosen = report_df["model"].head(3).tolist()
            for name in chosen:
                ensemble_names.append(f"{name}__base")
                ensemble_oofs.append(oof_store[name])
                ensemble_scores.append(float(report_df.loc[report_df["model"] == name, "cv_score"].values[0]))

        ens_oof, ens_weights = weighted_ensemble(ensemble_oofs, ensemble_scores)
        ens_score = competition_metric(Y, ens_oof, target_names=target_names)

        mlflow.log_metric("ensemble_top3_cv_score", ens_score)
        mlflow.log_param("ensemble_members", json.dumps(ensemble_names))
        mlflow.log_param("ensemble_weights", json.dumps(ens_weights))

        np.save(outdir / "oof_ensemble_top3.npy", ens_oof)
        mlflow.log_artifact(str(outdir / "oof_ensemble_top3.npy"))

        print("\n\n===== ENSEMBLE TOP-3 =====")
        print("members:", ensemble_names)
        print("weights:", ens_weights)
        print("ensemble_cv_score:", ens_score)

        # ---------------------------
        # D) Optional: fit tuned models on full train and predict test
        # ---------------------------
        if test_df is not None and X_test is not None:
            pred_dir = outdir / "test_predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)

            # Fit selected models for final predictions
            final_models = []
            final_weights = []
            final_preds = []

            # Prefer tuned if available
            if len(tuned_models) > 0:
                ranked_for_test = tuned_df["model"].head(3).tolist() if len(tuned_df) > 0 else []
                for name in ranked_for_test:
                    if name in tuned_models:
                        final_models.append((name, tuned_models[name]))
                        # weight by tuned score
                        final_weights.append(float(tuned_df.loc[tuned_df["model"] == name, "cv_score"].values[0]))
            else:
                ranked_for_test = report_df["model"].head(3).tolist()
                for name in ranked_for_test:
                    final_models.append((name, models[name]))
                    final_weights.append(float(report_df.loc[report_df["model"] == name, "cv_score"].values[0]))

            # Predict each
            for name, m in final_models:
                p = fit_full_and_predict(m, X, Y, X_test)
                final_preds.append(p)

                # save
                np.save(pred_dir / f"test_pred_{name}.npy", p)
                mlflow.log_artifact(str(pred_dir / f"test_pred_{name}.npy"))

            # Ensemble on test
            test_ens, w = weighted_ensemble(final_preds, final_weights)
            np.save(pred_dir / "test_pred_ensemble_top3.npy", test_ens)
            mlflow.log_artifact(str(pred_dir / "test_pred_ensemble_top3.npy"))
            mlflow.log_param("test_ensemble_weights", json.dumps(w))

            # If you want a CSV submission:
            # - If the competition expects multiple columns for targets, write them
            if args.id_col and args.id_col in test_df.columns:
                sub = pd.DataFrame({args.id_col: test_df[args.id_col].values})
            else:
                sub = pd.DataFrame({"row_id": np.arange(len(test_df))})

            for i, tname in enumerate(target_names):
                sub[tname] = test_ens[:, i]

            sub_path = outdir / "submission_ensemble_top3.csv"
            sub.to_csv(sub_path, index=False)
            mlflow.log_artifact(str(sub_path))

            print(f"\nSaved submission: {sub_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
