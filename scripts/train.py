import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from tqdm import tqdm
import json
import inspect
import mlflow
import mlflow.pytorch
from metrics import competition_metric
from data_split import make_multitarget_strata
from warnings import filterwarnings 
import random
import numpy as np
from datetime import datetime
from loss_functions import WeightedRMSELoss
from utils import log_model_results
from mlflow.models import infer_signature
from models.timm_model.model import TimmModel
from models.vit_0.model import VitTransformer

filterwarnings("ignore")


class Config:
    IMG_SIZE = 1000
    PATCH = 16
    EMBED_DIM = 256
    HEADS = 4
    DEPTH = 1 # num of transformer blocks

    N_SPLITS = 5
    EPOCHS = 50
    BATCH = 4
    LR = 1e-4

    NUM_WORKERS = 16
    VAL_WORKERS = 16

    TRAIN_CSV = "dataset/train_df.csv"
    IMG_DIR = "."
    SEED = 42

    NUM_CLASSES = 3
    IN_CHANNELS = 3
    GRID = (2, 2)
    DROPOUT = 0.2

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    MODEL_MAP = dict(
        vit_scratch = VitTransformer,
        timm_model = TimmModel,
    )
    MODEL = "timm_model"
    TIMM_BASE_MODEL = 'vit_base_patch16_dinov3.lvd1689m'

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

class CSIRODataset(Dataset):
    """
    Returns:
        image: Tensor (C, H, W)
        targets: Tensor (3,) -> ["Dry_Total_g", "GDM_g", "Dry_Green_g"]
    """
    def __init__(self, df, img_dir=".", is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.is_train = is_train

        # order must match model outputs
        self.base_cols = ["Dry_Total_g", "GDM_g", "Dry_Green_g"] # Ordering is important

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    antialias=True
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    antialias=True
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])

    def __len__(self):
        return len(self.df)

    def get_n_patches(self, image, rows, cols):
        width, height = image.size

        tile_width = width // cols
        tile_height = height // rows

        patches = []
        for r in range(rows):
            for c in range(cols):
                x1 = c * tile_width
                y1 = r * tile_height

                x2 = x1 + tile_width
                y2 = y1 + tile_height

                patch = image.crop(tuple(map(int, (x1, y1, x2, y2))))
                patches.append(patch)
        return patches

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Force numeric conversion here
        vals = row[self.base_cols].to_numpy(dtype="float32", na_value=0.0)
        targets = torch.from_numpy(vals)  # (3,)

        img_path = os.path.join(self.img_dir, row["image_path"])

        image = Image.open(img_path).convert("RGB")

        # converting the image into patches
        # left, right = self.get_n_patches(image, rows=1, cols = 2)
        w, h = image.size
        mid_w = w // 2
        left = image.crop((0, 0, mid_w, h))
        right = image.crop((mid_w, 0, w, h))

        left_t = self.transform(left)
        right_t = self.transform(right)

        return left_t, right_t, targets

class Runner:
    def __init__(self):
        # Setting seed for the run
        set_seed(seed = Config.SEED)

        # Setting up experiment directory
        parent_dir = "exps"
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.run_name = f"CSIRO-Biomass-ViT-{ts}"
        self.exp_dir = os.path.join(parent_dir, self.run_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # Initiailizing configs
        self.df = pd.read_csv(Config.TRAIN_CSV)
        self.target_cols = [
            "Dry_Green_g",
            "Dry_Dead_g",
            "Dry_Clover_g",
            "GDM_g",
            "Dry_Total_g",
        ]
        for c in self.target_cols:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        self.df["strata"] = make_multitarget_strata(
            self.df,
            cols=["Dry_Green_g", "Dry_Dead_g", "Dry_Total_g"],
            n_bins=4,
            n_splits=Config.N_SPLITS,
        )

    def run(self):

        mlflow.set_experiment(f"CSIRO-Biomass-ViT")

        with mlflow.start_run(run_name=self.run_name) as parent_run:
            parent_run_id = parent_run.info.run_id

            mlflow.log_params(Config.to_dict())
            
            config_path = os.path.join(self.exp_dir, "config_dump.json")
            with open(config_path, "w") as f: json.dump(Config.to_dict(), f, indent=4)
            mlflow.log_artifact(config_path)

            script_path = inspect.getfile(Runner)
            mlflow.log_artifact(script_path)

            skf = StratifiedKFold(
                n_splits=Config.N_SPLITS,
                shuffle=True,
                random_state=Config.SEED,
            )

            for fold, (tr, va) in enumerate(skf.split(self.df, self.df["strata"])):
                print(f"\n========== Fold {fold+1}/{Config.N_SPLITS} ==========")

                train_df = self.df.iloc[tr].reset_index(drop=True)
                val_df   = self.df.iloc[va].reset_index(drop=True)

                with mlflow.start_run(
                    run_name = f"{self.run_name}_fold{fold + 1}",
                    nested = True,
                ) as fold_run:
                    
                    fold_run_id = fold_run.info.run_id
                    print(f"    Fold run_id: {fold_run_id}")
                    mlflow.log_param("fold", fold + 1)

                    train_ds = CSIRODataset(train_df, Config.IMG_DIR, True)
                    val_ds   = CSIRODataset(val_df, Config.IMG_DIR, False)

                    train_loader = DataLoader(
                        train_ds,
                        batch_size=Config.BATCH,
                        shuffle=True,
                        num_workers=Config.NUM_WORKERS,
                        pin_memory=True,
                        prefetch_factor=2,
                        persistent_workers=True,
                    )

                    val_loader = DataLoader(
                        val_ds,
                        batch_size=Config.BATCH,
                        shuffle=False,
                        num_workers=Config.VAL_WORKERS,
                        pin_memory=True,
                    )

                    model = Config.MODEL_MAP[Config.MODEL](Config).to(Config.DEVICE, )
                    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

                    loss_weights = [0.5, 0.2, 0.1]
                    loss_fn = WeightedRMSELoss(weights=loss_weights).to(Config.DEVICE)
                    scaler = torch.amp.GradScaler(device = Config.DEVICE)

                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=5
                    )
                    best_comp_score = float('-inf')
                    best_ckpt_path = os.path.join(self.exp_dir, f"best_model_fold{fold+1}.pth")

                    # ================= TRAINING ==================
                    for epoch in range(Config.EPOCHS):
                        model.train()
                        total_loss = 0.0

                        for (left, right, targets) in tqdm(
                            train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", ncols=100
                        ):
                            left = left.to(Config.DEVICE, non_blocking=True)
                            right = right.to(Config.DEVICE, non_blocking=True)
                            targets = targets.to(Config.DEVICE, non_blocking=True)

                            optimizer.zero_grad()

                            with torch.amp.autocast(device_type = Config.DEVICE):
                                preds = model(left, right)  # (B, 3)
                                loss = loss_fn(preds, targets)

                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                            total_loss += loss.item()

                        train_loss_mean = total_loss / len(train_loader)
                        mlflow.log_metric("train_loss_mean", train_loss_mean, step=epoch)
                        print(f"train_loss_mean (3 targets): {train_loss_mean:.4f}")

                        # ================= VALIDATION ==================
                        model.eval()
                        vloss = 0.0
                        all_val_preds = []
                        with torch.no_grad():
                            for (left, right, targets) in tqdm(val_loader, desc=f"eval {epoch+1}", ncols=100):
                                left = left.to(Config.DEVICE, non_blocking=True)
                                right = right.to(Config.DEVICE, non_blocking=True)
                                targets = targets.to(Config.DEVICE, non_blocking=True)

                                preds = model(left, right)  # (B, 3)
                                all_val_preds.append(preds.detach().cpu())
                                vloss += loss_fn(preds, targets).item()

                        val_loss_mean = vloss / len(val_loader)
                        mlflow.log_metric("val_loss_mean", val_loss_mean, step=epoch)
                        print(f"val_loss_mean (3 targets): {val_loss_mean:.4f}")

                        # ===== COMPETITION METRIC =====
                        preds_3 = torch.cat(all_val_preds, dim=0)

                        # GT for all 5 targets in this fold, same row order as val_df
                        targets_5 = torch.tensor(
                            val_df[self.target_cols].values, dtype=torch.float32
                        )

                        val_comp_score = competition_metric(preds_3, targets_5)
                        mlflow.log_metric("val_competition_metric", val_comp_score, step=epoch)
                        print(f"Val competition metric: {val_comp_score:.4f}")

                        # ===== LR Decay + ModelCheckpointing =====
                        current_lr = optimizer.param_groups[0]['lr']
                        mlflow.log_metric("lr", current_lr, step=epoch)
                        scheduler.step(val_loss_mean)

                        # Checkpointing: if best comp score so far, save
                        if val_comp_score > best_comp_score:
                            best_comp_score = val_comp_score
                            print(f" >> New best comp metric {best_comp_score:.4f}, saving checkpoint to {best_ckpt_path}")
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'val_competition_metric': best_comp_score,
                            }, best_ckpt_path)

                            mlflow.log_artifact(best_ckpt_path, artifact_path=f"best_checkpoints_fold{fold+1}")

                    # ===== Training + Validation results =====
                    model.eval()
                    vloss = 0.0
                    all_val_preds = []
                    all_train_preds = []
                    with torch.no_grad():
                    
                        for (left, right, _) in train_loader:
                            left = left.to(Config.DEVICE, non_blocking=True)
                            right = right.to(Config.DEVICE, non_blocking=True)
                            preds = model(left, right)
                            all_train_preds.append(preds.detach().cpu())

                        for (left, right, _) in val_loader:
                            left = left.to(Config.DEVICE, non_blocking=True)
                            right = right.to(Config.DEVICE, non_blocking=True)
                            preds = model(left, right) 
                            all_val_preds.append(preds.detach().cpu())

                    train_preds = torch.cat(all_train_preds, dim = 0).numpy()
                    val_preds = torch.cat(all_val_preds, dim = 0).numpy()

                    log_model_results(
                        mlflow_instance=mlflow,
                        train_preds = train_preds,
                        val_preds = val_preds,
                        target_cols = ["Dry_Total_g", "GDM_g", "Dry_Green_g"],
                        train_df = train_df,
                        val_df = val_df,
                        exp_dir = self.exp_dir,
                        fold = fold,
                    )

                    # Save model weights as artifact
                    save_path = os.path.join(self.exp_dir, f"fold_{fold+1}.pth")
                    torch.save(model.state_dict(), save_path)

                    # Save as MLflow PyTorch model

                    model.eval()

                    left_batch, right_batch, _ = next(iter(val_loader))
                    left_batch = left_batch.to(Config.DEVICE)
                    right_batch = right_batch.to(Config.DEVICE)

                    with torch.no_grad():
                        example_output = model(left_batch, right_batch)

                    example_input = {
                        "left": left_batch[:1].detach().cpu().numpy(),
                        "right": right_batch[:1].detach().cpu().numpy()
                    }
                    example_output_np = example_output[:1].detach().cpu().numpy()

                    signature = infer_signature(
                        model_input=example_input, 
                        model_output=example_output_np
                    )
                    mlflow.pytorch.log_model(
                        pytorch_model= model, 
                        artifact_path = f"fold_{fold+1}",
                        input_example = example_input,
                        signature = signature
                    )


if __name__ == "__main__":
    Runner().run()
