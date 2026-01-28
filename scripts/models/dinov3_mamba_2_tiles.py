import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.dinov3_multi_reg import LocalMambaBlock


class DinoV3HybridConfig:
    def __init__(
        self,
        model_id="vit_huge_plus_patch16_dinov3.lvd1689m",
        pretrained_backbone=True,
        dropout=0.2,
        mamba_depth=2,
        use_grad_checkpointing=False,
        freeze_backbone=True,
        # blending
        init_mix_logits=(-1.0, -1.0),  # (dead_mix, clover_mix) negative => prefer derived early
    ):
        self.model_id = model_id
        self.pretrained_backbone = pretrained_backbone
        self.dropout = dropout
        self.mamba_depth = mamba_depth
        self.use_grad_checkpointing = use_grad_checkpointing
        self.freeze_backbone = freeze_backbone
        self.init_mix_logits = init_mix_logits


class DinoV3Hybrid(nn.Module):
    """
    Token-level fusion (left/right), separate heads + ratio heads.

    Direct heads:
      - green_raw, gdm_raw, total_raw  (Softplus)
      - optionally dead_raw, clover_raw (Softplus)  [kept here]

    Ratio heads:
      - dead_ratio in (0,1)
      - clover_ratio in (0,1)

    Derived:
      - dead_from_core   = relu(total - gdm)
      - clover_from_core = relu(gdm - green)
      - dead_from_ratio  = dead_ratio * total
      - clover_from_ratio= clover_ratio * gdm

    Final:
      dead   = mix_dead * dead_direct + (1-mix_dead) * dead_from_core   (or ratio)
      clover = mix_clover * clover_direct + (1-mix_clover) * clover_from_core (or ratio)

    Outputs pred5: [Green, Dead, Clover, GDM, Total]
    """
    def __init__(self, cfg: DinoV3HybridConfig):
        super().__init__()
        self.cfg = cfg

        # keep patch tokens: (B, N, D)
        self.backbone = timm.create_model(
            cfg.model_id,
            pretrained=cfg.pretrained_backbone,
            num_classes=0,
            global_pool="",  # IMPORTANT: keep tokens
        )

        if hasattr(self.backbone, "set_grad_checkpointing") and cfg.use_grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)

        self.D = self.backbone.num_features

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # Mamba fusion over token sequence
        blocks = []
        for _ in range(cfg.mamba_depth):
            blocks.append(LocalMambaBlock(self.D, kernal_size=5, dropout=cfg.dropout))
        self.fusion = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # ------- Heads -------
        def reg_head():
            return nn.Sequential(
                nn.LayerNorm(self.D),
                nn.Linear(self.D, self.D // 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(self.D // 2, 1),
            )

        def ratio_head():
            return nn.Sequential(
                nn.LayerNorm(self.D),
                nn.Linear(self.D, self.D // 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(self.D // 2, 1),
            )

        # direct regressors (raw -> softplus)
        self.head_green = reg_head()
        self.head_gdm   = reg_head()
        self.head_total = reg_head()
        self.head_dead  = reg_head()    # direct dead (optional but useful)
        self.head_clover= reg_head()    # direct clover (optional)

        # ratio heads
        self.head_dead_ratio   = ratio_head()
        self.head_clover_ratio = ratio_head()

        # learnable mixing (logits)
        # sigmoid(mix_logit) = weight on direct prediction
        dead_mix0, clover_mix0 = cfg.init_mix_logits
        self.dead_mix_logit   = nn.Parameter(torch.tensor(float(dead_mix0)))
        self.clover_mix_logit = nn.Parameter(torch.tensor(float(clover_mix0)))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> dict:
        # tokens: (B, N, D)
        if self.cfg.freeze_backbone:
            with torch.no_grad():
                x_l = self.backbone(left)
                x_r = self.backbone(right)
        else:
            x_l = self.backbone(left)
            x_r = self.backbone(right)

        # concat token sequences: (B, 2N, D)
        x = torch.cat([x_l, x_r], dim=1)

        # fuse
        x = self.fusion(x)  # (B, 2N, D)

        # pool: (B, D)
        x_pool = self.pool(x.transpose(1, 2)).squeeze(-1)

        # --- direct preds ---
        green = F.softplus(self.head_green(x_pool).squeeze(1))
        gdm_d = F.softplus(self.head_gdm(x_pool).squeeze(1))
        total_d= F.softplus(self.head_total(x_pool).squeeze(1))
        dead_d = F.softplus(self.head_dead(x_pool).squeeze(1))
        clover_d=F.softplus(self.head_clover(x_pool).squeeze(1))

        # --- ratio preds ---
        dead_ratio = torch.sigmoid(self.head_dead_ratio(x_pool).squeeze(1))
        clover_ratio = torch.sigmoid(self.head_clover_ratio(x_pool).squeeze(1))

        # --- derived candidates (two options) ---
        dead_from_core   = F.relu(total_d - gdm_d)     # ensures >=0
        clover_from_core = F.relu(gdm_d - green)       # ensures >=0

        dead_from_ratio  = dead_ratio * total_d
        clover_from_ratio= clover_ratio * gdm_d

        # pick which derived style you want:
        # - core-derived enforces equations and avoids "invisible dead"
        # - ratio-derived keeps your ratio heads meaningful
        # You can also blend core-derived and ratio-derived; keeping it simple:
        dead_derived = 0.5 * dead_from_core + 0.5 * dead_from_ratio
        clover_derived = 0.5 * clover_from_core + 0.5 * clover_from_ratio

        # --- mix direct vs derived ---
        mix_dead = torch.sigmoid(self.dead_mix_logit)       # scalar
        mix_clover = torch.sigmoid(self.clover_mix_logit)   # scalar

        dead = mix_dead * dead_d + (1.0 - mix_dead) * dead_derived
        clover = mix_clover * clover_d + (1.0 - mix_clover) * clover_derived

        # recompose for consistency (optional)
        # If you want strict: force gdm = green + clover, total = gdm + dead
        gdm = green + clover
        total = gdm + dead

        pred5 = torch.stack([green, dead, clover, gdm, total], dim=1)

        return {
            "green_pred": green,
            "gdm_pred": gdm,
            "total_pred": total,
            "dead_pred": dead,
            "clover_pred": clover,
            "dead_ratio_pred": dead_ratio,
            "clover_ratio_pred": clover_ratio,
            "mix_dead": mix_dead.detach(),
            "mix_clover": mix_clover.detach(),
            "pred5": pred5,
        }
