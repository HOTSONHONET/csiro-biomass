import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.dinov3_multi_reg import LocalMambaBlock
from PIL import Image, ImageDraw, ImageFont


class DinoV3StructuredConfig:
    def __init__(
        self,
        model_id: str = "vit_huge_plus_patch16_dinov3.lvd1689m",
        img_size: int = 512,
        tiles: int = 2,          # 2 = left/right, 8 = your tile idea
        freeze_backbone: bool = True,
        dropout: float = 0.1,
        use_mamba: bool = True,
        mamba_depth: int = 2,
    ):
        self.timm_id = model_id
        self.img_size = img_size
        self.tiles = tiles
        self.freeze_backbone = freeze_backbone
        self.dropout = dropout
        self.use_mamba = use_mamba
        self.mamba_depth = mamba_depth


class DinoV3Structured(nn.Module):
    """
    Input: full image tensor (B,3,H,W) already resized consistently (e.g. 2S x 4S).
    We tile inside the model into N tiles, encode each tile with frozen timm ViT,
    fuse tile embeddings, then output structured predictions.

    Outputs:
      dict with:
        green_pred, total_pred, dead_ratio_pred, clover_ratio_pred
        dead_pred, gdm_pred, clover_pred
        pred5 in order [Green, Dead, Clover, GDM, Total]
    """
    def __init__(self, cfg: DinoV3StructuredConfig):
        super().__init__()
        self.cfg = cfg

        # timm backbone (offline ok if weights are available; else set pretrained=False)
        self.encoder = timm.create_model(
            cfg.timm_id,
            pretrained=True,
            num_classes=0,            # removes classification head
            global_pool="avg",        # gives (B, D)
        )

        self.D = self.encoder.num_features

        if cfg.freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        # Fusion (tile tokens -> one vector)
        # You can run Mamba blocks on (B, T, D) then pool.
        if cfg.use_mamba:
            blocks = []
            for _ in range(cfg.mamba_depth):
                blocks.append(LocalMambaBlock(self.D, kernal_size=5, dropout=cfg.dropout))
            self.fusion = nn.Sequential(*blocks)
        else:
            self.fusion = nn.Identity()

        self.tile_pool = nn.AdaptiveAvgPool1d(1)  # (B, D, T) -> (B, D, 1)

        # Heads
        # 1) regression heads (green, total) — force non-negative via softplus
        self.head_reg = nn.Sequential(
            nn.LayerNorm(self.D),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.D, self.D // 2),
            nn.GELU(),
            nn.Linear(self.D // 2, 2),  # [green_raw, total_raw]
        )

        # 2) ratio heads (dead_ratio, clover_ratio) — sigmoid
        self.head_ratio = nn.Sequential(
            nn.LayerNorm(self.D),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.D, self.D // 2),
            nn.GELU(),
            nn.Linear(self.D // 2, 2),  # [dead_ratio_logit, clover_ratio_logit]
        )

    # -------------------------
    # Tiling utilities
    # -------------------------
    def _make_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W)
        returns tiles: (B*T, 3, tileH, tileW)
        Supported:
          tiles=2  -> split width into left/right
          tiles=8  -> 2 rows x 4 cols grid
        """
        B, C, H, W = x.shape
        if self.cfg.tiles == 2:
            mid = W // 2
            left = x[:, :, :, :mid]
            right = x[:, :, :, mid:]
            tiles = torch.cat([left, right], dim=0)  # (2B,3,H,W/2)
            return tiles

        if self.cfg.tiles == 8:
            # 2x4 grid
            rows, cols = 2, 4
            th, tw = H // rows, W // cols
            tile_list = []
            for r in range(rows):
                for c in range(cols):
                    tile = x[:, :, r*th:(r+1)*th, c*tw:(c+1)*tw]
                    tile_list.append(tile)
            tiles = torch.cat(tile_list, dim=0)  # (B*8, 3, th, tw)
            return tiles

        raise ValueError(f"Unsupported tiles={self.cfg.tiles}. Use 2 or 8.")

    def _encode_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        """
        tiles: (B*T,3,tH,tW) -> embeddings (B*T,D)
        """
        if self.cfg.freeze_backbone:
            with torch.no_grad():
                z = self.encoder(tiles)  # (B*T, D)
        else:
            z = self.encoder(tiles)
        return z

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x_full: torch.Tensor) -> dict:
        B = x_full.size(0)
        T = self.cfg.tiles

        tiles = self._make_tiles(x_full)          # (B*T,3,th,tw)
        z = self._encode_tiles(tiles)             # (B*T, D)

        # reshape to (B, T, D)
        z = z.view(T, B, self.D).transpose(0, 1)  # (B, T, D)

        # fusion expects (B, T, D) for mamba blocks (you already used that pattern)
        z = self.fusion(z)                        # (B, T, D)

        # pool over T -> (B, D)
        z = z.transpose(1, 2)                     # (B, D, T)
        z = self.tile_pool(z).squeeze(-1)         # (B, D)

        # heads
        reg_raw = self.head_reg(z)                # (B,2)
        ratio_logit = self.head_ratio(z)          # (B,2)

        green = F.softplus(reg_raw[:, 0])         # >=0
        total = F.softplus(reg_raw[:, 1])         # >=0

        # optional: enforce total >= green softly
        # total = green + F.softplus(reg_raw[:, 1])

        dead_ratio = torch.sigmoid(ratio_logit[:, 0])    # (0,1)
        clover_ratio = torch.sigmoid(ratio_logit[:, 1])  # (0,1)

        dead = dead_ratio * total
        gdm = total - dead
        clover = clover_ratio * gdm

        pred5 = torch.stack([green, dead, clover, gdm, total], dim=1)

        return {
            "green_pred": green,
            "total_pred": total,
            "dead_ratio_pred": dead_ratio,
            "clover_ratio_pred": clover_ratio,
            "dead_pred": dead,
            "gdm_pred": gdm,
            "clover_pred": clover,
            "pred5": pred5,
        }
