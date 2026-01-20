# scripts/models/dinov3_multi_reg.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModel
import timm

@dataclass
class Dinov3Config:
    model_id: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m"  # HF (may be gated)
    timm_id: str = "vit_huge_plus_patch16_dinov3.lvd1689m"           # fallback timm
    patch_size: int = 16

    # tiling setup: 2x4 = 8 tiles
    tile_rows: int = 2
    tile_cols: int = 4
    tile_size: int = 512   # S (must be divisible by patch_size)

class LocalMambaBlock(nn.Module):
    def __init__(self, dim: int, kernal_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(
            dim, dim,
            kernel_size=kernal_size,
            padding=kernal_size // 2,
            groups=dim  # depthwise conv
        )
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        shortcut = x
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))  # (B,T,D)
        x = x * g

        x = x.transpose(1, 2)            # (B,D,T)
        x = self.dwconv(x)               # (B,D,T)
        x = x.transpose(1, 2)            # (B,T,D)

        x = self.proj(x)                 # (B,T,D)
        x = self.dropout(x)
        return shortcut + x

class Dinov3MultiReg(nn.Module):
    """
    Input:
      img_full: (B, 3, H, W) where:
        H = tile_rows * tile_size
        W = tile_cols * tile_size
      Example for 2x4 tiles: H=2S, W=4S

    Internally:
      tiles: (B, N, 3, S, S) where N=tile_rows*tile_cols
      encode tiles in one shot: (B*N, T, D)
      reshape + concat tokens: (B, N*T, D)
      fusion -> pool -> heads -> (B,5)
    """
    def __init__(self, cfg: Dinov3Config):
        super().__init__()
        self.cfg = cfg

        self.backend = None
        self.encoder = None

        # -------- try HF first --------
        try:
            self.encoder = AutoModel.from_pretrained(cfg.model_id)
            self.backend = "hf"
        except Exception as e:
            print(f"[WARN] HF load failed for {cfg.model_id} ({type(e).__name__}: {e})")
            print(f"[WARN] Falling back to timm backbone: {cfg.timm_id}")
            self.encoder = timm.create_model(cfg.timm_id, pretrained=True)
            self.backend = "timm"

        # Freeze backbone
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # Figure out D
        D = getattr(self.encoder, "num_features", None)
        if D is None:
            if self.backend == "hf":
                D = self.encoder.config.hidden_size
            else:
                # timm usually has num_features
                D = getattr(self.encoder, "num_features", None)
                if D is None:
                    raise RuntimeError("Could not infer hidden size D from timm model.")
        self.num_features = D

        print(
            f"Encoder trainable params: {sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)}"
            f" | backend: {self.backend} | D: {self.num_features}"
        )

        self.fusion = nn.Sequential(
            LocalMambaBlock(self.num_features, kernal_size=5, dropout=0.1),
            LocalMambaBlock(self.num_features, kernal_size=5, dropout=0.1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head_green = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features // 2, 1),
            nn.Softplus(),
        )
        self.head_dead = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features // 2, 1),
            nn.Softplus(),
        )
        self.head_clover = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features // 2, 1),
            nn.Softplus(),
        )

    @torch.no_grad()
    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,S,S)
        returns tokens: (B,T,D)
        """
        if self.backend == "hf":
            out = self.encoder(pixel_values=x)
            return out.last_hidden_state  # (B,T,D)

        # timm path: prefer forward_features
        if hasattr(self.encoder, "forward_features"):
            feats = self.encoder.forward_features(x)
        else:
            feats = self.encoder(x)

        # Many timm ViTs return (B,T,D). Some return (B,D).
        if feats.ndim == 3:
            return feats
        if feats.ndim == 2:
            # fallback: treat as pooled embedding, make it look like one token
            return feats.unsqueeze(1)  # (B,1,D)
        raise RuntimeError(f"Unexpected timm features shape: {feats.shape}")

    def _tile(self, img_full: torch.Tensor) -> torch.Tensor:
        """
        img_full: (B,3,H,W) where H=R*S, W=C*S
        returns tiles: (B,N,3,S,S)
        """
        B, C, H, W = img_full.shape
        R, Cc, S = self.cfg.tile_rows, self.cfg.tile_cols, self.cfg.tile_size

        expected_h = R * S
        expected_w = Cc * S
        if H != expected_h or W != expected_w:
            raise AssertionError(
                f"Expected img_full H,W=({expected_h},{expected_w}) "
                f"for tile_rows={R}, tile_cols={Cc}, tile_size={S}, "
                f"but got ({H},{W}). Ensure your dataset resize matches."
            )

        # reshape to grid then flatten tiles
        # (B,3,R,S,Cc,S) -> (B,R,Cc,3,S,S) -> (B,N,3,S,S)
        x = img_full.view(B, 3, R, S, Cc, S)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        tiles = x.view(B, R * Cc, 3, S, S)
        return tiles

    def forward(self, img_full: torch.Tensor) -> torch.Tensor:
        """
        img_full: (B,3,2S,4S) for 2x4 tiles
        """
        # ensure backbone stays deterministic even when model.train()
        self.encoder.eval()

        tiles = self._tile(img_full)  # (B,N,3,S,S)
        B, N, C, S, S2 = tiles.shape  # S2 == S

        # Encode all tiles together
        tiles_flat = tiles.view(B * N, 3, S, S)         # (B*N,3,S,S)
        tok = self._encode_tokens(tiles_flat)           # (B*N,T,D)

        # reshape back + concat tokens across tiles
        T = tok.shape[1]
        tok = tok.view(B, N * T, self.num_features)     # (B, N*T, D)

        # fuse + pool
        tok = self.fusion(tok)                          # (B, N*T, D)
        pooled = self.pool(tok.transpose(1, 2)).flatten(1)  # (B,D)

        green = self.head_green(pooled)   # (B,1)
        dead = self.head_dead(pooled)     # (B,1)
        clover = self.head_clover(pooled) # (B,1)

        gdm = green + clover
        total = gdm + dead
        return torch.cat([green, dead, clover, gdm, total], dim=1)  # (B,5)
