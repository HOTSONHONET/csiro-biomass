import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from dataclasses import dataclass

@dataclass
class Dinov3Config:
    model_id: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
    timm_model_id: str = "vit_huge_plus_patch16_dinov3.lvd1689m"
    patch_size: int = 16


class LocalMambaBlock(nn.Module):
    def __init__(self, dim: int, kernal_size: int = 5, dropout: float = 0.1):
        """
        dim = D (hidden size / feature dimension)
        Input/Output: (B, L, D)
          - B = batch
          - L = token length (sequence length)
          - D = embedding dimension
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Conv1d expects (B, C, L). We'll transpose to make C=D and L=token_length.
        # groups=dim => depthwise conv: each channel has its own 1D kernel
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernal_size,
            padding=kernal_size // 2,
            groups=dim
        )

        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        returns: (B, L, D)
        """
        shortcut = x  # (B, L, D)

        x = self.norm(x)  # (B, L, D)

        g = torch.sigmoid(self.gate(x))  # (B, L, D) gate values in [0,1]

        x = x * g  # (B, L, D) gated token features

        # For Conv1d: (B, C, L) where C=D and L=token_length
        x = x.transpose(1, 2)  # (B, D, L)

        x = self.dwconv(x)  # (B, D, L) depthwise local mixing along token axis

        x = x.transpose(1, 2)  # (B, L, D)

        x = self.proj(x)  # (B, L, D) feature/channel mixing

        x = self.dropout(x)  # (B, L, D)

        return shortcut + x  # (B, L, D) residual


class Dinov3MultiReg(nn.Module):
    def __init__(self, cfg: Dinov3Config):
        super().__init__()

        self.use_timm = False  # flag to control forward()

        # 1) Try Hugging Face backbone
        try:
            self.encoder = AutoModel.from_pretrained(cfg.model_id)
            self.num_features = getattr(self.encoder, "num_features", None)
            if self.num_features is None:
                self.num_features = self.encoder.config.hidden_size  # D
        except Exception as e:
            print(f"[WARN] HF load failed for {cfg.model_id} ({type(e).__name__}: {e})")
            print("[WARN] Falling back to timm backbone: vit_huge_plus_patch16_dinov3.lvd1689m")

            # 2) Fallback: timm backbone
            self.use_timm = True
            self.encoder = timm.create_model(
                cfg.timm_model_id,
                pretrained=True,
                num_classes=0,
                global_pool="",   # IMPORTANT: keep token output
            )
            self.num_features = self.encoder.num_features  # D

        # Freeze backbone weights
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Keep backbone in eval mode
        self.encoder.eval()

        print(
            "Encoder trainable params:",
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
            "| backend:",
            "timm" if self.use_timm else "hf",
            "| D:",
            self.num_features,
        )

        # Token fusion
        self.fusion = nn.Sequential(
            LocalMambaBlock(self.num_features, kernal_size=5, dropout=0.1),
            LocalMambaBlock(self.num_features, kernal_size=5, dropout=0.1),
        )

        # AdaptiveAvgPool1d expects (B, C, L) and returns (B, C, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Heads: map pooled D -> 1
        self.head_green = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),  # (B, D) -> (B, D/2)
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features // 2, 1),                  # (B, D/2) -> (B, 1)
            nn.Softplus()
        )
        self.head_dead = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features // 2, 1),
            nn.Softplus()
        )
        self.head_clover = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features // 2, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        """
        x must be a tuple: (left, right)

        left/right: (B, 3, H, W)   (pixel_values)
        returns:    (B, 5)
        """
        if not isinstance(x, tuple) or len(x) != 2:
            raise ValueError("Input should be (left_half, right_half)")

        left, right = x  # each: (B, 3, H, W)

        # Transformer outputs:
        # last_hidden_state: (B, T, D)
        #   - T = number of tokens = 1 (CLS) + num_patches (+ maybe register tokens)
        #   - D = hidden size
        x_l = self.encoder(pixel_values=left).last_hidden_state   # (B, T, D)
        x_r = self.encoder(pixel_values=right).last_hidden_state  # (B, T, D)

        # Concatenate tokens from left and right along token axis
        x_cat = torch.cat([x_l, x_r], dim=1)  # (B, 2T, D)

        # Token fusion / local mixing
        x_fused = self.fusion(x_cat)  # (B, 2T, D)

        # Pool over token dimension to get a single vector per sample
        # pool expects (B, C, L), so transpose: (B, D, 2T)
        x_pool = self.pool(x_fused.transpose(1, 2))  # (B, D, 1)
        x_pool = x_pool.flatten(1)                   # (B, D)

        # Heads
        green  = self.head_green(x_pool)   # (B, 1)
        dead   = self.head_dead(x_pool)    # (B, 1)
        clover = self.head_clover(x_pool)  # (B, 1)

        # Derived targets
        gdm = green + clover               # (B, 1)
        total = gdm + dead                 # (B, 1)

        # Final output
        return torch.cat([green, dead, clover, gdm, total], dim=1)  # (B, 5)
