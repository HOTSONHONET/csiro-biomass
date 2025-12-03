import torch.nn as nn
import torch
import numpy as np


class PatchEmbedding(nn.Module):
    def __init__(self, num_filters, image_size, patch_size, in_channels):
        super().__init__()
        self.num_filters = num_filters
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.linear_project = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.num_filters,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        x = self.linear_project(x)
        x = x.flatten(2)  # flatten patches
        x = x.transpose(1, 2)  # (B, n_patches, num_filters)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, num_filters, max_seq_len):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, num_filters))

        pe = torch.zeros(max_seq_len, num_filters)

        for pos in range(max_seq_len):
            for i in range(num_filters):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos / 1000 ** (2*i / num_filters))
                else:
                    pe[pos][i] = np.cos(pos / 1000 ** ((2*i + 1) / num_filters))

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        num_batches = x.size(0)

        cls_batch = self.cls_token.expand(num_batches, -1, -1)
        x = torch.cat((cls_batch, x), dim=1)  # prepend cls token

        x = x + self.pe[:, : x.size(1), :]
        return x


class AttentionHead(nn.Module):
    def __init__(self, num_filters, head_size):
        super().__init__()

        self.head_size = head_size
        self.query, self.key, self.value = (
            nn.Linear(num_filters, head_size) for _ in range(3)
        )

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = Q @ K.transpose(-2, -1)
        attention /= (self.head_size ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        attention = attention @ V
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, num_filters, n_heads):
        super().__init__()

        self.head_size = num_filters // n_heads
        self.W_o = nn.Linear(num_filters, num_filters)

        self.heads = nn.ModuleList(
            [
                AttentionHead(num_filters=num_filters, head_size=self.head_size)
                for _ in range(n_heads)
            ]
        )

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.W_o(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_filters, n_heads, r_mlp=4):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(num_filters)
        self.mha = MultiHeadAttention(num_filters=num_filters, n_heads=n_heads)

        self.layer_norm2 = nn.LayerNorm(num_filters)
        self.mlp = nn.Sequential(
            nn.Linear(num_filters, num_filters * r_mlp),
            nn.GELU(),
            nn.Linear(num_filters * r_mlp, num_filters),
        )

    def forward(self, x):
        out = x + self.mha(self.layer_norm1(x))
        out = out + self.mlp(self.layer_norm2(out))
        return out


class VitTransformer(nn.Module):
    """
    ViT that predicts 3 base biomass components:

    Order of outputs:
        0: Dry_Clover_g
        1: Dry_Dead_g
        2: Dry_Green_g
    """

    def __init__(self, cfg):
        super().__init__()

        self.num_filters = cfg.EMBED_DIM
        self.n_classes = cfg.NUM_CLASSES         # 3 targets
        self.image_size = (cfg.IMG_SIZE, cfg.IMG_SIZE)
        self.patch_size = (cfg.PATCH, cfg.PATCH)
        self.in_channels = cfg.IN_CHANNELS
        self.n_heads = cfg.HEADS
        self.n_layers = getattr(cfg, "DEPTH", 6)

        assert (
            self.image_size[0] % self.patch_size[0] == 0
            and self.image_size[1] % self.patch_size[1] == 0
        ), "Image size must be divisible by patch size"

        patch_area = self.patch_size[0] * self.patch_size[1]
        image_area = self.image_size[0] * self.image_size[1]

        self.n_patches = image_area // patch_area
        self.max_seq_len = self.n_patches + 1  # +1 for CLS token

        self.patch_embedding = PatchEmbedding(
            num_filters=self.num_filters,
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
        )

        self.positional_encoding = PositionalEncoding(
            num_filters=self.num_filters,
            max_seq_len=self.max_seq_len,
        )

        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoder(
                    num_filters=self.num_filters,
                    n_heads=self.n_heads,
                )
                for _ in range(self.n_layers)
            ]
        )

        # Regressor head -> 3 outputs (Clover, Dead, Green)
        self.regressor = nn.Linear(self.num_filters, self.n_classes)

    def forward(self, images):
        # print("images.shape: ", images.shape)
        x = self.patch_embedding(images)      # (B, n_patches, C)
        # print("x.shape: ", x.shape)
        x = self.positional_encoding(x)       # (B, 1 + n_patches, C)
        # print("x.shape: ", x.shape)
        x = self.transformer_encoder(x)       # (B, 1 + n_patches, C)
        # print("x.shape: ", x.shape)
        cls_token = x[:, 0]                   # (B, C)
        # print("cls_token.shape: ", cls_token.shape)
        y = self.regressor(cls_token)         # (B, 3)
        # print("y.shape: ", y.shape)
        return y

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        
        Method to return the full sequence of tokens after the transformer encoder
        Shape: (B, 1 + n_patches, c)
        
        """

        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        return x

    def forward_with_tokens(self, images: torch.Tensor):
        """
        
        Returns:
             preds: (B, 3)
             tokens: (B, 1 + n_patches, C)

        """
        x = self._encode(images)
        cls_token = x[:, 0]
        y = self.regressor(cls_token)
        return y, x

