import timm
import torch.nn as nn
import torch
import torch.nn.functional as F

class TimmModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            model_name = cfg.TIMM_BASE_MODEL,
            pretrained = True,
            num_classes = 0,
            global_pool = "avg",
        )
        self.grid = cfg.GRID
        self.device = cfg.DEVICE
        self.drop_out = cfg.DROPOUT
        self.n_outputs = cfg.N_OUTPUTS
        self.tile_size, self.num_feats = self._get_model_info()
        print(f"[INFO] Using {cfg.TIMM_BASE_MODEL} | tile_size: {self.tile_size} | num_feats: {self.num_feats}")

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.num_feats, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.drop_out),
                nn.Linear(256, 1)
            )
            for _ in range(self.n_outputs)
        ])
        self.softplus = nn.Softplus(beta=1.0)
        self.grid_dim = (2, 2)
        self.query = nn.Parameter(torch.randn(self.num_feats))

        # initializing weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _get_model_info(self):
        # Collecting num_features info
        num_features = self.backbone.num_features

        # Collecting img_size info
        img_size = None
        if hasattr(self.backbone, "patch_embed") and hasattr(self.backbone.patch_embed, "img_size"):
            size = self.backbone.patch_embed.img_size
            img_size = int(size if isinstance(size, (int, float)) else size[0])
        
        if hasattr(self.backbone, "img_size"):
            size = self.backbone.img_size
            img_size = int(size if isinstance(size, (int, float)) else size[0])
        
        dc = getattr(self.backbone, "default_cfg", {}) or {}
        ins = dc.get("input_size", None)
        if ins:
            if isinstance(ins, (tuple, list)) and len(ins) >= 2:
                img_size = int(ins[1])
            else:
                img_size = int(ins if isinstance(ins, (int, float)) else 224)

        name = getattr(self.backbone, "default_cfg", {}).get("architecture", "") or str(type(self.backbone))
        img_size = 518 if ("dinov2" in name.lower()) else 224
        
        return img_size, num_features
    
    
    def _tile_and_embed(self, x):
        """
        x: (B, C, H, W)  — one half-image (left or right)
        returns: (B, T, feat_dim) — embeddings of all tiles
        """
        B, C, H, W = x.shape
        r, c = self.grid
        h_step = H // r
        w_step = W // c

        tile_feats = []
        for i in range(r):
            for j in range(c):
                ys = i * h_step
                ye = (i + 1) * h_step if (i < r - 1) else H
                xs = j * w_step
                xe = (j + 1) * w_step if (j < c - 1) else W
                tile = x[:, :, ys:ye, xs:xe]

                # resize tile if necessary
                if tile.shape[-2:] != (self.tile_size, self.tile_size):
                    tile = F.interpolate(tile, size=(self.tile_size, self.tile_size),
                                         mode="bilinear", align_corners=False)

                feat = self.backbone(tile)  # (B, feat_dim)
                tile_feats.append(feat)

        feats = torch.stack(tile_feats, dim=1)  # (B, T, feat_dim)
        return feats

    def forward(self, left, right):

        feats_l = self._tile_and_embed(left)
        feats_r = self._tile_and_embed(right)


        feats = torch.cat([feats_l, feats_r], dim=1) 

        B, T, D = feats.shape
        q = self.query.unsqueeze(0).unsqueeze(1).expand(B, -1, -1)  # (B,1,D)
        attn_scores = (q * feats).sum(dim=-1, keepdim=True)         # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)           # (B, T, 1)
        pooled = (feats * attn_weights).sum(dim=1)  
        
        outs = [
            self.softplus(
                head(pooled).squeeze(1)
            ) 
            for head in self.heads
        ]
        return torch.stack(outs, dim = 1)

if __name__ == "__main__":
    models = timm.list_models(filter = "*dinov3*")
    print(models)