import torch
import torch.nn as nn

class WeightedRMSELoss(nn.Module):
    """
    preds, targets: (batch_size, n_targets)
    n_targets should be 5 for this competition:
        0: Dry_Green_g
        1: Dry_Dead_g
        2: Dry_Clover_g
        3: GDM_g
        4: Dry_Total_g
    """
    def __init__(self, weights, eps: float = 1e-8):
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float32)
        w = w / w.sum()    # Normalizing the weights
        self.register_buffer("weights", w)
        self.eps = eps

    def forward(self, preds, targets):
        # squared error per sample, per target
        se = (preds - targets) ** 2             # (B, C)

        # weighted MSE per sample
        weighted_se = se * self.weights       # (B, C)
        weighted_mse_per_sample = weighted_se.sum(dim=1)   # (B,)
        return weighted_mse_per_sample.mean()
