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

class CSIROMultiTaskLoss(nn.Module):
    """
    Combined loss for:
      - 5 biomass targets (weighted MSE)
      - 2 auxiliary targets: Pre_GSHH_NDVI, Height_Ave_cm

    preds_all, targets_all: (B, 7) in the order:
      [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g,
       Pre_GSHH_NDVI, Height_Ave_cm]
    """
    def __init__(
        self,
        biomass_weights=None,
        aux_ndvi_weight: float = 0.3,
        aux_height_weight: float = 0.3,
    ):
        super().__init__()
        if biomass_weights is None:
            biomass_weights = [0.1, 0.1, 0.1, 0.2, 0.5]

        self.biomass_loss = WeightedRMSELoss(biomass_weights)
        self.aux_ndvi_weight = aux_ndvi_weight
        self.aux_height_weight = aux_height_weight

        # simple unweighted MSE for aux targets
        self.mse = nn.MSELoss()

    def forward(self, preds_all, targets_all):
        # split
        preds_bio   = preds_all[:, :5]
        targets_bio = targets_all[:, :5]

        preds_ndvi   = preds_all[:, 5]
        targets_ndvi = targets_all[:, 5]

        preds_height   = preds_all[:, 6]
        targets_height = targets_all[:, 6]

        # 1) biomass weighted loss
        loss_bio = self.biomass_loss(preds_bio, targets_bio)

        # 2) aux losses
        loss_ndvi   = self.mse(preds_ndvi, targets_ndvi)
        loss_height = self.mse(preds_height, targets_height)

        loss_aux = (
            self.aux_ndvi_weight * loss_ndvi
            + self.aux_height_weight * loss_height
        )

        total_loss = loss_bio + loss_aux

        return total_loss, {
            "loss_biomass": loss_bio.detach().item(),
            "loss_ndvi": loss_ndvi.detach().item(),
            "loss_height": loss_height.detach().item(),
            "loss_aux": loss_aux.detach().item(),
        }
