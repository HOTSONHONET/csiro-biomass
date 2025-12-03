import torch

def competition_metric(preds_3: torch.Tensor, targets_5: torch.Tensor) -> float:
    """
    CSIRO Biomass competition metric: weighted R^2 over 5 targets.

    Args
    ----
    preds_3 : (N, 3) tensor
        Order MUST be: [Dry_Total_g, GDM_g, Dry_Green_g]
        (this matches your training targets + loss_weights)

    targets_5 : (N, 5) tensor
        Order: [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]

    Returns
    -------
    float
        Weighted R^2 score.
    """

    # Move to CPU and make sure dtype is float32
    preds_3 = preds_3.detach().cpu().float()
    targets_5 = targets_5.detach().cpu().float()

    # unpack base predictions
    pred_total = preds_3[:, 0]   # Dry_Total_g
    pred_gdm   = preds_3[:, 1]   # GDM_g
    pred_green = preds_3[:, 2]   # Dry_Green_g

    # derive the remaining components
    pred_clover = pred_gdm  - pred_green       # Dry_Clover_g = GDM - Green
    pred_dead   = pred_total - pred_gdm        # Dry_Dead_g   = Total - GDM

    # stack predictions to match targets_5 order:
    # [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
    preds_5 = torch.stack(
        [pred_green, pred_dead, pred_clover, pred_gdm, pred_total], dim=1
    )

    # target weights for the 5 components
    weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], dtype=torch.float32)

    eps = 1e-9
    r2_per_target = []

    for j in range(5):
        y_true = targets_5[:, j]
        y_pred = preds_5[:, j]

        y_mean = y_true.mean()
        ss_res = torch.sum((y_pred - y_true) ** 2)
        ss_tot = torch.sum((y_true - y_mean) ** 2) + eps

        r2 = 1.0 - ss_res / ss_tot
        r2_per_target.append(r2)

    r2_per_target = torch.stack(r2_per_target)  # (5,)
    score = torch.sum(weights * r2_per_target).item()
    return score
