import torch

def competition_metric_for_pred3(preds_3: torch.Tensor, targets_5: torch.Tensor) -> float:
    """
    Kaggle-style CSIRO metric: single globally weighted R^2
    over all (sample, target) pairs.

    preds_3: (N, 3) [Dry_Total_g, GDM_g, Dry_Green_g]
    targets_5: (N, 5) [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
    """

    preds_3 = preds_3.detach().cpu().float()
    targets_5 = targets_5.detach().cpu().float()

    # unpack base predictions
    pred_total = preds_3[:, 0]   # Dry_Total_g
    pred_gdm   = preds_3[:, 1]   # GDM_g
    pred_green = preds_3[:, 2]   # Dry_Green_g

    # reconstruct remaining components
    pred_clover = pred_gdm - pred_green      # Dry_Clover_g
    pred_dead   = pred_total - pred_gdm      # Dry_Dead_g

    # stack predictions to match targets_5 order
    preds_5 = torch.stack(
        [pred_green, pred_dead, pred_clover, pred_gdm, pred_total], dim=1
    )  # (N, 5)

    # weights per target (broadcast to shape (1,5))
    w = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], dtype=torch.float32).view(1, 5)

    # global weighted mean over all (i, j)
    N = targets_5.shape[0]
    y_true = targets_5
    y_pred = preds_5

    w_sum = w.sum()
    y_mean = (w * y_true).sum() / (w_sum * N)

    ss_res = (w * (y_true - y_pred) ** 2).sum()
    ss_tot = (w * (y_true - y_mean) ** 2).sum() + 1e-9

    score = (1.0 - ss_res / ss_tot).item()
    return score


def competition_metric(preds_5: torch.Tensor, targets_5: torch.Tensor) -> float:
    """
    CSIRO competition metric: weighted R^2 over 5 biomass targets.

    preds_5, targets_5: (N, 5) in order
      [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
    """
    preds_5   = preds_5.detach().cpu().float()
    targets_5 = targets_5.detach().cpu().float()

    # weights per target (broadcastable to (N,5))
    w = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], dtype=torch.float32).view(1, 5)
    w_sum = w.sum()

    N = targets_5.shape[0]

    y_true = targets_5
    y_pred = preds_5

    # global weighted mean of y_true over all samples & targets
    y_mean = (w * y_true).sum() / (w_sum * N)

    ss_res = (w * (y_true - y_pred) ** 2).sum()
    ss_tot = (w * (y_true - y_mean) ** 2).sum() + 1e-9

    score = 1.0 - ss_res / ss_tot
    return score.item()