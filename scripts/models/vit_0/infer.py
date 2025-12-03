import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


def generate_biomass_heatmaps(model, pil_img: Image.Image, cfg, device="cuda"):
    """
    Grad-CAM style heatmaps for each of the 3 outputs:

        0: Dry_Clover_g
        1: Dry_Dead_g
        2: Dry_Green_g

    For each output:
        - compute grad of that scalar w.r.t. input image
        - aggregate |grad| over channels
        - normalize to [0,1]
        - blend with a strong color overlay
    """

    model.eval()
    model.to(device)

    vis_transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    overlays = {}
    raw_heat = {}

    # big, visible colors (you can change later)
    base_colors = {
        "Dry_Green_g":  torch.tensor([0.0, 1.0, 0.0], device=device),  # green
        "Dry_Clover_g": torch.tensor([0.7, 0.4, 0.0], device=device),  # brown
        "Dry_Dead_g":   torch.tensor([1.0, 0.0, 0.0], device=device),  # red
    }

    target_indices = {
        "Dry_Clover_g": 0,
        "Dry_Dead_g": 1,
        "Dry_Green_g": 2,
    }

    def make_overlay(base_img_3chw, heat_1hw, color_vec_3):
        """
        base_img_3chw: (3,H,W) in [0,1]
        heat_1hw:      (1,H,W) in [0,1]
        color_vec_3:   (3,)
        """
        # boost contrast
        heat = torch.clamp(heat_1hw ** 0.5, 0.0, 1.0)
        thresh = 0.2
        heat = torch.clamp((heat - thresh) / (1.0 - thresh + 1e-6), 0.0, 1.0)

        color_vec = color_vec_3.view(3, 1, 1)
        heat_3 = heat.expand_as(base_img_3chw)
        color_layer = color_vec * torch.ones_like(base_img_3chw)

        blended = (1.0 - heat_3) * base_img_3chw + heat_3 * color_layer
        blended = blended.clamp(0.0, 1.0)
        return TF.to_pil_image(blended.detach().cpu())

    for name in ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g"]:
        idx = target_indices[name]

        # fresh forward for this target, with autograd on
        with torch.enable_grad():
            img = vis_transform(pil_img).unsqueeze(0).to(device)  # (1,3,H,W)
            img.requires_grad_(True)

            preds = model(img)          # (1,3)
            score = preds[0, idx]       # scalar

            model.zero_grad(set_to_none=True)
            if img.grad is not None:
                img.grad.zero_()

            score.backward()

            grad = img.grad.detach()[0]      # (3,H,W)
            # importance per pixel: mean |grad| over channels
            importance = grad.abs().mean(dim=0, keepdim=True)  # (1,H,W)

            # normalize to [0,1]
            imp_min = importance.min()
            imp_max = importance.max()
            if float(imp_max - imp_min) == 0.0:
                print(f"[WARN] {name}: image-grad importance is flat; model may be constant for this sample.")
            heat = (importance - imp_min) / (imp_max - imp_min + 1e-6)  # (1,H,W)

        print(f"{name} image-grad heat range: {float(heat.min()):.4f} - {float(heat.max()):.4f}")

        raw_heat[name] = TF.to_pil_image(heat.detach().cpu())
        overlays[name] = make_overlay(img[0].detach(), heat, base_colors[name])

    return overlays, raw_heat
