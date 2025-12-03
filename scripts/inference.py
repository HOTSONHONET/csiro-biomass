from PIL import Image
from models.vit_0.model import VitTransformer
from train import Config 
from models.vit_0.infer import generate_biomass_heatmaps
import torch

cfg = Config()

model = VitTransformer(cfg)
state = torch.load("vit_fold_5.pth", map_location="cuda")
model.load_state_dict(state)

pil_img = Image.open("dataset/Dry_Total_g/ID4464212.jpg").convert("RGB")

overlays, raw_heat = generate_biomass_heatmaps(model, pil_img, cfg, device="cuda")

overlays["Dry_Green_g"].save("heatmap_green_overlay.png")
overlays["Dry_Clover_g"].save("heatmap_clover_overlay.png")
overlays["Dry_Dead_g"].save("heatmap_dead_overlay.png")

# grayscale heatmaps (no original image)
raw_heat["Dry_Green_g"].save("heat_green_raw.png")
raw_heat["Dry_Clover_g"].save("heat_clover_raw.png")
raw_heat["Dry_Dead_g"].save("heat_dead_raw.png")
