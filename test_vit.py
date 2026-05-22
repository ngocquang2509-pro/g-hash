import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import create_data_loaders
from src.models.ghash import GHashModel

ckpt = torch.load('experiments/runs/20260326-093024/best_model.pth', map_location='cuda', weights_only=False)
config = ckpt['config']
model = GHashModel(config).to('cuda')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

_, test_loader, _, _ = create_data_loaders(config)

with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to('cuda')
        features = model.image_encoder(images)
        print("ViT features shape:", features.shape)
        
        diff = torch.abs(features[0] - features[1]).sum().item()
        print(f"Difference between image 1 and 2: {diff}")
        
        continuous_hash = model.img_hash_fc(features)
        print("Continuous Hash shape:", continuous_hash.shape)
        diff_hash = torch.abs(continuous_hash[0] - continuous_hash[1]).sum().item()
        print(f"Diff between continuous hash 1 and 2: {diff_hash}")
        break
