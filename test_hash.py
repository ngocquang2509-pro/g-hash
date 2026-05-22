import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.models.ghash import GHashModel
from src.data.dataset import create_data_loaders

# Load checkpoint
ckpt = torch.load('experiments/runs/20260326-093024/best_model.pth', weights_only=False)
config = ckpt['config']
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model
model = GHashModel(config).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Load dataloader
_, test_loader, _, _ = create_data_loaders(config)

all_codes = []
with torch.no_grad():
    for images, _, _ in test_loader:
        images = images.to(device)
        codes = model.generate_hash_code(images)
        all_codes.append(codes)
        break

all_codes = torch.cat(all_codes, dim=0)

print(f"Batch Hash Codes Shape: {all_codes.shape}")
print(f"Unique Hash Codes in Batch: {torch.unique(all_codes, dim=0).shape[0]}")
print(f"First 5 elements of first hash code: {all_codes[0][:5]}")
print(f"First 5 elements of second hash code: {all_codes[1][:5]}")
