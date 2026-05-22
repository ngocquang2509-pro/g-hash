import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.visualization import plot_loss_components

print("Fixing empty total loss in loss_components.png...")
checkpoint = torch.load('experiments/runs/20260322-184233/best_model.pth', map_location='cpu', weights_only=False)
history = checkpoint['history']

# Fix the missing 'total' in loss_components
loss_comps = history['loss_components']
loss_comps['total'] = history['train_loss']

plot_loss_components(
    loss_history=loss_comps, 
    save_path='experiments/runs/20260322-184233/loss_components.png'
)
print("Fixed!")
