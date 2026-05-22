import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.visualization import plot_training_curves

print("Fixing empty plot...")
checkpoint = torch.load('experiments/runs/20260322-184233/best_model.pth', map_location='cpu', weights_only=False)
history = checkpoint['history']

plot_training_curves(
    train_losses=history['train_loss'], 
    save_path='experiments/runs/20260322-184233/training_curves.png'
)
print("Fixed!")
