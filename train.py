#!/usr/bin/env python3
"""
Main training script for G-hash Educational Image Retrieval System
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.multiprocessing
import numpy as np

# Fix for "Too many open files" error when using DataLoader with many workers
torch.multiprocessing.set_sharing_strategy('file_system')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config, set_seed
from src.models.ghash import GHashModel, BaselineModel
from src.data.dataset import create_data_loaders
from src.training.losses import GHashLoss
from src.training.trainer import Trainer
from src.utils.visualization import create_experiment_report


def compute_pos_weight(train_loader, device):
    """Estimate per-class positive weights from the training set."""
    dataset_labels = getattr(train_loader.dataset, 'labels', None)
    if dataset_labels is None:
        return None

    labels = torch.as_tensor(np.asarray(dataset_labels), dtype=torch.float32)
    pos_counts = labels.sum(dim=0)
    neg_counts = labels.size(0) - pos_counts

    pos_weight = neg_counts / torch.clamp(pos_counts, min=1.0)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=10.0)
    return pos_weight.to(device)


def main(args):
    # Load configuration
    print("Loading configuration...")
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.hash_bits:
        config.config['model']['hash_bits'] = args.hash_bits
    if args.batch_size:
        config.config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config.config['training']['num_epochs'] = args.epochs
    
    # Set random seed
    set_seed(config['seed'])
    
    # Get device
    device = config.device
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, test_loader, query_loader, num_classes = create_data_loaders(config.config)
    
    # Update num_classes in config
    config.config['dataset']['num_classes'] = num_classes
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Query samples: {len(query_loader.dataset)}")
    
    # Create model
    print("\nBuilding model...")
    if args.baseline:
        print("Using baseline model (ResNet50 without GAT)")
        model = BaselineModel(config.config)
    else:
        print("Using G-hash model (ViT + GAT)")
        model = GHashModel(config.config)
    
    model = model.to(device)
    
    # [THÊM MỚI] Load pretrained checkpoint for fine-tuning
    if 'checkpoint' in config.config.get('model', {}) and config.config['model']['checkpoint']:
        ckpt_path = config.config['model']['checkpoint']
        print(f"\n[TRANSFER LEARNING] Nạp kiến thức cũ từ: {ckpt_path}")
        checkpoint_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Cắt gọt những phần não lệch kích thước (vd: 21 nhãn gốc vs 5 nhãn mới)
        model_state = model.state_dict()
        checkpoint_state = checkpoint_data['model_state_dict']
        filtered_state = {}
        
        for k, v in checkpoint_state.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                print(f"  > [CẮT BỎ] Lớp màng '{k}' do lệch số lượng Nhãn ({v.shape}).")
                
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        print(f"✅ Đã nạp ViT thành công! (Ráp lại {len(missing_keys)} đuôi rỗng cho EDU đợt này)")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create loss function
    loss_config = config['loss']
    pos_weight = compute_pos_weight(train_loader, device)
    if pos_weight is not None:
        print("Using class-balanced BCE with capped pos_weight.")

    criterion = GHashLoss(
        alpha=loss_config['alpha_similarity'],
        beta=loss_config['beta_quantization'],
        gamma=loss_config['gamma_classification'],
        delta=loss_config.get('delta_bit_balance', 0.5),
        eta=loss_config.get('eta_retrieval', 0.5),
        pos_weight=pos_weight
    ).to(device)
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        query_loader=query_loader,
        criterion=criterion,
        config=config.config,
        device=device
    )
    
    # Load checkpoint if requested
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        
    # Train model
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    history = trainer.train()
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80 + "\n")
    
    final_metrics = trainer.evaluate(epoch='final')
    
    # Create experiment report
    print("\nGenerating experiment report...")
    create_experiment_report(
        config=config.config,
        metrics=final_metrics,
        loss_history=history['loss_components'],
        save_dir=trainer.save_dir
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults saved to: {trainer.save_dir}")
    print(f"Best mAP: {trainer.best_map:.4f}")
    print("\nGenerated files:")
    print(f"  - best_model.pth (trained model)")
    print(f"  - metrics.txt (evaluation results)")
    print(f"  - training_curves.png")
    print(f"  - loss_components.png")
    print(f"  - topk_metrics.png")
    print(f"  - pr_curve.png")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train G-hash model for image retrieval')
    
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--hash-bits', type=int, default=None,
                       help='Number of hash bits (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--baseline', action='store_true',
                       help='Train baseline model instead of G-hash')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    main(args)
