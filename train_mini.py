#!/usr/bin/env python3
"""
Training script for NUS-WIDE-MINI dataset
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config, set_seed
from src.models.ghash import GHashModel, BaselineModel
from src.data.nuswide2_dataset import create_nuswide2_loaders
from src.training.losses import GHashLoss
from src.training.trainer import Trainer
from src.utils.visualization import create_experiment_report


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
    print("\nCreating data loaders for NUS-WIDE 2...")
    train_loader, test_loader, query_loader, num_classes = create_nuswide2_loaders(config.config)
    
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
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create loss function
    loss_config = config['loss']
    criterion = GHashLoss(
        alpha=loss_config['alpha_similarity'],
        beta=loss_config['beta_quantization'],
        gamma=loss_config['gamma_classification']
    )
    
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
    print("STARTING TRAINING - NUS-WIDE 2 (Real Images)")
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
    parser = argparse.ArgumentParser(description='Train G-hash model on NUS-WIDE-MINI')
    
    parser.add_argument('--config', type=str, default='configs/config_mini.yaml',
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
