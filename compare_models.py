#!/usr/bin/env python3
"""
Script to compare G-hash model vs Baseline model
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config, set_seed
from src.models.ghash import GHashModel, BaselineModel
from src.data.dataset import create_data_loaders
from src.training.losses import GHashLoss
from src.training.trainer import Trainer


def train_and_evaluate(model_type, config, train_loader, test_loader, query_loader, device):
    """Train and evaluate a model"""
    print(f"\n{'='*80}")
    print(f"Training {model_type} Model")
    print(f"{'='*80}\n")
    
    # Create model
    if model_type == 'baseline':
        model = BaselineModel(config.config)
    else:
        model = GHashModel(config.config)
    
    model = model.to(device)
    
    # Create loss and trainer
    loss_config = config['loss']
    criterion = GHashLoss(
        alpha=loss_config['alpha_similarity'],
        beta=loss_config['beta_quantization'],
        gamma=loss_config['gamma_classification']
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        query_loader=query_loader,
        criterion=criterion,
        config=config.config,
        device=device
    )
    
    # Train
    history = trainer.train()
    
    # Final evaluation
    metrics = trainer.evaluate(epoch='final')
    
    return {
        'model_type': model_type,
        'metrics': metrics,
        'history': history,
        'best_map': trainer.best_map
    }


def plot_comparison(results, save_path=None):
    """Plot comparison between models"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    model_names = [r['model_type'].capitalize() for r in results]
    map_scores = [r['best_map'] for r in results]
    
    # 1. mAP Comparison
    colors = ['#2ecc71', '#e74c3c']
    bars = axes[0].bar(model_names, map_scores, color=colors, alpha=0.8, edgecolor='black')
    for bar, score in zip(bars, map_scores):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('mAP', fontsize=12)
    axes[0].set_title('Mean Average Precision', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, max(map_scores) * 1.2)
    axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Training Loss Comparison
    for result in results:
        losses = result['history']['train_loss']
        epochs = range(1, len(losses) + 1)
        axes[1].plot(epochs, losses, marker='o', label=result['model_type'].capitalize(), linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Precision@K Comparison
    k_values = []
    for key in results[0]['metrics'].keys():
        if key.startswith('P@'):
            k_values.append(int(key.split('@')[1]))
    
    k_values.sort()
    x = np.arange(len(k_values))
    width = 0.35
    
    for i, result in enumerate(results):
        precisions = [result['metrics'][f'P@{k}'] for k in k_values]
        axes[2].bar(x + i*width, precisions, width, 
                   label=result['model_type'].capitalize(),
                   alpha=0.8, edgecolor='black')
    
    axes[2].set_xlabel('Top-K', fontsize=12)
    axes[2].set_ylabel('Precision', fontsize=12)
    axes[2].set_title('Precision @ Top-K', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x + width / 2)
    axes[2].set_xticklabels(k_values)
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main(args):
    print("="*80)
    print("G-hash vs Baseline Model Comparison")
    print("="*80 + "\n")
    
    # Load config
    config = Config(args.config)
    set_seed(config['seed'])
    device = config.device
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, test_loader, query_loader, num_classes = create_data_loaders(config.config)
    config.config['dataset']['num_classes'] = num_classes
    
    results = []
    
    # Train both models if comparison mode
    if args.compare:
        for model_type in ['baseline', 'ghash']:
            result = train_and_evaluate(
                model_type, config, train_loader, test_loader, query_loader, device
            )
            results.append(result)
        
        # Plot comparison
        save_path = Path('experiments') / 'model_comparison.png'
        plot_comparison(results, save_path=save_path)
        
        # Print summary
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        for result in results:
            print(f"\n{result['model_type'].upper()} Model:")
            print(f"  Best mAP: {result['best_map']:.4f}")
            for key, value in result['metrics'].items():
                if not isinstance(value, dict):
                    print(f"  {key}: {value:.4f}")
        
        # Compute improvement
        if len(results) == 2:
            improvement = (results[1]['best_map'] - results[0]['best_map']) / results[0]['best_map'] * 100
            print(f"\n{'='*80}")
            print(f"G-hash Improvement over Baseline: {improvement:+.2f}%")
            print(f"{'='*80}\n")
    else:
        # Train single model
        model_type = 'baseline' if args.baseline else 'ghash'
        result = train_and_evaluate(
            model_type, config, train_loader, test_loader, query_loader, device
        )
        print(f"\n{model_type.upper()} Model mAP: {result['best_map']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare G-hash vs Baseline models')
    
    parser.add_argument('--config', type=str, default='configs/config_test.yaml',
                       help='Path to configuration file')
    parser.add_argument('--baseline', action='store_true',
                       help='Train only baseline model')
    parser.add_argument('--compare', action='store_true',
                       help='Train and compare both models')
    
    args = parser.parse_args()
    
    main(args)
