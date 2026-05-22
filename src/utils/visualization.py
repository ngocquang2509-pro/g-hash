import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_training_curves(train_losses, val_losses=None, save_path=None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_loss_components(loss_history, save_path=None):
    """Plot individual loss components over time"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    components = ['total', 'classification', 'similarity', 'quantization']
    titles = ['Total Loss', 'Classification Loss', 'Similarity Loss', 'Quantization Loss']
    
    for idx, (component, title) in enumerate(zip(components, titles)):
        ax = axes[idx // 2, idx % 2]
        
        if component in loss_history:
            values = loss_history[component]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved loss components to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_map_comparison(results, save_path=None):
    """Plot mAP comparison across different hash bit sizes or methods"""
    plt.figure(figsize=(10, 6))
    
    methods = list(results.keys())
    map_scores = [results[m]['mAP'] for m in methods]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    bars = plt.bar(methods, map_scores, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, score in zip(bars, map_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.ylabel('mAP', fontsize=12)
    plt.title('Mean Average Precision Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved mAP comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_precision_recall_curve(pr_data, save_path=None):
    """Plot Precision-Recall curve at different Hamming radii"""
    plt.figure(figsize=(10, 6))
    
    radii = sorted(pr_data.keys())
    precisions = [pr_data[r]['precision'] for r in radii]
    recalls = [pr_data[r]['recall'] for r in radii]
    
    plt.plot(recalls, precisions, 'b-o', linewidth=2, markersize=8, label='P-R Curve')
    
    # Annotate each point with Hamming radius
    for r, rec, prec in zip(radii, recalls, precisions):
        plt.annotate(f'R={r}', (rec, prec), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall at Different Hamming Radii', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved P-R curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_topk_metrics(metrics_dict, save_path=None):
    """Plot Precision and Recall at different Top-K values"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract Top-K values and metrics
    precision_data = {k: v for k, v in metrics_dict.items() if k.startswith('P@')}
    recall_data = {k: v for k, v in metrics_dict.items() if k.startswith('R@')}
    
    if precision_data:
        k_values = [int(k.split('@')[1]) for k in precision_data.keys()]
        precisions = list(precision_data.values())
        
        ax1.plot(k_values, precisions, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Top-K', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('Precision @ Top-K', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)
    
    if recall_data:
        k_values = [int(k.split('@')[1]) for k in recall_data.keys()]
        recalls = list(recall_data.values())
        
        ax2.plot(k_values, recalls, 'r-o', linewidth=2, markersize=8)
        ax2.set_xlabel('Top-K', fontsize=12)
        ax2.set_ylabel('Recall', fontsize=12)
        ax2.set_title('Recall @ Top-K', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Top-K metrics to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_hash_code_distribution(hash_codes, save_path=None):
    """Plot distribution of hash code values"""
    if isinstance(hash_codes, np.ndarray):
        hash_codes = hash_codes
    else:
        hash_codes = hash_codes.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of all values
    axes[0].hist(hash_codes.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Hash Code Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Hash Code Values', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axvline(x=-1, color='r', linestyle='--', label='Target: -1')
    axes[0].axvline(x=1, color='r', linestyle='--', label='Target: +1')
    axes[0].legend()
    
    # Bit balance (how many +1 vs -1 for each bit position)
    binary_codes = np.sign(hash_codes)
    bit_balance = binary_codes.mean(axis=0)
    
    axes[1].bar(range(len(bit_balance)), bit_balance, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Bit Position')
    axes[1].set_ylabel('Mean Value')
    axes[1].set_title('Bit Balance (ideally close to 0)', fontweight='bold')
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved hash code distribution to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_retrieval_results(query_images, retrieved_images, query_labels, 
                                retrieved_labels, top_k=5, save_path=None):
    """
    Visualize retrieval results for a few query examples
    
    Args:
        query_images: Query images (N_query, C, H, W)
        retrieved_images: Retrieved images (N_query, top_k, C, H, W)
        query_labels: Query labels
        retrieved_labels: Retrieved labels (N_query, top_k, num_classes)
        top_k: Number of retrieved images to show
        save_path: Path to save figure
    """
    # This is a placeholder - actual implementation would show real images
    # For now, just print statistics
    print(f"Visualization: Showing top-{top_k} retrieved images for each query")
    print(f"Query set size: {len(query_images)}")
    print("(Image visualization would be implemented with actual image data)")


def create_experiment_report(config, metrics, loss_history, save_dir):
    """
    Create comprehensive experiment report with all visualizations
    
    Args:
        config: Configuration dict
        metrics: Evaluation metrics
        loss_history: Training loss history
        save_dir: Directory to save report
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GENERATING EXPERIMENT REPORT")
    print(f"{'='*60}\n")
    
    # 1. Training curves
    plot_training_curves(
        loss_history.get('total', []),
        save_path=save_dir / 'training_curves.png'
    )
    
    # 2. Loss components
    plot_loss_components(
        loss_history,
        save_path=save_dir / 'loss_components.png'
    )
    
    # 3. Top-K metrics
    plot_topk_metrics(
        metrics,
        save_path=save_dir / 'topk_metrics.png'
    )
    
    # 4. Precision-Recall curve
    if 'PR_at_radius' in metrics:
        plot_precision_recall_curve(
            metrics['PR_at_radius'],
            save_path=save_dir / 'pr_curve.png'
        )
    
    # 5. Save metrics to text file
    with open(save_dir / 'metrics.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("="*60 + "\n\n")
        
        for key, value in metrics.items():
            if not isinstance(value, dict):
                f.write(f"{key}: {value:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("CONFIGURATION\n")
        f.write("="*60 + "\n\n")
        
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✓ Experiment report saved to: {save_dir}")
    print(f"  - training_curves.png")
    print(f"  - loss_components.png")
    print(f"  - topk_metrics.png")
    print(f"  - pr_curve.png")
    print(f"  - metrics.txt")
    print()
