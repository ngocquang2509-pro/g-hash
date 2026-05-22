#!/usr/bin/env python3
"""
Test script - automatically tests all images in specified folder
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from inference_mini import NUSWide2Retriever


def create_visualization(query_path, results, pred_labels, save_path):
    """Create visualization"""
    
    fig = plt.figure(figsize=(18, 6))
    
    # Query image
    ax_query = plt.subplot2grid((1, 7), (0, 0))
    query_img = Image.open(query_path)
    ax_query.imshow(query_img)
    ax_query.axis('off')
    
    query_name = Path(query_path).stem
    ax_query.set_title(f"QUERY\n{query_name}", fontsize=14, fontweight='bold', color='blue')
    
    # Predictions
    pred_text = "Predictions:\n"
    for i, label in enumerate(pred_labels[:3], 1):
        pred_text += f"{i}. {label['label']}: {label['confidence']:.3f}\n"
    
    ax_query.text(0.5, -0.08, pred_text, transform=ax_query.transAxes,
                  ha='center', va='top', fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Top 6 retrieved
    for i, result in enumerate(results[:6]):
        ax = plt.subplot2grid((1, 7), (0, i+1))
        
        try:
            img = Image.open(result['image_path'])
            ax.imshow(img)
        except:
            pass
        
        ax.axis('off')
        
        dist = result['hamming_distance']
        color = 'red' if dist == 0 else 'orange' if dist <= 5 else 'green'
        
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        
        ax.set_title(f"#{i+1}\nDist: {int(dist)}", fontsize=10, color=color, fontweight='bold')
        ax.text(0.5, -0.02, Path(result['image_path']).name, 
               transform=ax.transAxes, ha='center', va='top', fontsize=7)
    
    plt.suptitle(f'Query: {query_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    """Test all images in specified folder"""
    
    # Setup
    images_dir = Path(args.images_dir)
    data_root = Path("data/NUS-WIDE-MINI")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find query images
    query_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        query_images.extend(list(images_dir.glob(ext)))
    
    if not query_images:
        print(f"❌ No images found in {images_dir}/")
        return
    
    # Find model
    mini_runs = Path("experiments/mini_runs")
    runs = sorted(mini_runs.glob("*"))
    if not runs:
        print("❌ No trained model found! Run: python train_mini.py")
        return
    
    checkpoint_path = str(runs[-1] / "best_model.pth")
    
    # Load database
    with open(data_root / "test_img.txt", 'r') as f:
        test_images = [line.strip() for line in f.readlines()]
    database_images = [str(data_root / img) for img in test_images]
    
    # Initialize retriever (suppress verbose output)
    import sys
    import io
    
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    retriever = NUSWide2Retriever(checkpoint_path, "configs/config_mini.yaml")
    
    sys.stdout = old_stdout
    print(f"Testing {len(query_images)} images...\n")
    
    # Test each image
    for i, query_path in enumerate(query_images, 1):
        query_name = query_path.stem
        
        print(f"[{i}/{len(query_images)}] {query_path.name}")
        
        # Predict (suppress output)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        pred_labels = retriever.predict_labels(str(query_path), top_k=3)
        top_pred = pred_labels[0]
        
        # Check if correct
        sys.stdout = old_stdout
        correct = "✓" if query_name.lower() == top_pred['label'] else "✗"
        print(f"  {correct} Predict: {top_pred['label']} ({top_pred['confidence']:.2f})")
        
        # Retrieve (suppress output)
        sys.stdout = io.StringIO()
        results = retriever.retrieve_similar_images(str(query_path), database_images, top_k=10)
        sys.stdout = old_stdout
        
        distances = [r['hamming_distance'] for r in results]
        zero_count = sum(1 for d in distances if d == 0)
        
        print(f"  Retrieval: {zero_count}/10 zeros, range [{min(distances):.0f}-{max(distances):.0f}]")
        
        # Save visualization
        viz_path = output_dir / f"{query_name}.png"
        create_visualization(query_path, results, pred_labels, viz_path)
        print(f"  Saved: {viz_path}\n")
    
    # Summary
    print(f"✓ Completed! Results in {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test image retrieval with images from specified folder'
    )
    
    parser.add_argument(
        '--images-dir',
        type=str,
        default='images',
        help='Path to folder containing query images (default: images/)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_results',
        help='Path to save test results (default: test_results/)'
    )
    
    args = parser.parse_args()
    main(args)
