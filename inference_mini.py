#!/usr/bin/env python3
"""
Inference script for NUS-WIDE-MINI trained model
Supports multiple test runs with organized folder structure
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.models.ghash import GHashModel
from src.utils.config import Config
from src.evaluation.metrics import hamming_distance


class NUSWide2Retriever:
    """Image retrieval for NUS-WIDE-MINI trained model"""
    
    def __init__(self, checkpoint_path, config_path='configs/config_mini.yaml'):
        self.config = Config(config_path)
        self.device = self.config.device
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = GHashModel(self.config.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load label names (7 classes for NUS-WIDE-MINI)
        # Corresponds to class indices [0,1,2,3,4,5,6] from original 21 classes
        self.label_names = [
            'airport', 'animal', 'beach', 'bear', 'birds', 'boats', 'book'
        ]
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully!")
        print(f"Device: {self.device}")
        print(f"Hash bits: {self.config['model']['hash_bits']}")
        print(f"Labels: {len(self.label_names)} concepts")
    
    def encode_image(self, image_path):
        """Encode a single image to hash code"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            hash_code = self.model.generate_hash_code(image_tensor)
        
        return hash_code.cpu().numpy()[0]
    
    def predict_labels(self, image_path, top_k=5):
        """Predict top-K labels for an image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            img_features = self.model.image_encoder(image_tensor)
            logits = self.model.classifier(img_features)
            probs = torch.sigmoid(logits)[0]
        
        top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, len(self.label_names)))
        
        results = []
        for prob, idx in zip(top_k_probs.cpu().numpy(), top_k_indices.cpu().numpy()):
            results.append({
                'label': self.label_names[idx],
                'confidence': float(prob)
            })
        
        return results
    
    def retrieve_similar_images(self, query_image_path, database_images, top_k=5):
        """Retrieve top-K similar images from database"""
        print(f"\nEncoding query image...")
        query_code = self.encode_image(query_image_path)
        
        print(f"Encoding {len(database_images)} database images...")
        database_codes = []
        valid_db_images = []
        
        for img_path in database_images:
            try:
                code = self.encode_image(img_path)
                database_codes.append(code)
                valid_db_images.append(img_path)
            except Exception as e:
                print(f"Error encoding {img_path}: {e}")
        
        database_codes = np.array(database_codes)
        
        # Compute Hamming distances
        query_code = query_code.reshape(1, -1)
        distances = hamming_distance(query_code, database_codes)[0]
        
        # Get top-K nearest neighbors
        top_k_indices = np.argsort(distances)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_k_indices):
            results.append({
                'rank': rank + 1,
                'image_path': valid_db_images[idx],
                'hamming_distance': int(distances[idx])
            })
        
        return results
    
    def visualize_predictions(self, image_path, predictions, save_path):
        """Visualize image with predicted labels"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Show image
        image = Image.open(image_path)
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Query Image', fontsize=14, fontweight='bold')
        
        # Show predictions as bar chart
        labels = [p['label'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
        bars = ax2.barh(labels, confidences, color=colors, edgecolor='black')
        
        for bar, conf in zip(bars, confidences):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{conf:.3f}',
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_title('Top Predicted Labels', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1.0)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {save_path}")
    
    def visualize_retrieval(self, query_image_path, retrieved_results, save_path):
        """Visualize query image and retrieved similar images"""
        num_results = len(retrieved_results)
        fig = plt.figure(figsize=(16, 4))
        
        # Query image
        ax = plt.subplot(1, num_results + 1, 1)
        query_img = Image.open(query_image_path)
        ax.imshow(query_img)
        ax.set_title('Query Image', fontsize=12, fontweight='bold', color='red')
        ax.axis('off')
        
        # Retrieved images
        for i, result in enumerate(retrieved_results):
            ax = plt.subplot(1, num_results + 1, i + 2)
            try:
                img = Image.open(result['image_path'])
                ax.imshow(img)
                ax.set_title(f"Rank {result['rank']}\nDist: {result['hamming_distance']}",
                           fontsize=10)
            except:
                ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Retrieval visualization saved to {save_path}")


def create_test_folder():
    """Create a new timestamped test folder"""
    tests_dir = Path('tests')
    tests_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    test_folder = tests_dir / f'test_{timestamp}'
    test_folder.mkdir(exist_ok=True)
    
    return test_folder


def main(args):
    # Initialize retriever
    retriever = NUSWide2Retriever(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )
    
    # Create test folder for this run
    test_folder = create_test_folder()
    print(f"\n{'='*60}")
    print(f"Test folder: {test_folder}")
    print(f"{'='*60}\n")
    
    # Create test metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': args.checkpoint,
        'config': args.config,
        'mode': args.mode,
        'query_image': args.image,
        'top_k': args.top_k
    }
    
    if args.mode == 'predict':
        # Predict labels for single image
        print(f"Predicting labels for: {args.image}")
        
        predictions = retriever.predict_labels(args.image, top_k=args.top_k)
        
        print("\nTop Predicted Labels:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred['label']:20s} - Confidence: {pred['confidence']:.4f}")
        
        # Save results
        metadata['predictions'] = predictions
        
        # Save metadata
        with open(test_folder / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save predictions to text file
        with open(test_folder / 'predictions.txt', 'w') as f:
            f.write(f"Query Image: {args.image}\n")
            f.write(f"Timestamp: {metadata['timestamp']}\n\n")
            f.write("Top Predicted Labels:\n")
            for i, pred in enumerate(predictions, 1):
                f.write(f"  {i}. {pred['label']:20s} - Confidence: {pred['confidence']:.4f}\n")
        
        # Visualize and save
        viz_path = test_folder / 'predictions_visualization.png'
        retriever.visualize_predictions(args.image, predictions, save_path=viz_path)
    
    elif args.mode == 'retrieve':
        # Retrieve similar images
        if not args.database:
            print("Error: --database argument required for retrieval mode")
            return
        
        # Load database image paths
        database_images = []
        db_path = Path(args.database)
        
        if db_path.is_dir():
            # Directory of images
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                database_images.extend(list(db_path.glob(ext)))
            database_images = [str(p) for p in database_images]
        else:
            # Text file with image paths
            with open(args.database, 'r') as f:
                database_images = [line.strip() for line in f.readlines()]
        
        print(f"Retrieving similar images for: {args.image}")
        print(f"Database size: {len(database_images)} images")
        
        results = retriever.retrieve_similar_images(
            args.image, database_images, top_k=args.top_k
        )
        
        print("\nTop Retrieved Images:")
        for result in results:
            print(f"  Rank {result['rank']}: {Path(result['image_path']).name}")
            print(f"    Hamming Distance: {result['hamming_distance']}")
        
        # Save results
        metadata['database'] = args.database
        metadata['database_size'] = len(database_images)
        metadata['retrieval_results'] = results
        
        # Save metadata
        with open(test_folder / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save retrieval results to text file
        with open(test_folder / 'retrieval_results.txt', 'w') as f:
            f.write(f"Query Image: {args.image}\n")
            f.write(f"Database: {args.database}\n")
            f.write(f"Database Size: {len(database_images)} images\n")
            f.write(f"Timestamp: {metadata['timestamp']}\n\n")
            f.write("Top Retrieved Images:\n")
            for result in results:
                f.write(f"  Rank {result['rank']}: {Path(result['image_path']).name}\n")
                f.write(f"    Path: {result['image_path']}\n")
                f.write(f"    Hamming Distance: {result['hamming_distance']}\n\n")
        
        # Visualize and save
        viz_path = test_folder / 'retrieval_visualization.png'
        retriever.visualize_retrieval(args.image, results, save_path=viz_path)
    
    print(f"\n{'='*60}")
    print(f"Test completed successfully!")
    print(f"Results saved to: {test_folder}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NUS-WIDE 2 Image Inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config_nuswide2.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['predict', 'retrieve'],
                       default='predict', help='Inference mode')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to query image')
    parser.add_argument('--database', type=str, default=None,
                       help='Database images directory or file list (for retrieve mode)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top results to return')
    
    args = parser.parse_args()
    
    main(args)
