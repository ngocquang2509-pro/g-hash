#!/usr/bin/env python3
"""
Inference script for G-hash Image Retrieval
Test trained model with real images
"""

import sys
import argparse
from pathlib import Path
import re
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

sys.path.insert(0, str(Path(__file__).parent))

from src.models.ghash import GHashModel
from src.utils.config import Config
from src.evaluation.metrics import hamming_distance


class ImageRetriever:
    """Image retrieval using trained G-hash model"""
    
    def __init__(self, checkpoint_path, config_path='configs/config_m4pro.yaml'):
        self.config = Config(config_path)
        self.device = self.config.device
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = GHashModel(self.config.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load label names (Hỗ trợ nhãn tự định nghĩa)
        custom_label_file = self.config['dataset'].get('label_file')
        fallback_label_file = Path(self.config['dataset']['data_root']) / 'ConceptsList' / 'Concepts81.txt'
        
        if custom_label_file and Path(custom_label_file).exists():
            with open(custom_label_file, 'r') as f:
                self.label_names = [line.strip() for line in f.readlines()]
        elif fallback_label_file.exists():
            with open(fallback_label_file, 'r') as f:
                self.label_names = [line.strip() for line in f.readlines()]
        else:
            # Fallback for old custom datasets
            num_classes = self.config['dataset'].get('num_classes', 21)
            self.label_names = [f"Concept_ID_{i}" for i in range(num_classes)]
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Database of hash codes (will be populated)
        self.database_codes = None
        self.database_labels = None
        self.database_features = None
        self.database_probabilities = None
        self.database_images = None
        self.rerank_k = self.config['evaluation'].get('rerank_k', 100)
        eval_cfg = self.config.get('evaluation', {})
        self.query_auto_focus = eval_cfg.get('query_auto_focus', True)
        self.mmr_lambda = float(eval_cfg.get('mmr_lambda', 0.75))
        self.max_per_video_in_topk = int(eval_cfg.get('max_per_video_in_topk', 2))
        self.min_frame_gap_in_topk = int(eval_cfg.get('min_frame_gap_in_topk', 45))
        self.person_detector = None
        
        print(f"Model loaded successfully!")
        print(f"Device: {self.device}")
        print(f"Hash bits: {self.config['model']['hash_bits']}")
        print(f"Labels: {len(self.label_names)} concepts")
    
    def _encode_tensor(self, image_tensor):
        """Encode preprocessed image tensor into retrieval representations."""
        with torch.no_grad():
            img_features = self.model.image_encoder(image_tensor)
            continuous_hash = torch.tanh(self.model.img_hash_fc(img_features))
            binary_hash = torch.sign(continuous_hash)
            binary_hash[binary_hash == 0] = 1
            logits = self.model.classifier(img_features)
            probs = torch.sigmoid(logits)

        return {
            'binary_hash': binary_hash.cpu().numpy()[0],
            'continuous_hash': continuous_hash.cpu().numpy()[0],
            'probabilities': probs.cpu().numpy()[0]
        }

    def _encode_pil_image(self, image: Image.Image, return_details=False):
        image = image.convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        encoded = self._encode_tensor(image_tensor)
        if return_details:
            return encoded
        return encoded['binary_hash']

    def encode_image(self, image_path, return_details=False):
        """Encode a single image for retrieval."""
        image = Image.open(image_path).convert('RGB')
        return self._encode_pil_image(image, return_details=return_details)

    def _ensure_person_detector(self):
        if self.person_detector is not None:
            return self.person_detector
        if YOLO is None:
            return None
        try:
            self.person_detector = YOLO("yolov8n.pt")
        except Exception:
            self.person_detector = None
        return self.person_detector

    def _focus_query_person(self, query_image_path):
        """
        If query image is full-scene, auto-crop largest confident person
        to reduce mismatch between scene query and person-crop database.
        """
        if not self.query_auto_focus:
            return Image.open(query_image_path).convert('RGB')

        detector = self._ensure_person_detector()
        image = Image.open(query_image_path).convert('RGB')
        if detector is None:
            return image

        arr = np.array(image)
        h, w = arr.shape[:2]
        frame_area = max(1, h * w)

        try:
            results = detector.predict(arr, verbose=False)
        except Exception:
            return image

        if not results or results[0].boxes is None:
            return image

        best_box = None
        best_score = -1.0
        for box in results[0].boxes:
            if int(box.cls[0]) != 0:
                continue
            conf = float(box.conf[0])
            if conf < 0.45:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            area_ratio = (bw * bh) / frame_area
            # Prefer reasonably large and confident person.
            score = 0.7 * area_ratio + 0.3 * conf
            if score > best_score:
                best_score = score
                best_box = (x1, y1, x2, y2)

        if best_box is None:
            return image

        x1, y1, x2, y2 = best_box
        pad_w = int(0.12 * (x2 - x1))
        pad_h = int(0.12 * (y2 - y1))
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(w, x2 + pad_w)
        cy2 = min(h, y2 + pad_h)

        # Only use focused crop if reasonably large.
        if (cx2 - cx1) * (cy2 - cy1) < 0.04 * frame_area:
            return image
        crop = arr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return image
        return Image.fromarray(crop)

    @staticmethod
    def _parse_video_frame(image_path):
        """
        Parse video id + frame id from filenames like:
        near_D01_20240223064932_f002850_p00_00000023.jpg
        """
        name = Path(image_path).name
        parts = name.split('_')
        video_id = None
        frame_id = None
        if len(parts) >= 3:
            video_id = f"{parts[1]}_{parts[2]}"
        for p in parts:
            if p.startswith('f') and len(p) > 1 and p[1:].isdigit():
                frame_id = int(p[1:])
                break
        if frame_id is None:
            m = re.search(r"_f(\d+)", name)
            if m:
                frame_id = int(m.group(1))
        return video_id, frame_id

    def _select_diverse_topk(self, candidate_indices, relevance_scores, feature_matrix, top_k):
        """
        MMR-based selection with per-video and frame-gap constraints
        to reduce repeated near-duplicates in top-k.
        """
        if len(candidate_indices) == 0:
            return []

        selected = []
        remaining = list(range(len(candidate_indices)))
        video_count = {}
        selected_frames = {}

        sim_matrix = feature_matrix @ feature_matrix.T
        mmr_lambda = float(np.clip(self.mmr_lambda, 0.0, 1.0))

        while remaining and len(selected) < top_k:
            best_local = None
            best_score = -1e9

            for local_idx in remaining:
                db_idx = candidate_indices[local_idx]
                img_path = self.database_images[db_idx]
                video_id, frame_id = self._parse_video_frame(img_path)

                # Hard diversity constraints.
                if video_id is not None and self.max_per_video_in_topk > 0:
                    if video_count.get(video_id, 0) >= self.max_per_video_in_topk:
                        continue
                if video_id is not None and frame_id is not None and self.min_frame_gap_in_topk > 0:
                    prior_frames = selected_frames.get(video_id, [])
                    if any(abs(frame_id - pf) < self.min_frame_gap_in_topk for pf in prior_frames):
                        continue

                rel = relevance_scores[local_idx]
                if not selected:
                    mmr_score = rel
                else:
                    max_sim = max(sim_matrix[local_idx, s] for s in selected)
                    mmr_score = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_local = local_idx

            if best_local is None:
                # Relax constraints: fill remaining slots by relevance.
                remaining_sorted = sorted(remaining, key=lambda i: relevance_scores[i], reverse=True)
                for local_idx in remaining_sorted:
                    selected.append(local_idx)
                    if len(selected) >= top_k:
                        break
                break

            selected.append(best_local)
            remaining.remove(best_local)
            sel_db_idx = candidate_indices[best_local]
            sel_video, sel_frame = self._parse_video_frame(self.database_images[sel_db_idx])
            if sel_video is not None:
                video_count[sel_video] = video_count.get(sel_video, 0) + 1
                if sel_frame is not None:
                    selected_frames.setdefault(sel_video, []).append(sel_frame)

        return selected[:top_k]
    
    def predict_labels(self, image_path, top_k=5):
        """Predict top-K labels for an image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            img_features = self.model.image_encoder(image_tensor)
            logits = self.model.classifier(img_features)
            probs = torch.sigmoid(logits)[0]
        
        # Get top-K predictions
        top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, len(self.label_names)))
        
        results = []
        for prob, idx in zip(top_k_probs.cpu().numpy(), top_k_indices.cpu().numpy()):
            results.append({
                'label': self.label_names[idx],
                'confidence': float(prob)
            })
        
        return results
    
    def retrieve_similar_images(self, query_image_path, database_images, top_k=5):
        """
        Retrieve top-K similar images from database
        
        Args:
            query_image_path: Path to query image
            database_images: List of database image paths
            top_k: Number of similar images to retrieve
        """
        print(f"\nEncoding query image...")
        query_image = self._focus_query_person(query_image_path)
        query_data = self._encode_pil_image(query_image, return_details=True)

        self.build_database_index(database_images)

        # Stage 1: fast coarse search using Hamming distance.
        query_code = query_data['binary_hash'].reshape(1, -1)
        distances = hamming_distance(query_code, self.database_codes)[0]
        coarse_k = min(max(top_k * 10, self.rerank_k), len(self.database_images))
        coarse_indices = np.argsort(distances)[:coarse_k]

        query_path = Path(query_image_path).resolve()
        filtered_indices = []
        for idx in coarse_indices:
            candidate_path = Path(self.database_images[idx]).resolve()
            if candidate_path != query_path:
                filtered_indices.append(idx)
        coarse_indices = np.asarray(filtered_indices[:coarse_k], dtype=int)

        if coarse_indices.size == 0:
            return []

        # Stage 2: rerank with continuous hash + predicted label distributions.
        query_feat = query_data['continuous_hash']
        query_feat = query_feat / (np.linalg.norm(query_feat) + 1e-8)
        db_feats = self.database_features[coarse_indices]
        db_feats = db_feats / (np.linalg.norm(db_feats, axis=1, keepdims=True) + 1e-8)
        feature_scores = db_feats @ query_feat

        query_probs = query_data['probabilities']
        query_probs = query_probs / (np.linalg.norm(query_probs) + 1e-8)
        db_probs = self.database_probabilities[coarse_indices]
        db_probs = db_probs / (np.linalg.norm(db_probs, axis=1, keepdims=True) + 1e-8)
        label_scores = db_probs @ query_probs

        hamming_scores = 1.0 - (distances[coarse_indices] / max(1, self.config['model']['hash_bits']))
        rerank_scores = 0.55 * feature_scores + 0.30 * label_scores + 0.15 * hamming_scores

        selected_locals = self._select_diverse_topk(
            candidate_indices=coarse_indices,
            relevance_scores=rerank_scores,
            feature_matrix=db_feats,
            top_k=top_k
        )
        top_k_indices = coarse_indices[selected_locals]
        
        results = []
        for rank, idx in enumerate(top_k_indices):
            local_idx = int(np.where(coarse_indices == idx)[0][0])
            video_id, frame_id = self._parse_video_frame(self.database_images[idx])
            results.append({
                'rank': rank + 1,
                'image_path': self.database_images[idx],
                'hamming_distance': int(distances[idx]),
                'rerank_score': float(rerank_scores[local_idx]),
                'video_id': video_id,
                'frame_id': frame_id
            })
        
        return results

    def build_database_index(self, database_images):
        """Cache database encodings so repeated queries are consistent and fast."""
        normalized_paths = [str(Path(p)) for p in database_images]
        if self.database_images == normalized_paths and self.database_codes is not None:
            return

        print(f"Encoding {len(normalized_paths)} database images...")
        database_codes = []
        database_features = []
        database_probabilities = []

        for img_path in normalized_paths:
            try:
                encoded = self.encode_image(img_path, return_details=True)
                database_codes.append(encoded['binary_hash'])
                database_features.append(encoded['continuous_hash'])
                database_probabilities.append(encoded['probabilities'])
            except Exception as e:
                print(f"Error encoding {img_path}: {e}")
                database_codes.append(np.zeros(self.config['model']['hash_bits']))
                database_features.append(np.zeros(self.config['model']['hash_bits']))
                database_probabilities.append(np.zeros(len(self.label_names)))

        self.database_images = normalized_paths
        self.database_codes = np.asarray(database_codes, dtype=np.float32)
        self.database_features = np.asarray(database_features, dtype=np.float32)
        self.database_probabilities = np.asarray(database_probabilities, dtype=np.float32)
    
    def visualize_predictions(self, image_path, predictions, save_path=None):
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
        
        # Add value labels
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
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_retrieval(self, query_image_path, retrieved_results, save_path=None):
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
                ax.set_title(
                    f"Rank {result['rank']}\nDist: {result['hamming_distance']}\nScore: {result.get('rerank_score', 0.0):.3f}",
                           fontsize=10)
            except:
                ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Retrieval visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main(args):
    # Initialize retriever
    retriever = ImageRetriever(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )
    
    if args.mode == 'predict':
        # Predict labels for single image
        print(f"\n{'='*60}")
        print(f"Predicting labels for: {args.image}")
        print(f"{'='*60}\n")
        
        predictions = retriever.predict_labels(args.image, top_k=args.top_k)
        
        print("Top Predicted Labels:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred['label']:20s} - Confidence: {pred['confidence']:.4f}")
        
        # Visualize
        if args.save_viz:
            save_path = Path('experiments') / 'predictions.png'
            retriever.visualize_predictions(args.image, predictions, save_path=save_path)
    
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
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                database_images.extend(list(db_path.glob(ext)))
        else:
            # Text file with image paths
            db_root = db_path.parent
            with open(args.database, 'r') as f:
                database_images = [str(db_root / line.strip()) for line in f.readlines()]
        
        print(f"\n{'='*60}")
        print(f"Retrieving similar images for: {args.image}")
        print(f"Database size: {len(database_images)} images")
        print(f"{'='*60}\n")
        
        results = retriever.retrieve_similar_images(
            args.image, database_images, top_k=args.top_k
        )
        
        print("\nTop Retrieved Images:")
        for result in results:
            print(f"  Rank {result['rank']}: {result['image_path']}")
            print(f"    Hamming Distance: {result['hamming_distance']}")
        
        # Visualize
        if args.save_viz:
            save_path = Path('experiments') / 'retrieval_results.png'
            retriever.visualize_retrieval(args.image, results, save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='G-hash Image Inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config_m4pro.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['predict', 'retrieve'],
                       default='predict', help='Inference mode')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to query image')
    parser.add_argument('--database', type=str, default=None,
                       help='Database images directory or file list (for retrieve mode)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top results to return')
    parser.add_argument('--save-viz', action='store_true',
                       help='Save visualization to file')
    
    args = parser.parse_args()
    
    main(args)
