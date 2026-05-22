#!/usr/bin/env python3
"""
Create NUS-WIDE-MINI: A smaller subset for quick experiments
- Select top 5-7 most frequent classes
- ~500 training images
- ~100 test images
"""

import os
import random
import shutil
from pathlib import Path
from collections import Counter
import numpy as np


def load_data(data_root, split='database'):
    """Load image paths and labels"""
    data_root = Path(data_root)
    
    if split == 'database':
        img_file = data_root / 'database_img.txt'
        label_file = data_root / 'database_label_onehot.txt'
    else:
        img_file = data_root / 'test_img.txt'
        label_file = data_root / 'test_label_onehot.txt'
    
    # Read image paths
    with open(img_file, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    # Read labels
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            label_vec = [int(x) for x in line.strip().split()]
            labels.append(label_vec)
    
    labels = np.array(labels, dtype=np.int32)
    
    return image_paths, labels


def select_top_classes(labels, num_classes=7):
    """Select top N most frequent classes"""
    class_counts = labels.sum(axis=0)
    top_indices = np.argsort(class_counts)[::-1][:num_classes]
    return sorted(top_indices.tolist())


def filter_and_sample(image_paths, labels, selected_classes, max_samples):
    """Filter images that have at least one selected class and sample"""
    # Create mask for selected classes
    label_mask = labels[:, selected_classes]
    has_class = label_mask.sum(axis=1) > 0
    
    # Filter
    filtered_paths = [p for i, p in enumerate(image_paths) if has_class[i]]
    filtered_labels = labels[has_class][:, selected_classes]
    
    print(f"  Filtered {len(filtered_paths)} images with selected classes")
    
    # Sample
    if len(filtered_paths) > max_samples:
        indices = random.sample(range(len(filtered_paths)), max_samples)
        filtered_paths = [filtered_paths[i] for i in indices]
        filtered_labels = filtered_labels[indices]
        print(f"  Sampled {max_samples} images")
    
    return filtered_paths, filtered_labels


def create_mini_dataset():
    """Create NUS-WIDE-MINI dataset"""
    
    # Configuration
    SOURCE_ROOT = Path("data/NUS-WIDE 2")
    TARGET_ROOT = Path("data/NUS-WIDE-MINI")
    NUM_CLASSES = 7
    TRAIN_SIZE = 500
    TEST_SIZE = 100
    
    random.seed(42)
    
    print("="*80)
    print("Creating NUS-WIDE-MINI Dataset")
    print("="*80)
    
    # Create target directory
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    (TARGET_ROOT / "images").mkdir(exist_ok=True)
    
    # Load training data
    print("\nLoading training data...")
    train_paths, train_labels = load_data(SOURCE_ROOT, 'database')
    print(f"  Total: {len(train_paths)} images, {train_labels.shape[1]} classes")
    
    # Load test data
    print("\nLoading test data...")
    test_paths, test_labels = load_data(SOURCE_ROOT, 'test')
    print(f"  Total: {len(test_paths)} images, {test_labels.shape[1]} classes")
    
    # Select top classes
    print(f"\nSelecting top {NUM_CLASSES} classes...")
    selected_classes = select_top_classes(train_labels, NUM_CLASSES)
    print(f"  Selected class indices: {selected_classes}")
    
    # Calculate class frequencies
    train_freq = train_labels[:, selected_classes].sum(axis=0)
    print(f"  Training frequencies: {train_freq.tolist()}")
    
    # Filter and sample training data
    print(f"\nProcessing training data (target: {TRAIN_SIZE} samples)...")
    mini_train_paths, mini_train_labels = filter_and_sample(
        train_paths, train_labels, selected_classes, TRAIN_SIZE
    )
    
    # Filter and sample test data
    print(f"\nProcessing test data (target: {TEST_SIZE} samples)...")
    mini_test_paths, mini_test_labels = filter_and_sample(
        test_paths, test_labels, selected_classes, TEST_SIZE
    )
    
    # Copy images
    print("\nCopying images...")
    all_paths = mini_train_paths + mini_test_paths
    copied = 0
    for img_path in all_paths:
        src = SOURCE_ROOT / img_path
        dst = TARGET_ROOT / img_path
        
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
    
    print(f"  Copied {copied} unique images")
    
    # Save training files
    print("\nSaving training files...")
    with open(TARGET_ROOT / "database_img.txt", 'w') as f:
        f.write('\n'.join(mini_train_paths) + '\n')
    
    with open(TARGET_ROOT / "database_label_onehot.txt", 'w') as f:
        for label in mini_train_labels:
            f.write(' '.join(map(str, label)) + ' \n')
    
    # Save test files
    print("Saving test files...")
    with open(TARGET_ROOT / "test_img.txt", 'w') as f:
        f.write('\n'.join(mini_test_paths) + '\n')
    
    with open(TARGET_ROOT / "test_label_onehot.txt", 'w') as f:
        for label in mini_test_labels:
            f.write(' '.join(map(str, label)) + ' \n')
    
    # Create README
    print("Creating README...")
    with open(TARGET_ROOT / "README.txt", 'w') as f:
        f.write("NUS-WIDE-MINI Dataset\n")
        f.write("="*50 + "\n\n")
        f.write(f"A small subset of NUS-WIDE for quick experiments\n\n")
        f.write(f"Classes: {NUM_CLASSES}\n")
        f.write(f"Selected class indices: {selected_classes}\n")
        f.write(f"Training samples: {len(mini_train_paths)}\n")
        f.write(f"Test samples: {len(mini_test_paths)}\n")
        f.write(f"\nClass distribution (training):\n")
        for i, cls_idx in enumerate(selected_classes):
            count = mini_train_labels[:, i].sum()
            f.write(f"  Class {cls_idx}: {count} images\n")
    
    print("\n" + "="*80)
    print("NUS-WIDE-MINI Created Successfully!")
    print("="*80)
    print(f"\nLocation: {TARGET_ROOT}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Training: {len(mini_train_paths)} samples")
    print(f"Test: {len(mini_test_paths)} samples")
    print(f"\nFiles created:")
    print(f"  - database_img.txt")
    print(f"  - database_label_onehot.txt")
    print(f"  - test_img.txt")
    print(f"  - test_label_onehot.txt")
    print(f"  - images/ (directory with {copied} images)")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    create_mini_dataset()
