"""
Utility Script: Train/Test Dataset Splitter
-------------------------------------------
Dependencies:
    pip install pandas shutil

Usage:
    python split_dataset.py --csv_file data/draft_annotations.csv --dataset_dir data/dataset --train_ratio 0.9
"""

import os
import shutil
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Split annotated Multi-Label CSV into Train and Test directories.")
    parser.add_argument("--csv_file", type=str, default="data/draft_annotations.csv", help="Path to the finalized CSV annotations")
    parser.add_argument("--dataset_root", type=str, default="data/final_dataset", help="Root directory to generate final dataset")
    parser.add_argument("--src_img_dir", type=str, default="data", help="Root tracking directory for relative paths inside the CSV")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of data to put in train set (e.g., 0.9 = 90%)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"❌ Error: CSV file '{csv_path}' not found!")
        return
        
    print(f"[*] Reading dataset annotations from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split
    train_size = int(len(df) * args.train_ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"[*] Total Images: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}")
    
    # Create Directories
    dataset_root = Path(args.dataset_root)
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    src_root = Path(args.src_img_dir)
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Helper to copy images
    def copy_split(split_df, dest_dir, split_name):
        valid_rows = []
        missing_count = 0
        
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"):
            img_rel_path = row['Image_Path']
            src_img = src_root / img_rel_path
            
            if src_img.exists():
                dst_img = dest_dir / src_img.name
                shutil.copy2(src_img, dst_img)
                
                # Update path in the DataFrame for the new CSV to point directly to the image filename
                row_copy = row.copy()
                row_copy['Image_Path'] = src_img.name
                valid_rows.append(row_copy)
            else:
                missing_count += 1
                
        if missing_count > 0:
            print(f"⚠️ Warning: {missing_count} images were missing from the disk and skipped.")
            
        return pd.DataFrame(valid_rows)

    # Execute Copy and Path updates
    final_train_df = copy_split(train_df, train_dir, "Train Set")
    final_test_df = copy_split(test_df, test_dir, "Test Set")
    
    # Save CSVs inside train and test folders respectively
    train_csv_path = dataset_root / "train.csv"
    test_csv_path = dataset_root / "test.csv"
    
    final_train_df.to_csv(train_csv_path, index=False)
    final_test_df.to_csv(test_csv_path, index=False)
    
    print(f"\n🎉 Dataset Split Successful!")
    print(f"Train annotations: {train_csv_path}")
    print(f"Test annotations:  {test_csv_path}")

if __name__ == "__main__":
    main()
