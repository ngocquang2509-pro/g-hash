"""
Semi-Supervised Labeling Pipeline
==================================
Quy trình:
  1. Đọc file seed_annotations.csv (500 ảnh do CON NGƯỜI gán nhãn)
  2. Train bộ phân loại CNN (ResNet18) trên 500 ảnh seed đó
  3. Dùng CNN đã train để dự đoán nhãn cho 444k ảnh còn lại
  4. Áp dụng Class Balancing, xuất ra golden_annotations.csv (10k ảnh)

Usage:
    python semi_supervised_label.py
"""

import os
import csv
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

CLASSES = [
    'using_phone', 'dozing_off', 'turning_sideways',
    'turning_back', 'raising_hand', 'opening_book',
    'reading', 'writing', 'listening', 'head_down',
    'sitting', 'standing', 'walking', 'interacting'
]
NUM_CLASSES = len(CLASSES)
MAX_PER_CLASS = 2000  # Giới hạn cân bằng nhãn


# =====================================================
# BƯỚC 1: Dataset cho CNN Classifier
# =====================================================
class SeedDataset(Dataset):
    def __init__(self, csv_file, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []

        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            img_path = os.path.join(data_root, row['Image_Path'])
            labels = [int(row[c]) for c in CLASSES]
            if os.path.exists(img_path):
                self.samples.append((img_path, labels))

        print(f"  Đã tải {len(self.samples)} ảnh seed hợp lệ")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.FloatTensor(labels)


# =====================================================
# BƯỚC 2: Train CNN Classifier trên Seed Data
# =====================================================
def train_classifier(seed_csv, data_root, epochs=20, batch_size=32, lr=1e-3):
    print("\n" + "=" * 60)
    print("📚 BƯỚC 1: TRAIN CNN CLASSIFIER TRÊN SEED DATA")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data augmentation mạnh giúp 500 ảnh "nhân bản" hiệu quả
    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = SeedDataset(seed_csv, data_root, transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # ResNet18 pre-trained + thay fc layer cho multi-label
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
        nn.Sigmoid()  # Multi-label: mỗi nhãn ra xác suất độc lập [0, 1]
    )
    model = model.to(device)

    criterion = nn.BCELoss()  # Binary Cross Entropy cho Multi-Label
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Tính accuracy (ngưỡng 0.5)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()

        scheduler.step()
        acc = correct / total * 100
        avg_loss = total_loss / len(loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")

    print(f"  ✅ Training hoàn tất! Accuracy cuối: {acc:.1f}%")
    return model


# =====================================================
# BƯỚC 3: Dùng CNN để gán nhãn 10k ảnh
# =====================================================
def propagate_labels(model, img_dir, output_csv, max_images=10000, threshold=0.4):
    print("\n" + "=" * 60)
    print("🔄 BƯỚC 2: TRUYỀN NHÃN TỚI 10,000 ẢNH (PROPAGATION)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    infer_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Quét tất cả ảnh
    all_imgs = glob.glob(f"{img_dir}/*.jpg")
    random.seed(42)
    random.shuffle(all_imgs)
    print(f"  Tìm thấy {len(all_imgs)} ảnh")

    # Mở CSV
    f_csv = open(output_csv, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f_csv)
    writer.writerow(['Image_Path'] + CLASSES)

    class_count = Counter()
    saved = 0
    skipped = 0

    # Cài bộ lọc Average Hash (A-Hash) rà soát trùng lặp
    def get_ahash(image):
        img_resized = image.resize((8, 8), Image.Resampling.LANCZOS).convert('L')
        pixels = np.array(img_resized.getdata())
        avg = pixels.mean()
        bits = "".join(['1' if p > avg else '0' for p in pixels])
        return hex(int(bits, 2))

    seen_hashes = set()
    pbar = tqdm(total=max_images, desc="Truyền nhãn")

    for img_path in all_imgs:
        if saved >= max_images:
            break

        try:
            img = Image.open(img_path).convert('RGB')
            if img.size[0] < 50 or img.size[1] < 50:
                continue

            # Lọc bỏ ảnh trùng lặp nhan nhản
            img_hash = get_ahash(img)
            if img_hash in seen_hashes:
                skipped += 1
                continue
            seen_hashes.add(img_hash)
        except Exception:
            continue

        # Inference
        img_tensor = infer_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(img_tensor)[0].cpu().numpy()  # (10,)

        # Gán nhãn: Top-2 nhãn có xác suất cao nhất (đảm bảo mỗi ảnh có ít nhất 1 nhãn)
        top2_idx = np.argsort(probs)[-2:][::-1]

        binary = [0] * NUM_CLASSES
        for idx in top2_idx:
            if probs[idx] >= threshold or idx == top2_idx[0]:
                # Luôn gán top-1, top-2 chỉ khi vượt ngưỡng
                binary[idx] = 1

        active = [CLASSES[i] for i, v in enumerate(binary) if v == 1]

        # Class Balancing
        if all(class_count[c] >= MAX_PER_CLASS for c in active):
            skipped += 1
            continue

        # Lọc nhãn đã đầy
        for i, cls in enumerate(CLASSES):
            if binary[i] == 1 and class_count[cls] >= MAX_PER_CLASS:
                binary[i] = 0

        if sum(binary) == 0:
            skipped += 1
            continue

        rel_path = f"cropped_students/{os.path.basename(img_path)}"
        writer.writerow([rel_path] + binary)

        for i, cls in enumerate(CLASSES):
            if binary[i] == 1:
                class_count[cls] += 1
        saved += 1
        pbar.update(1)

    pbar.close()
    f_csv.close()

    # Báo cáo
    print("\n" + "=" * 60)
    print(f"🎉 HOÀN TẤT! Đã gán nhãn {saved} ảnh → {output_csv}")
    print("=" * 60)
    print("\n📊 PHÂN BỐ NHÃN CUỐI CÙNG:")
    print("-" * 50)
    for cls in CLASSES:
        c = class_count[cls]
        bar = "█" * (c // 40)
        print(f"  {cls:25s}: {c:5d}  {bar}")
    print("-" * 50)
    print(f"  {'TỔNG ẢNH':25s}: {saved:5d}")


def main():
    seed_csv = 'data/seed_annotations_quality10k.csv'
    data_root = 'data'
    img_dir = 'data/cropped_students'
    output_csv = 'data/golden_annotations.csv'

    # Kiểm tra file seed
    if not os.path.exists(seed_csv):
        print("❌ Chưa có file seed_annotations.csv!")
        print("   Chạy trước: python annotate_golden_dataset.py")
        print("   (Gán tay 500 ảnh, chỉ mất ~30 phút)")
        return

    df = pd.read_csv(seed_csv)
    print(f"[*] Tìm thấy {len(df)} ảnh seed đã gán nhãn thủ công")

    if len(df) < 100:
        print(f"⚠️ Chỉ có {len(df)} ảnh seed. Cần ít nhất 100 ảnh để CNN học được.")
        print("   Tiếp tục gán nhãn: python annotate_golden_dataset.py")
        return

    # In phân bố nhãn seed
    print("\n📊 Phân bố nhãn Seed (do con người gán):")
    for c in CLASSES:
        print(f"  {c:25s}: {int(df[c].sum()):5d}")

    # BƯỚC 1: Train CNN
    model = train_classifier(seed_csv, data_root, epochs=30, batch_size=16)

    # BƯỚC 2: Propagate
    propagate_labels(model, img_dir, output_csv, max_images=10000)

    print("\n🚀 TIẾP THEO, chạy 2 lệnh:")
    print("   python split_dataset_50k.py")
    print("   python train.py --config configs/et_edu_config.yaml")


if __name__ == "__main__":
    main()
