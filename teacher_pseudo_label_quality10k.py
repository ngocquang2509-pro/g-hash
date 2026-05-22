import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models


CLASSES = [
    "using_phone",
    "dozing_off",
    "turning_sideways",
    "turning_back",
    "raising_hand",
    "opening_book",
    "reading",
    "writing",
    "listening",
    "head_down",
    "sitting",
    "standing",
    "walking",
    "interacting",
]


class SeedDataset(Dataset):
    def __init__(self, csv_path: Path, data_root: Path, transform=None):
        self.transform = transform
        self.samples: List[Tuple[Path, List[int]]] = []
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            rel_path = row["Image_Path"]
            img_path = data_root / rel_path
            if not img_path.exists():
                continue
            labels = [int(row[c]) for c in CLASSES]
            self.samples.append((img_path, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(labels, dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train teacher on seed labels, then pseudo-label remaining quality10k images."
    )
    parser.add_argument("--seed-csv", default="data/seed_annotations_quality10k.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--input-dir", default="data/cropped_students_quality_10k")
    parser.add_argument("--rel-prefix", default="cropped_students_quality_10k")
    parser.add_argument("--pseudo-output-csv", default="data/teacher_pseudo_quality10k.csv")
    parser.add_argument("--merged-output-csv", default="data/annotations_quality10k_teacher_merged.csv")
    parser.add_argument("--teacher-ckpt", default="experiments/teacher_quality10k_seed500.pth")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--max-per-class", type=int, default=2500)
    parser.add_argument("--max-new-images", type=int, default=0, help="0 = all remaining images")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_teacher_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def train_teacher(args, device):
    train_tf = T.Compose(
        [
            T.Resize((256, 256)),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(12),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = SeedDataset(Path(args.seed_csv), Path(args.data_root), transform=train_tf)
    if len(dataset) == 0:
        raise RuntimeError("No valid seed samples found for teacher training.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Compute pos_weight from seed set to reduce class imbalance bias.
    y = np.array([labels for _, labels in dataset.samples], dtype=np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pos_weight = np.clip(neg / np.clip(pos, 1.0, None), 1.0, 15.0)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    model = build_teacher_model(len(CLASSES)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    print(f"Teacher train samples: {len(dataset)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch:02d}/{args.epochs} - loss: {avg_loss:.4f}")

        if args.save_every_epoch:
            ckpt_tmp = Path(args.teacher_ckpt).with_name(f"{Path(args.teacher_ckpt).stem}_e{epoch:02d}.pth")
            ckpt_tmp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "classes": CLASSES}, ckpt_tmp)

    Path(args.teacher_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "classes": CLASSES}, args.teacher_ckpt)
    return model


def list_remaining_images(seed_csv: Path, input_dir: Path, rel_prefix: str):
    df = pd.read_csv(seed_csv)
    labeled = set(df["Image_Path"].tolist())
    files = sorted(input_dir.glob("*.jpg"))
    remaining = []
    for p in files:
        rel = f"{rel_prefix}/{p.name}"
        if rel in labeled:
            continue
        remaining.append((p, rel))
    return remaining


def pseudo_label_remaining(model, args, device):
    infer_tf = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    seed_df = pd.read_csv(args.seed_csv)
    class_count = Counter()
    for cls in CLASSES:
        class_count[cls] = int(seed_df[cls].sum())

    remaining = list_remaining_images(Path(args.seed_csv), Path(args.input_dir), args.rel_prefix)
    random.shuffle(remaining)
    if args.max_new_images > 0:
        remaining = remaining[: args.max_new_images]

    pseudo_rows = []
    model.eval()
    with torch.no_grad():
        for img_path, rel_path in tqdm(remaining, desc="Teacher pseudo-label"):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            x = infer_tf(img).unsqueeze(0).to(device)
            logits = model(x)[0]
            probs = torch.sigmoid(logits).cpu().numpy()

            top_idx = np.argsort(probs)[-args.top_k :][::-1]
            binary = [0] * len(CLASSES)
            for idx in top_idx:
                if probs[idx] >= args.threshold or idx == top_idx[0]:
                    binary[idx] = 1

            # class balancing against current merged counts.
            for i, cls in enumerate(CLASSES):
                if binary[i] == 1 and class_count[cls] >= args.max_per_class:
                    binary[i] = 0

            if sum(binary) == 0:
                continue

            for i, cls in enumerate(CLASSES):
                if binary[i] == 1:
                    class_count[cls] += 1

            pseudo_rows.append([rel_path] + binary)

    return pseudo_rows


def merge_seed_and_pseudo(seed_csv: Path, pseudo_rows: List[List], merged_csv: Path):
    seed_df = pd.read_csv(seed_csv)
    header = ["Image_Path"] + CLASSES
    seen = set()
    merged_rows = []

    for _, row in seed_df.iterrows():
        r = [row["Image_Path"]] + [int(row[c]) for c in CLASSES]
        if r[0] in seen:
            continue
        merged_rows.append(r)
        seen.add(r[0])

    for r in pseudo_rows:
        if r[0] in seen:
            continue
        merged_rows.append(r)
        seen.add(r[0])

    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    with merged_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(merged_rows)

    return len(merged_rows)


def write_csv(path: Path, rows: List[List]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_Path"] + CLASSES)
        w.writerows(rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = train_teacher(args, device)
    pseudo_rows = pseudo_label_remaining(model, args, device)
    write_csv(Path(args.pseudo_output_csv), pseudo_rows)
    merged_total = merge_seed_and_pseudo(
        seed_csv=Path(args.seed_csv),
        pseudo_rows=pseudo_rows,
        merged_csv=Path(args.merged_output_csv),
    )

    print("\nDone.")
    print(f"Teacher checkpoint: {args.teacher_ckpt}")
    print(f"Pseudo rows: {len(pseudo_rows)} -> {args.pseudo_output_csv}")
    print(f"Merged rows total: {merged_total} -> {args.merged_output_csv}")


if __name__ == "__main__":
    main()
