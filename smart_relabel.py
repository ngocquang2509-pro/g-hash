"""
Smart Re-Labeling Pipeline V3: Single-Call Comparative + Class Balancing
========================================================================
Khắc phục triệt để bằng cách:
  1. GỌI CLIP 1 LẦN DUY NHẤT với tất cả 10 nhãn → Điểm số SO SÁNH ĐƯỢC
  2. Top-K Assignment: Mỗi ảnh nhận top 2-3 nhãn mạnh nhất
  3. Class Balancing: Giới hạn tối đa mỗi nhãn để cân bằng

Usage:
    python smart_relabel.py --max_images 10000
"""

import os
import csv
import glob
import torch
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter

BEHAVIORAL_CLASSES = [
    'using_phone', 'dozing_off', 'turning_sideways',
    'turning_back', 'raising_hand', 'opening_book',
    'reading', 'writing', 'listening', 'head_down',
    'sitting', 'standing', 'walking', 'interacting'
]

# Prompt mô tả chi tiết nhất cho mỗi nhãn (1 prompt/nhãn, gọi chung 1 lần)
PROMPTS = [
    "a student holding and looking at a mobile phone",        # using_phone
    "a student sleeping or dozing off on the desk",           # dozing_off
    "a student turning sideways to look at nearby classmates",# turning_sideways
    "a student turning back to look behind",                  # turning_back
    "a student raising their hand in class",                  # raising_hand
    "a student opening a book or notebook",                   # opening_book
    "a student reading a book or paper",                      # reading
    "a student writing notes with a pen",                     # writing
    "a student listening attentively to lecture",             # listening
    "a student with head down on desk without writing",       # head_down
    "a student sitting in class",                             # sitting
    "a student standing in class",                            # standing
    "a student walking in the classroom",                     # walking
    "students interacting or talking to each other",          # interacting
]

MAX_PER_CLASS = 2000  # Giới hạn tối đa mỗi nhãn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/cropped_students")
    parser.add_argument("--output_csv", default="data/golden_annotations.csv")
    parser.add_argument("--max_images", type=int, default=10000)
    parser.add_argument("--top_k", type=int, default=2,
                        help="Số nhãn tối đa gán cho mỗi ảnh (mặc định: top-2)")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("🧠 SMART RE-LABELING V3: Single-Call Comparative")
    print("=" * 70)

    # 1. Quét tất cả ảnh, xáo trộn để lấy mẫu đều từ các video
    all_imgs = glob.glob(f"{args.input_dir}/*.jpg")
    random.seed(42)
    random.shuffle(all_imgs)
    print(f"[*] Tìm thấy {len(all_imgs)} ảnh")

    # 2. Load CLIP
    print(f"[*] Đang tải CLIP: {args.model}...")
    from transformers import pipeline as hf_pipeline
    device = 0 if torch.cuda.is_available() else -1
    clip = hf_pipeline("zero-shot-image-classification", model=args.model, device=device)
    print("[✓] CLIP sẵn sàng!")

    # 3. Khởi tạo CSV
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    f_csv = open(args.output_csv, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f_csv)
    writer.writerow(['Image_Path'] + BEHAVIORAL_CLASSES)

    # 4. Class Balancing counter
    class_count = Counter()
    saved = 0
    skipped = 0

    pbar = tqdm(total=args.max_images, desc="Gán nhãn")

    for img_path in all_imgs:
        if saved >= args.max_images:
            break

        try:
            img = Image.open(img_path).convert('RGB')
            if img.size[0] < 50 or img.size[1] < 50:
                continue
        except Exception:
            continue

        # ===== GỌI CLIP 1 LẦN DUY NHẤT VỚI TẤT CẢ 10 PROMPTS =====
        # Kết quả trả về đã được Softmax → điểm số SO SÁNH ĐƯỢC giữa các nhãn
        try:
            results = clip(img, candidate_labels=PROMPTS)
        except Exception:
            continue

        # Map kết quả về tên nhãn
        score_map = {}
        for r in results:
            idx = PROMPTS.index(r['label'])
            score_map[BEHAVIORAL_CLASSES[idx]] = r['score']

        # Sắp xếp theo điểm giảm dần, lấy top-K
        sorted_cls = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        top_labels = [cls for cls, _ in sorted_cls[:args.top_k]]

        # Kiểm tra Class Balancing: Bỏ qua nếu tất cả nhãn top-K đã đầy quota
        if all(class_count[cls] >= MAX_PER_CLASS for cls in top_labels):
            skipped += 1
            continue

        # Lọc bỏ nhãn đã đầy, giữ lại nhãn còn slot
        final_labels = [cls for cls in top_labels if class_count[cls] < MAX_PER_CLASS]
        if not final_labels:
            skipped += 1
            continue

        # Tạo vector nhị phân
        binary = [1 if cls in final_labels else 0 for cls in BEHAVIORAL_CLASSES]

        # Ghi CSV
        rel_path = f"cropped_students/{os.path.basename(img_path)}"
        writer.writerow([rel_path] + binary)

        for cls in final_labels:
            class_count[cls] += 1
        saved += 1
        pbar.update(1)

        if saved % 500 == 0:
            dist = {c: class_count[c] for c in BEHAVIORAL_CLASSES}
            pbar.set_postfix(saved=saved, skip=skipped)

    pbar.close()
    f_csv.close()

    # 5. Báo cáo kết quả
    print("\n" + "=" * 70)
    print(f"🎉 HOÀN TẤT! Đã gán nhãn {saved} ảnh → {args.output_csv}")
    print(f"   Bỏ qua {skipped} ảnh do Class Balancing")
    print("=" * 70)
    print("\n📊 PHÂN BỐ NHÃN CUỐI CÙNG:")
    print("-" * 50)
    for cls in BEHAVIORAL_CLASSES:
        c = class_count[cls]
        bar = "█" * (c // 40)
        print(f"  {cls:25s}: {c:5d}  {bar}")
    print("-" * 50)
    total_labels = sum(class_count.values())
    print(f"  {'TỔNG ẢNH':25s}: {saved:5d}")
    print(f"  {'TỔNG NHÃN':25s}: {total_labels:5d}")
    print(f"  {'TB NHÃN/ẢNH':25s}: {total_labels/max(saved,1):.1f}")


if __name__ == "__main__":
    main()
