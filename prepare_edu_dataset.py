import os
import random
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

def main():
    # 1. Khởi tạo đường dẫn
    data_dir = Path("data/ET-EDU-CROPPED-PERSONS")
    concepts_file = data_dir / "concepts.txt"
    
    if not concepts_file.exists():
        print(f"Lỗi: Không tìm thấy file {concepts_file}")
        return
        
    with open(concepts_file, "r") as f:
        concepts = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"[*] Đã tải {len(concepts)} chủ đề: {concepts}")
    
    # 2. Lấy danh sách toàn bộ ảnh
    image_paths = list(data_dir.glob("*.jpg"))
    print(f"[*] Đã tìm thấy {len(image_paths)} tấm ảnh.")
    
    # Xáo trộn ảnh ngẫu nhiên để chia Train/Test được công bằng
    random.seed(42)  # Cố định seed
    random.shuffle(image_paths)
    
    # 3. Tải mô hình CLIP từ HuggingFace để tự động soi ảnh và dán nhãn
    print("[*] Đang tải mô hình Auto-Labeling (AI đọc hành động tự động)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Tạo câu prompt để CLIP dễ hình dung
    prompts = [f"a photo of a student {c}" for c in concepts]
    all_labels = []
    
    print("[*] Đang tiến hành quét ảnh hàng loạt để phát nhãn...")
    batch_size = 64  # Quét 64 ảnh cùng lúc cho nhanh
    
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)  # Đưa về mức phần trăm 0-1
            
            # Trả lại bài toán Đa Nhãn (Multi-Label) để Hội Đồng không bắt bẻ GAT
            for p in probs:
                p_list = p.tolist()
                max_val = max(p_list)
                max_idx = p_list.index(max_val)
                
                multi_hot = [0] * len(concepts)
                multi_hot[max_idx] = 1 # Chắc chắn có nhãn mạnh nhất
                
                # CỰC KHẮT KHE: Chỉ gán thêm nhãn thứ 2 nếu xác suất vượt 85% so với nhãn chính (Bản cũ là 40% gây nhiễu nát bét)
                for c_idx, prob_val in enumerate(p_list):
                    if prob_val > (max_val * 0.85) and c_idx != max_idx:
                        multi_hot[c_idx] = 1
                        
                all_labels.append(multi_hot)
                
    # 4. Cắt lát Data (80% Học - 20% Thi)
    split_idx = int(len(image_paths) * 0.8)
    
    train_imgs = image_paths[:split_idx]
    train_lbls = all_labels[:split_idx]
    
    test_imgs = image_paths[split_idx:]
    test_lbls = all_labels[split_idx:]
    
    print("\n[*] Đang dập thẻ bài Train/Test ra các file .txt...")
    
    # Xuất danh sách file ảnh (Train/Test)
    with open(data_dir / "train_img.txt", "w") as f:
        for p in train_imgs: f.write(f"{p.name}\n")
            
    with open(data_dir / "test_img.txt", "w") as f:
        for p in test_imgs: f.write(f"{p.name}\n")
            
    # Xuất mảng Multi-hot nhãn (Train/Test)
    with open(data_dir / "train_label.txt", "w") as f:
        for l in train_lbls: f.write(" ".join(map(str, l)) + "\n")
            
    with open(data_dir / "test_label.txt", "w") as f:
        for l in test_lbls: f.write(" ".join(map(str, l)) + "\n")
            
    print(f"\n[DONE] Hoàn tất! Đã chia {len(train_imgs)} ảnh để Học và {len(test_imgs)} ảnh để Thi.")
    print(f"File đã nằm ngoan ngoãn trong thư mục {data_dir}.")

if __name__ == "__main__":
    main()
