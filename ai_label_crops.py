import os
import glob
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# 5 Khái niệm cốt lõi cho Đồ án (Các trạng thái của sinh viên)
CONCEPTS = [
    "sitting",       # Đang ngồi
    "standing",      # Đang đứng
    "raising hand",  # Đang giơ tay 
    "writing",       # Đang cắm cúi viết bài
    "using phone"    # Đang lướt điện thoại
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"==================================================")
print(f"Đang nạp Siêu AI OpenAI CLIP lên bộ xử lý: {device.upper()}")
print(f"==================================================")
# Tải mô hình CLIP từ máy chủ Mạng
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def process_folder(folder_name, output_img, output_lbl):
    src_dir = f"data/ET-EDU-CROPPED-PERSONS/{folder_name}"
    images = glob.glob(os.path.join(src_dir, "*.jpg")) + glob.glob(os.path.join(src_dir, "*.png"))
    
    # Sắp xếp đúng thứ tự từ trên xuống dưới
    images.sort()
    
    with open(output_img, 'w') as f_img, open(output_lbl, 'w') as f_lbl:
        print(f"\n⏳ Bắt đầu phân tích {len(images)} ảnh của rổ dữ liệu [{folder_name.upper()}]...")
        
        # Xử lý theo chùm (Batch size 32 ảnh 1 lần cho lẹ)
        batch_size = 32
        
        for i in tqdm(range(0, len(images), batch_size)):
            batch_paths = images[i:i+batch_size]
            batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            
            # Gợi ý cực kỳ thông minh cho AI: "Một bức ảnh sinh viên đang..."
            text_prompts = [f"a photo of a student {c}" for c in CONCEPTS]
            
            inputs = processor(text=text_prompts, images=batch_imgs, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image  # Điểm tương đồng giữa ảnh và text
                probs = logits_per_image.softmax(dim=1).cpu().numpy() # Ép về dạng \%
            
            # Xử lý kết quả in ra file Txt
            for j, p in enumerate(batch_paths):
                # 1. Điền đường dẫn của ảnh
                rel_path = p.replace("\\", "/") 
                f_img.write(f"{rel_path}\n")
                
                # 2. Xây dựng Vector Nhãn Multi-hot (ví dụ: [1, 0, 1, 0, 0] là Ngồi + Giơ tay)
                # THAY ĐỔI: Dùng Ngưỡng xác suất (Threshold = 35%). 
                # Mức 15% quá dễ dãi khiến 90% Dataset dính chữ "Ngồi" làm mAP lạm phát lên 0.9012.
                # Lên 35% sẽ ép Siêu AI CLIP phải cực kỳ chắc chắn mới dám thả nhãn thứ 2!
                threshold = 0.35
                hot_vector = [1 if prob >= threshold else 0 for prob in probs[j]]
                
                # Đề phòng AI quá tự ti không nhãn nào qua 15%, ta vẫn lấy 1 nhãn cao nhất
                if sum(hot_vector) == 0:
                    best_idx = probs[j].argmax()
                    hot_vector[best_idx] = 1
                
                # Ghi Vector ra cách nhau bởi dấu cách (Chuẩn cấu trúc NUS-WIDE 2 đa nhãn)
                f_lbl.write(" ".join(map(str, hot_vector)) + "\n")

# === Bấm nút chạy bộ Lọc ===
process_folder("train", "data/ET-EDU-CROPPED-PERSONS/train_img.txt", "data/ET-EDU-CROPPED-PERSONS/train_label.txt")
process_folder("test", "data/ET-EDU-CROPPED-PERSONS/test_img.txt", "data/ET-EDU-CROPPED-PERSONS/test_label.txt")

# Chốt sổ danh sách 5 từ khóa Labels
with open("data/ET-EDU-CROPPED-PERSONS/concepts.txt", "w") as f:
    for c in CONCEPTS:
        f.write(f"{c}\n")

print("\n🎉 HOÀN TẤT CHIẾN DỊCH GÁN NHÃN TỰ ĐỘNG BẰNG AI!")
print("Đã tạo sẵn 5 file xương sống ở data/ET-EDU-CROPPED-PERSONS chuẩn bị cho Fine-tuning.")
