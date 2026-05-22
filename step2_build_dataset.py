import os
import random
from pathlib import Path

def build_ghash_dataset(work_dir, test_ratio=0.2):
    """
    Bước 2: Chạy quét các thư mục mà người dùng đã phân loại tay và sinh ra File txt huân luyện chuẩn
    """
    work_dir = Path(work_dir)
    
    if not work_dir.exists():
        print(f"❌ Thư mục {work_dir} không tồn tại! Vui lòng tạo thư mục này và chia ảnh vào các thư mục con.")
        return

    # Quét nhận diện Class (Tên các thư mục con trong SORTED)
    class_folders = [f for f in work_dir.iterdir() if f.is_dir()]
    if not class_folders:
        print(f"❌ Bạn cần tạo các thư mục con (Ví dụ: Giao_Vien_Giang, Sinh_Vien_Hoc) bên trong {work_dir} và dán ảnh vào đó!")
        return

    class_list = sorted([f.name for f in class_folders])
    num_classes = len(class_list)
    class_to_idx = {cls: i for i, cls in enumerate(class_list)}
    
    print(f"🏷️ Hệ thống nhận diện {num_classes} nhãn hành vi/môn học: {class_list}")
    
    # Gom Nhặt toàn bộ ảnh trong các thư mục đó
    dataset_records = []
    total_imgs = 0
    
    for folder in class_folders:
        class_name = folder.name
        class_idx = class_to_idx[class_name]
        
        # Lược lấy toàn bộ ảnh jpeg, jpg, png
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        
        for img_path in images:
            # Lưu đường dẫn dạng 'Sinh_Vien_Hoc/frame123.jpg'
            rel_path = img_path.relative_to(work_dir).as_posix()
            dataset_records.append((rel_path, class_idx))
            total_imgs += 1
            
        print(f"  - Lớp [{class_name}]: {len(images)} ảnh")
        
    if total_imgs == 0:
        print("❌ Chưa có tấm ảnh nào trong các thư mục này! Vui lòng copy ảnh vào.")
        return
        
    print(f"\n✅ Đã gom được {total_imgs} bức ảnh đã gắn nhãn!")
    
    # Sắp xếp và chia Data (Train / Test)
    random.seed(42)  # Đảm bảo trộn ngẫu nhiên sinh ra code giống nhau nếu chạy lại nhiều lần
    random.shuffle(dataset_records)
    
    split_idx = int(len(dataset_records) * (1 - test_ratio))
    train_records = dataset_records[:split_idx]
    test_records = dataset_records[split_idx:]
    
    print(f"✂️ Chia tỷ lệ bộ Data: Train={len(train_records)} ({100-test_ratio*100}%), Test={len(test_records)} ({test_ratio*100}%)\n")
    
    # Tạo Ma Trận Text TXT tiêu chuẩn cho Model
    def write_dataset_files(records, img_file, label_file):
        with open(work_dir / img_file, 'w') as f_img, open(work_dir / label_file, 'w') as f_label:
            for img_path, class_idx in records:
                f_img.write(f"{img_path}\n")
                
                # Mã hóa vector 0 1
                one_hot = [0] * num_classes
                one_hot[class_idx] = 1 
                f_label.write(" ".join(map(str, one_hot)) + " \n")
                
    write_dataset_files(train_records, "database_img.txt", "database_label_onehot.txt")
    write_dataset_files(test_records, "test_img.txt", "test_label_onehot.txt")
    
    # Sinh file danh sách nhóm class giống Concepts81.txt
    with open(work_dir / "conceptslist.txt", 'w', encoding='utf-8') as f:
        for i, cls in enumerate(class_list):
            f.write(f"Class {i}: {cls}\n")
            
    print("🚀 THÀNH CÔNG! ĐÃO TẠO SẴN SÀNG TOÀN BỘ MA TRẬN:")
    print(f"  - Các File đã được ném thẳng vào thư mục: {work_dir.absolute()}")
    print("👉 HƯỚNG DẪN KẾ TIẾP:")
    print(f" 👉 1. Mở file configs/config.yaml")
    print(f" 👉 2. Sửa thông số: data_root: \"{work_dir}\"")
    print(f" 👉 3. Chạy python train.py")

if __name__ == "__main__":
    # Thư mục gốc nơi bạn đã phân loại tay các ảnh!
    SORTED_DIR = "data/ET-EDU-SORTED"
    
    print("="*60)
    print("BƯỚC 2: BUILD DATASET TỪ ẢNH ĐÃ PHÂN LOẠI")
    print("="*60)
    
    build_ghash_dataset(SORTED_DIR, test_ratio=0.2)
