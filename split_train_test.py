import os
import glob
import random
import shutil

def split_dataset_by_video():
    src_dir = "data/ET-EDU-CROPPED-PERSONS"
    
    # 1. Khởi tạo hộc tủ chứa dữ liệu chia
    train_dir = os.path.join(src_dir, "train")
    test_dir = os.path.join(src_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 2. Quét toàn bộ ảnh
    images = glob.glob(os.path.join(src_dir, "*.jpg")) + glob.glob(os.path.join(src_dir, "*.png"))
    if not images:
        print("❌ Không tìm thấy ảnh nào trong thư mục. Vui lòng kiểm tra lại!")
        return
        
    print(f"🔍 Đã quét được tổng cộng {len(images)} ảnh sinh viên.")
    
    # 3. Phân cụm Nhóm ảnh theo Video ID
    video_groups = {}
    for img in images:
        filename = os.path.basename(img)
        parts = filename.split('_')
        if len(parts) >= 2:
            video_id = parts[0] + "_" + parts[1]
        else:
            video_id = "unknown_video"
            
        if video_id not in video_groups:
            video_groups[video_id] = []
        video_groups[video_id].append(img)
        
    videos = list(video_groups.keys())
    print(f"🎬 Khám phá ra {len(videos)} Video ID gốc:")
    for v in videos:
        print(f"   - Clip {v}: chứa {len(video_groups[v])} ảnh nhỏ")
        
    # 4. Trộn ngẫu nhiên danh sách Video (Fix seed ngẫu nhiên nhưng cố định kết quả)
    random.seed(42)
    random.shuffle(videos)
    
    # 5. Phân chẻ bánh: 10 Video đi Train, phần dôi dư đi Test
    split_index = 10
    train_videos = videos[:split_index]
    test_videos = videos[split_index:]
    
    print("\n" + "="*50)
    print(f"Phân rổ TRAIN gồm {len(train_videos)} clip: {train_videos}")
    print(f"Phân rổ TEST gồm {len(test_videos)} clip: {test_videos}")
    print("="*50 + "\n")
    
    # 6. Sao chép ảnh (Copy)
    print("⏳ Đang sao chép ảnh phân vào 2 hộc tủ 'train' và 'test'...")
    train_count = 0
    test_count = 0
    
    # Xếp đồ ảnh Train
    for v in train_videos:
        for img in video_groups[v]:
            shutil.copy2(img, os.path.join(train_dir, os.path.basename(img)))
            train_count += 1
            
    # Xếp đồ ảnh Test
    for v in test_videos:
        for img in video_groups[v]:
            shutil.copy2(img, os.path.join(test_dir, os.path.basename(img)))
            test_count += 1
            
    print("\n🎉 XONG PHI VỤ!")
    print(f"✅ Thư mục TRAIN  : {train_count} ảnh")
    print(f"✅ Thư mục TEST   : {test_count} ảnh")
    print(f"📁 Hãy truy cập: {src_dir} để xem thành quả!")

if __name__ == "__main__":
    split_dataset_by_video()
