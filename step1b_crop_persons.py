import os
import cv2
import numpy as np
from pathlib import Path
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Vui lòng cài đặt thư viện nhận diện người bằng lệnh: pip install ultralytics")
    exit(1)

def is_blurry(image, threshold=100.0):
    """
    Hàm kiểm tra xem ảnh cắt ra có bị nhòe nét hay không (nếu mờ quá thì bỏ đi).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def crop_persons_from_frames(source_dir, target_dir, min_size=(64, 64), conf_thresh=0.45):
    """
    Quét qua tất cả khung hình toàn cảnh, dùng AI (YOLOv8) bóc tách toàn bộ 
    học sinh/giáo viên trong lớp thành các tấm ảnh chân dung/cá nhân riêng lẻ.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Chỉ đọc .jpg, .png
    images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    
    if not images:
        print(f"❌ Không tìm thấy bức ảnh gốc nào ở {source_dir}")
        return
        
    print(f"Bắt đầu Load trí tuệ nhân tạo (YOLOv8)...")
    # Tự động tải model yolov8n (nano) nhẹ gọn và nhận diện nhanh nhất
    model = YOLO("yolov8n.pt")
    
    print(f"🔍 Quét {len(images)} bức ảnh toàn cảnh để bóc tách từng sinh viên...")
    
    total_persons_cropped = 0
    
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
            
        # Dùng YOLO tìm tất cả mọi thứ trong ảnh
        results = model.predict(frame, verbose=False)
        
        # results[0] chứa danh sách các hộp giới hạn (Bounding boxes)
        boxes = results[0].boxes
        
        person_count_in_frame = 0
        for i, box in enumerate(boxes):
            # class 0 là "person" (Người) trong COCO dataset
            if int(box.cls[0]) == 0 and float(box.conf[0]) >= conf_thresh:
                # Lấy tọa độ cắt x1, y1, x2, y2
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Cắt (Crop) khung hình đúng người đó ra
                crop_img = frame[y1:y2, x1:x2]
                
                # Bỏ qua những sinh viên ngồi quá xa, bé tí mờ nhạt
                h, w = crop_img.shape[:2]
                if w < min_size[0] or h < min_size[1]:
                    continue
                    
                # Bỏ qua ảnh người bị nhòe (Chụp lúc họ đang hất đầu/xoay mạnh)
                if is_blurry(crop_img, threshold=40.0):
                    continue
                
                # Lưu file ảnh cá thể (Tên file gốc + số thứ tự người)
                person_filename = f"{img_path.stem}_person_{person_count_in_frame:02d}.jpg"
                save_path = target_dir / person_filename
                
                cv2.imwrite(str(save_path), crop_img)
                
                person_count_in_frame += 1
                total_persons_cropped += 1
                
        if person_count_in_frame > 0:
            print(f"  -> {img_path.name}: Gắp ra được {person_count_in_frame} cá nhân")
            
    print("\n" + "="*60)
    print(f"✅ THÀNH CÔNG! Đã bóc tách tổng cộng {total_persons_cropped} học sinh/giáo viên!")
    print(f"Mời bạn vào tập dữ liệu siêu lọc cá nhân tại: {target_dir.absolute()}")
    print("="*60)

if __name__ == "__main__":
    # Đọc từ thư mục vừa được cắt xén từ video
    SOURCE_RAW_FRAMES = "data/ET-EDU-RAW-FRAMES"
    # Thư mục mới chuyên gắp sinh viên
    OUTPUT_CROPPED_PERSONS = "data/ET-EDU-CROPPED-PERSONS"
    
    crop_persons_from_frames(SOURCE_RAW_FRAMES, OUTPUT_CROPPED_PERSONS)
