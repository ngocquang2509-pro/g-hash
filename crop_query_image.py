import cv2
from pathlib import Path
from ultralytics import YOLO

def crop_people(image_path):
    print(f"\n[*] Đang nạp Mắt Thần YOLOv8 để cắt tỉa ảnh: {image_path}")
    
    # Gọi YOLOv8 (phiên bản siêu nhẹ Nano đang có sẵn trong thư mục của bạn)
    model = YOLO('yolov8n.pt') 
    
    # Đọc ảnh Mộc
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Lỗi: Không thể mở hay tìm thấy file ảnh {image_path}")
        return
        
    print("[*] Đang chẻ nhỏ khung hình để nhận diện từng khuôn mặt/dáng người...")
    # Chạy YOLO cho AI tự mò các Bounding Box
    results = model(img)
    
    count = 0
    # Lục tung các chiếc hộp (box) mà YOLO tìm thấy
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            # Chỉ lấy DUY NHẤT Class 0 (Trong danh sách COCO của YOLO, 0 = Con người / Person)
            if int(box.cls[0]) == 0:
                # Rút lấy 4 toạ độ (Góc trên trái và dưới phải)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Cắt xén (Crop) trực tiếp tấm ảnh gốc ra
                crop_img = img[y1:y2, x1:x2]
                
                # Phễu lọc: Bỏ đi lính lác liti phía xa mờ (Ví dụ bé hơn 50x50px)
                if crop_img.shape[0] < 50 or crop_img.shape[1] < 50:
                    continue
                    
                # Tạo tên file mới. VD: mắm__crop_0.jpg
                save_path = f"{image_path.rsplit('.', 1)[0]}_crop_{count}.jpg"
                cv2.imwrite(save_path, crop_img)
                count += 1
                
                print(f"  👉 Cắt thành công Nhân vật {count}: {save_path}")
                
    if count == 0:
        print("⚠️ Ồ... YOLOv8 không tìm thấy người nào đủ lớn hoặc tấm ảnh này chỉ là Bàn ghế trống!")
    else:
        print(f"\n🎉 HOÀN TẤT TUYỆT ĐỐI! Đã rọc ra được {count} người để làm Mồi nhử AI!")
        print("Bây giờ bạn chỉ cần copy dòng đường dẫn của nhân vật đẹp nhất...")
        print("...rồi thay vào đuôi chữ --image trong lệnh inference.py là Hệ thống sẽ tìm ra ngay!\n")
        
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        crop_people(sys.argv[1])
    else:
        crop_people("images/hocsinh.jpg")
