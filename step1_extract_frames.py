import os
import cv2
import numpy as np
from pathlib import Path

def is_different_enough(frame1, frame2, threshold=15.0):
    """
    So sánh độ lệch pixel giữa 2 khung hình bằng mức trung bình.
    Nếu ảnh quá tĩnh (giống hệt nhau), hàm này trả về False để không lưu rác.
    """
    if frame1 is None or frame2 is None:
        return True
    # Chuyển về đen trắng để so sánh cực lẹ
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Tính độ lệch trung bình giữa các Pixel
    mean_diff = np.mean(cv2.absdiff(gray1, gray2))
    return mean_diff > threshold

def extract_raw_frames(source_dir, target_dir, extract_interval_sec=5, diff_threshold=15.0):
    """
    Bước 1: Trích xuất thô KHÔNG PHÂN LOẠI từ tất cả các video lấy 1 khung hình mỗi `extract_interval_sec` giây.
    Có tích hợp thuật toán lọc ảnh trùng lặp!
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Tạo thư mục chứa thô
    target_dir.mkdir(parents=True, exist_ok=True)
    
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov']
    video_paths = []
    
    for ext in video_extensions:
        video_paths.extend(source_dir.rglob(f'*{ext}'))
        
    if not video_paths:
        print(f"❌ Không tìm thấy video nào trong {source_dir}")
        return
        
    print(f"🎥 Đã tìm thấy {len(video_paths)} videos. Bắt đầu trích xuất siêu lọc (khoảng cạch {extract_interval_sec} giây/ảnh)...")
    
    total_frames = 0
    for vid_path in video_paths:
        cap = cv2.VideoCapture(str(vid_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Nếu không lấy được fps, mặc định là 30
        if fps == 0 or fps != fps:
            fps = 30.0
            
        frame_interval = int(round(fps * extract_interval_sec))
        
        count = 0
        saved_count = 0
        last_saved_frame = None
        
        print(f"⏳ Cắt video: {vid_path.name}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % frame_interval == 0:
                # Kiểm tra thuật toán ảnh tĩnh, tránh lưu cả 10 ảnh thầy giáo chỉ đứng im
                if is_different_enough(last_saved_frame, frame, diff_threshold):
                    img_name = f"{vid_path.stem}_{saved_count:04d}.jpg"
                    img_save_path = target_dir / img_name
                    
                    # Để nguyên khung ảnh gốc sắc nét
                    cv2.imwrite(str(img_save_path), frame)
                    
                    # Cập nhật bộ nhớ đệm
                    last_saved_frame = frame.copy()
                    
                    saved_count += 1
                    total_frames += 1
                
            count += 1
            
        cap.release()
        print(f"  -> Sinh ra được {saved_count} ảnh (Đã tự động vứt vô số ảnh trùng lặp đứng im)")
        
    print(f"\n✅ Tổng cộng đã băm nhỏ {total_frames} ảnh! Mời bạn vào thư mục {target_dir} để phân loại bằng tay.")

if __name__ == "__main__":
    # Thay đổi thông số này theo ý bạn:
    # Lấy 1 bức ảnh cứ sau mỗi 3 giây video để tránh các hình chụp bị lặp lại y hệt
    SECONDS_PER_FRAME = 3  

    SOURCE_VIDEOS = "data/ET-EDU"
    RAW_FRAMES_OUTPUT = "data/ET-EDU-RAW-FRAMES"
    
    print("="*60)
    print("BƯỚC 1: TRÍCH XUẤT ẢNH THÔ TỪ VIDEO (KHÔNG GẮN NHÃN)")
    print("="*60)
    extract_raw_frames(SOURCE_VIDEOS, RAW_FRAMES_OUTPUT, SECONDS_PER_FRAME)
