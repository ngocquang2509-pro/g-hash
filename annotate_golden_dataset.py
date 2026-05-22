import cv2
import os
import csv
import glob
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Danh sách nhãn hành vi mở rộng
CLASSES = [
    'using phone', 'dozing off', 'turning sideways',
    'turning back', 'raising hand', 'opening book',
    'reading', 'writing', 'listening', 'head down',
    'sitting', 'standing', 'walking', 'interacting'
]

# Ánh xạ Phím -> Chỉ số nhãn
KEY_MAP = {
    ord('1'): 0, ord('2'): 1, ord('3'): 2, ord('4'): 3, ord('5'): 4,
    ord('6'): 5, ord('7'): 6, ord('8'): 7, ord('9'): 8, ord('0'): 9,
    ord('a'): 10, ord('b'): 11, ord('c'): 12, ord('d'): 13
}

# Chú thích tiếng Việt
CLASSES_VI = {
    'using phone':       'Dùng điện thoại',
    'dozing off':        'Ngủ gật',
    'turning sideways':  'Quay ngang',
    'turning back':      'Quay ra sau',
    'raising hand':      'Giơ tay',
    'opening book':      'Mở sách',
    'reading':           'Đọc sách',
    'writing':           'Ghi chép',
    'listening':         'Lắng nghe',
    'head down':         'Cúi đầu',
    'sitting':           'Ngồi',
    'standing':          'Đứng',
    'walking':           'Đi lại',
    'interacting':       'Tương tác',
}

KEY_CHARS = ['1','2','3','4','5','6','7','8','9','0','a','b','c','d']

# Load font Unicode hỗ trợ tiếng Việt
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
try:
    FONT_LARGE = ImageFont.truetype(FONT_PATH, 20)
    FONT_SMALL = ImageFont.truetype(FONT_PATH, 16)
except:
    FONT_LARGE = ImageFont.load_default()
    FONT_SMALL = ImageFont.load_default()


def draw_ui_pil(ui_width, ui_height, count, target, classes, classes_vi, key_chars, current_labels):
    """Vẽ bảng UI bằng PIL (hỗ trợ Unicode tiếng Việt có dấu)"""
    img_pil = Image.new('RGB', (ui_width, ui_height), (0, 0, 0))
    draw = ImageDraw.Draw(img_pil)

    # Tiêu đề
    draw.text((10, 5), f"ĐÃ GÁN: {count}/{target} ẢNH", font=FONT_LARGE, fill=(0, 255, 255))
    draw.text((10, 30), "Trạng thái (Bấm phím):", font=FONT_SMALL, fill=(255, 255, 255))

    for idx, cls_name in enumerate(classes):
        kc = key_chars[idx]
        vi = classes_vi[cls_name]
        is_active = current_labels[idx] == 1
        color = (0, 255, 0) if is_active else (120, 120, 120)
        marker = "■" if is_active else "□"
        text = f"[{kc}] {marker} {cls_name} - {vi}"
        y = 52 + idx * 22
        draw.text((10, y), text, font=FONT_SMALL, fill=color)

    # Hướng dẫn phía dưới
    draw.text((10, 52 + len(classes) * 22 + 8), "[Enter] Lưu  [Space] Bỏ qua  [U] Quay lại  [Q] Thoát", font=FONT_SMALL, fill=(180, 180, 180))

    return np.array(img_pil)


def parse_args():
    parser = argparse.ArgumentParser(description="Manual multi-label annotation tool.")
    parser.add_argument("--img-dir", default="data/cropped_students", help="Directory of images to annotate.")
    parser.add_argument("--csv-file", default="data/seed_annotations.csv", help="Output CSV path.")
    parser.add_argument("--target", type=int, default=500, help="Target number of annotated images.")
    parser.add_argument(
        "--rel-prefix",
        default=None,
        help="Relative prefix written to CSV (default: basename of --img-dir).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("🔥 CÔNG CỤ GÁN NHÃN - GOLDEN SUBSET (MULTI-LABEL) 🔥")
    print("="*60)
    print("CÁCH SỬ DỤNG BÀN PHÍM:")
    for idx, cls in enumerate(CLASSES):
        vi = CLASSES_VI[cls]
        print(f"  Phím [{KEY_CHARS[idx]}] : {cls} ({vi})")
    print("-" * 40)
    print("  [ENTER] : LƯU ảnh này và Chuyển ảnh tiếp theo")
    print("  [SPACE] : BỎ QUA ảnh này (Nếu rác/mờ)")
    print("  [U] : QUAY LẠI ảnh vừa gán gần nhất để gán lại")
    print("  [Q] hoặc [ESC] : THOÁT VÀ LƯU DỮ LIỆU")
    print("="*60)

    img_dir = args.img_dir
    csv_file = args.csv_file
    rel_prefix = args.rel_prefix or Path(img_dir).name
    target = args.target

    # 1. Tìm tất cả ảnh
    all_imgs = glob.glob(f"{img_dir}/*.jpg")
    random.shuffle(all_imgs)

    # Khôi phục trạng thái cũ (Resume)
    header = ['Image_Path'] + [c.replace(' ', '_') for c in CLASSES]
    annotations = {}
    annotation_order = []
    if os.path.exists(csv_file):
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
            if existing_header and existing_header == header:
                for row in reader:
                    if row:
                        rel_path = row[0]
                        labels = [int(v) for v in row[1:1 + len(CLASSES)]]
                        if len(labels) == len(CLASSES):
                            annotations[rel_path] = labels
                            annotation_order.append(rel_path)
            else:
                print("⚠️ Header file seed cũ không khớp schema hiện tại, tạo file mới.")
    if not os.path.exists(csv_file):
        os.makedirs('data', exist_ok=True)
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def save_annotations():
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for rel in annotation_order:
                if rel in annotations:
                    writer.writerow([rel] + annotations[rel])

    processed = set(annotation_order)

    print(f"[*] Đã quét được {len(all_imgs)} ảnh trong: {img_dir}")
    print(f"[*] CSV lưu tại: {csv_file}")
    print(f"[*] Prefix ảnh trong CSV: {rel_prefix}/")
    print(f"[*] Bạn đã gán nhãn được {len(processed)} / {target} ảnh.")

    pending_imgs = []
    for img_path in all_imgs:
        rel_path = f"{rel_prefix}/{os.path.basename(img_path)}"
        if rel_path not in processed:
            pending_imgs.append(img_path)
    rel_to_idx = {f"{rel_prefix}/{os.path.basename(p)}": i for i, p in enumerate(pending_imgs)}

    count = len(annotation_order)
    session_saved = []
    img_idx = 0
    while img_idx < len(pending_imgs):
        if count >= target:
            print(f"🚀 ĐÃ ĐẠT MỤC TIÊU {target} ẢNH SEED!")
            break

        img_path = pending_imgs[img_idx]
        rel_path = f"{rel_prefix}/{os.path.basename(img_path)}"
        img = cv2.imread(img_path)
        if img is None:
            img_idx += 1
            continue

        # Resize để dễ nhìn
        img = cv2.resize(img, (400, 400))

        # Khởi tạo nhãn theo dữ liệu hiện tại (nếu có), ngược lại là nhãn trống
        current_labels = annotations.get(rel_path, [0] * len(CLASSES)).copy()

        while True:
            # Vẽ UI bằng PIL (hỗ trợ tiếng Việt có dấu)
            ui_board = draw_ui_pil(430, 400, count, target, CLASSES, CLASSES_VI, KEY_CHARS, current_labels)

            # Ghép ảnh + UI
            combined = np.hstack((img, ui_board))
            cv2.imshow("Gan Nhan - [Enter] Luu, [Space] Bo qua, [Q] Thoat", combined)

            key = cv2.waitKey(0) & 0xFF

            if key in KEY_MAP:
                cls_idx = KEY_MAP[key]
                current_labels[cls_idx] = 1 - current_labels[cls_idx]

            elif key == 13:  # ENTER
                if sum(current_labels) > 0:
                    if rel_path not in annotations:
                        annotation_order.append(rel_path)
                    annotations[rel_path] = current_labels.copy()
                    if not session_saved or session_saved[-1] != rel_path:
                        session_saved.append(rel_path)
                    save_annotations()
                    count = len(annotation_order)
                    active = [CLASSES[i] for i, v in enumerate(current_labels) if v == 1]
                    print(f"✅ Đã lưu ảnh {count}/{target} | Nhãn: {active}")
                    img_idx += 1
                else:
                    print("⚠️ Chưa bật Nhãn nào. Bấm SPACE nếu muốn bỏ qua.")
                    continue
                break

            elif key == 32:  # SPACE
                print("⏭️ Bỏ qua ảnh này...")
                img_idx += 1
                break
            
            elif key == ord('u') or key == 8:  # U hoặc Backspace
                if not session_saved:
                    print("⚠️ Chưa có ảnh nào trong phiên này để quay lại.")
                    continue

                prev_rel = session_saved.pop()
                if prev_rel in annotations:
                    del annotations[prev_rel]
                if prev_rel in annotation_order:
                    annotation_order.remove(prev_rel)
                save_annotations()
                count = len(annotation_order)

                if prev_rel in rel_to_idx:
                    img_idx = rel_to_idx[prev_rel]
                else:
                    img_idx = max(img_idx - 1, 0)
                print(f"↩️ Quay lại ảnh trước: {prev_rel}")
                break

            elif key == ord('q') or key == 27:  # Q hoặc ESC
                print("🛑 ĐÃ THOÁT VÀ LƯU DỮ LIỆU.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("🎉 Đã Review xong!")

if __name__ == "__main__":
    main()
