"""
Công cụ Kiểm tra Nhãn AI (Review AI Labels)
=============================================
Hiển thị ảnh + nhãn mà CNN đã tự gán.
Bạn xem qua và:
  [Enter] = Đúng, chấp nhận
  [Space] = Sai, bỏ (xóa khỏi dataset)
  Phím 1-0, a-d = Sửa nhãn nếu cần
  [Q] = Thoát
"""

import cv2
import os
import csv
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CLASSES = [
    'using_phone', 'dozing_off', 'turning_sideways',
    'turning_back', 'raising_hand', 'opening_book',
    'reading', 'writing', 'listening', 'head_down',
    'sitting', 'standing', 'walking', 'interacting'
]

CLASSES_VI = {
    'using_phone':       'Dùng điện thoại',
    'dozing_off':        'Ngủ gật',
    'turning_sideways':  'Quay ngang',
    'turning_back':      'Quay ra sau',
    'raising_hand':      'Giơ tay',
    'opening_book':      'Mở sách',
    'reading':           'Đọc sách',
    'writing':           'Ghi chép',
    'listening':         'Lắng nghe',
    'head_down':         'Cúi đầu',
    'sitting':           'Ngồi',
    'standing':          'Đứng',
    'walking':           'Đi lại',
    'interacting':       'Tương tác',
}

KEY_MAP = {
    ord('1'): 0, ord('2'): 1, ord('3'): 2, ord('4'): 3, ord('5'): 4,
    ord('6'): 5, ord('7'): 6, ord('8'): 7, ord('9'): 8, ord('0'): 9,
    ord('a'): 10, ord('b'): 11, ord('c'): 12, ord('d'): 13
}
KEY_CHARS = ['1','2','3','4','5','6','7','8','9','0','a','b','c','d']

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
try:
    FONT_LARGE = ImageFont.truetype(FONT_PATH, 20)
    FONT_SMALL = ImageFont.truetype(FONT_PATH, 14)
except:
    FONT_LARGE = ImageFont.load_default()
    FONT_SMALL = ImageFont.load_default()


def draw_ui(ui_w, ui_h, idx, total, labels, accepted, rejected):
    img_pil = Image.new('RGB', (ui_w, ui_h), (0, 0, 0))
    draw = ImageDraw.Draw(img_pil)

    draw.text((10, 3), f"KIỂM TRA: {idx+1}/{total}", font=FONT_LARGE, fill=(0, 255, 255))
    draw.text((10, 26), f"✅ Đúng: {accepted}  ❌ Sai: {rejected}", font=FONT_SMALL, fill=(200, 200, 200))
    draw.text((10, 44), "Nhãn AI đã gán:", font=FONT_SMALL, fill=(255, 255, 0))

    for i, cls in enumerate(CLASSES):
        kc = KEY_CHARS[i]
        vi = CLASSES_VI[cls]
        active = labels[i] == 1
        color = (0, 255, 0) if active else (60, 60, 60)
        marker = "■" if active else "□"
        text = f"[{kc}] {marker} {cls} - {vi}"
        draw.text((10, 62 + i * 20), text, font=FONT_SMALL, fill=color)

    draw.text((10, 62 + len(CLASSES) * 20 + 8), "[Enter] Đúng  [Space] Sai/Xóa  [Q] Thoát", font=FONT_SMALL, fill=(180, 180, 180))
    draw.text((10, 62 + len(CLASSES) * 20 + 26), "Bấm phím 1-d để SỬA nhãn trước khi Enter", font=FONT_SMALL, fill=(180, 180, 180))

    return np.array(img_pil)


def parse_args():
    parser = argparse.ArgumentParser(description="Review and correct multi-label CSV with visual UI.")
    parser.add_argument("--csv-file", default="data/golden_annotations.csv", help="Input annotation CSV.")
    parser.add_argument("--data-root", default="data", help="Root folder for relative image paths in CSV.")
    parser.add_argument(
        "--save-path",
        default="data/golden_annotations_reviewed.csv",
        help="Output CSV after review.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_file = args.csv_file
    data_root = args.data_root
    save_path = args.save_path

    df = pd.read_csv(csv_file)
    print(f"[*] Đang tải {len(df)} ảnh từ {csv_file} để kiểm tra...")

    # Xáo trộn để kiểm tra ngẫu nhiên
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    accepted = 0
    rejected = 0
    keep_rows = []

    for idx, row in df.iterrows():
        img_path = os.path.join(data_root, row['Image_Path'])
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (400, 400))

        labels = [int(row[c]) for c in CLASSES]

        while True:
            ui = draw_ui(430, 400, idx, len(df), labels, accepted, rejected)
            combined = np.hstack((img, ui))
            cv2.imshow("Kiem tra Nhan AI - [Enter] Dung, [Space] Sai, [Q] Thoat", combined)

            key = cv2.waitKey(0) & 0xFF

            if key in KEY_MAP:
                ci = KEY_MAP[key]
                labels[ci] = 1 - labels[ci]

            elif key == 13:  # Enter = chấp nhận (đúng hoặc đã sửa)
                accepted += 1
                new_row = [row['Image_Path']] + labels
                keep_rows.append(new_row)
                break

            elif key == 32:  # Space = sai, xóa khỏi dataset
                rejected += 1
                break

            elif key == ord('q') or key == 27:
                print(f"\n🛑 THOÁT! Đã kiểm tra {accepted + rejected} ảnh.")
                print(f"   ✅ Chấp nhận: {accepted}")
                print(f"   ❌ Loại bỏ: {rejected}")
                print(f"   Tỷ lệ chính xác AI: {accepted/(accepted+rejected)*100:.1f}%" if (accepted+rejected) > 0 else "")

                # Lưu lại file đã lọc
                if keep_rows:
                    with open(save_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Image_Path'] + CLASSES)
                        writer.writerows(keep_rows)

                    # Ghép thêm phần chưa kiểm tra
                    remaining = df.iloc[idx+1:]
                    for _, r in remaining.iterrows():
                        keep_rows.append([r['Image_Path']] + [int(r[c]) for c in CLASSES])

                    with open(save_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Image_Path'] + CLASSES)
                        writer.writerows(keep_rows)

                    print(f"   📄 Đã lưu: {save_path} ({len(keep_rows)} ảnh)")

                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print(f"\n🎉 Đã kiểm tra TOÀN BỘ {len(df)} ảnh!")
    print(f"   ✅ Chấp nhận: {accepted}")
    print(f"   ❌ Loại bỏ: {rejected}")


if __name__ == "__main__":
    main()
