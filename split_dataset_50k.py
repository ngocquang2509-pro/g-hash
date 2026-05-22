import pandas as pd
from pathlib import Path
import os
import argparse

def main():
    parser = argparse.ArgumentParser("ET-EDU Golden Dataset Extractor")
    parser.add_argument("--csv_file", default="data/golden_annotations.csv")  # 🚨 CHỈ LẤY DATA VÀNG
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--total_samples", type=int, default=10000)
    args = parser.parse_args()

    print(f"[*] Đang tải File CSV Khối lượng lớn: {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    print(f"[*] Đã quét được: {len(df)} dòng học sinh.")

    df = df.dropna()

    total_samples = min(args.total_samples, len(df))
    print(f"[*] Đang tiến hành thuật toán Random Sampling (Lấy mẫu ngẫu nhiên) chọn xuất sắc {total_samples} ảnh...")
    
    # Thiết lập random_state=42 để lần sau chạy vẫn giữ nguyên tỷ lệ
    df_sampled = df.sample(n=total_samples, random_state=42).reset_index(drop=True)

    # Chia mẻ 90% Train - 10% Test
    train_size = int(total_samples * 0.9)
    df_train = df_sampled.iloc[:train_size]
    df_test = df_sampled.iloc[train_size:]

    print(f"[*] Kết quả Chia Lưới:")
    print(f"    -> Tập Train (Huấn luyện): {len(df_train)} ảnh")
    print(f"    -> Tập Test (Kiểm định):  {len(df_test)} ảnh")

    # Bóc tách 10 cột Nhãn
    label_cols = df.columns[1:]

    # Xuất file Định dạng chuẩn cho dataset.py của PyTorch
    print("[*] Đang chuyển đổi sang định dạng Text Nhị phân cho Pytorch DataLoader...")
    
    # Tập Train
    with open(os.path.join(args.output_dir, "train_img.txt"), "w") as f_img, \
         open(os.path.join(args.output_dir, "train_label.txt"), "w") as f_lbl:
        for _, row in df_train.iterrows():
            f_img.write(f"{row['Image_Path']}\n")
            f_lbl.write(" ".join([str(int(row[c])) for c in label_cols]) + "\n")

    # Tập Test
    with open(os.path.join(args.output_dir, "test_img.txt"), "w") as f_img, \
         open(os.path.join(args.output_dir, "test_label.txt"), "w") as f_lbl:
        for _, row in df_test.iterrows():
            f_img.write(f"{row['Image_Path']}\n")
            f_lbl.write(" ".join([str(int(row[c])) for c in label_cols]) + "\n")

    print("\n🎉 THÀNH CÔNG! Đã Xây Dựng Xong Hạ Tầng Dữ liệu ET-EDU.")
    print("   Các file sinh ra lập tức khớp với 'src/data/dataset.py'")

if __name__ == "__main__":
    main()
