import sys
from pathlib import Path
import warnings

# Tắt cảnh báo không cần thiết
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from inference import ImageRetriever

def main():
    images_dir = Path("images")
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)

    # === CONFIG: Model ET-EDU V4 (14 nhãn, mới train) ===
    checkpoint = "experiments/runs/20260420-115433/best_model.pth"
    config = "configs/et_edu_config.yaml"

    print("[*] Đang nạp Model ET-EDU V3 (10 hành vi học sinh)...")
    try:
        retriever = ImageRetriever(
            checkpoint_path=checkpoint,
            config_path=config
        )
    except Exception as e:
        print(f"❌ Lỗi nạp Model: {e}")
        return

    # Database = toàn bộ ảnh train + test ET-EDU
    print("[*] Đang xây dựng Database ET-EDU...")
    db_root = Path("data")
    database_images = []
    for txt_file in ["train_img.txt", "test_img.txt"]:
        txt_path = db_root / txt_file
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                for line in f:
                    img_path = str(db_root / line.strip())
                    database_images.append(img_path)
    print(f"   Database: {len(database_images)} ảnh")

    # Gom ảnh query từ thư mục images/
    extensions = ['*.jpg', '*.jpeg', '*.png']
    query_images = []
    for ext in extensions:
        query_images.extend(images_dir.glob(ext))

    print(f"\n🚀 Bắt đầu truy vấn {len(query_images)} ảnh...\n")

    for i, img_path in enumerate(query_images, 1):
        print(f"[{i}/{len(query_images)}] {img_path.name} ", end=">> ")
        try:
            results = retriever.retrieve_similar_images(
                query_image_path=str(img_path),
                database_images=database_images,
                top_k=10
            )

            save_path = results_dir / f"{img_path.stem}_result.png"
            retriever.visualize_retrieval(str(img_path), results, save_path=save_path)

        except Exception as e:
            print(f"[LỖI] {e}")

    print(f"\n🎉 HOÀN TẤT! Kết quả lưu tại: {results_dir}/")

if __name__ == "__main__":
    main()
