# Báo cáo vận hành — Toàn hệ CBIR G-hash: code chạy theo các bước nào?

Tài liệu **mới**, bổ sung cho báo cáo thuyết trình run cụ thể (`experiments/runs/…/BAO_CAO_THUYET_TRINH_ET_EDU.md`). Ở đây ta mô tả **cách toàn bộ hệ thống được tổ chức và thứ tự thực thi**.

---

## 1. Hai vùng mã trong repository

| Vùng | Đường dẫn | Ý nghĩa |
|------|-----------|---------|
| **Thư viện lõi** | `src/` | Mô hình, loader, loss, trainer, metric, viz — được `import` từ script ngoài. |
| **Kịch bản dự án / dữ liệu** | Các `.py` ở **gốc repo** và `labeling/` | Tiền xử lý video, CSV, chia tập, huấn luyện từ dòng lệnh, inference hàng loạt. |

Khi Python chạy `train.py` hoặc `inference.py`, thư mục gốc được chèn vào `sys.path` để có thể gọi `from src.models.ghash import ...`.

---

## 2. Sơ đồ “đường đi của dữ liệu → mô hình → kết quả”

```
Video / ảnh gốc
    → Script crop & lọc (một trong các pipeline §3)
    → ảnh JPG + nhãn (CSV hoặc txt)
    → (tuỳ) build_etedu_split_from_csv / build_edu_txt_from_labeled_csv
    → data/{train_img.txt, train_label.txt, test_img.txt, test_label.txt}
         │
         ▼
    train.py  →  Trainer  →  experiments/runs/<timestamp>/
                              (best_model.pth, metrics.txt, *.png báo cáo)
         │
         ▼
    inference.py / batch_test.py  →  ảnh truy vấn + visualization
```

---

## 3. Pipeline A — Theo từng bước “frame → crop học sinh → chất lượng” (kiểu cổ điển trong repo)

Các script **độc lập**, có thể chạy lần lượt hoặc chọn một nhánh:

| Thứ tự đề xuất | Script | Việc làm |
|----------------|--------|-----------|
| 1 | `step1_extract_frames.py` | Đọc video `data/ET-EDU`, ghi khung ảnh thô vào ví dụ `data/ET-EDU-RAW-FRAMES`. |
| 2 | `step1b_crop_persons.py` | Dùng YOLO cắt người từ frame thô → `data/ET-EDU-CROPPED-PERSONS`. |
| 3 | `step2_build_dataset.py` | Tổ chức / lọc thêm sang cấu trúc ví dụ `data/ET-EDU-SORTED`. |
| 4 | `crop_students_quality.py` | Lọc theo detector score, blur, phân bucket gần–xa, mặc định `--video-dir data/ET-EDU`; có optional enhance ảnh. |
| 5 | `prepare_video_dataset.py` | Cách một thể làm cụ thể khác crop từ video (FPS, threshold YOLO, hash trùng frame). |

Luồng này phù hợp báo cáo khi nhấn mạnh “tách bước trích khung → phát hiện → crop”.

---

## 4. Pipeline B — ET-EDU CBIR V2 (extract “một chỗ”)

Script **`extract_edu_cbir_v2.py`** là **pipeline đóng**: YOLO + tracking IoU + keyframe + lọc chất lượng + dedup Hamming histogram + chia train/val/test theo video + sinh các file `*_img*.txt`, `labels_template.csv`, v.v.

- **Tham khảo đầy đủ:** `ET_EDU_CBIR_V2_RUNBOOK.md`
- Đầu vào ví dụ: `--video-dir data/ET-EDU`
- Đầu ra ví dụ: `--output-root data/ET-EDU-CBIR-V2`
- Sau khi có CSV nhãn, dùng `build_edu_txt_from_labeled_csv.py --labeled-csv ... --output-root ...` để ra đúng cặp `train_img.txt` / `train_label.txt`… mà **`src/data/dataset.py`** đọc khi `dataset.name: ET-EDU`.

---

## 5. Pipeline gán nhãn và hợp nhất nhãn (script gốc, tuỳ dự án dùng cái nào)**

| Script (ví dụ) | Mục đích ngắn |
|----------------|---------------|
| `annotate_golden_dataset.py`, `labeling/make_labeling_review_csvs.py`, `labeling/make_label_studio_tasks.py`, `labeling/apply_label_studio_export.py`, `labeling/qc_labeled_csv.py` | Gán tay / QA / Xuất-import Label Studio. |
| `ai_label_crops.py` | Chuỗi gán/auto với cụ thể ET-EDU-CROPPED-PERSONS. |
| `auto_label_diverse_remaining.py` | Lan nhãn tự động có kiểm soát độ đa dạng. |
| `teacher_pseudo_label_quality10k.py` | Teacher pseudo-label sau khi có seed tay. |
| `semi_supervised_label.py`, `smart_relabel.py`, `review_labels.py`, … | Tuỳ quy trình team. |

Sau đó có thể **`build_etedu_split_from_csv.py`**: nhận CSV merged, **chia theo cụm video_id** trong tên file, ghi các txt + label tương ứng dưới `--output-root` (thường `data/`).

Hoặc nếu dùng **`build_edu_txt_from_labeled_csv.py`**, CSV đã có cột **split / retrieval_role**, script ghi thẳng `train_*` và `test_*` theo cột đó.

**Checkpoint cho huấn luyện:** Dưới `data/` (khớp `dataset.data_root` trong YAML, thường `"data"`):

- `train_img.txt`, `train_label.txt` — mỗi dòng: đường dẫn ảnh tương đối; vector nhị phân 0/1 cách nhau bằng khoảng trắng.
- `test_img.txt`, `test_label.txt` — tương tự cho tập đánh giá / query trong trainer.

Loader đọc bằng class **`NUSWIDE2Dataset`** trong `src/data/dataset.py` khi **`dataset.name == "ET-EDU"`**.

---

## 6. Luồng huấn luyện: `train.py` gọi gì lần lượt?

### 6.1. Khởi động

1. **`Config(args.config)`** (`src/utils/config.py`): đọc YAML (ví dụ `configs/et_edu_config.yaml`).
2. **`set_seed(config['seed'])`**: cố định ngẫu nhiên.
3. **`create_data_loaders(config.config)`**: tạo 3 DataLoader trong `src/data/dataset.py`  
   - train shuffle + augment train  
   - test + query: cùng file test nhưng dùng ở hai vai trò (test batch vs query batch khi trainer đánh giá).
4. Cập nhật `config.dataset.num_classes = num_classes` từ file label (dataset biết số cột nhãn).
5. **`GHashModel(config.config)`** hoặc `BaselineModel` nếu `--baseline` (`src/models/ghash.py`, encoder trong `vision_encoder.py`, GAT trong `gat.py`).
6. **[Tuỳ chọn]** `--resume` hoặc trường `model.checkpoint` trong YAML để partial load ViT/state dict (filter shape).
7. **`compute_pos_weight(train_loader, device)`** → truyền vào **`GHashLoss`** với các hệ số `alpha_similarity`, `beta_quantization`, `gamma_classification`, `delta_bit_balance`, `eta_retrieval` đọc từ YAML.
8. **`Trainer(...)`** (`src/training/trainer.py`).

### 6.2. Bên trong `Trainer`

- **`__init__`**: optimizer Adam, cosine scheduler, **`_build_adjacency_matrix()`** lặp train_loader gom nhãn rồi `build_label_cooccurrence_matrix` (`src/data/label_graph.py`), tạo **`database_eval_loader`** nếu `evaluation.database_split == "train"` (Ảnh trong train nhưng **transform như query** để đánh giá công bằng).
- **`train()`**: mỗi epoch **`train_epoch`**: forward `model(images, labels, adj_matrix)`, backward `GHashLoss`, logging thành phần loss. Mỗi 5 epoch (hoặc epoch cuối) gọi **`evaluate()`**.
- **`evaluate()`**: encode toàn DB và query bằng **`model.generate_hash_code`** (±1 binary), **`compute_retrieval_metrics`** trong `src/evaluation/metrics.py` (ranking Hamming, multi-label “cùng ít nhất một nhãn”).
- **`train()` tiếp**: so sánh `metrics['mAP']` vs `best_map`, lưu `best_model.pth`, early stopping theo patience.
- Kết thúc `trainer.train()`, **`create_experiment_report`** (`src/utils/visualization.py`) ghi:
  - `training_curves.png`, `loss_components.png`, `topk_metrics.png`, `pr_curve.png`
  - **`metrics.txt`** = block metric + YAML dump của config đã chạy

Thư mục lưu: **`experiments/runs/<YYYYMMDD-HHMMSS>/`** (`Trainer.save_dir`).

### 6.3. Lệnh mẫu

```bash
python train.py --config configs/et_edu_config.yaml
# Tuỳ chọn:
# python train.py --config configs/et_edu_config.yaml --batch-size 64 --epochs 80
```

**Phiên rút gọn dataset:** `train_mini.py` (cùng ý như train nhưng thường thu nhỏ dữ liệu / để smoke test).

---

## 7. Luồng suy luận: `inference.py` và `batch_test.py`

### 7.1. `inference.py` — entry hai chế độ

- **`main(args)`** nhận `--checkpoint`, `--config`, `--mode predict|retrieve`, `--image`, `--database` (retrieve), `--top-k`, `--save-viz`.

**Luồng nạp vào:**
1. **`ImageRetriever(checkpoint, config)`**  
   - `Config` đọc YAML.  
   - `GHashModel` + **`load_state_dict(checkpoint['model_state_dict'])`**.  
   - Tên nhãn: `label_file` trong config, hoặc `Concepts81.txt`, hoặc fallback `Concept_ID_i`.

**Predict (`--mode predict`):**
- `predict_labels` → softmax/sigmoid logits qua classifier, in top-k và Optional `visualize_predictions`.

**Retrieve (`--mode retrieve`):**
1. **`build_database`** (hoặc tương đương): encode từng ảnh trong list/thư mục → binary hash + continuous hash + probs.
2. Query: encode ảnh hỏi; **Stage 1** sắp xếp theo Hamming trong tập coarse.
3. **Stage 2 re-rank** (blend `feature_scores`, `label_scores`, `hamming_scores` — không dùng trong `Trainer.evaluate`).
4. **Tuỳ cấu hình:** `query_auto_focus` có thể bật YOLO crop người (nếu `ultralytics` có).
5. MMR / giới hạn `max_per_video_in_topk`, `min_frame_gap_in_topk` để đa dạng kết quả.
6. **`visualize_retrieval`** → PNG.

### 7.2. `batch_test.py`

- Hard-code đường checkpoint + config ET-EDU ví dụ.  
- Tạo `ImageRetriever` giống trên.  
- **Database**: đọc nối **`data/train_img.txt`** và **`data/test_img.txt`** thành list đường dẫn tuyệt đối.  
- **Query**: mọi `*.jpg` trong thư mục `images/`.  
- Lặp từng query → `retrieve_similar_images` → lưu ảnh kết quả vào **`test_results/`**.

Đây là “chuỗi sản phẩm” một lần nhấn (sau khi đặt ảnh vào `images/`).

---

## 8. Bảng chức năng `src/` (tra cứu nhanh khi báo cáo)

| File | Chức năng |
|------|-----------|
| `src/utils/config.py` | Đọc YAML, device, save config. |
| `src/utils/visualization.py` | Vẽ curve, báo cáo cuối `create_experiment_report` → **`metrics.txt`**. |
| `src/data/dataset.py` | `NUSWIDE2Dataset`, `create_data_loaders` (ET-EDU nhánh riêng). |
| `src/data/label_graph.py` | Ma trận đồng xuất hiện nhãn → GAT. |
| `src/models/vision_encoder.py` | `VisionEncoder` (ViT), `TextEncoder` embedding nhãn. |
| `src/models/gat.py` | Lớp GAT nhận `adj_matrix`. |
| `src/models/ghash.py` | `GHashModel`, `BaselineModel`, `generate_hash_code`. |
| `src/training/losses.py` | `GHashLoss` (cls, sim, retrieval, quant, ortho, bit balance). |
| `src/training/trainer.py` | Vòng huấn luyện, evaluate, database loader, lưu checkpoint. |
| `src/evaluation/metrics.py` | Hamming, mAP/P/R@K, PR theo radius. |

---

## 9. Các script thử / so sánh (không bắt buộc khi báo cáo chính)

| Script | Ghi chú |
|--------|---------|
| `test.py`, `test_hash.py`, `test_vit.py`, `test_bug.py` | Kiểm tra từng phần. |
| `compare_models.py` | So sánh mô hình / checkpoint. |
| `prepare_finetune.py` | Chuẩn bị tinh chỉnh. |

---

## 10. Chuỗi lệnh tham chiếu “từ đầu đến cuối” (ET-EDU, minh họa)

Đây chỉ là **thứ tự logic**, bạn chỉnh đường dẫn theo máy và pipeline A hoặc B đang dùng.

```bash
# B1 — Sinh crop + metadata (Pipeline B ví dụ)
python extract_edu_cbir_v2.py --video-dir data/ET-EDU --output-root data/ET-EDU-CBIR-V2

# B2 — Gán nhãn (thủ công / tooling trong labeling/), rồi CSV → txt
python build_edu_txt_from_labeled_csv.py \
  --labeled-csv data/ET-EDU-CBIR-V2/labels_template.csv \
  --output-root data

# HOẶC: CSV merged → chia theo video
# python build_etedu_split_from_csv.py --csv-file data/annotations_....csv --output-root data

# B3 — Huấn luyện (data_root trong YAML phải trùng thư mục chứa *_img.txt và *_label.txt)
python train.py --config configs/et_edu_config.yaml

# B4 — Thuận tiện: copy best_model.pth về một run cố định nếu cần demo
# ...

# B5 — Truy vấn một ảnh
python inference.py --checkpoint experiments/runs/<timestamp>/best_model.pth \
  --config configs/et_edu_config.yaml --mode retrieve \
  --image path/to/query.jpg --database data/train_img.txt --top-k 8 --save-viz

# B6 — Batch query từ thư mục images/
python batch_test.py
```

**Lưu ý:** `inference.py` mặc định `--config configs/config_m4pro.yaml`; với ET-EDU luôn truyền **`--config configs/et_edu_config.yaml`** như trong `batch_test.py`.

---

## 11. Tóm tắt một đoạn cho hội đồng

> Hệ có **hai lớp**: (1) **script bash-level** làm dataset và TXT đa nhãn; **thư viện `src/`** chỉ lo **đọc config, nạp ảnh, forward-backward, và metric**. `train.py` là **điểm vào huấn luyện**; **`Trainer`** xây đồ thị nhãn từ train, học **`G-hash`**, và lưu run với **`metrics.txt`**. **`inference.py`** tái hiện sản phẩm người dùng: **Hamming nhanh rồi re-rank** và visualize — khác một bước so với **đánh giá offline trong trainer** chỉ Hamming để báo số học báo hashing.

---

*Tài liệu: `/home/ubuntu/Desktop/CBIR/g-hash-main/BAO_CAO_HE_THONG_VAN_HANH_CODE.md` — cập nhật theo trạng thái codebase tại thời điểm tạo file.*
