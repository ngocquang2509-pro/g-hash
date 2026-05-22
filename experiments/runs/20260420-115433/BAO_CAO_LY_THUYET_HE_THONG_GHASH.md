# BÁO CÁO LÝ THUYẾT – HỆ THỐNG G-HASH (CBIR ĐA NHÃN + GAT)

**Mục tiêu:** giải thích **đầy đủ, chi tiết** hệ thống hoạt động như thế nào (từ dữ liệu → nhãn → mô hình → hash → truy hồi → đánh giá), theo hướng “lý thuyết + trực giác + công thức”, và **chỉ rõ code nằm ở đâu** để đối chiếu.

**Tài liệu liên quan (đã có):**
- Báo cáo luồng chạy end-to-end theo script: [BAO_CAO_CODE_FLOW.md](BAO_CAO_CODE_FLOW.md)
- Báo cáo mô hình chi tiết + vị trí code: [BAO_CAO_MO_HINH_GHASH_CHI_TIET.md](BAO_CAO_MO_HINH_GHASH_CHI_TIET.md)

---

## 1) Bài toán: Multi-label Content-Based Image Retrieval (CBIR)

### 1.1 Ta muốn làm gì?
Có một **database** gồm rất nhiều ảnh học sinh (đã crop người). Khi có **query image** (một ảnh bất kỳ), hệ thống trả về **top-K ảnh giống nhất** theo nội dung/hành vi.

- Query: một ảnh $q$
- Database: $\{x_i\}_{i=1}^{N}$
- Output: danh sách $K$ ảnh gần nhất theo một độ đo giống nhau.

### 1.2 Multi-label nghĩa là gì?
Mỗi ảnh có thể có **nhiều nhãn cùng lúc** (multi-hot vector):

$$y \in \{0,1\}^C,\quad C=14\text{ (ET-EDU)}$$

Ví dụ: một học sinh có thể vừa **standing** vừa **using_phone**.

### 1.3 “Relevant” trong hệ thống được định nghĩa thế nào?
Trong code, hai ảnh được coi là **relevant** nếu **chia sẻ ít nhất 1 nhãn**:

$$\text{relevant}(q, x)=\mathbb{1}[y_q^T y_x > 0]$$

**Code:** [src/evaluation/metrics.py](../../../src/evaluation/metrics.py) – hàm `compute_similarity_matrix()`.

---

## 2) Vì sao dùng Deep Hashing thay vì embedding float (kNN cosine)?

### 2.1 Ý tưởng cốt lõi
Thay vì lưu một vector feature float dài (VD 768-dim) cho mỗi ảnh, ta học một hàm:

$$f(\cdot): \mathbb{R}^{3\times H\times W} \to \{-1,+1\}^b$$

Trong đó:
- $b$ = số bit hash (VD: 64)
- Output là **mã nhị phân** (binary code)

### 2.2 Lợi ích thực tế
- **Tốc độ**: dùng Hamming distance (đếm bit khác nhau) cực nhanh.
- **Bộ nhớ**: 64-bit/ảnh rất nhỏ.
- **Indexing**: phù hợp khi database rất lớn.

### 2.3 Hamming distance và dot-product (mẹo quan trọng)
Với $q,x\in\{-1,+1\}^b$:

- Nếu bit giống nhau → tích = +1
- Nếu bit khác nhau → tích = -1

Suy ra:

$$q^Tx = (\#same) - (\#diff) = b - 2\#diff$$

Do đó:

$$d_H(q,x)=\#diff=\frac{b-q^Tx}{2}$$

**Code:** [src/evaluation/metrics.py](../../../src/evaluation/metrics.py) – hàm `hamming_distance()`.

---

## 3) Tổng quan hệ thống (từ dữ liệu đến truy hồi)

### 3.1 Pipeline dữ liệu/nhãn (phần “đầu vào” cho model)
Hệ thống ET-EDU trong repo có 3 khâu lớn trước khi train:

1) **Crop người chất lượng cao từ video**
- Mục tiêu: biến video full-scene → ảnh “person crop” rõ nét, chuẩn kích thước và ít nhiễu.
- Code: [crop_students_quality.py](../../../crop_students_quality.py)

2) **Seed → train teacher → pseudo-label → merge**
- Mục tiêu: mở rộng nhãn từ một seed nhỏ, nhưng vẫn có kiểm soát (threshold + cap per-class).
- Code: [teacher_pseudo_label_quality10k.py](../../../teacher_pseudo_label_quality10k.py)

3) **Split train/test theo video**
- Mục tiêu: tránh leakage (ảnh cùng video vào cả train lẫn test).
- Code: [build_etedu_split_from_csv.py](../../../build_etedu_split_from_csv.py)

Sau bước 3, ta có 4 file chuẩn để train:
- `data/train_img.txt`, `data/train_label.txt`
- `data/test_img.txt`, `data/test_label.txt`

Loader đọc 4 file này để tạo DataLoader.
- Code: [src/data/dataset.py](../../../src/data/dataset.py)

---

## 4) Lý thuyết mô hình G-hash: vì sao “Image Hash + Label Hash + GAT”?

### 4.1 Vấn đề nếu chỉ học hash từ ảnh (image-only hashing)
Nếu chỉ học $f(x)$ sao cho ảnh giống nhau có hash gần nhau, ta gặp:
- Multi-label làm “similarity” phức tạp: ảnh A giống ảnh B theo nhãn 1, nhưng lại giống C theo nhãn 2.
- Nhãn trong giáo dục thường **có liên hệ**: “standing” hay đi kèm “walking”, “using_phone” có thể đi kèm “turning_sideways”, …

=> Ta cần mô hình hóa **quan hệ giữa các nhãn**, không chỉ quan hệ ảnh-ảnh.

### 4.2 Hai “không gian” song song
G-hash học **cùng lúc**:

1) **Image hash codes**: $h_x \in [-1,1]^b$ (continuous, dùng tanh)
2) **Label hash codes (prototype)**: $h_c \in [-1,1]^b$ cho mỗi label $c$ (14 nhãn)

Mục tiêu: ảnh có nhãn $c$ thì $h_x$ phải **gần** $h_c$.

### 4.3 Vì sao cần GAT ở nhánh nhãn?
Label embedding ban đầu là “learnable vector” cho mỗi nhãn. Nếu không dùng graph:
- mỗi nhãn học độc lập,
- không truyền thông tin giữa nhãn liên quan.

GAT cho phép “message passing” giữa nhãn dựa trên **đồ thị đồng-xuất-hiện** (co-occurrence).

**Model code:** [src/models/ghash.py](../../../src/models/ghash.py)
- Image stream: `VisionEncoder` + `img_hash_fc`
- Label stream: `TextEncoder` + `GAT` + `txt_hash_fc`
- Classifier: `classifier`

**GAT code:** [src/models/gat.py](../../../src/models/gat.py)

---

## 5) Lý thuyết đồ thị nhãn (Label Graph) và ma trận đồng-xuất-hiện

### 5.1 Label co-occurrence là gì?
Với tập train gồm $N$ mẫu, label matrix:

$$Y \in \{0,1\}^{N\times C}$$

Co-occurrence counts:

$$\text{co}(i,j) = \sum_{n=1}^N \mathbb{1}[Y_{n,i}=1 \wedge Y_{n,j}=1]$$

### 5.2 Cách hệ thống tạo adjacency matrix
Trong code, adjacency được tạo bằng cách chuẩn hóa co-occurrence theo **tần suất của nhãn hàng** (row-normalize):

$$A_{i,j} = \frac{\text{co}(i,j)}{\max(1,\text{count}(i))}$$

- Điều này gần với xác suất có điều kiện dạng “nhãn $j$ xuất hiện khi nhãn $i$ xuất hiện”.
- Luôn đặt self-loop: $A_{i,i}=1$.

**Code:** [src/data/label_graph.py](../../../src/data/label_graph.py) – hàm `build_label_cooccurrence_matrix()`.

### 5.3 Tại sao phải chuẩn hóa?
Nếu không chuẩn hóa:
- nhãn phổ biến sẽ có co-occurrence lớn với mọi thứ → graph bị “đè” bởi nhãn phổ biến.

Chuẩn hóa giúp giảm bias này (dù vẫn cần thêm xử lý imbalance bằng `pos_weight`).

---

## 6) Lý thuyết GAT (Graph Attention Network) trong hệ thống

### 6.1 Input/Output
- Input node features: embedding nhãn $H \in \mathbb{R}^{C\times d}$
- Output: embedding nhãn tăng cường $H' \in \mathbb{R}^{C\times d}$

### 6.2 Attention cơ bản
GAT học trọng số attention $\alpha_{i,j}$ (mức độ nhãn $j$ ảnh hưởng đến nhãn $i$):

1) Linear transform: $\tilde{h}_i = W h_i$
2) Score: $e_{i,j} = \text{LeakyReLU}(a^T[\tilde{h}_i \Vert \tilde{h}_j])$
3) Mask theo adjacency:

$$e_{i,j} = -\infty \text{ nếu } A_{i,j}=0$$

4) Softmax:

$$\alpha_{i,j} = \text{softmax}_j(e_{i,j})$$

5) Aggregate:

$$h'_i = \sigma\left(\sum_j \alpha_{i,j}\tilde{h}_j\right)$$

**Điểm then chốt:** mask giúp attention chỉ chạy trên những cạnh “có ý nghĩa” từ co-occurrence.

**Code:** [src/models/gat.py](../../../src/models/gat.py)
- `GraphAttentionLayer.forward()` dùng `torch.where(adj_matrix > 0, e, -9e15)` rồi softmax.

### 6.3 Multi-head attention
Thay vì 1 attention, dùng nhiều head:
- mỗi head học một kiểu quan hệ khác nhau,
- cuối cùng concat hoặc average.

**Code:** [src/models/gat.py](../../../src/models/gat.py) – lớp `MultiHeadGATLayer`.

---

## 7) Deep Hashing: continuous relaxation (tanh) và nhị phân (sign)

### 7.1 Vì sign không thể train trực tiếp?
Hàm $\text{sign}(\cdot)$ không khả vi (gradient gần như 0 mọi nơi), nên khó backprop.

Giải pháp phổ biến:
- Train continuous code $h \in [-1,1]^b$ bằng $\tanh$,
- Sau đó lượng tử hóa khi inference bằng sign.

### 7.2 Trong hệ thống, hash được tạo thế nào?
- Continuous image hash: $h_x = \tanh(\text{FC}(\text{ViT}(x)))$
- Continuous label hash: $h_c = \tanh(\text{FC}(\text{GAT}(\text{TextEmb}(c))))$

Binary codes dùng khi retrieval:

$$b_x = \text{sign}(h_x) \in \{-1,+1\}^b$$

**Code:**
- Model: [src/models/ghash.py](../../../src/models/ghash.py)
- Encoder: [src/models/vision_encoder.py](../../../src/models/vision_encoder.py)

---

## 8) Học mục tiêu: vì sao hệ thống cần nhiều loss (không chỉ BCE)?

Hệ thống cần đồng thời:
1) Dự đoán nhãn đúng (classification)
2) Hash ảnh phản ánh semantics multi-label (retrieval)
3) Hash gần nhị phân để dùng Hamming (quantization)
4) Tránh collapse (tất cả hash giống nhau)

Tất cả được đóng gói trong `GHashLoss`.
- Code: [src/training/losses.py](../../../src/training/losses.py)

### 8.1 Loss A – Classification (BCEWithLogits)
Mục tiêu: classifier dự đoán vector multi-label.

$$L_{cls} = \text{BCEWithLogits}(z, y)$$

Vì mất cân bằng lớp, hệ thống dùng **pos_weight**:
- Lớp hiếm → pos_weight cao → phạt nặng nếu đoán sai positive.

**Code:**
- Tính pos_weight: [train.py](../../../train.py) – hàm `compute_pos_weight()` (clamp 1..10)
- Dùng trong loss: [src/training/losses.py](../../../src/training/losses.py) – `BCEWithLogitsLoss(pos_weight=...)`

### 8.2 Loss B – Similarity (Image ↔ Label prototype)
Mục tiêu: ảnh có nhãn $c$ thì hash ảnh gần prototype $h_c$.

Trong code: dùng cosine similarity giữa `img_hash` và `txt_hash`, rồi hinge margin:
- tăng similarity với nhãn đúng
- giảm similarity với nhãn sai

**Trực giác:** biến label prototypes thành “neo” trong không gian hash.

**Code:** [src/training/losses.py](../../../src/training/losses.py) – hàm `similarity_loss()`.

### 8.3 Loss C – Image-to-Image retrieval (supervised)
Mục tiêu: nếu hai ảnh share ≥1 nhãn → hash gần nhau; nếu share 0 nhãn → hash xa.

Trong code:
- Positive mask: $(Y Y^T) > 0$
- Similarity logits: $s_{i,j} = \frac{h_i^T h_j}{b}$
- Dùng logistic/softplus để kéo/đẩy.

**Code:** [src/training/losses.py](../../../src/training/losses.py) – hàm `image_retrieval_loss()`.

### 8.4 Loss D – Quantization (continuous → binary)
Mục tiêu: giảm sai số lượng tử hóa.

$$L_{quant} = \|h - \text{sign}(h)\|_2^2$$

**Code:** [src/training/losses.py](../../../src/training/losses.py) – hàm `quantization_loss()`.

### 8.5 Loss E – Orthogonality (chống mode collapse của label prototypes)
Nếu GAT làm mọi nhãn “na ná nhau” thì prototype hash của nhãn bị dính chùm → retrieval kém.

Trong code: chuẩn hóa `txt_hash`, tính ma trận cosine giữa nhãn với nhau, ép gần Identity.

**Code:** [src/training/losses.py](../../../src/training/losses.py) – phần `loss_ortho`.

### 8.6 Loss F – Bit balance (chống collapse của image hash bits)
Nếu một bit luôn +1 hoặc luôn -1 trên batch → bit đó “vô dụng”.

Bit-balance ép trung bình mỗi bit gần 0.

**Code:** [src/training/losses.py](../../../src/training/losses.py) – phần `loss_bit_balance`.

---

## 9) Vì sao “GAT + prototype label hash” giúp retrieval tốt hơn?

### 9.1 Multi-label gây mâu thuẫn cho metric học trực tiếp
Ví dụ:
- A: {standing, using_phone}
- B: {standing}
- C: {using_phone}

A “giống” B theo nhãn đứng, và “giống” C theo nhãn phone.
Nếu chỉ học ảnh-ảnh, mô hình dễ bị kéo theo hướng không ổn định.

### 9.2 Prototype hash tạo neo cho từng nhãn
Khi có hash prototype cho mỗi nhãn:
- A được kéo về cả $h_{standing}$ và $h_{using\_phone}$
- B kéo về $h_{standing}$
- C kéo về $h_{using\_phone}$

Kết quả: không gian hash “tách theo semantics” tốt hơn.

### 9.3 GAT làm prototype “có ngữ cảnh”
Nếu hai nhãn thường đi cùng nhau, prototype của chúng cũng được phép gần hơn có kiểm soát.
Nếu hai nhãn ít liên quan, orthogonality loss đẩy chúng xa.

=> cân bằng giữa “học quan hệ nhãn thật” và “không dính chùm”.

---

## 10) Lý thuyết training protocol và các cơ chế ổn định

### 10.1 Adj matrix được build ở đâu và vì sao build 1 lần?
Adjacency dựa trên thống kê train set, nên thường build ở đầu training để:
- cố định graph theo dữ liệu,
- giảm noise do batch nhỏ.

**Code:** [src/training/trainer.py](../../../src/training/trainer.py) – `_build_adjacency_matrix()`.

### 10.2 Gradient clipping
Hash losses + attention có thể làm gradient lớn. Hệ thống clip để ổn định:

$$\|g\|_2 \le 5.0$$

**Code:** [src/training/trainer.py](../../../src/training/trainer.py) – `clip_grad_norm_`.

### 10.3 Cosine annealing LR
Giảm LR theo cos giúp fine-tune ViT ổn định hơn.

**Code:** [src/training/trainer.py](../../../src/training/trainer.py) – `CosineAnnealingLR`.

### 10.4 Early stopping
Dừng khi mAP không tăng sau một số lần đánh giá (patience).

**Code:** [src/training/trainer.py](../../../src/training/trainer.py) – biến `patience`, `patience_counter`.

---

## 11) Lý thuyết inference/retrieval trong hệ thống

### 11.1 Encode: 3 đại diện cho 1 ảnh
Trong inference, hệ thống encode 1 ảnh ra:
- `binary_hash` (để Hamming coarse)
- `continuous_hash` (để rerank)
- `probabilities` (để rerank theo nhãn)

**Code:** [inference.py](../../../inference.py) – `ImageRetriever._encode_tensor()`.

### 11.2 Query auto-focus (tại sao cần?)
Database ảnh là **person-crop**, nhưng query có thể là ảnh toàn cảnh.
Auto-focus dùng YOLO cắt người lớn nhất để giảm mismatch.

**Code:** [inference.py](../../../inference.py) – `ImageRetriever._focus_query_person()`.

### 11.3 Retrieval 2-stage: coarse rồi fine
1) **Coarse (Hamming)**: tìm nhanh top-N gần nhất
2) **Fine rerank**: dùng continuous hash + label probability similarity để sắp lại

**Code:** [inference.py](../../../inference.py)

### 11.4 Diversity (MMR + constraint)
Vì data video có nhiều frame gần nhau, top-K dễ toàn ảnh “gần như trùng”.
Hệ thống thêm:
- giới hạn số ảnh trên mỗi video
- tối thiểu frame-gap
- MMR để cân bằng relevance vs diversity

**Code:** [inference.py](../../../inference.py) – `ImageRetriever._select_diverse_topk()`.

---

## 12) Đánh giá chất lượng retrieval (mAP, P@K, R@K, PR@radius)

### 12.1 mAP (Mean Average Precision)
Với mỗi query, tính Average Precision dựa trên thứ tự retrieval; lấy trung bình theo query.

**Code:** [src/evaluation/metrics.py](../../../src/evaluation/metrics.py) – `mean_average_precision()`.

### 12.2 Precision@K / Recall@K
- P@K: trong top-K có bao nhiêu ảnh relevant
- R@K: trong tất cả ảnh relevant, top-K cover được bao nhiêu

**Code:** [src/evaluation/metrics.py](../../../src/evaluation/metrics.py) – `precision_at_k()`, `recall_at_k()`.

### 12.3 PR theo Hamming radius
Đánh giá kiểu “lấy tất cả ảnh trong bán kính Hamming ≤ r”:
- radius nhỏ → ít ảnh, precision cao
- radius lớn → nhiều ảnh, recall tăng

**Code:** [src/evaluation/metrics.py](../../../src/evaluation/metrics.py) – `precision_recall_curve_at_hamming_radius()`.

---

## 13) Các điểm dễ sai (và hệ thống đã xử lý bằng gì?)

### 13.1 Mất cân bằng lớp (class imbalance)
- Lớp hiếm (VD dozing_off) có rất ít positive → model dễ “bỏ qua”.
- Teacher/pseudo cũng dễ bias về lớp phổ biến.

**Cơ chế xử lý trong hệ thống:**
- `pos_weight` trong BCE (train teacher + train G-hash)
  - Teacher: [teacher_pseudo_label_quality10k.py](../../../teacher_pseudo_label_quality10k.py)
  - G-hash: [train.py](../../../train.py)
- giới hạn `max_per_class` khi pseudo-label để tránh tràn lớp phổ biến.
  - Code: [teacher_pseudo_label_quality10k.py](../../../teacher_pseudo_label_quality10k.py)

### 13.2 Mode collapse (hash codes giống nhau)
- Nếu mọi ảnh ra cùng một hash → retrieval vô nghĩa.

**Cơ chế chống collapse:**
- `loss_ortho` (label prototype tách nhau)
- `loss_bit_balance` (bit 50/50)
- `loss_quant` (đẩy về ±1 rõ ràng)

**Code:** [src/training/losses.py](../../../src/training/losses.py)

### 13.3 Leakage train/test (video overlap)
Nếu train/test có cùng video, model có thể học background/video id → mAP ảo.

**Cơ chế chống leakage:**
- split theo video_id trích từ tên file.

**Code:** [build_etedu_split_from_csv.py](../../../build_etedu_split_from_csv.py)

### 13.4 Mismatch query vs database (full-scene vs crop)
**Cơ chế:** auto-focus query bằng YOLO.

**Code:** [inference.py](../../../inference.py)

---

## 14) Bảng “đối chiếu lý thuyết ↔ code” (1 trang)

| Khối lý thuyết | Ý nghĩa | Code triển khai |
|---|---|---|
| Data loader | đọc `train/test_*.txt`, transform | [src/data/dataset.py](../../../src/data/dataset.py) |
| Label graph | build co-occurrence adjacency | [src/data/label_graph.py](../../../src/data/label_graph.py) |
| Vision encoder | ViT từ timm, lấy CLS | [src/models/vision_encoder.py](../../../src/models/vision_encoder.py) |
| GAT | attention + mask theo adjacency | [src/models/gat.py](../../../src/models/gat.py) |
| GHashModel | image hash + label hash + classifier | [src/models/ghash.py](../../../src/models/ghash.py) |
| Loss tổng | cls + sim + ret + quant + ortho + balance | [src/training/losses.py](../../../src/training/losses.py) |
| Training loop | train/eval/checkpoint/early stop | [src/training/trainer.py](../../../src/training/trainer.py) |
| Metrics | Hamming, mAP, P@K, R@K, PR@radius | [src/evaluation/metrics.py](../../../src/evaluation/metrics.py) |
| Inference | coarse Hamming + rerank + MMR diversity | [inference.py](../../../inference.py) |

---

## 15) Kết luận (dạng trình bày)

- Hệ thống dùng **deep hashing** để truy hồi nhanh bằng Hamming.
- G-hash học đồng thời **hash ảnh** và **prototype hash nhãn**.
- **GAT** dùng đồ thị co-occurrence để làm nhãn “có ngữ cảnh”, giúp multi-label retrieval ổn định hơn.
- Bộ loss được thiết kế để vừa đúng nhãn, vừa đúng quan hệ retrieval, vừa nhị phân hóa tốt và chống collapse.
- Inference dùng coarse-to-fine + diversity để cho top-K vừa đúng vừa đa dạng theo video.
