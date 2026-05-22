# Báo cáo mô hình G-hash — Kiến trúc, luồng dữ liệu và **chỉ dẫn mã nguồn**

Tài liệu mô tả **toàn bộ mô hình G-hash** trong repo này: ý tưởng, các thành phần, và **chính xác file + đoạn code** đang thực hiện từng bước (xử lý dữ liệu, ma trận đồng xuất hiện, GAT, hashing, loss, đánh giá).

---

## 1. G-hash là gì (trong codebase này)?

**G-hash** = **G**raph (GAT trên đồ thị nhãn) + **hash** (mã nhị phân gọn cho truy vấn nhanh), kết hợp **ViT** trích đặc trưng ảnh.

- **Nhánh ảnh:** ViT → vector đặc trưng → MLP → hash liên tục (tanh) → khi cần nhị phân dùng `sign`.
- **Nhánh nhãn:** Embedding từng lớp → **GAT** trên ma trận kề **đồng xuất hiện nhãn** (tính từ tập train) → vector nhãn “có ngữ cảnh” → MLP → hash nhãn.
- **Đa nhãn:** Classifier trên đặc trưng ảnh + **BCE** với nhãn multi-hot; loss bổ sung căn chỉnh ảnh–nhãn, ảnh–ảnh, lượng tử hóa, v.v.

**Điểm vào mô hình đầy đủ:** `GHashModel` trong `src/models/ghash.py`.

---

## 2. Xử lý dữ liệu đầu vào (ảnh + nhãn) — ở đâu?

### 2.1. Đọc danh sách ảnh và vector nhãn từ file `.txt`

**File:** `src/data/dataset.py` — class `NUSWIDE2Dataset` (tên lịch sử; với ET-EDU vẫn dùng class này).

- Với `dataset_name == "ET-EDU"`: đọc `train_img.txt` / `train_label.txt` (train) và `test_img.txt` / `test_label.txt` (test/query), đường dẫn tương đối so với `data_root`.

```16:40:src/data/dataset.py
        if dataset_name == "ET-EDU":
            if split == "database":
                img_file = self.data_root / "train_img.txt"
                label_file = self.data_root / "train_label.txt"
            else:
                img_file = self.data_root / "test_img.txt"
                label_file = self.data_root / "test_label.txt"
        ...
        with open(img_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines() if line.strip()]
            
        with open(label_file, 'r') as f:
            self.labels = []
            for line in f.readlines():
                if line.strip():
                    self.labels.append([int(x) for x in line.strip().split()])
                    
        self.labels = np.array(self.labels, dtype=np.float32)
```

### 2.2. Mở ảnh, áp transform (train vs test)

**Cùng file:** `__getitem__` — PIL RGB, augment hoặc resize+normalize.

```45:57:src/data/dataset.py
    def __getitem__(self, idx):
        img_path = self.data_root / self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0,0,0))
            
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx])
        return image, label, idx
```

### 2.3. Tạo DataLoader (augment train, không shuffle test)

**File:** `create_data_loaders` trong `src/data/dataset.py`.

```59:92:src/data/dataset.py
def create_data_loaders(config):
    ...
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ...
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, ...)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, ...)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, ...)
```

**Gọi từ:** `train.py` → `create_data_loaders(config.config)`.

```64:66:train.py
    print("\nCreating data loaders...")
    train_loader, test_loader, query_loader, num_classes = create_data_loaders(config.config)
```

> **Lưu ý:** Chuẩn bị file `*_img.txt` / `*_label.txt` từ CSV hay video là việc của script **gốc repo** (ví dụ `build_edu_txt_from_labeled_csv.py`, `build_etedu_split_from_csv.py`, `extract_edu_cbir_v2.py`) — không nằm trong `src/data/`.

---

## 3. Ma trận đồng xuất hiện nhãn (label co-occurrence) — ở đâu tính, ở đâu dùng?

### 3.1. Công thức trong mã

**File:** `src/data/label_graph.py` — hàm `build_label_cooccurrence_matrix`.

- `co_matrix[i,j]` = số lần nhãn `i` và `j` cùng bật trên cùng một ảnh (tổng qua toàn bộ mẫu đưa vào).
- Chuẩn hóa theo **số ảnh có nhãn cột `j`** (`class_counts`), thêm **self-loop** trên đường chéo = 1.

```4:27:src/data/label_graph.py
def build_label_cooccurrence_matrix(labels):
    ...
    labels = labels.float()
    co_matrix = torch.matmul(labels.t(), labels) # (C, C)
    class_counts = labels.sum(dim=0).unsqueeze(1) # (C, 1)
    class_counts[class_counts == 0] = 1
    adj_matrix = co_matrix / class_counts
    adj_matrix.fill_diagonal_(1.0)
    return adj_matrix
```

### 3.2. Ai gọi hàm này?

**File:** `src/training/trainer.py` — khi khởi tạo `Trainer`, gom **toàn bộ nhãn** từ `train_loader` một lần, ghép batch, rồi gọi `build_label_cooccurrence_matrix`.

```72:84:src/training/trainer.py
    def _build_adjacency_matrix(self):
        """Build label co-occurrence adjacency matrix from training data"""
        print("Building label co-occurrence graph...")
        
        all_labels = []
        for _, labels, _ in self.train_loader:
            all_labels.append(labels)
        
        all_labels = torch.cat(all_labels, dim=0)
        adj_matrix = build_label_cooccurrence_matrix(all_labels)
        
        print(f"Label graph built: {adj_matrix.shape}")
        return adj_matrix.to(self.device)
```

Ma trận này được lưu trong `self.adj_matrix` và **đưa vào forward mỗi batch**:

```117:122:src/training/trainer.py
        for batch_idx, (images, labels, _) in enumerate(pbar):
            ...
            img_hash, txt_hash, pred_labels = self.model(images, labels, self.adj_matrix)
```

---

## 4. Vision Transformer (ViT) — ở đâu?

**File:** `src/models/vision_encoder.py` — class `VisionEncoder`.

- Dùng **timm** `create_model`, bỏ head phân loại, lấy token CLS (hoặc chiều cuối tương đương).

```12:47:src/models/vision_encoder.py
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
    ...
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) == 3:
            features = features[:, 0]  # CLS token
        return features
```

**Gắn vào G-hash:** `GHashModel.__init__` tạo `self.image_encoder = VisionEncoder(...)` trong `src/models/ghash.py` (khoảng dòng 26–30).

**Forward ảnh:** `img_features = self.image_encoder(images)` trong `GHashModel.forward` (dòng 91).

---

## 5. Embedding nhãn (TextEncoder) — ở đâu?

**File:** `src/models/vision_encoder.py` — class `TextEncoder`.

- `nn.Embedding(num_classes, embed_dim)` + MLP chiếu lên `hidden_dim`.

```61:94:src/models/vision_encoder.py
    def __init__(self, num_classes, embed_dim=300, hidden_dim=512):
        ...
        self.embeddings = nn.Embedding(num_classes, embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, label_indices=None):
        if label_indices is None:
            label_indices = torch.arange(self.embeddings.num_embeddings, 
                                        device=self.embeddings.weight.device)
        embeds = self.embeddings(label_indices)
        embeds = self.projection(embeds)
        return embeds
```

**Trong G-hash:** `label_embeddings = self.text_encoder()` — trả về tensor `(num_classes, hidden_dim)` cho **tất cả** lớp, rồi đưa vào GAT (`src/models/ghash.py`, dòng 98).

---

## 6. GAT (Graph Attention Network) — ở đâu xử lý?

**File:** `src/models/gat.py`.

- **`GraphAttentionLayer`:** tính hệ số attention cặp nút, **mask** bằng `adj_matrix > 0`, softmax theo hàng, tổng hợp đặc trưng láng giềng.

```29:64:src/models/gat.py
    def forward(self, h, adj_matrix):
        Wh = self.W(h)
        N = Wh.size(0)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(self.a(a_input).squeeze(-1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
```

- **`MultiHeadGATLayer`:** ghép hoặc trung bình nhiều head.
- **`GAT`:** xếp chồng nhiều lớp (cấu hình `num_layers`, `num_heads` từ YAML).

```159:173:src/models/gat.py
    def forward(self, x, adj_matrix):
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, adj_matrix)
            if i < len(self.gat_layers) - 1:
                x = self.dropout(x)
        return x
```

**Khởi tạo trong G-hash:**

```40:48:src/models/ghash.py
        self.gat = GAT(
            in_features=self.hidden_dim,
            hidden_features=config['gat']['hidden_dim'],
            out_features=self.hidden_dim,
            num_heads=config['gat']['num_heads'],
            num_layers=config['gat']['num_layers'],
            dropout=config['gat']['dropout']
        )
```

**Áp dụng lên embedding nhãn:**

```96:104:src/models/ghash.py
        label_embeddings = self.text_encoder()
        enhanced_labels = self.gat(label_embeddings, adj_matrix)
        txt_hash = torch.tanh(self.txt_hash_fc(enhanced_labels))
```

→ **Tóm lại:** Ma trận kề đồng xuất hiện **không** chạy trong `gat.py`; nó được tính ở `label_graph.py` và truyền vào `GAT.forward` như `adj_matrix`.

---

## 7. Hash ảnh, hash nhãn, phân loại đa nhãn — ở đâu?

**File:** `src/models/ghash.py` — `GHashModel`.

- **MLP hash ảnh / nhãn** và **classifier** (logits đa nhãn):

```50:70:src/models/ghash.py
        self.img_hash_fc = nn.Sequential(
            nn.LayerNorm(img_feat_dim),
            nn.Linear(img_feat_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(self.hidden_dim, self.hash_bits)
        )
        self.txt_hash_fc = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hash_bits)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(img_feat_dim),
            nn.Linear(img_feat_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
```

- **Forward đầy đủ (train):**

```89:110:src/models/ghash.py
        img_features = self.image_encoder(images)
        img_hash = torch.tanh(self.img_hash_fc(img_features))
        label_embeddings = self.text_encoder()
        enhanced_labels = self.gat(label_embeddings, adj_matrix)
        txt_hash = torch.tanh(self.txt_hash_fc(enhanced_labels))
        pred_labels = self.classifier(img_features)
```

- **Sinh mã nhị phân ảnh (evaluate / retrieval offline trong Trainer):** `generate_hash_code` — `sign` sau `img_hash_fc` (không qua tanh ở bước cuối; continuous_hash trước sign).

```123:140:src/models/ghash.py
    def generate_hash_code(self, images):
        self.eval()
        with torch.no_grad():
            img_features = self.image_encoder(images)
            continuous_hash = self.img_hash_fc(img_features)
            binary_hash = torch.sign(continuous_hash)
            binary_hash[binary_hash == 0] = 1
        return binary_hash
```

---

## 8. Hàm mất GHashLoss — từng thành phần ở đâu?

**File:** `src/training/losses.py` — class `GHashLoss`.

| Thành phần | Ý nghĩa | Trong mã |
|------------|---------|----------|
| Classification | BCE đa nhãn (có `pos_weight` tùy chọn) | `loss_cls = self.bce_loss(...)` |
| Similarity | Căn hash ảnh với hash các nhãn đúng/sai | `similarity_loss` |
| Retrieval (ảnh–ảnh) | Cặp cùng ≥1 nhãn vs khác nhãn | `image_retrieval_loss` |
| Quantization | Ép gần ±1 | `quantization_loss` |
| Orthogonality | Hash nhãn không dính chùm | `loss_ortho` từ `txt_hash_norm @ txt_hash_norm.t()` vs `I` |
| Bit balance | Cân bit hash ảnh | `mean(img_hash.mean(dim=0)**2)` |

**Tổng hợp:**

```75:81:src/training/losses.py
        total_loss = (self.gamma * loss_cls + 
                     self.alpha * loss_sim + 
                     self.eta * loss_retrieval +
                     self.beta * loss_quant +
                     loss_ortho +
                     self.delta * loss_bit_balance)
```

**Cặp ảnh positive/negative cho retrieval:**

```152:161:src/training/losses.py
    def image_retrieval_loss(self, img_hash, labels):
        ...
        similarity_targets = (labels @ labels.t()) > 0
```

**Gắn hệ số từ YAML:** `train.py` khởi tạo `GHashLoss` với `alpha_similarity`, `beta_quantization`, v.v. (khoảng dòng 116–122).

**Cân nhãn (pos_weight):** `compute_pos_weight` trong `train.py` (dòng 29–41).

---

## 9. Vòng huấn luyện (forward / backward / metric) — ở đâu?

**File:** `src/training/trainer.py`.

- Mỗi batch: forward `model(..., self.adj_matrix)`, `criterion`, `backward`, clip gradient.

```117:132:src/training/trainer.py
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            img_hash, txt_hash, pred_labels = self.model(images, labels, self.adj_matrix)
            loss, loss_dict = self.criterion(img_hash, txt_hash, pred_labels, labels)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
```

- **Đánh giá retrieval:** encode query + database bằng `generate_hash_code`, gọi `compute_retrieval_metrics` (`src/evaluation/metrics.py`).

---

## 10. Hamming, mAP, định nghĩa “ảnh liên quan” — ở đâu?

**File:** `src/evaluation/metrics.py`.

- **Khoảng cách Hamming** (mã ±1):

```6:27:src/evaluation/metrics.py
def hamming_distance(code1, code2):
    ...
    hash_bits = code1.shape[1]
    hamming_dist = (hash_bits - code1 @ code2.T) / 2
    return hamming_dist.astype(int)
```

- **Ground truth multi-label:** có **ít nhất một nhãn chung**.

```47:48:src/evaluation/metrics.py
    similarity = (labels1 @ labels2.T) > 0
```

- **mAP:** sắp xếp theo Hamming tăng dần, tính AP trên chuỗi relevant, trung bình query — xem `mean_average_precision` (dòng 53–108).

**Tổng hợp nhiều K:** `compute_retrieval_metrics` (cuối file).

---

## 11. Điểm vào chương trình và so sánh Baseline

| Việc | File |
|------|------|
| Huấn luyện G-hash | `train.py` — `GHashModel` |
| Huấn luyện không GAT (ResNet) | `train.py --baseline` — `BaselineModel` trong `src/models/ghash.py` (dòng 162+) |
| Cấu hình YAML | `src/utils/config.py` — class `Config`; ví dụ `configs/et_edu_config.yaml` |

---

## 12. Suy luận ngoài Trainer (tùy sản phẩm)

**File:** `inference.py` — class `ImageRetriever`: nạp checkpoint, encode query/database, **tìm kiếm thô Hamming** rồi có thể **re-rank** (khác pipeline metric trong `Trainer.evaluate` — đã nêu trong các báo cáo vận hành khác).

Encoder trong inference tái sử dụng `self.model.image_encoder`, `self.model.img_hash_fc`, `self.model.classifier` (xem `_encode_tensor` trong `inference.py`).

---

## 13. Bảng tra cứu nhanh “Xử lý X ở file Y”

| Nội dung | File chính |
|----------|------------|
| Đọc ảnh + nhãn từ txt, augment | `src/data/dataset.py` |
| Ma trận đồng xuất hiện nhãn | `src/data/label_graph.py` |
| Gom nhãn train → `adj_matrix` | `src/training/trainer.py` → `_build_adjacency_matrix` |
| ViT | `src/models/vision_encoder.py` → `VisionEncoder` |
| Embedding nhãn | `src/models/vision_encoder.py` → `TextEncoder` |
| GAT (attention, mask kề) | `src/models/gat.py` |
| Ghép ViT + GAT + hash + classifier | `src/models/ghash.py` → `GHashModel` |
| Loss tổng hợp | `src/training/losses.py` → `GHashLoss` |
| Train loop + evaluate | `src/training/trainer.py` |
| mAP / Hamming / P@K / R@K | `src/evaluation/metrics.py` |
| Khởi chạy train | `train.py` |

---

*Tài liệu:* `BAO_CAO_MO_HINH_GHASH_VA_DAN_CHUNG_CODE.md` — mô tả đúng theo mã trong repo tại thời điểm tạo.*
