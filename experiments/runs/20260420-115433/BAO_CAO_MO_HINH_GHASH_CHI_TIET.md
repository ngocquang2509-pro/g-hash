# BÁO CÁO KỸ THUẬT: MÔ HÌNH G-HASH (CHI TIẾT + VỊ TRÍ CODE)

**Mục tiêu báo cáo:** Mô tả **toàn bộ kiến trúc G-hash** + **chỉ rõ code ở đâu** thực hiện từng xử lý cụ thể (ViT → hash, GAT xử lý nhãn, mã nhị phân, loss function, v.v.).

---

## 0) Tóm tắt 10 giây: G-hash là gì?

**G-hash = Image Hash + Text Hash + GAT Label Learning**

- **Image Stream**: ViT CLS token → FC → tanh → hash code ảnh
- **Label Stream**: 14 embedding nhãn → GAT học co-occurrence → hash code nhãn
- **Hashing**: Continuous (tanh) ∈ [-1,1] → Binary (sign) ∈ {-1,+1}
- **Retrieval**: Hamming distance (không cần ViT ở inference, chỉ cần sign!)
- **Training**: 6 loss components để học ảnh/nhãn/similarity/quantize/balance/orthogonal

---

## 1) KIẾN TRÚC TỔNG QUÁT

### 1.1 Sơ đồ khối model

```
┌──────────────────────────────────────────────────────────────────────┐
│                         GHashModel (nn.Module)                       │
│                      [src/models/ghash.py:1-220]                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─ IMAGE STREAM ─┐              ┌─ LABEL STREAM ─┐                │
│  │                │              │                │                │
│  │ Input images   │              │ 14 label class │                │
│  │ (B,3,224,224)  │              │ (one per nhãn) │                │
│  │       ↓        │              │       ↓        │                │
│  │  VisionEncoder │              │  TextEncoder   │                │
│  │  (ViT CLS)     │              │  (embed_dim)   │                │
│  │  (B, img_feat) │              │  (num_cls,hid) │                │
│  │       ↓        │              │       ↓        │                │
│  │ img_hash_fc    │              │     GAT        │                │
│  │  tanh → hash   │              │  + adj_matrix  │                │
│  │ (B, hash_bits) │              │ (num_cls,hid)  │                │
│  └────────────────┘              │       ↓        │                │
│                                  │ txt_hash_fc    │                │
│  classifier (B,num_classes)      │  tanh → hash   │                │
│         ↓                         │ (num_cls,bits) │                │
│  pred_labels (logits)             └────────────────┘                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
        ↓                ↓                    ↓
   (img_hash)    (txt_hash)         (pred_labels)
        ↓                ↓                    ↓
   GHashLoss (6 terms: cls + sim + ret + quant + ortho + balance)
```

### 1.2 File model chính

| File | Dòng | Nội dung |
|---|---|---|
| [src/models/ghash.py](src/models/ghash.py) | 1-220 | `GHashModel` class: forward, generate_hash_code |
| [src/models/gat.py](src/models/gat.py) | 1-150+ | `GraphAttentionLayer`, `MultiHeadGATLayer`, `GAT` |
| [src/models/vision_encoder.py](src/models/vision_encoder.py) | - | `VisionEncoder` (ViT), `TextEncoder` |
| [src/training/losses.py](src/training/losses.py) | 1-250+ | `GHashLoss` (6 loss components) |
| [src/training/trainer.py](src/training/trainer.py) | 1-250+ | `Trainer`: train_epoch, evaluate, train loop |

---

## 2) THÀNH PHẦN 1: IMAGE STREAM (ViT + Hash)

### 2.1 ViT encoder để trích xuất feature ảnh

**File:** [src/models/vision_encoder.py](src/models/vision_encoder.py)  
**Class:** `VisionEncoder`

```python
class VisionEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        # Load pretrained ViT từ timm
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.output_dim = self.model.embed_dim  # 768 cho ViT-base
    
    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.model.forward_features(x)  # (B, 197, 768) → pool CLS
        return x  # (B, 768)
```

**Vị trí code:**
- Đầu vào ảnh: shape `(B, 3, H, W)` (mặc định H=W=224)
- **Output:** CLS token feature `(B, img_feat_dim)` (ví dụ: 768 nếu ViT-base)

### 2.2 Image hash FC layers

**File:** [src/models/ghash.py](src/models/ghash.py)  
**Lines:** 54-59

```python
self.img_hash_fc = nn.Sequential(
    nn.LayerNorm(img_feat_dim),          # Line 55: Normalize ViT output
    nn.Linear(img_feat_dim, self.hidden_dim),
    nn.GELU(),                           # Line 57: Activation
    nn.Dropout(config['model']['dropout']),
    nn.Linear(self.hidden_dim, self.hash_bits)  # Line 59: Project to hash_bits
)
```

**Xử lý:**
1. **LayerNorm** (line 55): Chuẩn hóa output ViT để chống explosion gradient
2. **Linear1** → **GELU** → **Dropout** (lines 56-58): Expand + activate
3. **Linear2** (line 59): Project xuống `hash_bits` (mặc định 64)

### 2.3 Continuous hash từ tanh

**File:** [src/models/ghash.py](src/models/ghash.py)  
**Line:** 89

```python
img_hash = torch.tanh(self.img_hash_fc(img_features))  # (B, hash_bits)
```

**Kết quả:** `img_hash` ∈ [-1, +1], continuous values

### 2.4 Binary hash từ sign()

**File:** [src/models/ghash.py](src/models/ghash.py)  
**Lines:** 138-145

```python
def generate_hash_code(self, images):
    # Inference mode: đổi sang binary {-1, +1}
    self.eval()
    with torch.no_grad():
        img_features = self.image_encoder(images)
        continuous_hash = self.img_hash_fc(img_features)
        binary_hash = torch.sign(continuous_hash)  # Sign function
        binary_hash[binary_hash == 0] = 1  # Handle zeros → +1
    return binary_hash
```

**Chi tiết:**
- `torch.sign()` chuyển từ continuous → {-1, 0, +1}
- Line 143: Gán `0 → +1` để toàn bộ là {-1, +1}

---

## 3) THÀNH PHẦN 2: LABEL STREAM (Text Encoder + GAT)

### 3.1 Text encoder để tạo embedding nhãn

**File:** [src/models/vision_encoder.py](src/models/vision_encoder.py)  
**Class:** `TextEncoder`

```python
class TextEncoder(nn.Module):
    def __init__(self, num_classes=14, embed_dim=256, hidden_dim=512):
        # Mỗi class có 1 learnable embedding vector
        self.embeddings = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Linear(embed_dim, hidden_dim)
    
    def forward(self):
        # Không có input, chỉ lấy toàn bộ 14 embedding
        class_indices = torch.arange(self.num_classes)  # [0,1,2,...,13]
        embeddings = self.embeddings(class_indices)  # (14, embed_dim)
        return self.fc(embeddings)  # (14, hidden_dim)
```

**Output:** `(num_classes, hidden_dim)` = `(14, hidden_dim)`

### 3.2 Xây dựng adjacency matrix (label co-occurrence)

**File:** [src/data/label_graph.py](src/data/label_graph.py)

```python
def build_label_cooccurrence_matrix(labels):
    # labels: (N, num_classes) - multi-hot vectors từ train set
    
    # Compute co-occurrence: Y^T @ Y
    co_matrix = labels.t() @ labels  # (num_classes, num_classes)
    
    # Normalize by class counts (chống bias class hiếm)
    class_counts = labels.sum(dim=0).float()
    adj = co_matrix / (class_counts.unsqueeze(1) + 1e-8)
    
    # Set diagonal = 1 (mỗi class tự co-occur với nó 100%)
    adj.fill_diagonal_(1.0)
    
    return adj
```

**Mục tiêu:** 
- Nếu nhãn "Đứng" và "Dùng điện thoại" xuất hiện cùng nhau → adj[i,j] cao
- GAT dùng adjacency này để học "nhãn liên quan" nên phải enhance feature

**Vị trí tính toán:** [src/training/trainer.py](src/training/trainer.py) line 53

```python
def _build_adjacency_matrix(self):
    # Đọc toàn bộ label từ train_loader, tính co-occurrence
    all_labels = torch.cat([labels for _, labels, _ in self.train_loader])
    adj_matrix = build_label_cooccurrence_matrix(all_labels)
    return adj_matrix.to(self.device)
```

### 3.3 GAT: Graph Attention Network

**File:** [src/models/gat.py](src/models/gat.py)  
**Lines:** 1-150+

#### 3.3.1 Single GAT layer

```python
class GraphAttentionLayer(nn.Module):
    def forward(self, h, adj_matrix):
        # h: (N, in_features) - embedding 14 nhãn
        # adj_matrix: (N, N) - co-occurrence
        
        Wh = self.W(h)  # Linear transform: (14, out_features)
        
        # Attention scores: e[i,j] = LeakyReLU(a^T [W*h_i || W*h_j])
        e = self.leakyrelu(self.a(pairwise_concat))  # (14, 14)
        
        # MASK: chỉ để attention ≠0 nơi có edge (adj > 0)
        attention = torch.where(adj_matrix > 0, e, -9e15)  # (14, 14)
        
        # Normalize attention
        attention = F.softmax(attention, dim=1)  # Softmax theo hàng
        
        # Aggregate
        h_prime = attention @ Wh  # (14, out_features)
        
        return F.elu(h_prime)
```

**Chi tiết:**
- Line (mask): **"torch.where(adj_matrix > 0, e, -9e15)"** ← Đây là nơi GAT biết rằng "chỉ attend qua những nhãn có co-occur"
- **Softmax** normalize attention ở từng nhãn

#### 3.3.2 Multi-head GAT

**File:** [src/models/gat.py](src/models/gat.py)  
**Lines:** 87-110

```python
class MultiHeadGATLayer(nn.Module):
    def forward(self, h, adj_matrix):
        # Chạy nhiều head
        h_prime = torch.cat([att(h, adj_matrix) for att in self.attentions], dim=1)
        # Concatenate: (14, out*num_heads)
        return h_prime
```

**Lý do:** Mỗi head có attention khác nhau, learn các mối quan hệ khác nhau

#### 3.3.3 Xếp chồng GAT layers

**File:** [src/models/gat.py](src/models/gat.py)  
**Lines:** 130-155

```python
class GAT(nn.Module):
    def __init__(self, num_layers=2, num_heads=4, ...):
        # Layer 1: (hidden_dim) → (hidden_dim*num_heads) via multi-head
        # Layer 2: (hidden_dim*num_heads) → (out_dim) 
        self.gat_layers = nn.ModuleList([...])
    
    def forward(self, h, adj):
        for gat_layer in self.gat_layers:
            h = gat_layer(h, adj)
        return h  # (14, out_features)
```

### 3.4 Gọi GAT trong model forward

**File:** [src/models/ghash.py](src/models/ghash.py)  
**Line:** 94

```python
# ===== Label Path =====
label_embeddings = self.text_encoder()  # (14, hidden_dim)

# Apply GAT to learn label correlations
enhanced_labels = self.gat(label_embeddings, adj_matrix)  # (14, hidden_dim)
```

**Kết quả:** Mỗi nhãn giờ có "context" từ những nhãn liên quan

---

## 4) THÀNH PHẦN 3: TEXT HASH (Nhãn → Mã Hash)

### 4.1 Text hash FC layer

**File:** [src/models/ghash.py](src/models/ghash.py)  
**Lines:** 61-63

```python
self.txt_hash_fc = nn.Sequential(
    nn.LayerNorm(self.hidden_dim),
    nn.Linear(self.hidden_dim, self.hash_bits)
)
```

### 4.2 Continuous text hash

**File:** [src/models/ghash.py](src/models/ghash.py)  
**Line:** 98

```python
txt_hash = torch.tanh(self.txt_hash_fc(enhanced_labels))  # (num_classes, hash_bits)
```

**Kết quả:** 
- Shape: `(14, 64)` (14 nhãn, mỗi nhãn 64-bit code)
- Value: ∈ [-1, +1]

### 4.3 Binary text hash

**File:** [src/models/ghash.py](src/models/ghash.py)  
**Lines:** 157-168

```python
def get_label_hash_codes(self, adj_matrix):
    # Lấy binary hash cho 14 nhãn
    label_embeddings = self.text_encoder()
    enhanced_labels = self.gat(label_embeddings, adj_matrix)
    continuous_hash = self.txt_hash_fc(enhanced_labels)
    binary_hash = torch.sign(continuous_hash)
    binary_hash[binary_hash == 0] = 1
    return binary_hash  # (14, hash_bits)
```

---

## 5) THÀNH PHẦN 4: CLASSIFIER (Dự đoán Nhãn)

### 5.1 Classifier FC layers

**File:** [src/models/ghash.py](src/models/ghash.py)  
**Lines:** 65-70

```python
self.classifier = nn.Sequential(
    nn.LayerNorm(img_feat_dim),
    nn.Linear(img_feat_dim, self.hidden_dim),
    nn.GELU(),
    nn.Dropout(config['model']['dropout']),
    nn.Linear(self.hidden_dim, self.num_classes)
)
```

**Input:** ViT features `(B, img_feat_dim)`  
**Output:** Logits `(B, num_classes)` để sau đó dùng sigmoid/BCE

### 5.2 Gọi trong forward

**File:** [src/models/ghash.py](src/models/ghash.py)  
**Line:** 103

```python
pred_labels = self.classifier(img_features)  # (B, num_classes)
```

---

## 6) THÀNH PHẦN 5: LOSS FUNCTION (6 Thành Phần)

**File:** [src/training/losses.py](src/training/losses.py)  
**Lines:** 1-250+

### 6.1 Forward của GHashLoss

```python
def forward(self, img_hash, txt_hash, pred_labels, true_labels):
```

**Inputs:**
- `img_hash`: (B, hash_bits) - continuous từ tanh
- `txt_hash`: (num_classes, hash_bits) - continuous từ tanh
- `pred_labels`: (B, num_classes) - logits
- `true_labels`: (B, num_classes) - multi-hot ground truth

### 6.2 Loss 1: Classification Loss

**File:** [src/training/losses.py](src/training/losses.py)  
**Lines:** 37-40

```python
loss_cls = self.bce_loss(pred_labels, true_labels)
# BCEWithLogitsLoss(pos_weight=pos_weight) từ line 28
```

**Mục tiêu:** Ép `pred_labels` dự đoán đúng multi-hot labels  
**Formula:** $L_{cls} = -\frac{1}{B} \sum_i \left[ y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right] \times pos\_weight_i$

**pos_weight tính toán:** [src/training/trainer.py](src/training/trainer.py) trong `train.py` → `compute_pos_weight()` (cân bằng class mất cân bằng)

### 6.3 Loss 2: Similarity Loss (Nhãn-Ảnh co-occurrence)

**File:** [src/training/losses.py](src/training/losses.py)  
**Lines:** 78-103

```python
def similarity_loss(self, img_hash, txt_hash, labels):
    # Normalize hash codes
    img_hash_norm = F.normalize(img_hash, p=2, dim=1)
    txt_hash_norm = F.normalize(txt_hash, p=2, dim=1)
    
    # Cosine similarity matrix: (B, num_classes)
    similarity_matrix = img_hash_norm @ txt_hash_norm.t()
    
    # Positive: ảnh có nhãn, similarity phải cao
    pos_sim = (similarity_matrix * labels).sum() / (labels.sum() + 1e-8)
    
    # Negative: ảnh không có nhãn, similarity phải thấp
    neg_sim = (similarity_matrix * (1-labels)).sum() / ((1-labels).sum() + 1e-8)
    
    # Margin-based loss
    margin = 0.5
    loss = torch.clamp(margin + neg_sim - pos_sim, min=0)
    return loss
```

**Mục tiêu:** Ảnh → Hash ảnh, Nhãn → Hash nhãn, cosine distance gần nếu co-occur

### 6.4 Loss 3: Image Retrieval Loss (Ảnh-Ảnh similarity)

**File:** [src/training/losses.py](src/training/losses.py)  
**Lines:** 120-148

```python
def image_retrieval_loss(self, img_hash, labels):
    # Positive: 2 ảnh share ≥1 nhãn
    similarity_targets = (labels @ labels.t()) > 0
    
    # Pairwise similarity (via hash codes)
    sim_logits = torch.matmul(img_hash, img_hash.t()) / img_hash.size(1)
    
    # Loss: push positive pairs close, negative pairs far
    losses = []
    if pos_mask.any():
        losses.append(F.softplus(-sim_logits[pos_mask]).mean())  # softplus(-x) ↓ khi x↑
    if neg_mask.any():
        losses.append(F.softplus(sim_logits[neg_mask]).mean())   # softplus(x) ↓ khi x↓
    return sum(losses) / len(losses)
```

**Mục tiêu:** Ảnh cùng nhãn → hash gần; ảnh khác nhãn → hash xa

### 6.5 Loss 4: Quantization Loss

**File:** [src/training/losses.py](src/training/losses.py)  
**Lines:** 110-119

```python
def quantization_loss(self, hash_codes):
    # hash_codes từ tanh ∈ [-1, 1]
    binary_codes = torch.sign(hash_codes)  # {-1, 0, +1}
    binary_codes[binary_codes == 0] = 1
    
    loss = F.mse_loss(hash_codes, binary_codes.detach())
    # = mean((hash_codes - sign(hash_codes))^2)
    return loss
```

**Mục tiêu:** Ép continuous hash gần với binary version (sign)  
**Formula:** $L_{quant} = \frac{1}{B \cdot bits} \sum_{i,k} (h^i_k - \text{sign}(h^i_k))^2$

### 6.6 Loss 5: Orthogonality Loss (Chống Mode Collapse)

**File:** [src/training/losses.py](src/training/losses.py)  
**Lines:** 67-73

```python
# Ép text hash codes của các nhãn KHÁC NHAU phải trực giao
txt_hash_norm = F.normalize(txt_hash, p=2, dim=1)
txt_sim = torch.matmul(txt_hash_norm, txt_hash_norm.t())  # (14, 14)
eye = torch.eye(txt_hash.size(0), device=txt_hash.device)

loss_ortho = F.mse_loss(txt_sim, eye) * 2.0
# Penalty: txt_sim[i,j] phải ≈ eye[i,j]
# → diagonal = 1 (tự tương tự 100%), off-diagonal = 0 (khác nhãn không giống)
```

**Mục tiêu:** Chống situation đó GAT làm tất cả nhãn collapse → toàn 0000 hay 1111

### 6.7 Loss 6: Bit Balance Loss

**File:** [src/training/losses.py](src/training/losses.py)  
**Lines:** 74-77

```python
# Ép phân phối bit cân bằng (50% bit=1, 50% bit=-1)
loss_bit_balance = torch.mean(img_hash.mean(dim=0) ** 2)
# mean(img_hash, dim=0) = (hash_bits,) - mean value của mỗi bit qua batch
# Bình phương: penalize khi bit bias về +1 hay -1
```

**Mục tiêu:** Chống mode collapse ảnh (vd: toàn +1 bit)

### 6.8 Total Loss

**File:** [src/training/losses.py](src/training/losses.py)  
**Lines:** 63-66

```python
total_loss = (self.gamma * loss_cls +          # classification weight
             self.alpha * loss_sim +           # similarity weight
             self.eta * loss_retrieval +       # retrieval weight
             self.beta * loss_quant +          # quantization weight
             loss_ortho +
             self.delta * loss_bit_balance)
```

**Default weights (từ config):**
- gamma: 1.0
- alpha: 1.0
- beta: 0.1
- delta: 0.5
- eta: 0.5

---

## 7) THÀNH PHẦN 6: TRAINING LOOP

### 7.1 Initialize trainer

**File:** [train.py](train.py)  
**Các dòng chính:**

```python
# 1. Load config
config = Config(args.config)

# 2. Create dataloaders
train_loader, test_loader, query_loader = create_data_loaders(config.config)

# 3. Compute pos_weight (cân bằng class)
pos_weight = compute_pos_weight(train_loader)

# 4. Build model
if args.model == 'ghash':
    model = GHashModel(config.config)
else:
    model = BaselineModel(config.config)

# 5. Build loss
criterion = GHashLoss(pos_weight=pos_weight, ...)

# 6. Create trainer
trainer = Trainer(model, train_loader, test_loader, query_loader, criterion, config.config, device)

# 7. Train
trainer.train()
```

### 7.2 Train epoch

**File:** [src/training/trainer.py](src/training/trainer.py)  
**Lines:** 88-141

```python
def train_epoch(self, epoch):
    self.model.train()
    
    for batch_idx, (images, labels, _) in enumerate(self.train_loader):
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass (line 107)
        img_hash, txt_hash, pred_labels = self.model(images, labels, self.adj_matrix)
        
        # Compute loss (line 110)
        loss, loss_dict = self.criterion(img_hash, txt_hash, pred_labels, labels)
        
        # Backward (line 113-115)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        
        # Optimizer step (line 117)
        self.optimizer.step()
```

**Chi tiết:**
- **adj_matrix**: build mỗi lần khởi tạo trainer (line 53), cùng cho mọi epoch
- **Gradient clipping** line 115: max norm = 5.0
- **Accumulate loss** rồi average (line 121-123)

### 7.3 Evaluate

**File:** [src/training/trainer.py](src/training/trainer.py)  
**Lines:** 143-210

```python
def evaluate(self, epoch=0):
    self.model.eval()
    
    # Generate hash codes database
    db_codes = []
    for images, labels, _ in self.database_eval_loader:
        codes = self.model.generate_hash_code(images)  # Binary codes
        db_codes.append(codes.cpu())
    db_codes = torch.cat(db_codes, dim=0).numpy()
    
    # Generate hash codes queries
    query_codes = []
    for images, labels, _ in self.query_loader:
        codes = self.model.generate_hash_code(images)
        query_codes.append(codes.cpu())
    query_codes = torch.cat(query_codes, dim=0).numpy()
    
    # Compute metrics (line 185)
    metrics = compute_retrieval_metrics(
        query_codes, db_codes, query_labels, db_labels,
        top_k_list=self.config['evaluation']['top_k']
    )
    return metrics
```

**Tính Hamming distance:** [src/evaluation/metrics.py](src/evaluation/metrics.py)

### 7.4 Hamming Distance Computation

**File:** [src/evaluation/metrics.py](src/evaluation/metrics.py)

```python
def hamming_distance(code1, code2):
    # code1, code2 ∈ {-1, +1}^bits
    # Hamming = số bit khác nhau
    # Formula: (bits - dot_product) / 2
    return (code1.shape[1] - np.dot(code1, code2.T)) / 2
```

**Ví dụ:**
- code1 = [-1, -1, +1, +1] (4 bits)
- code2 = [-1, +1, +1, +1]
- dot_product = (-1)*(-1) + (-1)*(+1) + (+1)*(+1) + (+1)*(+1) = 1 - 1 + 1 + 1 = 2
- Hamming = (4 - 2) / 2 = 1 (1 bit khác)

### 7.5 Main training loop

**File:** [src/training/trainer.py](src/training/trainer.py)  
**Lines:** 213-260

```python
def train(self):
    for epoch in range(1, num_epochs + 1):
        # 1. Train 1 epoch
        train_loss, loss_components = self.train_epoch(epoch)
        
        # 2. Step scheduler
        self.scheduler.step()
        
        # 3. Evaluate mỗi 5 epoch
        if epoch % 5 == 0 or epoch == num_epochs:
            metrics = self.evaluate(epoch)
            self.history['eval_metrics'].append(metrics)
            
            # 4. Check best model
            if metrics['mAP'] > self.best_map:
                self.best_map = metrics['mAP']
                torch.save(self.model.state_dict(), self.save_dir / 'best_model.pth')
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        # 5. Checkpoint mỗi 10 epoch
        if epoch % 10 == 0:
            torch.save(self.model.state_dict(), 
                      self.save_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # 6. Early stopping
        if self.patience_counter >= self.patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

**Artifacts lưu:** 
- `best_model.pth` (theo mAP cao nhất)
- `checkpoint_epoch_*.pth` (định kỳ)
- `metrics.txt`, plot files (ở `create_experiment_report()`)

---

## 8) THÀNH PHẦN 7: RETRIEVAL (INFERENCE)

### 8.1 Encode query

**File:** [inference.py](inference.py)

```python
def _encode_tensor(self, images):
    with torch.no_grad():
        # Generate binary hash (dùng cho Hamming stage 1)
        binary_hash = model.generate_hash_code(images)  # {-1, +1}
        
        # Generate continuous hash (dùng cho rerank)
        continuous_hash = img_hash_fc(encoder(images))  # tanh → [-1, 1]
        
        # Generate probabilities (dùng cho rerank)
        probabilities = torch.sigmoid(classifier(encoder(images)))
    return binary_hash, continuous_hash, probabilities
```

### 8.2 Stage 1: Coarse retrieval bằng Hamming

```python
# Compute Hamming distance tới mỗi ảnh trong database
hamming_dists = compute_hamming_distances(query_binary, db_binary)  # (N,)

# Lấy top-K gần nhất
_, top_indices = torch.topk(-hamming_dists, k=coarse_k)
```

**Tính Hamming:** Line ở [src/evaluation/metrics.py](src/evaluation/metrics.py)

### 8.3 Stage 2: Rerank (Fine-grained)

```python
# Reranking score = weighted combination
score = (0.55 * cosine_sim_continuous +
         0.30 * cosine_sim_probs +
         0.15 * (1 - normalized_hamming))

# Sort lại coarse_k candidates
final_ranking = sort(score)[:top_k]
```

**Rerank weights:** Tunable từ config (mặc định: 55% continuous + 30% probs + 15% Hamming)

### 8.4 Stage 3: Diversity (MMR + constraint)

```python
# Diverse selection: Maximum Marginal Relevance + per-video constraint
selected = []
remaining = top_k_candidates

for i in range(k):
    # Chọn candidate nào:
    # 1) Có relevance cao (từ reranking)
    # 2) Khác với những candidate đã chọn (diverse)
    # 3) Từ video khác (tránh redup)
    
    best = select_diverse(remaining, selected, video_ids)
    selected.append(best)
    remaining.remove(best)
```

**Mục tiêu:** Top-K không bị "nhóm" quá nhiều ảnh từ 1 video

---

## 9) DATA LOADING (VỊ TRÍ XỬ LÝ DỮ LIỆU)

### 9.1 Dataset class

**File:** [src/data/dataset.py](src/data/dataset.py)

```python
class NUSWIDE2Dataset(nn.Module):
    def __init__(self, dataset_name, split, transform=None):
        # Với ET-EDU:
        if dataset_name == 'ET-EDU':
            self.img_file = f'{data_root}/{split}_img.txt'
            self.label_file = f'{data_root}/{split}_label.txt'
    
    def __getitem__(self, idx):
        # Load ảnh từ disk
        img = Image.open(self.img_paths[idx]).convert('RGB')
        
        # Apply transform (crop, flip, normalize)
        if self.transform:
            img = self.transform(img)
        
        # Load label (14-dim multi-hot)
        label = self.labels[idx]
        
        return img_tensor, label_tensor, idx
```

### 9.2 Transform pipeline (xử lý dữ liệu ảnh)

**File:** [src/data/dataset.py](src/data/dataset.py)

```python
# Train transform
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Test transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

---

## 10) BÀN TÓM TẮT: "VỊ TRÍ CODE" CHO MỖI XỨNG LỰ

| **Xử lý** | **File** | **Line(s)** | **Chi tiết** |
|---|---|---|---|
| **Tải config** | [train.py](train.py) | ~ | `Config(args.config)` từ YAML |
| **ViT encode ảnh** | [src/models/vision_encoder.py](src/models/vision_encoder.py) | - | `VisionEncoder.forward()` lấy CLS token |
| **Image hash FC** | [src/models/ghash.py](src/models/ghash.py) | 54-59 | LayerNorm + Linear + GELU + Linear |
| **Continuous image hash (tanh)** | [src/models/ghash.py](src/models/ghash.py) | 89 | `torch.tanh(img_hash_fc(...))` |
| **Binary image hash (sign)** | [src/models/ghash.py](src/models/ghash.py) | 138-145 | `torch.sign()` → {-1, +1} |
| **Text embedding** | [src/models/vision_encoder.py](src/models/vision_encoder.py) | - | `TextEncoder.forward()` → (14, hidden_dim) |
| **Build label co-occurrence** | [src/data/label_graph.py](src/data/label_graph.py) | - | `Y^T @ Y`, normalize, set diag=1 |
| **GAT single layer** | [src/models/gat.py](src/models/gat.py) | 8-65 | `GraphAttentionLayer.forward()` với mask |
| **Multi-head GAT** | [src/models/gat.py](src/models/gat.py) | 87-110 | Concatenate nhiều head |
| **Full GAT stack** | [src/models/gat.py](src/models/gat.py) | 130-155+ | Multi-layer GAT |
| **Call GAT trong model** | [src/models/ghash.py](src/models/ghash.py) | 94 | `enhanced_labels = gat(label_embeddings, adj_matrix)` |
| **Text hash FC** | [src/models/ghash.py](src/models/ghash.py) | 61-63 | Tương tự image hash |
| **Continuous text hash** | [src/models/ghash.py](src/models/ghash.py) | 98 | `torch.tanh(txt_hash_fc(...))` → (14, bits) |
| **Classifier (label prediction)** | [src/models/ghash.py](src/models/ghash.py) | 65-70, 103 | `classifier(img_features)` → logits |
| **BCE loss** | [src/training/losses.py](src/training/losses.py) | 28, 37-40 | `BCEWithLogitsLoss(pos_weight=...)` |
| **Similarity loss (nhãn-ảnh)** | [src/training/losses.py](src/training/losses.py) | 78-103 | Cosine sim, margin-based |
| **Retrieval loss (ảnh-ảnh)** | [src/training/losses.py](src/training/losses.py) | 120-148 | Pairwise, softplus |
| **Quantization loss** | [src/training/losses.py](src/training/losses.py) | 110-119 | MSE(continuous - binary) |
| **Orthogonality loss (chống collapse)** | [src/training/losses.py](src/training/losses.py) | 67-73 | MSE(txt_sim, identity) |
| **Bit balance loss** | [src/training/losses.py](src/training/losses.py) | 74-77 | MSE(mean(img_hash), 0) |
| **Total loss** | [src/training/losses.py](src/training/losses.py) | 63-66 | Weighted sum 6 terms |
| **Train epoch** | [src/training/trainer.py](src/training/trainer.py) | 88-141 | Forward, loss.backward(), optimizer.step() |
| **Evaluate (Hamming encoding)** | [src/training/trainer.py](src/training/trainer.py) | 143-210 | `generate_hash_code()` → binary codes |
| **Hamming distance** | [src/evaluation/metrics.py](src/evaluation/metrics.py) | - | `(bits - dot_prod) / 2` |
| **mAP / Retrieval metrics** | [src/evaluation/metrics.py](src/evaluation/metrics.py) | - | Average precision @ K |
| **Main training loop** | [src/training/trainer.py](src/training/trainer.py) | 213-260 | Epoch loop, checkpoint, early stop |
| **Create report** | [src/utils/visualization.py](src/utils/visualization.py) | 260-295 | Plots + metrics.txt |
| **Query encoding (inference)** | [inference.py](inference.py) | - | `_encode_tensor()` → 3 versions of hash |
| **Hamming stage 1 (coarse)** | [inference.py](inference.py) | - | Compute Hamming distances |
| **Rerank stage 2 (fine)** | [inference.py](inference.py) | - | Weighted score combination |
| **Diversity stage 3 (MMR)** | [inference.py](inference.py) | - | Per-video constraint + MMR |

---

## 11) FLOW CHART TOÀN HỆ THỐNG

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ [train.py] ENTRY POINT                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
  │
  ├─ Load config (YAML)
  │   └─> [Config.py]
  │
  ├─ Create DataLoaders
  │   └─> [src/data/dataset.py] NUSWIDE2Dataset
  │       ├─ Load train_img.txt + train_label.txt
  │       └─ Apply transform (crop/flip/normalize)
  │
  ├─ Compute pos_weight (class imbalance)
  │
  ├─ Build Model [src/models/ghash.py] GHashModel
  │   ├─ VisionEncoder (ViT) [src/models/vision_encoder.py]
  │   ├─ TextEncoder (14 embeddings) [src/models/vision_encoder.py]
  │   ├─ GAT (2 layers, 4 heads) [src/models/gat.py]
  │   ├─ img_hash_fc (LayerNorm + FC + GELU + FC → tanh)
  │   ├─ txt_hash_fc (LayerNorm + FC → tanh)
  │   └─ classifier (FC)
  │
  ├─ Build Loss [src/training/losses.py] GHashLoss
  │   └─ 6 components: cls + sim + ret + quant + ortho + balance
  │
  └─ Trainer [src/training/trainer.py]
      │
      ├─ Build adj_matrix [src/data/label_graph.py]
      │   └─ Y^T @ Y (co-occurrence), normalize, diag=1
      │
      ├─ FOR each epoch:
      │   │
      │   ├─ train_epoch()
      │   │   ├─ FOR each batch:
      │   │   │   ├─ model.forward(img, labels, adj_matrix)
      │   │   │   │   ├─ img → ViT → img_hash_fc → tanh → img_hash (cont)
      │   │   │   │   ├─ text_encoder() → label_emb
      │   │   │   │   ├─ gat(label_emb, adj_matrix) → enhanced_label
      │   │   │   │   ├─ txt_hash_fc(enhanced_label) → tanh → txt_hash (cont)
      │   │   │   │   └─ classifier(img_feat) → pred_labels (logits)
      │   │   │   │
      │   │   │   ├─ loss(img_hash, txt_hash, pred_labels, labels)
      │   │   │   │   ├─ loss_cls: BCE(pred_labels, labels)
      │   │   │   │   ├─ loss_sim: margin based (img_hash vs txt_hash)
      │   │   │   │   ├─ loss_ret: pairwise (img vs img)
      │   │   │   │   ├─ loss_quant: MSE(cont, sign)
      │   │   │   │   ├─ loss_ortho: ép txt_hash trực giao
      │   │   │   │   └─ loss_balance: ép bit cân bằng
      │   │   │   │
      │   │   │   └─ backward() + optimizer.step()
      │   │   │
      │   │   └─ Accumulate loss (logging)
      │   │
      │   ├─ scheduler.step() (CosineAnnealingLR)
      │   │
      │   └─ IF epoch % 5 == 0:
      │       │
      │       └─ evaluate()
      │           ├─ model.generate_hash_code(db_images)
      │           │   ├─ img_feat = ViT(img)
      │           │   ├─ cont_hash = img_hash_fc(img_feat)
      │           │   └─ binary_hash = sign(cont_hash)  ← USE FOR RETRIEVAL!
      │           │
      │           ├─ model.generate_hash_code(query_images) → query_binary
      │           │
      │           ├─ Hamming distances [src/evaluation/metrics.py]
      │           │   └─ d = (bits - query @ db.T) / 2
      │           │
      │           └─ compute_retrieval_metrics() → mAP, P@K, R@K
      │               ├─ Relevant = share ≥1 label
      │               └─ Average Precision @ K
      │
      ├─ Save best_model.pth (highest mAP)
      ├─ Save checkpoint_epoch_*.pth (mỗi 10 epoch)
      └─ Early stopping (patience)
          │
          └─ create_experiment_report()
              ├─ Plot training_curves.png
              ├─ Plot loss_components.png
              ├─ Plot topk_metrics.png
              ├─ Plot pr_curve.png
              └─ Write metrics.txt
```

---

## 12) VÍ DỤ CỤ THỂ: Một lần forward pass

**Input:**
- Batch 8 ảnh lớp học
- Nhãn: ảnh 0,1 = [đứng, dùng điện thoại], ảnh 2-7 = [ngồi, ...]

**Thực thi:**

```
1) [src/models/ghash.py line 75-89] Image Stream
   img (B=8, 3, 224, 224)
   └─> VisionEncoder (ViT)
       └─> img_features (8, 768)
           └─> img_hash_fc
               └─> img_hash_cont (8, 64) ∈ [-1, +1]

2) [src/models/ghash.py line 92-98] Label Stream
   Text Encoder
   └─> label_emb (14, 512)
       └─> GAT (with adj_matrix from co-occurrence)
           └─> enhanced_label (14, 512)
               └─> txt_hash_fc
                   └─> txt_hash_cont (14, 64) ∈ [-1, +1]

3) [src/models/ghash.py line 103] Classifier
   img_features (8, 768)
   └─> classifier
       └─> pred_labels (8, 14) logits

4) [src/training/losses.py line 37-66] Loss Computation
   img_hash_cont (8, 64), txt_hash_cont (14, 64), 
   pred_labels (8, 14), true_labels (8, 14)
   
   ├─ loss_cls = BCE(pred_labels, true_labels)
   │           = "ảnh 0 predicted [1, 1, 0, ...] nhưng true [1, 1, ...]"
   │           → predict sai → loss cao
   │
   ├─ loss_sim = similarity(img_hash, txt_hash, true_labels)
   │           = "hash ảnh 0 gần hash nhãn 'đứng' & 'dùng phone' (true label)"
   │           = "hash ảnh 0 xa hash nhãn 'ngồi' (false label)"
   │
   ├─ loss_ret = image_retrieval(img_hash, true_labels)
   │           = "ảnh 0 & 1 share 2 label → hash gần"
   │           = "ảnh 0 & 2 share 0 label → hash xa"
   │
   ├─ loss_quant = MSE(img_hash_cont, sign(img_hash_cont))
   │             = MSE(8, 64 continuous values, 8*64 binary {-1,+1})
   │
   ├─ loss_ortho = MSE(txt_hash_norm @ txt_hash_norm.T, Identity)
   │             = "hash nhãn 'đứng' ⊥ hash nhãn 'ngồi'" (trực giao)
   │
   └─ loss_balance = MSE(mean(img_hash, dim=0), 0)
                   = "mỗi bit trong batch (8 ảnh) = trung bình 0"

5) Total Loss = 1.0 * loss_cls + 1.0 * loss_sim + 0.5 * loss_ret 
              + 0.1 * loss_quant + loss_ortho + 0.5 * loss_balance

6) Backward pass [src/training/trainer.py line 113]
   loss.backward()
   ├─ Compute gradients cho tất cả parameters
   │   ├─ VisionEncoder (pretrained ViT) - có lr
   │   ├─ TextEncoder embeddings - có gradient
   │   ├─ GAT weights - có gradient
   │   ├─ FC layers - có gradient
   │   └─ classifier - có gradient
   │
   └─ Clip grad norm (max 5.0) [src/training/trainer.py line 115]

7) Optimizer step [src/training/trainer.py line 117]
   optimizer.step() ← Adam update all parameters
```

---

## 13) VÍ DỤ CỤ THỂ: Inference (Retrieval)

**Input:**
- Query ảnh: sinh viên dùng điện thoại đứng (1 ảnh)
- Database: 1000 ảnh lớp học

**Thực thi:**

```
1) Encode query [inference.py]
   query_img (1, 3, 224, 224)
   └─> ViT features (1, 768)
       ├─> img_hash_fc → tanh → continuous_hash (1, 64)
       ├─> sign(continuous_hash) → binary_hash (1, 64) {-1,+1}
       └─> classifier → pred_probs (1, 14) sigmoid

2) Encode database [inference.py]
   FOR each batch in db (batch_size=128):
       db_img (128, 3, 224, 224)
       └─> generate_hash_code()
           └─> binary_hash_db (1000, 64) ← Concatenated từ mọi batch

3) Stage 1: Coarse retrieval (Hamming) [inference.py]
   Compute Hamming distance: (64 - binary_query @ binary_db.T) / 2
   └─> distances (1, 1000)
       └─> topk(-distances, k=100) → coarse_indices (1, 100)
           ← Chỉ cần 100 candidate tốt nhất (rất nhanh!)

4) Stage 2: Rerank (Fine-grained) [inference.py]
   FOR candidate in coarse_indices:
       score = 0.55 * cosine(continuous_query, continuous_candidate)
             + 0.30 * cosine(probs_query, probs_candidate)
             + 0.15 * (1 - normalized_hamming)
   
   └─> sort score → top 20 candidates

5) Stage 3: Diversity [inference.py]
   Maximum Marginal Relevance + per-video constraint
   └─> Remove candidates từ cùng video
       └─> Final top-5: [ảnh1, ảnh2, ảnh3, ảnh4, ảnh5]
           ← Từ video khác nhau, không bị "cluster"
```

---

## 14) KẾT LUẬN: "Mô hình G-hash là gì" (Dạng trình bày)

**Để nói trước hội đồng:**

1. **Vấn đề:** Multi-label image retrieval trong video (lớp học) → retrieval nhanh (Hamming).

2. **Giải pháp G-hash = 3 thành phần chính:**
   - **Image stream**: ViT → FC → tanh hash (64-bit)
   - **Label stream**: Text embedding → GAT (học co-occurrence) → FC → tanh hash
   - **Unified loss**: 6 thành phần train img/txt/sim/quant/ortho/balance

3. **Inference siêu nhanh:**
   - Binary hash {-1,+1}: mỗi ảnh = 64 bit → Hamming distance rất nhanh
   - Coarse + Rerank + Diversity → top-K đa dạng

4. **Tại sao GAT?**
   - Nhãn không độc lập: "đứng" + "dùng phone" thường co-occur
   - GAT học adjacency matrix (co-occurrence) → enhance related labels
   - Kết quả: text hash codes thêm semantic → ép image hash learn đúng

5. **Tại sao 6 losses?**
   - **cls**: dự đoán nhãn đúng
   - **sim**: ảnh & nhãn đúng → hash gần
   - **ret**: ảnh cùng nhãn → hash gần
   - **quant**: ép continuous → binary
   - **ortho**: chống mode collapse (toàn 0 hoặc 1)
   - **balance**: đảm bảo diversity bit

---

**Báo cáo này viết xong!** Bạn có thể dùng để:**
- Trình bày chi tiết về mô hình
- Giải thích "code ở đâu" cho từng xử lý
- Tra cứu implementation khi cần
