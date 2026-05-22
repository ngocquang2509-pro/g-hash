# G-hash: Educational Image Retrieval System

Deep learning-based image retrieval system using Graph Attention Networks (GAT) integrated with Vision Transformers for generating compact binary hash codes.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This project implements the **G-hash** architecture from the paper "G-hash: Educational image retrieval based on GAT integrated with deep hashing". The system combines:

- **Vision Transformer (ViT)** for global image feature extraction
- **Graph Attention Networks (GAT)** for learning label co-occurrence patterns
- **Binary Hashing** for efficient similarity search and storage

### Key Features

- 🎯 Multi-label image classification with 81 concepts
- 🔍 Fast similarity search using 64-bit binary codes
- 📊 Comprehensive evaluation metrics (mAP, Precision@K, Recall@K)
- 🚀 Production-ready inference API
- 📈 Rich visualization of training and evaluation results

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    G-hash Model                          │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Image Path              Label Path                      │
│  ┌──────────┐           ┌────────────┐                  │
│  │ ViT      │           │ Text       │                  │
│  │ Encoder  │           │ Embeddings │                  │
│  └────┬─────┘           └──────┬─────┘                  │
│       │                        │                         │
│       │ 768-dim           512-dim │                      │
│       ▼                        ▼                         │
│  ┌──────────┐           ┌────────────┐                  │
│  │ Hash     │           │ GAT        │                  │
│  │ Layer    │◄──────────│ Network    │                  │
│  └────┬─────┘           └────────────┘                  │
│       │                                                  │
│       ▼                                                  │
│  64-bit Binary Hash Code                                │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.9+
- Git LFS (for downloading dataset)
- CUDA-capable GPU (optional, can run on CPU/MPS)
- 10GB+ free disk space

### Setup

**1. Clone the repository**

```bash
git clone https://github.com/thangvb168/g-hash.git
cd G-hash_Educational_Image_Retrieval_System
```

**2. Create and activate virtual environment**

```bash
# Create virtual environment
python3 -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Download NUS-WIDE dataset**

Clone the dataset from HuggingFace:

```bash
# Ensure you're in the project root directory
cd data

# Clone NUS-WIDE dataset from HuggingFace
git clone https://huggingface.co/datasets/Lxyhaha/NUS-WIDE

# The dataset will be downloaded to data/NUS-WIDE/
# Total size: ~8GB
```

**5. Extract dataset files**

After downloading, extract the compressed files:

```bash
cd NUS-WIDE

# Extract image archives
unzip -q NUS-WIDE.zip

# Extract label files
unzip -q NUS_WID_Tags.zip

# Verify the structure
ls -la
# You should see:
# - Groundtruth/
# - ConceptsList/
# - Images/ (or Low_Resolution_Images/)
```

**6. Verify installation**

```bash
# Return to project root
cd ../..

# Test the installation
python -c "import torch; import timm; print('✓ Dependencies installed successfully')"
```

## 🚀 Quick Start

### 1. Create Mini Dataset

Create a smaller dataset for quick testing:

```bash
# Activate virtual environment first
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Create NUS-WIDE-MINI (7 classes, 500 train, 100 test)
python create_mini_dataset.py
```

This creates a mini dataset at `data/NUS-WIDE-MINI/` with:
- 7 classes: airport, animal, beach, bear, birds, boats, book
- 500 training images
- 100 test images

### 2. Training

**Train with mini dataset:**

```bash
# Train with mini config (faster for testing)
python train_mini.py --config configs/config_mini.yaml
```

**Custom training configuration:**

```bash
python train_mini.py \
    --config configs/config_mini.yaml \
    --hash-bits 32 \
    --epochs 30 \
    --batch-size 16
```

**Train baseline model (without GAT):**

```bash
python train_mini.py --config configs/config_mini.yaml --baseline
```

### 3. Testing

**Test with images from a folder:**

```bash
# Test images in default 'images/' folder
python test.py

# Test images from custom folder
python test.py --images-dir path/to/your/images

# Custom output directory
python test.py --images-dir my_images --output-dir my_results
```

The script automatically:
- Finds all `.jpg`, `.jpeg`, `.png` images in the folder
- Predicts labels for each image
- Retrieves similar images from database
- Saves visualizations to output directory

**Example usage:**

```bash
# Add your test images
mkdir images
cp ~/Downloads/beach.jpg images/
cp ~/Downloads/bear.jpg images/

# Run test
python test.py

# View results
open test_results/beach.png
open test_results/bear.png
```

## 📁 Project Structure

```
G-hash_Educational_Image_Retrieval_System/
├── src/
│   ├── models/          # Model architectures (GAT, ViT, G-hash)
│   ├── data/            # Dataset loaders and preprocessing
│   ├── training/        # Training loop and loss functions
│   ├── evaluation/      # Evaluation metrics (mAP, P@K, R@K)
│   └── utils/           # Utilities (config, visualization)
├── configs/             # Configuration files
│   ├── config_mini.yaml      # Mini dataset config (recommended)
│   └── config_nuswide2.yaml  # Full dataset config
├── data/                # Dataset directory
│   ├── NUS-WIDE-MINI/   # Mini dataset (7 classes)
│   └── NUS-WIDE 2/      # Full dataset (21 classes)
├── experiments/         # Training outputs
│   └── mini_runs/       # Timestamped training runs
├── images/              # Your test images (put images here)
├── test_results/        # Test output visualizations
├── create_mini_dataset.py  # Create mini dataset
├── train_mini.py        # Training script
├── inference_mini.py    # Inference module
├── test.py              # Testing script
└── requirements.txt     # Python dependencies
```

## 🎯 Configuration

Edit `configs/config_mini.yaml` to customize:

```yaml
dataset:
  name: "NUS-WIDE-MINI"
  data_root: "data/NUS-WIDE-MINI"
  num_classes: 7
  train_size: 500
  test_size: 100

model:
  hash_bits: 32  # Binary code length (16/32/64/128)
  image_encoder: "vit_base_patch16_224"

gat:
  num_heads: 2  # Multi-head attention
  num_layers: 2  # GAT depth

training:
  batch_size: 16
  num_epochs: 30
  learning_rate: 0.0001
  weight_decay: 0.001

loss:
  alpha_similarity: 1.0
  beta_quantization: 1.0
  gamma_classification: 1.0
```

## 📊 Evaluation Metrics

The system provides comprehensive evaluation:

- **mAP** (Mean Average Precision)
- **mAP@K** (mAP at top-K retrievals)
- **Precision@K** and **Recall@K**
- **Precision-Recall curves**
- **Hamming distance analysis**

Example output:

```
mAP@100:    0.2564
Precision@100: 0.1740
Recall@100:    0.0989
```

## 🎨 Visualizations

Training automatically generates:

1. **Training Curves** - Loss over epochs
2. **Loss Components** - Classification, Similarity, Quantization
3. **Top-K Metrics** - Precision and Recall at different K
4. **P-R Curves** - Precision-Recall trade-offs
5. **Predictions** - Visual predictions with confidence scores

All visualizations are saved to `experiments/runs/TIMESTAMP/`

## 🔬 Model Details

### Architecture

- **Backbone**: Vision Transformer (ViT-Base)
- **Label Encoder**: Graph Attention Network (4-head, 2-layer)
- **Hash Size**: 64-bit binary codes
- **Parameters**: ~88M total
- **Training**: End-to-end with combined loss

### Loss Function

```
L_total = γ·L_cls + α·L_sim + β·L_quant

- L_cls:   Classification loss (BCE)
- L_sim:   Similarity preservation loss
- L_quant: Quantization loss
```

Default weights: γ=1.0, α=1.0, β=0.1

## 📈 Performance

Trained on 2K samples with real label structure:

| Metric          | Value           |
| --------------- | --------------- |
| mAP             | 0.2246          |
| mAP@100         | 0.2564          |
| Precision@100   | 0.1740          |
| Training Time   | ~30 minutes     |
| Inference Speed | ~0.5s per image |

**Note**: Performance with synthetic images. Using real images would significantly improve results (expected mAP ~0.87).

## 🛠️ Development

### Running Tests

Test the model on diverse scenarios:

```bash
python -m pytest tests/
```

### Code Style

The project follows PEP 8 guidelines. Format code with:

```bash
black src/
isort src/
```

## 📝 Dataset

This project uses the **NUS-WIDE** dataset:

- **Source**: Flickr web images
- **Size**: 269,648 images
- **Labels**: 81 concept classes (multi-label)
- **Format**: Pre-split into Train/Test sets
- **Download**: Available on [HuggingFace](https://huggingface.co/datasets/Lxyhaha/NUS-WIDE)

**Dataset Structure:**

```
data/NUS-WIDE/
├── Groundtruth/
│   └── TrainTestLabels/      # Label files for each concept
├── ConceptsList/
│   └── Concepts81.txt         # List of 81 concepts
└── Images/                    # Image files
```

**Citation:**

```bibtex
@inproceedings{chua2009nus,
  title={Nus-wide: a real-world web image database from national university of singapore},
  author={Chua, Tat-Seng and Tang, Jinhui and Hong, Richang and Li, Haojie and Luo, Zhiping and Zheng, Yantao},
  booktitle={Proceedings of the ACM international conference on image and video retrieval},
  year={2009}
}
```

## 📄 License

This project is licensed under the MIT License.
