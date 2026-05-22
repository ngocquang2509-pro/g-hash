# G-Hash: Classroom Educational Behavior Image Retrieval System (ET-EDU CBIR)

Deep learning-based multi-label Content-Based Image Retrieval (CBIR) system designed for classroom settings. By combining **Vision Transformers (ViT)** for rich visual feature extraction and **Graph Attention Networks (GAT)** to model label co-occurrence patterns, G-Hash produces compact binary codes (64-bit) for high-precision, low-latency student behavior similarity searching.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask WebApp](https://img.shields.io/badge/Flask-Web_UI-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Classroom Behavior Categories](#-classroom-behavior-categories)
3. [Architecture](#-architecture)
4. [Data Curation & Processing Pipeline](#-data-curation--processing-pipeline)
5. [Installation & Setup](#-installation--setup)
6. [Training & Evaluation](#-training--evaluation)
7. [Running the Interactive Web App](#-running-the-interactive-web-app)
8. [Command-Line Inference](#-command-line-inference)
9. [Performance Metrics](#-performance-metrics)
10. [Project Structure](#-project-structure)

---

## 📋 Overview

In educational environments, analyzing student classroom behaviors is crucial for understanding engagement and teaching effectiveness. **G-Hash** translates student crops from classroom cameras into **64-bit compact binary hash codes** via deep hashing, enabling:
- **Fast Similarity Search**: Retrieve near-duplicate behaviors or similar student activities from large database frames within milliseconds using Hamming distance.
- **Relational Context**: Leverages **GAT** to model how behaviors frequently co-occur (e.g., *listening* often co-occurs with *sitting*, while *writing* rarely co-occurs with *dozing off*).
- **Lightweight Storage**: Compresses high-dimensional features into lightweight 64-bit keys.

---

## 🎯 Classroom Behavior Categories

The model is trained on the **ET-EDU dataset** featuring **14 classroom behavior/action classes**:

| behavior ID | Behavior Name | Emoji | behavior ID | Behavior Name | Emoji |
|---|---|---|---|---|---|
| **1** | `using phone` | 📱 | **8** | `writing` | ✍️ |
| **2** | `dozing off` | 😴 | **9** | `listening` | 👂 |
| **3** | `turning sideways` | ↪️ | **10** | `head down` | 🙇 |
| **4** | `turning back` | ↩️ | **11** | `sitting` | 🪑 |
| **5** | `raising hand` | 🙋‍♂️ | **12** | `standing` | 🧍 |
| **6** | `opening book` | 📖 | **13** | `walking` | 🚶 |
| **7** | `reading` | 📚 | **14** | `interacting` | 🗣️ |

---

## 🏗️ Architecture

The G-Hash network processes an image branch and a label relationship graph in parallel to align visual features and behavioral semantics in a unified hashing space:

```
                  ┌───────────────────────────────┐
                  │       G-Hash Architecture     │
                  └───────────────┬───────────────┘
                                  │
      [Image Path]                                [Label Graph]
      ┌──────────┐                               ┌─────────────┐
      │  Image   │                               │    Label    │
      │ (Student)│                               │ Embeddings  │
      └────┬─────┘                               └──────┬──────┘
           │                                            │
           ▼ (timm ViT)                                 ▼ (Text Embedding)
      ┌──────────┐                               ┌─────────────┐
      │ ViT-Base │                               │  Graph Node │
      │ Encoder  │                               │ Features    │
      └────┬─────┘                               └──────┬──────┘
           │ (768-dim)                                  │
           ▼                                            ▼
      ┌──────────┐ (Image features mapping)      ┌─────────────┐
      │   MLP    ├──────────────────────────────►│    GAT      │ (Models co-occurrence
      │  Layer   │◄──────────────────────────────┤  Network    │  via adjacency matrix)
      └────┬─────┘                               └──────┬──────┘
           │                                            │
           ▼                                            ▼
      ┌──────────┐                               ┌─────────────┐
      │ Continuous│                              │ Label Hash  │
      │  Hash    │                              │   Vectors   │
      └────┬─────┘                               └─────────────┘
           │
           ▼ (sign function for retrieval)
      ┌─────────────────────────┐
      │ 64-bit Binary Hash Code │
      └─────────────────────────┘
```

### Deep Multi-Task Loss Formulation
The model is optimized end-to-end using a joint loss function:
$$\mathcal{L}_{\text{total}} = \gamma \mathcal{L}_{\text{cls}} + \alpha \mathcal{L}_{\text{sim}} + \eta \mathcal{L}_{\text{retrieval}} + \beta \mathcal{L}_{\text{quant}} + \mathcal{L}_{\text{ortho}} + \delta \mathcal{L}_{\text{bit\_balance}}$$

- **$\mathcal{L}_{\text{cls}}$**: Binary Cross Entropy Loss for multi-label behavior classification.
- **$\mathcal{L}_{\text{sim}}$**: Aligns continuous image hash with the corresponding text label embedding hash.
- **$\mathcal{L}_{\text{retrieval}}$**: Forces images sharing at least one behavior to have closer Hamming distances.
- **$\mathcal{L}_{\text{quant}}$**: Quantization penalty forcing continuous features closer to $\pm1$ bounds.
- **$\mathcal{L}_{\text{ortho}}$**: Orthogonality penalty ensuring behavior label hashes are distinct.
- **$\mathcal{L}_{\text{bit\_balance}}$**: Prevents hash code collapse by encouraging uniform bit distribution.

---

## 🔄 Data Curation & Processing Pipeline

The project features a dedicated curation script to build high-quality, non-duplicate crops of student frames from classroom video files.

```
 Video Files (.mp4) ──► YOLO Detection ──► IoU Tracking ──► Blur/Quality Filter ──► Hamming Deduplication ──► Split & Annotate
```

### Step 1: Extraction & Deduplication
Run `extract_edu_cbir_v2.py` to extract high-quality, keyframe-based student crops from classroom camera recordings while discarding duplicate frames and blurry outputs:
```bash
python extract_edu_cbir_v2.py --video-dir data/ET-EDU --output-root data/ET-EDU-CBIR-V2
```

### Step 2: Split and Label Alignment
Once behaviors have been annotated in the generated template CSV, convert it into standard label index files read by the dataset loader:
```bash
python build_edu_txt_from_labeled_csv.py \
  --labeled-csv data/ET-EDU-CBIR-V2/labels_template.csv \
  --output-root data
```
This produces the database index files:
- `data/train_img.txt` & `data/train_label.txt`
- `data/test_img.txt` & `data/test_label.txt`
- `data/edu_labels.txt` (the list of 14 classes)

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (highly recommended for training; CPU/MPS supported for inference)

### Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/ngocquang2509-pro/g-hash.git
cd g-hash

# Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 🚀 Training & Evaluation

### Training the G-Hash Model
To train the G-Hash model on the ET-EDU dataset using GAT and ViT architectures, run:
```bash
python train.py --config configs/et_edu_config.yaml
```

### Customizing Hyperparameters
You can adjust epochs, batch-size, and other options on the fly:
```bash
python train.py --config configs/et_edu_config.yaml --epochs 100 --batch-size 32 --learning-rate 0.00003
```

### Train Baseline Model (Without GAT)
To train a model without the Graph Attention Network (using only visual features) for ablation comparison:
```bash
python train.py --config configs/et_edu_config.yaml --baseline
```

---

## 🌐 Running the Interactive Web App

G-Hash includes an elegant, real-time web application to let teachers and researchers upload classroom crops, classify student behaviors, and search for similar activities across the entire historical database in real-time.

```bash
# Start the Flask web application
python app.py
```
Open your browser and navigate to `http://localhost:5000` to interact with the visual dashboard!

---

## 🔍 Command-Line Inference

### 1. Retrieve similar behaviors for a single image:
Run a retrieval query for a specific query image to find the top-K matching behaviors in the database:
```bash
python inference.py \
  --checkpoint experiments/runs/20260420-115433/best_model.pth \
  --config configs/et_edu_config.yaml \
  --mode retrieve \
  --image data/cropped_students_quality_10k/mid_D01_20240223064932_f007068_p00_00000132.jpg \
  --database data/train_img.txt \
  --top-k 8 \
  --save-viz
```
The results and comparison plots will be saved as query reports.

### 2. Batch Test Query Images:
Place your query images inside the `images/` directory and execute:
```bash
python batch_test.py
```
This retrieves matching educational behaviors and places visualized outputs in `test_results/`.

---

## 📊 Performance Metrics

The G-Hash model achieves excellent retrieval accuracy and robustness on the ET-EDU classroom behavior dataset (64-bit hash codes):

| Retrieval Metric | Score (%) | Explanation |
|---|---|---|
| **mAP@10** | **90.60%** | Mean Average Precision for top 10 retrieved behaviors |
| **mAP@50** | **85.51%** | Mean Average Precision for top 50 retrieved behaviors |
| **mAP@100** | **83.92%** | Mean Average Precision for top 100 retrieved behaviors |
| **Precision@10** | **77.24%** | Accuracy of the top 10 retrieved student behaviors |
| **Overall mAP** | **64.47%** | Strict global Mean Average Precision across all test queries |

*Evaluation metrics demonstrate that modeling label relationships using GAT alongside deep visual hashing significantly boosts retrieval accuracy by up to **+15% mAP** compared to vision-only baselines.*

---

## 📁 Project Structure

```
g-hash/
├── src/
│   ├── models/            # ViT-Base encoders, Text embedding, GAT & G-Hash Model
│   ├── data/              # Dataset loader, dataloaders & label-graph builders
│   ├── training/          # Training Trainer class and multi-task loss (GHashLoss)
│   ├── evaluation/        # Metrics computing (mAP, Hamming, Precision@K, Recall@K)
│   └── utils/             # Config loaders and visualization modules
├── configs/               # YAML configurations for ET-EDU & ablations
│   ├── et_edu_config.yaml         # Optimized 14-class G-Hash configuration
│   └── et_edu_config_no_gat.yaml  # Vision-only baseline configuration
├── data/                  # Workspace datasets, image splits & class list
│   └── edu_labels.txt     # List of 14 valid behavior classes
├── experiments/           # Logged runs, weights checkpoints & PR curves
├── images/                # Input folder for batch testing query images
├── test_results/          # Visual output files for batch query runs
├── uploads/               # Temporary uploaded query images for Flask Web UI
├── app.py                 # Flask web dashboard application
├── train.py               # Main training script
├── inference.py           # Core CLI inference & search engine
├── batch_test.py          # Script for batch-evaluating test images
└── requirements.txt       # Python dependencies
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
