# Intelligent Strawberry Freshness and Quality Assessment Using ViT, GNN/GAT, and Explainable AI

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org/)
[![Vision Transformer](https://img.shields.io/badge/ViT-timm-green.svg)](https://github.com/huggingface/pytorch-image-models)
[![Graph Attention Network](https://img.shields.io/badge/GNN-PyG%20GAT-purple.svg)](https://pytorch-geometric.readthedocs.io/)
[![Streamlit App](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io/)
[![Test Suite](https://img.shields.io/badge/Tests-7%20Passed-brightgreen.svg)](#testing--verification)

> **Academic Project Title:**  
> *An Intelligent Strawberry Freshness/Quality Assessment Framework Using Vision Transformer (ViT), Graph Neural Network (GNN/GAT), and Explainable AI (XAI)*

---

## 📌 Executive Summary

Strawberries (*Fragaria × ananassa*) are delicate, high-value horticultural commodities vulnerable to rapid post-harvest degradation caused by foliar bacteria and necrotrophic fungal rots (*Botrytis cinerea*, *Colletotrichum acutatum*, *Podosphaera aphanis*, *Xanthomonas fragariae*).

Standard Convolutional Neural Networks (CNNs) treat strawberry images as flat grids of independent pixels and suffer from fixed local receptive fields. In contrast, this project introduces a **hybrid neuro-symbolic framework**:
1. **Vision Transformer (ViT):** Tokenizes images into spatial patches ($14 \times 14 = 196$ patches) and extracts 192-dimensional contextual feature representations.
2. **Patch Graph Modeling:** Represents 16x16 image patches as graph nodes connected through spatial 8-neighborhood topology and semantic feature cosine similarity.
3. **Multi-Head Graph Attention Network (GAT):** Learns inter-lesion propagation dynamics and regional dependencies via attention coefficients $\alpha_{ij}$.
4. **Multimodal Fusion & Classification:** Fuses graph-level pooled representations with global ViT CLS context to classify strawberry pathologies and predict severity.
5. **Explainable AI (XAI):** Generates transparent image-level gradient saliency heatmaps and maps attended GAT edges directly back to physical strawberry tissues.
6. **Commercial Quality Grading:** Maps pathological diagnoses into agricultural quality tiers (Grade A: Prime Fresh, Grade B: Early Infection, Grade C: Severe Decay).

---

## 🏗️ Architecture Pipeline

```
Strawberry Input Image (224x224 RGB)
              │
              ▼
Vision Transformer Backbone (vit_tiny_patch16_224)
              │
      ┌───────┴──────────────────────┐
      ▼                              ▼
  CLS Token (1x192)         196 Patch Tokens (196x192)
                                     │
                                     ▼
                      Attributed Patch Graph Construction
                      - 196 Nodes (16x16 image regions)
                      - 8-Connected Spatial Edges + Top-K Similarity
                                     │
                                     ▼
                      Multi-Head Graph Attention Network (GAT)
                      - Layer 1: 4 Heads x 128 Channels + LayerNorm + ELU
                      - Layer 2: 4 Heads x 64 Channels + Dropout
                                     │
                                     ▼
                      Graph Readout Pooling (MeanPool || MaxPool = 128-dim)
                                     │
      ┌──────────────────────────────┘
      ▼
Multimodal Feature Fusion [Graph (128) || CLS (192)] = 320-dim
      │
      ├──────────────────────────────┬──────────────────────────────┐
      ▼                              ▼                              ▼
Disease Classification Head   Severity Assessment Head     Explainability Engine (XAI)
(7 Pathology Classes)         (Level 1 vs Level 2)         - ViT Gradient Saliency Overlay
                                                          - Top Attended GAT Edges & Nodes
```

---

## 📊 Dataset Structure & Verified Splits

The system was evaluated on a verified strawberry pathology dataset comprising **3,243 total images** with corresponding Labelme JSON polygon annotations:

- **Train Set:** 1,450 images
- **Validation Set:** 307 images
- **Held-Out Test Set:** 743 images
- **Severity-Rated Test Set:** 743 images (Level 1: 206 images, Level 2: 537 images)

### Ground Truth Pathology Classes:
1. `angular_leafspot` (*Xanthomonas fragariae*) - 582 images
2. `anthracnose_fruit_rot` (*Colletotrichum acutatum*) - 130 images
3. `blossom_blight` (*Botrytis / Colletotrichum*) - 270 images
4. `gray_mold` (*Botrytis cinerea*) - 622 images
5. `leaf_spot` (*Mycosphaerella fragariae*) - 777 images
6. `powdery_mildew_fruit` (*Podosphaera aphanis*) - 178 images
7. `powdery_mildew_leaf` (*Podosphaera aphanis*) - 684 images

---

## 📈 Real Experimental Results (Held-Out Test Set)

> Evaluated on the held-out test split of **743 images** without data leakage:

| Metric | Score |
|---|---|
| **Overall Test Accuracy** | **89.64%** |
| **Macro-Averaged Precision** | **85.62%** |
| **Macro-Averaged Recall** | **83.63%** |
| **Macro-Averaged F1-Score** | **84.30%** |
| **Weighted F1-Score** | **89.31%** |

### Per-Class Test Breakdown:
| Pathology Class | Precision | Recall | F1-Score | Test Support |
|---|---|---|---|---|
| **Blossom Blight** | 96.88% | **100.00%** | **98.41%** | 62 |
| **Angular Leafspot** | 92.16% | 95.92% | **94.00%** | 147 |
| **Powdery Mildew Leaf** | **99.26%** | 88.74% | **93.71%** | 151 |
| **Leaf Spot** | 91.82% | 90.12% | **90.97%** | 162 |
| **Gray Mold** | 82.84% | 96.55% | **89.17%** | 145 |
| **Anthracnose Fruit Rot** | 64.52% | 60.61% | 62.50% | 33 |
| **Powdery Mildew Fruit** | 71.88% | 53.49% | 61.33% | 43 |

---

## 📁 Repository Structure

```
strawberry-ai/
├── app.py                      # Interactive Streamlit Web Application (7 pages)
├── config.yaml                 # Central configuration for dataset, model, and training
├── requirements.txt            # Pinned Python package dependencies
├── README.md                   # Complete documentation and user guide
├── REPORT_OUTLINE.md           # College project academic report blueprint
├── .gitignore                  # Source control hygiene exclusions
├── data/
│   ├── README.md               # Dataset details and pathology descriptions
│   ├── raw/                    # Extracted raw JPG images and JSON annotations
│   └── processed/
│       └── dataset_manifest.json # Verified dataset index and split manifest
├── src/
│   ├── __init__.py             # Package initializer
│   ├── utils.py                # Device detection, seed setting, YAML loader
│   ├── dataset_inspector.py    # CRC verification, image checking, manifest builder
│   ├── preprocessing.py        # ImageNet normalization and augmentation pipeline
│   ├── data_loader.py          # PyTorch Dataset, DataLoader, class-weight balancing
│   ├── vit_encoder.py          # ViT-Tiny patch token extractor & coordinate mapper
│   ├── graph_builder.py        # Spatial 8-connectivity and similarity graph builder
│   ├── gat_model.py            # Multi-head GAT layers (PyG + pure-PyTorch fallback)
│   ├── classifier.py           # Unified ViT-GAT end-to-end classifier
│   ├── train.py                # Training loop, AdamW, CosineAnnealing, checkpointing
│   ├── evaluate.py             # Test set evaluation, metrics, and confusion matrix
│   ├── explainability.py       # ViT saliency rollout and GAT edge attention XAI
│   └── inference.py            # Single-image inference and quality grading
├── scripts/
│   ├── verify_dataset.py       # Check ZIP archive CRC and display dataset summary
│   ├── prepare_dataset.py      # Extract raw archive and build manifest
│   ├── run_smoke_test.py       # Mini end-to-end pipeline smoke test
│   └── test_inference.py       # Verify prediction and XAI on test images
├── tests/
│   ├── test_dataset.py         # Transforms and PyTorch Dataset unit tests
│   ├── test_graph.py           # Adjacency matrix and batch graph tests
│   └── test_model.py           # GAT, ViT-GAT forward pass and backprop tests
├── checkpoints/
│   └── best_strawberry_gat.pth # Trained model weights and metadata
└── outputs/
    ├── figures/
    │   ├── training_history.png  # Loss and accuracy curves
    │   └── confusion_matrix.png  # 7x7 confusion matrix heatmap
    └── metrics/
        ├── train_history.json    # Epoch-by-epoch loss and F1 logs
        └── test_metrics.json     # Test accuracy, per-class metrics, confusion matrix
```

---

## 🚀 Step-by-Step Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Dataset Archive
Checks the CRC checksum of the ZIP file and parses `Dataset.txt`:
```bash
python scripts/verify_dataset.py
```

### 3. Prepare and Extract Dataset
Extracts images into `data/raw/` and generates `data/processed/dataset_manifest.json`:
```bash
python scripts/prepare_dataset.py
```

### 4. Run Unit Test Suite
Executes the automated unit tests covering transforms, graphs, and gradients:
```bash
python -m pytest tests/
```

### 5. Run Fast Smoke Test
Verifies mini-batch training, validation, checkpoint saving/reloading, and single-image inference:
```bash
python scripts/run_smoke_test.py
```

### 6. Train the Model
Trains the ViT-GAT model with class-weighted cross-entropy and cosine annealing:
```bash
python src/train.py
```
*Outputs are saved to `checkpoints/best_strawberry_gat.pth` and `outputs/figures/training_history.png`.*

### 7. Evaluate on Held-Out Test Set
Computes accuracy, per-class metrics, and confusion matrix on the 743 held-out test images:
```bash
python src/evaluate.py
```
*Outputs are saved to `outputs/metrics/test_metrics.json` and `outputs/figures/confusion_matrix.png`.*

### 8. Run Single-Image Inference CLI
Tests prediction and explanation generation on individual images:
```bash
python scripts/test_inference.py
```

### 9. Launch the Streamlit Web Application
Starts the interactive browser interface:
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 🔍 Explainability & Quality Grading Details

- **ViT Gradient Saliency:** Gradient norms $\lVert \nabla_{\mathbf{x}_i} \hat{y}_c \rVert_2$ are computed across patch tokens, normalized, and upsampled via bicubic interpolation onto the strawberry image.
- **GAT Edge Attention:** Attention coefficients $\alpha_{ij}$ from the multi-head GAT attention layer highlight which interacting plant regions drove the diagnosis.
- **Commercial Quality Grading:**
  - **Grade A (Prime Fresh / Healthy):** No significant fungal lesions; suitable for immediate premium retail.
  - **Grade B (Commercial / Early Infection):** Foliar infection present (e.g., Angular Leafspot, Leaf Spot); fruit salvageable with protective fungicidal management.
  - **Grade C (Substandard / Severe Spoilage):** Active necrotrophic rot (Gray Mold, Anthracnose Fruit Rot); unmarketable and subject to immediate culling.

---

## 🛠️ Troubleshooting & Common Errors

1. **`CUDA Out of Memory`:**
   - The framework automatically selects CPU or CUDA based on availability. Set `device: "cpu"` in `config.yaml` for memory-safe execution on laptops.
2. **`UnicodeEncodeError` in Windows Command Prompt:**
   - Handled automatically via `sys.stdout.reconfigure(encoding="utf-8")` and clean ASCII console markers.
3. **`Checkpoint Not Found`:**
   - Run `python src/train.py` before launching inference or Streamlit.
