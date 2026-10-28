# Academic Project Report Outline

## Title: An Intelligent Strawberry Freshness and Quality Assessment Framework Using Vision Transformer (ViT), Graph Neural Network (GNN/GAT), and Explainable AI (XAI)

---

### Abstract
Post-harvest losses in commercial strawberry production frequently exceed 25–40% due to rapid spore germination and bacterial foliar blights. Conventional deep learning approaches rely on standard convolutional backbones that process images as unstructured pixel grids, lacking the ability to model spatial relationships between isolated lesions and offering minimal interpretability. This research develops a hybrid Vision Transformer (ViT) and Graph Attention Network (GAT) framework for automated strawberry pathology assessment and commercial quality grading. Images (224×224 RGB) are tokenized into 196 spatial patch embeddings using a pretrained `vit_tiny_patch16_224` backbone. A dynamic patch graph is constructed combining 8-neighborhood spatial adjacency with semantic cosine feature similarity. A 2-layer Graph Attention Network with 4 attention heads performs message passing across lesion patches. The graph representation is fused with the global ViT CLS token to classify 7 strawberry disease classes and assess severity. Evaluated on 743 held-out test images from a verified dataset of 3,243 images, the model achieved an overall accuracy of **89.64%** and a macro-averaged F1-score of **84.30%**. Predictions are interpreted using gradient-based patch saliency overlays and GAT edge attention maps, translating pathology findings into commercial freshness grades (Grade A, B, and C).

---

### Chapter 1: Introduction
1. **Background & Motivation**
   - Economic significance of strawberry (*Fragaria × ananassa*) crops.
   - High perishability, rapid senescence, and susceptibility to fungal/bacterial decay.
2. **Problem Statement**
   - Limitations of standard CNNs in capturing non-local lesion dependencies.
   - The "black box" dilemma in precision agriculture and trust deficits among growers.
3. **Research Objectives**
   - Implement patch-level ViT representation extraction.
   - Construct attributed spatial-semantic patch graphs.
   - Design a Multi-Head GAT for relational lesion learning.
   - Develop multi-scale visual and graph-level XAI explanations.
   - Deploy an end-to-end interactive decision-support web dashboard.
4. **Scope & Delimitations**
   - Focus on 7 verified pathology classes and 2-tier severity levels.
   - Distinguishing pathological degradation from physiological shelf-life metrics.

---

### Chapter 2: Literature Review
1. **Traditional Machine Learning & Color Handcrafted Features in Horticulture**
   - Color histograms, GLCM texture analysis, and SVM classifiers.
2. **Convolutional Neural Networks in Plant Pathology**
   - ResNet, MobileNet, and EfficientNet architectures.
   - Inductive biases and receptive field limitations.
3. **Vision Transformers in Agriculture**
   - Self-attention mechanisms: $Q, K, V$ formulations.
   - Patch tokenization and global receptive fields.
4. **Graph Neural Networks (GNN) and Graph Attention Networks (GAT)**
   - Representing images as graphs of superpixels or patches.
   - Attention coefficients: $\alpha_{ij} = \text{Softmax}(\text{LeakyReLU}(\mathbf{a}^T [W h_i \,\|\, W h_j]))$.
5. **Explainable AI in Agricultural Vision**
   - Grad-CAM, Attention Rollout, GNNExplainer, and clinical trust.

---

### Chapter 3: Methodology and Architecture
1. **Dataset Acquisition & Preprocessing**
   - Dataset verification, CRC integrity testing, and Labelme JSON parsing.
   - Deterministic 3-way split: 1,450 train / 307 val / 743 test images.
   - Normalization matching ImageNet distribution and data augmentations.
2. **Vision Transformer Feature Extractor**
   - Backbone selection (`vit_tiny_patch16_224`, 192 embedding dimensions).
   - Patch decomposition: 16×16 pixels $\to$ 14×14 grid ($N = 196$).
   - Disentanglement of the CLS token $\mathbf{z}_{\text{cls}}$ from regional patch tokens $\mathbf{z}_{\text{patches}}$.
3. **Patch Graph Construction**
   - Node feature matrix: $\mathbf{X} \in \mathbb{R}^{196 \times 192}$.
   - Spatial 8-connectivity grid adjacency matrix: $\mathbf{A}_{\text{spatial}}$.
   - Dynamic semantic similarity edges: Cosine similarity thresholding ($\tau = 0.5$).
   - Batched graph indexing for variable mini-batches.
4. **Graph Attention Network (GAT)**
   - Multi-head attention aggregation ($K = 4$ heads).
   - Layer normalization and non-linear ELU activation.
   - Readout pooling: $\mathbf{z}_{\text{graph}} = [\text{MeanPool}(\{h_i\}) \,\|\, \text{MaxPool}(\{h_i\})] \in \mathbb{R}^{128}$.
5. **Multimodal Fusion & Classification Head**
   - Representation fusion: $\mathbf{z}_{\text{fused}} = [\mathbf{z}_{\text{graph}} \,\|\, \mathbf{z}_{\text{cls}}] \in \mathbb{R}^{320}$.
   - Multi-layer perceptron with dropout ($p = 0.25$) and GELU activation.
   - Dual-head design: 7-class disease logits and 2-class severity rating.
6. **Explainability Engine (XAI)**
   - Image-level patch gradient saliency: $S_i = \lVert \nabla_{\mathbf{x}_i} \hat{y}_c \rVert_2$.
   - Upsampling via bicubic interpolation and jet colormap blending.
   - Graph edge extraction: Edge attention weight ranking and bounding box mapping.
7. **Commercial Freshness & Quality Mapping**
   - Agronomic grading rules (Grade A Prime, Grade B Commercial, Grade C Substandard).

---

### Chapter 4: Experimental Setup & Results
1. **Experimental Environment & Implementation Details**
   - Hardware: Intel Core i7-1255U (CPU execution mode).
   - Software: Python 3.13, PyTorch 2.14, timm, torch-geometric, scikit-learn.
   - Hyperparameters: Batch size = 32, AdamW, $\text{lr} = 10^{-3}$, CosineAnnealingLR.
   - Class-weighted CrossEntropyLoss to address class imbalance.
2. **Training & Convergence Analysis**
   - Loss reduction from 0.9659 (Epoch 1) to 0.2384 (Epoch 5).
   - Validation macro-F1 progression: 0.8063 $\to$ 0.8962.
3. **Held-Out Test Set Performance (743 Images)**
   - Overall Accuracy: **89.64%**
   - Macro-Averaged Precision: **85.62%**
   - Macro-Averaged Recall: **83.63%**
   - Macro-Averaged F1-Score: **84.30%**
4. **Per-Class Pathology Evaluation**
   - Blossom Blight: 98.41% F1 (100% Recall).
   - Angular Leafspot: 94.00% F1.
   - Powdery Mildew Leaf: 93.71% F1.
   - Leaf Spot: 90.97% F1.
   - Gray Mold: 89.17% F1.
   - Fruit-specific infections (Anthracnose: 62.50% F1, Powdery Mildew Fruit: 61.33% F1) reflecting class imbalance and subtle early sporulation cues.
5. **Confusion Matrix Analysis**
   - Inter-class confusions between foliar leaf spot variations.

---

### Chapter 5: Explainability Case Studies
1. **Case Study 1: Angular Leafspot**
   - Saliency focuses sharply on water-soaked polygonal lesions bounded by veinlets.
2. **Case Study 2: Gray Mold (*Botrytis cinerea*)**
   - Attended graph edges connect fruit calyx margins to soft decaying berry tissue.
3. **Case Study 3: Powdery Mildew Leaf**
   - Graph attention clusters along upwardly curled foliar margins.

---

### Chapter 6: System Implementation & Web Interface
1. **Software Architecture**
   - Modular design pattern (`data_loader`, `vit_encoder`, `graph_builder`, `gat_model`, `classifier`, `explainability`).
2. **Streamlit Web Application**
   - 7 core modules: Home, Dataset Explorer, Model Training, Strawberry Assessment, XAI, Evaluation, About.
   - Real-time image upload and instantaneous visual feedback.

---

### Chapter 7: Discussion, Limitations & Future Work
1. **Technical Findings & Discussion**
   - Superiority of patch graph attention over flat pooling.
   - Significance of preserving spatial topology in biological lesion tracking.
2. **Limitations**
   - Laboratory dataset vs. complex field backgrounds with occlusions and varying sunlight.
   - Distinction between external pathological decay and internal biochemical Brix/firmness metrics.
3. **Future Work**
   - Integration of multi-spectral or hyperspectral imaging.
   - Edge deployment on Raspberry Pi / NVIDIA Jetson devices for field harvesters.
   - Temporal graph learning for tracking disease progression over storage time.

---

### References
- Dosovitskiy et al. (2020), *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR.
- Veličković et al. (2018), *Graph Attention Networks*, ICLR.
- Howard et al. (2019), *Searching for MobileNetV3*, ICCV.
- Plant Pathology references cited in `Dataset.txt` (Università di Bologna, Bugwood, USDA ARS).
