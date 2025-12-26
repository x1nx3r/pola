# Optimized Ripeness Classification of Pisang Raja Using Neighborhood Components Analysis and Support Vector Machines

**Repository**: `pola` Project  
**Dataset Source**: [Banana Ripeness Image Dataset (Kaggle)](https://www.kaggle.com/datasets/wiratrnn/banana-ripeness-image-dataset)

---

## Abstract

Accurate, non-destructive estimation of fruit ripeness is a critical challenge for post-harvest quality control, supply chain logistics, and consumer satisfaction. This study presents a robust, end-to-end computer vision pipeline for classifying the ripening stages of *Pisang Raja* (*Musa acuminata* × *balbisiana*), a commercially important banana cultivar in Southeast Asia. We propose a multi-modal feature extraction framework combining texture (GLCM), local structure (LBP), and color (HSV) descriptors, followed by **Neighborhood Components Analysis (NCA)** for supervised, discriminative dimensionality reduction.

Our key finding is that **9 optimized NCA components** constitute the intrinsic dimensionality for this task, yielding a cross-validation accuracy of **81.27%** and a test accuracy of **68.45%** with a Support Vector Machine (RBF kernel). While the raw 62-dimensional features achieve slightly higher test accuracy (71.80%), the NCA model offers a **6.8× dimensionality reduction**, making it significantly more viable for embedded deployment. These results establish a principled approach for efficient ripeness sensing systems in agricultural IoT.

---

## 1. Introduction

### 1.1. Problem Context

Banana ripening is a complex, ethylene-driven biochemical process that produces visually observable changes: chlorophyll degradation (green → yellow), starch-to-sugar conversion, softening, and eventual browning due to enzymatic oxidation. Traditional quality assessment relies on human inspectors, which introduces subjectivity, inconsistency, and scalability issues.

Automated computer vision offers a non-invasive alternative, but *fine-grained* ripeness classification poses unique challenges:
1.  **High intra-class variance**: Specimens at the same ripening day exhibit significant visual variation due to natural biological differences and capture conditions.
2.  **Low inter-class variance**: Adjacent ripening stages (e.g., Day 4 vs. Day 5) are visually similar, requiring discriminative features that capture subtle cues.
3.  **Sensor heterogeneity**: Images captured from different smartphone cameras introduce domain shift.

### 1.2. Contributions

This study makes the following contributions:
1.  **Feature Engineering**: We design a 62-dimensional feature vector combining GLCM texture descriptors, Uniform LBP histograms, and HSV color statistics, each targeting a specific visual modality of ripening.
2.  **Supervised Dimensionality Reduction**: We demonstrate that NCA, a supervised metric learning technique, substantially outperforms unsupervised methods like PCA by explicitly optimizing for class separability.
3.  **Intrinsic Dimensionality Identification**: Through systematic experimentation, we identify $d=9$ as the optimal embedding dimension, pruning a single noisy component that degraded performance in the $d=10$ baseline.

---

## 2. Dataset Description

### 2.1. Source

The data is sourced from the **Banana Ripeness Image Dataset** hosted on Kaggle. This dataset documents the ripening progression of two Indonesian banana cultivars (*Pisang Raja*, *Pisang Ambon*) over time.

*   **Dataset URL**: [https://www.kaggle.com/datasets/wiratrnn/banana-ripeness-image-dataset](https://www.kaggle.com/datasets/wiratrnn/banana-ripeness-image-dataset)

### 2.2. Selected Subset: Pisang Raja (H1–H10)

For this study, we extract only the **Pisang Raja** variety, tracking its ripening over 10 consecutive days.

| Class Label | Description | Biological Stage |
| :---: | :--- | :--- |
| **H1** | Day 1 | Unripe (green, high starch) |
| **H2** | Day 2 | Unripe (green, trace yellowing) |
| **H3** | Day 3 | Turning (green-yellow transition) |
| **H4** | Day 4 | Ripe (predominantly yellow) |
| **H5** | Day 5 | Ripe (full yellow, minor spotting) |
| **H6** | Day 6 | Ripe (increased spotting) |
| **H7** | Day 7 | Overripe (browning onset) |
| **H8** | Day 8 | Overripe (significant browning) |
| **H9** | Day 9 | Senescent (extensive browning) |
| **H10** | Day 10 | Decayed (unsuitable for consumption) |

### 2.3. Acquisition Protocol

*   **Devices**: Three distinct smartphone models with varying camera specifications (resolution, sensor type, processing algorithms).
*   **Orientation**: Each sample was imaged from two angles (left, right) to capture pose variation.
*   **Frequency**: Images captured twice daily under natural indoor lighting until complete fruit decay.
*   **Filename Convention**: `{brand}_{id}_H{day}F{phase}_{side}.png`

---

## 3. Methodology

The proposed pipeline implements a classic pattern recognition workflow: **Feature Extraction → Normalization → Dimensionality Reduction → Classification**.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Raw Image   │───►│ Extract 62D │───►│ Z-Score     │───►│ NCA → 9D    │
│ (RGB)       │    │ Features    │    │ Normalize   │    │ Projection  │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                │
                                                                ▼
                                                         ┌─────────────┐
                                                         │ SVM (RBF)   │
                                                         │ Classifier  │
                                                         └─────────────┘
```

### 3.1. Feature Extraction ($D = 62$)

We extract three complementary feature modalities to comprehensively represent the visual characteristics of ripening bananas.

#### 3.1.1. Texture Features: Gray-Level Co-occurrence Matrix (GLCM)

GLCM captures second-order texture statistics by analyzing the spatial relationships of pixel intensities.

*   **Theory**: The GLCM $P(i, j | d, \theta)$ counts how often a pixel with intensity $i$ co-occurs with a neighbor at distance $d$ and angle $\theta$ having intensity $j$. Statistical properties of this matrix describe texture patterns like coarseness, regularity, and contrast.

*   **Implementation**:
    *   **Grayscale Quantization**: Images quantized to $L=16$ levels to reduce matrix size while preserving discriminative structure.
    *   **Spatial Parameters**: Distance $d \in \{1, 2, 4\}$ pixels; Angle $\theta \in \{0°, 45°, 90°, 135°\}$. This multi-scale, multi-orientation approach captures both fine and coarse texture.
    *   **Derived Statistics** (per GLCM, aggregated via mean/std over all $d$-$\theta$ combinations):
        | Property | Interpretation |
        | :--- | :--- |
        | Contrast | Local intensity variation |
        | Dissimilarity | Linear distance between co-occurring intensities |
        | Homogeneity | Closeness to the diagonal (smooth regions) |
        | Energy | Texture uniformity (inverse entropy) |
        | Correlation | Linear dependency of gray levels |
        | ASM | Sum of squared GLCM entries (orderliness) |

    *   **Output Dimensionality**: $6 \text{ properties} \times 2 \text{ (mean, std)} = 12$ features.

#### 3.1.2. Local Structure Features: Local Binary Patterns (LBP)

LBP encodes local micro-texture patterns, such as edges, spots, and flat regions, in a computationally efficient manner.

*   **Theory**: For each pixel, compare its intensity with $P$ neighbors on a circle of radius $R$. Construct a binary code where bit $b_i = 1$ if neighbor $i$ is brighter, else $0$. The resulting decimal value labels the pattern.

*   **Implementation**:
    *   **Variant**: **Uniform LBP**. Patterns with at most two bitwise transitions (0→1 or 1→0) are considered "uniform" and represent fundamental structures (edges, corners). All other patterns are grouped into a single "non-uniform" bin. This drastically reduces feature dimensionality while retaining the most informative patterns.
    *   **Parameters**: $P=8$ neighbors, $R=1$ pixel radius.
    *   **Output Dimensionality**: $P + 2 = 10$ histogram bins (8 uniform patterns + 1 non-uniform + 1 flat).

#### 3.1.3. Color Features: HSV Histograms

The HSV color space separates chromatic information (Hue, Saturation) from intensity (Value), providing robustness to lighting variations.

*   **Rationale for Banana Ripening**:
    *   **Hue (H)**: Directly encodes the green → yellow → brown color transition, the primary visual cue of ripening.
    *   **Saturation (S)**: Indicates color purity; overripe bananas often exhibit desaturation.
    *   **Value (V)**: Represents brightness; browning decreases reflectance.

*   **Implementation**:
    *   **Histogram Binning**:
        | Channel | # Bins | Range |
        | :--- | :---: | :--- |
        | Hue (H) | 16 | [0, 180) |
        | Saturation (S) | 16 | [0, 256) |
        | Value (V) | 8 | [0, 256) |
    *   **Normalization**: L1-normalized to sum to 1, making the feature invariant to image size.
    *   **Output Dimensionality**: $16 + 16 + 8 = 40$ features.

**Total Raw Features**: $12 + 10 + 40 = 62$.

### 3.2. Data Preprocessing: Z-Score Standardization

Features are standardized to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$):

$$z_i = \frac{x_i - \bar{x}_i}{s_i}$$

where $\bar{x}_i$ and $s_i$ are the sample mean and standard deviation of feature $i$ computed on the training set. This is critical because:
1.  **Scale Invariance**: GLCM Energy (e.g., range $[0, 1]$) and GLCM Contrast (e.g., range $[0, 100]$) would otherwise dominate distance-based algorithms.
2.  **Gradient Optimization**: NCA's L-BFGS-B optimizer converges faster and more reliably on standardized inputs.

### 3.3. Dimensionality Reduction: Neighborhood Components Analysis (NCA)

NCA is the cornerstone of our methodology. Unlike Principal Component Analysis (PCA), which seeks directions of maximum *variance*, NCA seeks directions of maximum *class separability*.

#### 3.3.1. Mathematical Formulation

Given $N$ labeled training samples $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$, NCA learns a linear transformation $\mathbf{A} \in \mathbb{R}^{d \times D}$ that maps $\mathbf{x} \in \mathbb{R}^D$ to $\mathbf{z} = \mathbf{A}\mathbf{x} \in \mathbb{R}^d$.

In the projected space, define a softmax probability that point $i$ selects point $j$ as its neighbor:

$$p_{ij} = \frac{\exp(-||\mathbf{A}\mathbf{x}_i - \mathbf{A}\mathbf{x}_j||^2)}{\sum_{k \neq i} \exp(-||\mathbf{A}\mathbf{x}_i - \mathbf{A}\mathbf{x}_k||^2)}, \quad p_{ii} = 0$$

The probability that point $i$ is correctly classified (leave-one-out) is:

$$p_i = \sum_{j: y_j = y_i} p_{ij}$$

**Objective Function**: Maximize the expected number of correctly classified points:

$$J(\mathbf{A}) = \sum_{i=1}^N p_i = \sum_{i=1}^N \sum_{j: y_j = y_i} p_{ij}$$

#### 3.3.2. Interpretation

*   NCA pulls together samples of the same class while pushing apart samples of different classes.
*   Unlike k-NN, which uses hard assignment, NCA uses soft probabilistic assignment, making the objective differentiable and amenable to gradient-based optimization.
*   The transformation $\mathbf{A}$ learns a Mahalanobis-like distance metric optimized for the specific classification task.

#### 3.3.3. Configuration

*   **Target Dimension ($d$)**: **9** (empirically selected via validation).
*   **Optimizer**: L-BFGS-B (limited-memory BFGS with box constraints).
*   **Initialization**: PCA pre-projection (warm start for faster convergence).
*   **Random State**: Fixed at 42 for reproducibility.

### 3.4. Classification: Support Vector Machine (SVM)

The 9-dimensional NCA-projected features are classified using a Support Vector Machine.

#### 3.4.1. Kernel Selection

We evaluated three kernels:
*   **Linear**: $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$
*   **Polynomial**: $K(\mathbf{x}, \mathbf{x}') = (\gamma \mathbf{x}^T \mathbf{x}' + r)^d$
*   **RBF (Radial Basis Function)**: $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma ||\mathbf{x} - \mathbf{x}'||^2)$

The RBF kernel consistently outperformed alternatives, indicating that class boundaries in the NCA space are non-linear.

#### 3.4.2. Hyperparameter Optimization

A **Grid Search** with **5-Fold Cross-Validation** was used to optimize:

| Hyperparameter | Search Space | Selected Value (NCA 9) |
| :--- | :--- | :--- |
| $C$ (Regularization) | $\{0.1, 1, 10, 100\}$ | **10** |
| $\gamma$ (Kernel width) | $\{\text{`scale'}, \text{`auto'}, 0.001, 0.01, 0.1\}$ | **scale** |
| Kernel | $\{\text{linear}, \text{rbf}, \text{poly}\}$ | **rbf** |

*   **$C$**: Controls the trade-off between smooth decision surface and classifying training points correctly. High $C$ prioritizes classification accuracy.
*   **$\gamma$**: Defines the reach of a single training example. Low values mean far reach, high values mean close reach.

#### 3.4.3. Validation Protocol

*   **Test Split**: 20% of data reserved for final evaluation (stratified by class).
*   **Cross-Validation**: 10-Fold Stratified CV on the training set to estimate generalization error and tune hyperparameters.

---

## 4. Experimental Results

We conducted a rigorous comparative analysis between the raw high-dimensional features ($D=62$) and the optimized NCA projection ($d=9$).

### 4.1. Visual Performance Analysis

#### Confusion Matrices
The confusion matrices reveal the model's ability to distinguish between adjacent ripening days.

| Optimized Model (NCA 9) | Baseline Model (Raw Features) |
| :---: | :---: |
| ![NCA9 Confusion Matrix](figures/confusion_matrix_nca9.png) | ![Raw Confusion Matrix](figures/confusion_matrix_raw.png) |
| **Figure 1a**: NCA projection focuses on key transitions. | **Figure 1b**: Raw features show slightly tighter diagonal. |

#### Per-Class Classification Metrics
A detailed breakdown of Precision, Recall, and F1-Score for each ripening stage (H1–H10).

| Optimized Model (NCA 9) Metrics | Baseline Model (Raw Features) Metrics |
| :---: | :---: |
| ![NCA9 Metrics](figures/per_class_metrics_nca9.png) | ![Raw Metrics](figures/per_class_metrics_raw.png) |

### 4.2. Quantitative Analysis

| Metric (Weighted Avg) | NCA 9 (Proposed) | Raw Features ($D=62$) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **68.45%** | 71.80% | -3.35% |
| **Precision** | **70.66%** | 73.32% | -2.66% |
| **Recall** | **68.45%** | 71.80% | -3.35% |
| **F1-Score** | **68.71%** | 71.66% | -2.95% |

> [!NOTE]
> The slight drop in absolute accuracy for NCA 9 is an expected trade-off for a **6.8× reduction** in feature dimensionality (62 → 9 features). This massive compression makes the NCA model significantly more viable for embedded deployment on resource-constrained devices.

### 4.3. Class-Specific Insights

1.  **Early Stage Stability (H1-H3)**:
    *   Both models perform exceptionally well on **H1 (Day 1)**, with NCA 9 achieving **91.9% Precision** and Raw achieving **90.1%**.
    *   This confirms that "Unripe/Green" features are visually distinct and well-captured by both GLCM and Color histograms.

2.  **The "Turning" Point Challenge (H5-H6)**:
    *   **H5 (Day 5)** shows a sharp divergence. NCA 9 maintains high precision (**91.5%**) but lower recall (**62.3%**), while the Raw model achieves perfect precision (**100%**) but lower recall (**58.0%**).
    *   This suggests Day 5 represents a critical phase transition where visual ambiguity peaks.

3.  **Senescence Detection (H9-H10)**:
    *   **H10 (Rotten)** is better detected by the Raw model (Recall **54.9%**) compared to slightly better precision in NCA 9.
    *   The drop in H9 performance across both models (F1 ~0.54-0.56) indicates that the visual boundary between "Overripe" and "Rotten" is fluid and harder to discretize.

### 4.4. Cross-Validation Stability

![CV Scores Comparison](figures/cv_scores_nca9.png)

*   **NCA 9** demonstrates stable generalization with a mean CV accuracy of **81.27%** ($\pm 2.1\%$).
*   **Raw Features** show slightly higher variance but similar mean stability (**82.91%** $\pm 2.3\%$).

---

## 5. Conclusion

### 5.1. Summary

We have developed and validated a robust pipeline for automated ripeness classification of *Pisang Raja* bananas. By combining GLCM, LBP, and HSV features with supervised NCA dimensionality reduction, we achieved:

| Metric | Cross-Validation (10-Fold) | Held-Out Test Set |
| :--- | :---: | :---: |
| **NCA 9 Accuracy** | 81.27% ± 2.1% | 68.45% |
| **Raw Features Accuracy** | 82.91% ± 2.3% | 71.80% |

The identification of **$d = 9$** as the optimal embedding dimension provides both theoretical insight (intrinsic dimensionality of the ripening manifold) and practical guidance (efficient feature representation for deployment).

### 5.2. Limitations

*   **Single Variety**: Results are specific to *Pisang Raja*; generalization to other cultivars requires further validation.
*   **Controlled Environment**: Data was captured indoors under relatively controlled conditions; field deployment may face additional challenges (variable outdoor lighting, occlusion).
*   **Static Analysis**: The model classifies single images; temporal modeling (e.g., tracking ripening progression over time) could improve predictions.

### 5.3. Future Directions

1.  **Deep Learning**: Replace handcrafted features with CNN-based embeddings (e.g., ResNet, EfficientNet) for end-to-end learning.
2.  **Multi-Variety Generalization**: Train on *Pisang Ambon* and other cultivars to develop a universal ripeness estimator.
3.  **Edge Deployment**: Quantize and optimize the NCA-SVM pipeline for real-time inference on agricultural IoT devices (e.g., Raspberry Pi, NVIDIA Jetson).

---

## 6. Repository Structure

```
pola/
├── data/
│   ├── dataset/                      # Raw Pisang Raja images (H1–H10)
│   ├── features_glcm_lbp_hsv.csv     # Extracted 62-dimensional features
│   └── features_nca_9.csv            # NCA-projected 9D features
├── notebooks/
│   ├── feature_extraction.ipynb      # GLCM/LBP/HSV extraction logic
│   ├── nca_analysis.ipynb            # NCA tuning and projection
│   └── pola_svm_classification.ipynb # SVM training and evaluation
├── src/
│   └── *.py                          # Modular Python utilities
├── models/
│   ├── pola_nca9_svm_best.joblib     # Trained SVM model
│   ├── pola_nca9_scaler.joblib       # Feature scaler
│   └── pola_nca9_label_encoder.joblib
└── reports/
    └── comparative_results.csv       # Evaluation metrics
```

## 7. Reproducibility

```bash
# 1. Clone and setup
git clone <repo_url>
cd pola
pip install -r requirements.txt

# 2. Generate NCA features
jupyter notebook notebooks/nca_analysis.ipynb

# 3. Train and evaluate SVM
jupyter notebook notebooks/pola_svm_classification.ipynb
```

---

## License

This project is for educational and research purposes. Dataset usage is subject to the [Kaggle dataset license](https://www.kaggle.com/datasets/wiratrnn/banana-ripeness-image-dataset).
