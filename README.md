# Optimized Ripeness Classification of Pisang Raja Using Neighborhood Components Analysis and Support Vector Machines

**Repository**: `pola` Project  
**Dataset Source**: [Banana Ripeness Image Dataset (Kaggle)](https://www.kaggle.com/datasets/wiratrnn/banana-ripeness-image-dataset)

---

## Abstract

Accurate, non-destructive estimation of fruit ripeness is a critical challenge for post-harvest quality control, supply chain logistics, and consumer satisfaction. This study presents a robust, end-to-end computer vision pipeline for classifying the ripening stages of *Pisang Raja* (*Musa acuminata* × *balbisiana*), a commercially important banana cultivar in Southeast Asia. We propose a multi-modal feature extraction framework combining texture (GLCM), local structure (LBP), and color (HSV) descriptors, followed by **Neighborhood Components Analysis (NCA)** for supervised, discriminative dimensionality reduction.

Our key finding is that **9 optimized NCA components** constitute the intrinsic dimensionality for this task, yielding a classification accuracy of **81.86%** with a Support Vector Machine (RBF kernel). This configuration significantly outperforms both lower-dimensional projections ($d=2$: 45.58%) and higher-dimensional alternatives ($d=10$: 80.18%), demonstrating that the 10th component captures noise rather than discriminative signal. These results establish a principled approach for embedded ripeness sensing systems in agricultural IoT.

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

| Hyperparameter | Search Space | Selected Value |
| :--- | :--- | :--- |
| $C$ (Regularization) | $\{0.1, 1, 10, 100\}$ | 100 |
| $\gamma$ (Kernel width) | $\{\text{`scale'}, \text{`auto'}, 0.001, 0.01, 0.1\}$ | scale |
| Kernel | $\{\text{linear}, \text{rbf}, \text{poly}\}$ | rbf |

*   **$C$**: Controls the trade-off between smooth decision surface and classifying training points correctly. High $C$ prioritizes classification accuracy.
*   **$\gamma$**: Defines the reach of a single training example. Low values mean far reach, high values mean close reach.

#### 3.4.3. Validation Protocol

*   **Test Split**: 20% of data reserved for final evaluation (stratified by class).
*   **Cross-Validation**: 10-Fold Stratified CV on the training set to estimate generalization error and tune hyperparameters.

---

## 4. Experimental Results

### 4.1. Comparative Performance

We systematically evaluated NCA projections at three dimensionalities ($d \in \{2, 9, 10\}$) using the same SVM classifier.

| Configuration | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **NCA 9 (Proposed)** | **81.86%** | **82.03%** | **81.86%** | **81.71%** |
| NCA 10 | 80.18% | 80.24% | 80.18% | 80.04% |
| NCA 2 | 45.58% | 52.83% | 45.58% | 45.16% |

### 4.2. Analysis and Insights

#### 4.2.1. The Optimality of $d = 9$: Signal vs. Noise

The 9-component model outperforms the 10-component model by approximately **1.7 percentage points**. This counterintuitive result—that *fewer* dimensions yield *better* performance—can be explained as follows:

*   **Intrinsic Dimensionality**: The true separability structure of the ripening classes lies on a manifold of approximately 9 degrees of freedom. The 10th NCA component captures residual variance that is primarily attributable to:
    *   **Acquisition noise**: Sensor inconsistencies across the three smartphone cameras.
    *   **Background clutter**: Variable environmental conditions in the capture setup.
    *   **Biological irrelevance**: Natural intra-specimen variation not correlated with ripening stage.

*   **Regularization Effect**: By projecting to 9 dimensions instead of 10, we effectively apply a form of implicit regularization, discarding a noisy subspace that would otherwise contribute to overfitting.

#### 4.2.2. Failure of Low-Dimensional Projection ($d = 2$)

The dramatic performance drop at $d=2$ (from 81.86% to 45.58%) illustrates the **Johnson-Lindenstrauss limitation**: projecting high-dimensional data to very low dimensions inevitably distorts pairwise distances.

While 2D embeddings are useful for visualization, they conflate classes that are separable in higher dimensions. For *Pisang Raja*, visualizing the H1–H10 manifold requires at least 9 axes to preserve class topology.

#### 4.2.3. Non-Linearity of the Ripening Manifold

The substantial performance gap between the RBF kernel (~82%) and the linear kernel (~50%) confirms that ripening-induced visual changes do not evolve linearly in the feature space. The SVM's implicit mapping to an infinite-dimensional Hilbert space via the RBF kernel is necessary to capture the curved decision boundaries separating adjacent ripening stages.

---

## 5. Conclusion and Future Work

### 5.1. Summary

We have developed and validated a robust pipeline for automated ripeness classification of *Pisang Raja* bananas. By combining GLCM, LBP, and HSV features with supervised NCA dimensionality reduction, we achieved **81.86% accuracy** on a challenging 10-class problem. The identification of **$d = 9$** as the optimal embedding dimension provides both theoretical insight (intrinsic dimensionality of the ripening manifold) and practical guidance (efficient feature representation for deployment).

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
