# Pola Dataset Analysis: Advanced Classification Pipeline

## 1. Project Overview

This project implements a state-of-the-art computer vision pipeline for classifying images of **Pisang Ambon** from the "Pola" dataset. The primary objective is to accurately categorize images into their respective ripening stages by leveraging high-dimensional feature extraction, supervised dimensionality reduction, and robust classification algorithms.

The core innovation of this pipeline is the transition from a standard 10-component analysis to an optimized **9-component Neighborhood Components Analysis (NCA)**, which has been empirically proven to maximize classification accuracy (81.86%).

## 2. Dataset Description

The analysis focuses on a specific high-quality subset of a larger banana ripeness dataset.

### Source Data
*   **Varieties**: Pisang Ambon.
*   **Devices**: Three distinct smartphone cameras with varying specifications.
*   **Methodology**: Images captured twice daily from two different angles (left/right) until complete decay.
*   **Filename Format**: `{phone_brand}_{image_initial}_H{day}F{phase}{side}`.

### Selected Subset: "Pola" (Pisang Ambon)
For this project, we utilize **Pisang Ambon** samples exclusively, tracking their ripening process over a 10-day period.

*   **Classes**: 10 distinct classes representing daily ripeness stages.
    *   **H1**: Day 1 (Unripe/Green)
    *   ...
    *   **H10**: Day 10 (Overripe/Rotten)
*   **Input**: RGB Images captured under natural conditions.
*   **Applications**: Ripeness estimation, shelf-life prediction, and automated quality control.

---

## 3. Methodology: From 0 to 100

The analysis pipeline follows a strict four-stage process: **Extraction $\rightarrow$ Standardization $\rightarrow$ Reduction $\rightarrow$ Classification**.

### Stage 1: Raw Feature Extraction (The "0 to 60")

We transform raw image pixels into a meaningful tabular dataset by extracting distinct visual descriptors. The raw feature vector has a dimensionality of **$D=62$**.

#### A. Texture Features (GLCM)
We utilize the **Gray-Level Co-occurrence Matrix (GLCM)** to capture second-order statistical texture features.
*   **Preprocessing**: Images converted to grayscale and quantized to **16 gray levels** to reduce sparsity.
*   **Parameters**:
    *   **Distances**: `[1, 2, 4]` pixels (capturing fine to coarse texture).
    *   **Angles**: `[0, 45, 90, 135]` degrees (rotation invariance).
*   **Descriptors**: For each GLCM, we compute the *Mean* and *Standard Deviation* of:
    1.  Contrast
    2.  Dissimilarity
    3.  Homogeneity
    4.  Energy
    5.  Correlation
    6.  ASM (Angular Second Moment)

#### B. Local Structure (LBP)
We employ **Local Binary Patterns (LBP)** to detect local micro-structures (edges, corners, flat spots).
*   **Configuration**: Uniform LBP with variables $P=8$ (neighbors) and $R=1$ (radius).
*   **Output**: A normalized histogram of invariant patterns.

#### C. Color Statistics (HSV)
We analyze color distribution in the **HSV (Hue, Saturation, Value)** color space, which is more robust to lighting variations than RGB.
*   **Quantization**:
    *   **Hue**: 16 bins
    *   **Saturation**: 16 bins
    *   **Value**: 8 bins
*   **Output**: Concatenated, normalized histograms.

---

### Stage 2: Standardization

Before dimensionality reduction, all 62 features are standardized using **Z-score normalization**:
$$z = \frac{x - \mu}{\sigma}$$
This ensures that features with larger scales (e.g., Energy) do not dominate the objective function of the gradient-based NCA algorithm.

---

### Stage 3: Dimensionality Reduction (NCA)

This is the critical optimization step. We apply **Neighborhood Components Analysis (NCA)** to project the 62-dimensional data into a lower-dimensional subspace ($d < D$).

*   **Why Not PCA?**: Principal Component Analysis (PCA) maximizes *variance*, which is unsupervised and ignores class labels.
*   **Why NCA?**: NCA is *supervised*. It learns a linear transformation matrix $A$ such that the expected leave-one-out classification accuracy of a stochastic Nearest Neighbor (k-NN) rule is maximized in the transformed space.

**Configuration**:
*   **Input Dimension**: 62
*   **Target Dimension ($d$)**: **9** (Empirically determined optimal)
*   **Solver**: `l-bfgs-b` (Limited-memory BFGS) optimizer.
*   **Initialization**: Auto (PCA-based initialization).

The resulting 9 features (`nca0` ... `nca8`) represent the most discriminative axes of the data.

---

### Stage 4: Classification (SVM)

We classify the 9-dimensional vectors using a **Support Vector Machine (SVM)**.

*   **Model Selection**: We perform an exhaustive **Grid Search** with 5-fold Cross-Validation to optimize:
    *   **C (Regularization)**: `[0.1, 1, 10, 100]`
    *   **Gamma (Kernel Coefficient)**: `['scale', 'auto', 0.001 ... 0.1]`
    *   **Kernel**:
        *   *Linear*: For linearly separable data.
        *   *RBF (Radial Basis Function)*: For non-linear boundaries (Selected Best).
        *   *Poly*: Polynomial mapping.

*   **Validation**:
    *   **Test Split**: 20% of data held out for final testing (`random_state=42`).
    *   **Cross-Validation**: 10-Fold Stratified CV on the training set to ensure robustness.

---

## 4. Performance Analysis

We systematically evaluated feature sets with dimensions $d=2$, $d=9$, and $d=10$.

### Comparative Results

| Metric | NCA 2 | NCA 9 (Optimal) | NCA 10 |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 45.58% | **81.86%** | 80.18% |
| **Precision** | 52.83% | **82.03%** | 80.24% |
| **Recall** | 45.58% | **81.86%** | 80.18% |
| **F1-Score** | 45.16% | **81.71%** | 80.04% |

### Key Insights

1.  **The "Curse of Dimensionality" Reversed**: NCA 9 outperforms NCA 10. This indicates that the 10th component in the legacy model was likely capturing noise or variance irrelevant to class separation. Pruning it to 9 dimensions denoised the signal.
2.  **Failure of Visualization Dimensions**: The sharp drop in accuracy at $d=2$ (45%) confirms that the "Pola" dataset forms a complex manifold that cannot be flattened onto a 2D plane without significant loss of topological information.
3.  **Kernel Non-Linearity**: The RBF kernel consistently beat the Linear kernel (~50% accuracy), proving that the decision boundaries between Pola classes are curved and complex, even in the optimized NCA space.

---

## 5. Project Structure

The repository is organized for reproducibility:

```
pola/
├── data/
│   ├── dataset/             # Source images
│   └── features_nca_9.csv   # The optimized feature set
├── src/
│   ├── generate_nca9.py     # Script to run NCA transformation
│   └── evaluate_all_models.py # Comparative evaluation logic
├── notebooks/
│   ├── feature_extraction.ipynb      # Raw feature generation (GLCM/LBP/HSV)
│   ├── nca_analysis.ipynb            # Dimensionality reduction tuning
│   └── pola_svm_classification.ipynb # Final SVM pipeline
└── models/                  # Persisted .joblib model artifacts
```

## 6. Usage

To replicate the results:

1.  **Environment Setup**: Install dependencies via `pip install -r requirements.txt`.
2.  **Run Pipeline**:
    *   Generate features: Run `notebooks/nca_analysis.ipynb`.
    *   Train & Evaluate: Run `notebooks/pola_svm_classification.ipynb`.
3.  **Inference**: Load `models/pola_nca9_svm_best.joblib` and predict on new 62-dimensional feature vectors (normalized via scaler).
