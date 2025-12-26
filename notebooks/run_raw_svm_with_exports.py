# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import json
import warnings
import os
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("Libraries imported successfully")
print("Ready to build SVM classification pipeline for RAW features")

print("="*70)
print("STEP 1: DATA PREPARATION")
print("="*70)

# Load the features dataset
features_file = '../data/features_glcm_lbp_hsv.csv'
if not os.path.exists(features_file):
    raise FileNotFoundError(f"{features_file} not found locally.")

df = pd.read_csv(features_file)

print(f"\nLoaded dataset: {features_file}")
print(f"Dataset shape: {df.shape}")

# Extract Features
# We exclude metadata columns to get the raw feature matrix
exclude_cols = {'filename', 'label', 'label_code', 'Unnamed: 0'}
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols].values
feature_names = feature_cols

# Extract Targets (robust label extraction)
if 'label' in df.columns:
    y_raw = df['label'].values
    print("Using 'label' column for targets")
elif 'label_code' in df.columns:
    y_raw = df['label_code'].values
    print("Using 'label_code' column for targets")
else:
    # Fallback to filename extraction
    print("Extracting labels from filenames...")
    y_raw = df['filename'].astype(str).str.extract(r'(?i)(h\d+)')[0].str.upper().fillna('UNKNOWN').values

print(f"Number of species: {len(np.unique(y_raw))}")
print(f"\nSpecies distribution:")
print(pd.Series(y_raw).value_counts())

# Handle any NaN values
if np.isnan(X).any():
    print(f"\nWarning: Found {np.isnan(X).sum()} NaN values. Filling with column means.")
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])

print(f"\nFeature matrix shape: {X.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Number of samples: {len(y_raw)}")

# Encode species labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
species_names = le.classes_

print(f"\nEncoded labels: {dict(zip(species_names, range(len(species_names))))}")

# Split data into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"Train set distribution:")
train_dist = pd.Series(y_train).value_counts().sort_index()
for idx, count in train_dist.items():
    print(f"  {species_names[idx]}: {count}")

# Standardize features (fit on train, transform both)
scaler_svm = StandardScaler()
X_train_scaled = scaler_svm.fit_transform(X_train)
X_test_scaled = scaler_svm.transform(X_test)

print("\n✓ Features standardized (zero mean, unit variance)")
print("✓ Data preparation complete")

print("\n" + "="*70)
print("STEP 2: BASELINE SVM MODELS")
print("="*70)

# Test different SVM kernels
kernels = ['linear', 'rbf', 'poly']
baseline_results = {}

for kernel in kernels:
    print(f"\nTraining SVM with {kernel.upper()} kernel...")
    start_time = time.time()
    
    # Create and train SVM
    svm = SVC(kernel=kernel, random_state=42, gamma='scale')
    svm.fit(X_train_scaled, y_train)
    
    # Predict on test set
    y_pred = svm.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    training_time = time.time() - start_time
    
    # Store results
    baseline_results[kernel] = {
        'model': svm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'predictions': y_pred
    }
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Time:      {training_time:.2f}s")

# Compare baseline models
print("\n" + "-"*70)
print("BASELINE MODEL COMPARISON")
print("-"*70)

comparison_df = pd.DataFrame({
    'Kernel': list(baseline_results.keys()),
    'Accuracy': [baseline_results[k]['accuracy'] for k in baseline_results.keys()],
    'Precision': [baseline_results[k]['precision'] for k in baseline_results.keys()],
    'Recall': [baseline_results[k]['recall'] for k in baseline_results.keys()],
    'F1-Score': [baseline_results[k]['f1'] for k in baseline_results.keys()],
    'Time (s)': [baseline_results[k]['training_time'] for k in baseline_results.keys()]
})

print(comparison_df.round(4))

# Visualize baseline comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(kernels))
width = 0.2

for i, metric in enumerate(metrics):
    axes[0].bar(x + i*width, comparison_df[metric], width, label=metric)

axes[0].set_xlabel('Kernel')
axes[0].set_ylabel('Score')
axes[0].set_title('Baseline Model Performance Comparison')
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels([k.upper() for k in kernels])
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([0, 1.1])

# Plot training time
axes[1].bar(kernels, comparison_df['Time (s)'])
axes[1].set_xlabel('Kernel')
axes[1].set_ylabel('Training Time (seconds)')
axes[1].set_title('Training Time Comparison')
axes[1].set_xticklabels([k.upper() for k in kernels])
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Find best baseline model
best_kernel = max(baseline_results.keys(), key=lambda k: baseline_results[k]['accuracy'])
print(f"\n✓ Best baseline kernel: {best_kernel.upper()} (Accuracy: {baseline_results[best_kernel]['accuracy']:.4f})")

print("\n" + "="*70)
print("STEP 3: HYPERPARAMETER TUNING")
print("="*70)

# Define parameter grid for RBF and Linear kernels
param_grid = [
    {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    },
    {
        'kernel': ['linear'],
        'C': [0.1, 1, 10, 100]
    },
    {
        'kernel': ['poly'],
        'C': [0.1, 1, 10],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto']
    }
]

print("\nParameter grid defined:")
print(f"  RBF: C={[0.1, 1, 10, 100]}, gamma={['scale', 'auto', 0.001, 0.01, 0.1]}")
print(f"  Linear: C={[0.1, 1, 10, 100]}")
print(f"  Poly: C={[0.1, 1, 10]}, degree={[2, 3, 4]}, gamma={['scale', 'auto']}")

# Perform grid search with cross-validation
print("\nPerforming GridSearchCV (5-fold cross-validation)...")
print("This may take a few minutes...")

start_time = time.time()

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

grid_time = time.time() - start_time

print(f"\n✓ Grid search completed in {grid_time:.2f}s")
print(f"\nBest parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")

# Get the best model
best_svm = grid_search.best_estimator_

# Test on held-out test set
y_pred_best = best_svm.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred_best)

print(f"Test set accuracy: {test_accuracy:.4f}")

# Show top 10 parameter combinations
print("\nTop 10 parameter combinations:")
results_df = pd.DataFrame(grid_search.cv_results_)
top_results = results_df.nsmallest(10, 'rank_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
]
print(top_results)

print("\n" + "="*70)
print("STEP 4: CROSS-VALIDATION ANALYSIS")
print("="*70)

# Perform stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\nPerforming 10-fold stratified cross-validation...")
cv_scores = cross_val_score(best_svm, X_train_scaled, y_train, cv=skf, scoring='accuracy', n_jobs=-1)

print(f"\nCross-validation scores (10 folds):")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i:2d}: {score:.4f}")

print(f"\nCross-validation summary:")
print(f"  Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"  Median accuracy: {np.median(cv_scores):.4f}")
print(f"  Min accuracy: {cv_scores.min():.4f}")
print(f"  Max accuracy: {cv_scores.max():.4f}")

# Visualize cross-validation scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), cv_scores, 'o-', linewidth=2, markersize=8)
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
plt.fill_between(range(1, 11), 
                 cv_scores.mean() - cv_scores.std(), 
                 cv_scores.mean() + cv_scores.std(), 
                 alpha=0.2, color='r', label=f'±1 Std Dev')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('10-Fold Cross-Validation Scores')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([max(0, cv_scores.min() - 0.05), min(1, cv_scores.max() + 0.05)])
plt.tight_layout()
plt.show()

print("\n✓ Cross-validation analysis complete")

print("\n" + "="*70)
print("STEP 5: PERFORMANCE EVALUATION")
print("="*70)

# Get predictions on test set
y_pred_final = best_svm.predict(X_test_scaled)

# Calculate comprehensive metrics
accuracy = accuracy_score(y_test, y_pred_final)
precision = precision_score(y_test, y_pred_final, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_final, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_final, average='weighted', zero_division=0)

print("\nOverall Performance Metrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# Detailed classification report
print("\n" + "-"*70)
print("CLASSIFICATION REPORT (Per-Class Metrics)")
print("-"*70)
print(classification_report(y_test, y_pred_final, target_names=species_names, digits=4, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_final)

print("\n" + "-"*70)
print("CONFUSION MATRIX")
print("-"*70)

# Visualize confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Count matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=species_names, yticklabels=species_names,
            cbar_kws={'label': 'Count'}, ax=axes[0])
axes[0].set_xlabel('Predicted Species')
axes[0].set_ylabel('True Species')
axes[0].set_title('Confusion Matrix (Counts)')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

# Normalized confusion matrix (percentages)
# Add small epsilon to avoid division by zero if a class has no samples in test set
cm_sum = cm.sum(axis=1)[:, np.newaxis]
cm_normalized = cm.astype('float') / (cm_sum + 1e-9)

sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=species_names, yticklabels=species_names,
            cbar_kws={'label': 'Percentage'}, ax=axes[1])
axes[1].set_xlabel('Predicted Species')
axes[1].set_ylabel('True Species')
axes[1].set_title('Normalized Confusion Matrix (Percentages)')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()

# Per-class accuracy
print("\nPer-Class Accuracy:")
for i, species in enumerate(species_names):
    total = cm[i, :].sum()
    class_accuracy = cm[i, i] / total if total > 0 else 0
    print(f"  {species}: {class_accuracy:.4f} ({cm[i, i]}/{total})")

print("\n✓ Performance evaluation complete")

print("\n" + "="*70)
print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
print("="*70)

if best_svm.kernel == 'linear':
    print("\nAnalyzing feature importance for linear SVM...")
    
    # Get feature weights from linear SVM
    # For multi-class, we have one weight vector per class
    coef = best_svm.coef_
    
    print(f"Coefficient matrix shape: {coef.shape}")
    print(f"({coef.shape[0]} classes × {coef.shape[1]} features)")
    
    # Calculate mean absolute coefficient across all classes
    mean_abs_coef = np.abs(coef).mean(axis=0)
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_coef
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20))
    
    # Visualize top 20 features
    plt.figure(figsize=(12, 8))
    top_20 = feature_importance.head(20)
    plt.barh(range(len(top_20)), top_20['importance'])
    plt.yticks(range(len(top_20)), top_20['feature'])
    plt.xlabel('Mean Absolute Coefficient')
    plt.title('Top 20 Most Important Features (Linear SVM)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    # Analyze per-class feature importance
    print("\n" + "-"*70)
    print("PER-CLASS FEATURE IMPORTANCE")
    print("-"*70)
    
    n_classes = len(species_names)
    n_cols = 4
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.ravel() if n_classes > 1 else [axes]
    
    for i, species in enumerate(species_names):
        if i < len(axes) and i < len(coef):
            class_coef = np.abs(coef[i])
            # Show top 5 features
            top_k = min(10, len(feature_names))
            top_features_idx = np.argsort(class_coef)[-top_k:][::-1]
            
            axes[i].barh(range(top_k), class_coef[top_features_idx])
            axes[i].set_yticks(range(top_k))
            axes[i].set_yticklabels([feature_names[j] for j in top_features_idx], fontsize=8)
            axes[i].set_xlabel('Absolute Coefficient')
            axes[i].set_title(f'{species}')
            axes[i].invert_yaxis()
    
    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
else:
    print(f"\nFeature importance analysis not available for {best_svm.kernel} kernel")
    print("Use linear kernel to see feature weights")
    print("\nNote: For non-linear kernels, feature importance is implicit in the support vectors")
    print(f"Number of support vectors: {len(best_svm.support_)}")
    print(f"Support vector ratio: {len(best_svm.support_) / len(X_train_scaled):.2%}")

print("\n✓ Feature importance analysis complete")

print("\n" + "="*70)
print("STEP 7: ERROR ANALYSIS")
print("="*70)

# Find misclassified samples
misclassified_mask = y_test != y_pred_final
misclassified_indices = np.where(misclassified_mask)[0]
n_misclassified = len(misclassified_indices)

print(f"\nMisclassified samples: {n_misclassified}/{len(y_test)} ({n_misclassified/len(y_test)*100:.2f}%)")

if n_misclassified > 0:
    # Analyze misclassification patterns
    print("\n" + "-"*70)
    print("MISCLASSIFICATION PATTERNS")
    print("-"*70)
    
    misclass_df = pd.DataFrame({
        'True': [species_names[y_test[i]] for i in misclassified_indices],
        'Predicted': [species_names[y_pred_final[i]] for i in misclassified_indices]
    })
    
    # Count confusion pairs
    confusion_pairs = misclass_df.groupby(['True', 'Predicted']).size().sort_values(ascending=False)
    
    print("\nMost common misclassifications:")
    for (true_class, pred_class), count in confusion_pairs.head(10).items():
        print(f"  {true_class} → {pred_class}: {count} times")
    
    # Visualize confusion pairs
    if len(confusion_pairs) > 0:
        plt.figure(figsize=(12, 6))
        top_confusions = confusion_pairs.head(15)
        labels = [f"{true_cls} → {pred_cls}" for (true_cls, pred_cls) in top_confusions.index]
        plt.barh(range(len(top_confusions)), top_confusions.values)
        plt.yticks(range(len(top_confusions)), labels)
        plt.xlabel('Number of Misclassifications')
        plt.title('Most Common Misclassification Patterns')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    # Decision confidence for misclassified samples
    if hasattr(best_svm, 'decision_function'):
        print("\n" + "-"*70)
        print("DECISION CONFIDENCE ANALYSIS")
        print("-"*70)
        
        # Get decision function values
        decision_values = best_svm.decision_function(X_test_scaled)
        
        # For correctly classified samples
        correct_mask = ~misclassified_mask
        if decision_values.ndim > 1:
            correct_confidence = np.max(decision_values[correct_mask], axis=1)
            misclass_confidence = np.max(decision_values[misclassified_mask], axis=1)
        else:
            correct_confidence = np.abs(decision_values[correct_mask])
            misclass_confidence = np.abs(decision_values[misclassified_mask])
        
        print(f"\nDecision confidence (mean absolute value):")
        print(f"  Correct classifications: {np.mean(correct_confidence):.4f} ± {np.std(correct_confidence):.4f}")
        print(f"  Misclassifications: {np.mean(misclass_confidence):.4f} ± {np.std(misclass_confidence):.4f}")
        
        # Visualize confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(correct_confidence, bins=30, alpha=0.6, label='Correct', color='green')
        plt.hist(misclass_confidence, bins=30, alpha=0.6, label='Misclassified', color='red')
        plt.xlabel('Decision Function Confidence')
        plt.ylabel('Count')
        plt.title('Decision Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
else:
    print("\n✓ Perfect classification! No errors to analyze.")

print("\n✓ Error analysis complete")

print("\n" + "="*70)
print("STEP 8: SAVE MODEL AND GENERATE REPORT")
print("="*70)

model_filename = '../models/pola_raw_svm_best.joblib'
scaler_filename = '../models/pola_raw_scaler.joblib'
label_encoder_filename = '../models/pola_raw_label_encoder.joblib'
summary_filename = '../reports/pola_raw_svm_summary.json'

joblib.dump(best_svm, model_filename)
joblib.dump(scaler_svm, scaler_filename)
joblib.dump(le, label_encoder_filename)

print(f"\nSaved model artifacts:")
print(f"  Model: {model_filename}")
print(f"  Scaler: {scaler_filename}")
print(f"  Label encoder: {label_encoder_filename}")

# Generate comprehensive summary report
summary_report = {
    'Model Type': 'Support Vector Machine (SVM) - RAW Features',
    'Best Kernel': best_svm.kernel,
    'Best Parameters': best_svm.get_params(),
    'Training Samples': len(X_train),
    'Test Samples': len(X_test),
    'Number of Features': X.shape[1],
    'Number of Classes': len(species_names),
    'Class Names': list(species_names),
    'Test Accuracy': f"{accuracy:.4f}",
    'Test Precision': f"{precision:.4f}",
    'Test Recall': f"{recall:.4f}",
    'Test F1-Score': f"{f1:.4f}",
    'CV Mean Accuracy': f"{cv_scores.mean():.4f}",
    'CV Std Accuracy': f"{cv_scores.std():.4f}",
    'Number of Support Vectors': len(best_svm.support_),
    'Support Vector Ratio': f"{len(best_svm.support_) / len(X_train_scaled):.2%}",
}

# Save summary to JSON
with open(summary_filename, 'w') as f:
    json.dump(summary_report, f, indent=2, default=str)

print(f"\nSaved model summary: {summary_filename}")

# Print final summary
print("\n" + "="*70)
print("SVM CLASSIFICATION PIPELINE - FINAL SUMMARY")
print("="*70)

print(f"\n{'Model Configuration:':<30}")
print(f"  {'Kernel:':<28} {best_svm.kernel}")
if best_svm.kernel in ['rbf', 'poly']:
    print(f"  {'C parameter:':<28} {best_svm.C}")
    print(f"  {'Gamma:':<28} {best_svm.gamma}")
if best_svm.kernel == 'poly':
    print(f"  {'Degree:':<28} {best_svm.degree}")
elif best_svm.kernel == 'linear':
    print(f"  {'C parameter:':<28} {best_svm.C}")

print(f"\n{'Dataset Information:':<30}")
print(f"  {'Total samples:':<28} {len(X)}")
print(f"  {'Training samples:':<28} {len(X_train)}")
print(f"  {'Test samples:':<28} {len(X_test)}")
print(f"  {'Number of features:':<28} {X.shape[1]}")
print(f"  {'Number of species:':<28} {len(species_names)}")

print(f"\n{'Performance Metrics:':<30}")
print(f"  {'Test Accuracy:':<28} {accuracy:.4f}")
print(f"  {'Test Precision:':<28} {precision:.4f}")
print(f"  {'Test Recall:':<28} {recall:.4f}")
print(f"  {'Test F1-Score:':<28} {f1:.4f}")
print(f"  {'CV Accuracy (10-fold):':<28} {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print(f"\n{'Model Characteristics:':<30}")
print(f"  {'Support vectors:':<28} {len(best_svm.support_)}")
print(f"  {'Support vector ratio:':<28} {len(best_svm.support_) / len(X_train_scaled):.2%}")
print(f"  {'Misclassified (test):':<28} {n_misclassified}/{len(y_test)} ({n_misclassified/len(y_test)*100:.2f}%)")

print(f"\n{'Output Files:':<30}")
print(f"  • {model_filename}")
print(f"  • {scaler_filename}")
print(f"  • {label_encoder_filename}")
print(f"  • {summary_filename}")

print("\n" + "="*70)
print("✓ RAW FEATURES SVM CLASSIFICATION PIPELINE COMPLETE")
print("="*70)

# =======================================================================
# SAVE FIGURES AND METADATA (RAW FEATURES)
# =======================================================================
import os
import json
from datetime import datetime

# Create directories if they don't exist
figures_dir = '../figures'
reports_dir = '../reports'
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# --- Save Confusion Matrix ---
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=species_names, yticklabels=species_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - Raw Features (62D) + SVM')
plt.tight_layout()
plt.savefig(f'{figures_dir}/confusion_matrix_raw.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {figures_dir}/confusion_matrix_raw.png')

# --- Save Cross-Validation Scores Plot ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), cv_scores, 'o-', linewidth=2, markersize=8, color='forestgreen')
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
plt.fill_between(range(1, 11), 
                 cv_scores.mean() - cv_scores.std(), 
                 cv_scores.mean() + cv_scores.std(), 
                 alpha=0.2, color='r')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('10-Fold Cross-Validation Scores (Raw Features)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{figures_dir}/cv_scores_raw.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {figures_dir}/cv_scores_raw.png')

# --- Save Per-Class Metrics Bar Chart ---
report_dict = classification_report(y_test, y_pred, target_names=species_names, output_dict=True)
class_metrics = {k: v for k, v in report_dict.items() if k in species_names}
metrics_df = pd.DataFrame(class_metrics).T

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(species_names))
width = 0.25
ax.bar(x - width, metrics_df['precision'], width, label='Precision', color='forestgreen')
ax.bar(x, metrics_df['recall'], width, label='Recall', color='darkorange')
ax.bar(x + width, metrics_df['f1-score'], width, label='F1-Score', color='steelblue')
ax.set_ylabel('Score')
ax.set_xlabel('Class (Ripening Day)')
ax.set_title('Per-Class Classification Metrics (Raw Features)')
ax.set_xticks(x)
ax.set_xticklabels(species_names, rotation=45)
ax.legend()
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(f'{figures_dir}/per_class_metrics_raw.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {figures_dir}/per_class_metrics_raw.png')

# --- Save Metadata JSON ---
# Helper to convert numpy types to native Python
def convert_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    return obj

metadata = {
    'experiment_date': datetime.now().isoformat(),
    'dataset': 'features_glcm_lbp_hsv.csv',
    'n_samples': int(len(df)),
    'n_features': int(len(feature_cols)),
    'n_classes': int(len(species_names)),
    'class_names': list(species_names),
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test)),
    'best_params': convert_to_native(grid_search.best_params_),
    'cv_scores': cv_scores.tolist(),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'test_accuracy': float(accuracy_score(y_test, y_pred)),
    'test_precision_weighted': float(precision_score(y_test, y_pred, average='weighted')),
    'test_recall_weighted': float(recall_score(y_test, y_pred, average='weighted')),
    'test_f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
    'classification_report': convert_to_native(report_dict)
}

with open(f'{reports_dir}/raw_svm_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f'Saved: {reports_dir}/raw_svm_metadata.json')

print('\\n✓ All figures and metadata saved successfully!')