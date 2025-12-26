
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

print("="*70)
print("VERIFYING NCA 9 PIPELINE")
print("="*70)

features_file = '../data/features_nca_9.csv'

if not os.path.exists(features_file):
    print(f"Error: {features_file} not found.")
    exit(1)

# Load data
print(f"Loading {features_file}...")
df = pd.read_csv(features_file)
print(f"Shape: {df.shape}")

# Features
feature_cols = [c for c in df.columns if c.startswith('nca')]
X = df[feature_cols].values
print(f"Features: {len(feature_cols)}")

# Labels
if 'label_code' in df.columns:
    y = df['label_code'].values
else:
    print("Encoding labels...")
    le = LabelEncoder()
    y = le.fit_transform(df['label'])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM RBF
print("\nTraining SVM (RBF)...")
svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)

# Evaluate
acc = svm.score(X_test_scaled, y_test)
print(f"Test Accuracy: {acc:.4f}")

# Cross-validation
print("Running CV...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(svm, X_train_scaled, y_train, cv=skf)
print(f"CV Mean Accuracy: {scores.mean():.4f}")

# Save dummy report for consistency check
import json
report = {'accuracy': acc, 'cv_mean': scores.mean()}
with open('../reports/pola_nca9_svm_summary.json', 'w') as f:
    json.dump(report, f)
print("Saved summary report.")
