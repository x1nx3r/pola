
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load NCA 10
print("Loading features_nca_10.csv...")
try:
    df = pd.read_csv('../data/features_nca_10.csv')
    
    # Extract
    X = df[[c for c in df.columns if c.startswith('nca')]].values
    
    # Label extraction (reusing logic)
    if 'label' in df.columns:
        y = df['label'].values
    elif 'label_code' in df.columns:
        y = df['label_code'].values
    else:
        y = df['filename'].astype(str).str.extract(r'(?i)(h\d+)')[0].str.upper().fillna('UNKNOWN').values
        
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check Linear vs RBF
    print("Evaluating NCA 10...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    svm_lin = SVC(kernel='linear')
    scores_lin = cross_val_score(svm_lin, X_scaled, y_enc, cv=skf)
    print(f"Linear CV: {scores_lin.mean():.4f}")
    
    svm_rbf = SVC(kernel='rbf')
    scores_rbf = cross_val_score(svm_rbf, X_scaled, y_enc, cv=skf)
    print(f"RBF CV: {scores_rbf.mean():.4f}")

except Exception as e:
    print(f"Error checking NCA 10: {e}")

