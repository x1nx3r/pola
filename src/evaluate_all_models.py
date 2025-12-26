
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Configuration
DATA_DIR = '../data'
MODELS_DIR = '../models'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Models to evaluate
MODEL_CONFIGS = [
    {
        'name': 'NCA 2',
        'feature_file': 'features_nca_2.csv',
        'model_file': 'pola_nca2_svm_best.joblib',
        'scaler_file': 'pola_nca2_scaler.joblib',
        'encoder_file': 'pola_nca2_label_encoder.joblib'
    },
    {
        'name': 'NCA 9',
        'feature_file': 'features_nca_9.csv',
        'model_file': 'pola_nca9_svm_best.joblib',
        'scaler_file': 'pola_nca9_scaler.joblib',
        'encoder_file': 'pola_nca9_label_encoder.joblib'
    },
    {
        'name': 'NCA 10',
        'feature_file': 'features_nca_10.csv',
        'model_file': 'pola_nca10_svm_best.joblib',
        'scaler_file': 'pola_nca10_scaler.joblib',
        'encoder_file': 'pola_nca10_label_encoder.joblib'
    }
]

def load_data(feature_file):
    path = os.path.join(DATA_DIR, feature_file)
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None, None
    
    df = pd.read_csv(path)
    # Assuming 'label' and 'filename' are columns, and the rest are features
    # Excluding filename and label from X
    feature_cols = [c for c in df.columns if c not in ['label', 'filename']]
    X = df[feature_cols]
    y = df['label']
    return X, y

def evaluate_model(config):
    print(f"Evaluating {config['name']}...")
    X, y = load_data(config['feature_file'])
    if X is None:
        return None

    # Encode labels
    try:
        le = joblib.load(os.path.join(MODELS_DIR, config['encoder_file']))
        y_encoded = le.transform(y)
    except Exception as e:
        print(f"  Error loading label encoder: {e}. Fitting new one.")
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    # Load scaler and transform
    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, config['scaler_file']))
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        print(f"  Error loading scaler: {e}. Fitting new one (less accurate for validation).")
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_test_scaled = scaler.transform(X_test)

    # Load model
    try:
        model = joblib.load(os.path.join(MODELS_DIR, config['model_file']))
    except Exception as e:
        print(f"  Error loading model {config['model_file']}: {e}")
        return None

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    return metrics

def main():
    results = []
    for config in MODEL_CONFIGS:
        metrics = evaluate_model(config)
        if metrics:
            metrics['Model'] = config['name']
            results.append(metrics)
    
    if results:
        results_df = pd.DataFrame(results)
        # reorder columns
        cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        results_df = results_df[cols]
        print("\n\n=== COMPARATIVE RESULTS ===")
        print(results_df.to_markdown(index=False))
        results_df.to_csv('../reports/comparative_results.csv', index=False)
        print("\nSaved to ../reports/comparative_results.csv")

if __name__ == "__main__":
    main()
