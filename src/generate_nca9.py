
import pandas as pd
import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
import os

def generate_nca9():
    print("Loading features_glcm_lbp_hsv.csv...")
    try:
        # Load features CSV
        if os.path.exists('../data/features_glcm_lbp_hsv.csv'):
            csv_path = '../data/features_glcm_lbp_hsv.csv'
        elif os.path.exists('features_glcm_lbp_hsv.csv'):
            csv_path = 'features_glcm_lbp_hsv.csv'
        else:
            print("Error: Could not find features_glcm_lbp_hsv.csv")
            return

        df = pd.read_csv(csv_path)
        print(f"Shape: {df.shape}")

        # Extract label
        print("Extracting labels...")
        df['label'] = df['filename'].str.extract(r'(?i)(h\d+)')[0].str.upper()
        df['label'] = df['label'].fillna('UNKNOWN')
        df['label_code'] = pd.factorize(df['label'])[0]
        
        # Prepare X and y
        exclude = {'filename','label','label_code', 'Unnamed: 0'}
        feature_cols = [c for c in df.columns if c not in exclude]
        X = df[feature_cols].values
        y = df['label_code'].values
        
        # Scale
        print("Scaling features...")
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        
        # NCA 9
        n_components = 9
        print(f"Fitting NCA with {n_components} components...")
        nca = NeighborhoodComponentsAnalysis(n_components=n_components, random_state=42)
        nca.fit(Xs, y)
        X_nca = nca.transform(Xs)
        
        # Save
        df_nca = df[['filename','label', 'label_code']].copy()
        for i in range(n_components):
            df_nca[f'nca{i}'] = X_nca[:, i]
            
        out_csv = '../data/features_nca_9.csv'
        df_nca.to_csv(out_csv, index=False)
        print(f"Saved transformed features to {out_csv}")
        print(df_nca.head())
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_nca9()
