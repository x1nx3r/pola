
import os
import pandas as pd
import joblib
import json

print("Verifying paths from notebooks/ directory...")

# 1. Check Data
data_files = [
    '../data/features_nca_10.csv',
    '../data/features_nca_2.csv',
    '../data/features_glcm_lbp_hsv.csv'
]
for f in data_files:
    if os.path.exists(f):
        print(f"PASS: Found {f}")
        try:
            pd.read_csv(f, nrows=5)
            print(f"PASS: Read {f}")
        except Exception as e:
            print(f"FAIL: Could not read {f}: {e}")
    else:
        print(f"FAIL: Not found {f}")

# 2. Check Models (Writable)
try:
    joblib.dump('test', '../models/test_artifact.joblib')
    print("PASS: Wrote to ../models/")
    os.remove('../models/test_artifact.joblib')
except Exception as e:
    print(f"FAIL: Could not write to ../models/: {e}")

# 3. Check Reports (Writable)
try:
    with open('../reports/test_report.json', 'w') as f:
        json.dump({'status': 'ok'}, f)
    print("PASS: Wrote to ../reports/")
    os.remove('../reports/test_report.json')
except Exception as e:
    print(f"FAIL: Could not write to ../reports/: {e}")

print("Verification complete.")
