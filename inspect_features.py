#!/usr/bin/env python3
"""inspect_features.py

Quick CSV feature-spec inspector for `features_glcm_lbp_hsv.csv`.

Usage:
  python inspect_features.py [--csv PATH] [--sample N] [--json OUT]

Outputs basic schema: columns, dtypes, missing counts, numeric summaries,
unique counts for non-numeric, and simple grouping by name (GLCM, LBP, HSV).
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from typing import Dict, Any

# Prevent local files named `inspect.py` from shadowing the stdlib `inspect`
# (which breaks imports in packages like numpy/pandas). If a local `inspect.py`
# exists in the current working directory, temporarily remove it from the
# import path when importing heavy libraries.
_removed_sys_path0 = None
cwd = os.getcwd()
if os.path.exists(os.path.join(cwd, "inspect.py")) or os.path.exists(os.path.join(cwd, "inspect")):
    sys.stderr.write("Detected local 'inspect.py' that may shadow stdlib; adjusting sys.path temporarily...\n")
    if sys.path and (sys.path[0] == "" or os.path.abspath(sys.path[0]) == cwd):
        _removed_sys_path0 = sys.path.pop(0)

import pandas as pd
if _removed_sys_path0 is not None:
    sys.path.insert(0, _removed_sys_path0)


def analyze(df: pd.DataFrame, low_var_threshold: float = 1e-6) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["n_rows"] = int(df.shape[0])
    out["n_cols"] = int(df.shape[1])
    out["columns"] = []

    for col in df.columns:
        ser = df[col]
        col_info: Dict[str, Any] = {
            "name": str(col),
            "dtype": str(ser.dtype),
            "n_missing": int(ser.isna().sum()),
        }

        if pd.api.types.is_numeric_dtype(ser):
            desc = ser.describe()
            col_info.update(
                {
                    "type": "numeric",
                    "count": int(desc.get("count", 0)),
                    "mean": None if pd.isna(desc.get("mean")) else float(desc.get("mean")),
                    "std": None if pd.isna(desc.get("std")) else float(desc.get("std")),
                    "min": None if pd.isna(desc.get("min")) else float(desc.get("min")),
                    "max": None if pd.isna(desc.get("max")) else float(desc.get("max")),
                    "unique": int(ser.nunique(dropna=True)),
                }
            )
            # low-variance warning
            if col_info["unique"] <= 1 or (col_info["std"] is not None and col_info["std"] <= low_var_threshold):
                col_info["low_variance"] = True
        else:
            nunique = int(ser.nunique(dropna=True))
            top = None
            try:
                top = ser.mode().iat[0]
            except Exception:
                top = None
            col_info.update({"type": "categorical", "unique": nunique, "top": None if pd.isna(top) else top})

        out["columns"].append(col_info)

    # overall numeric summary
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric:
        out["numeric_columns"] = numeric
        out["numeric_summary"] = df[numeric].describe().to_dict()

    # group features by name heuristics
    groups = {"GLCM": [], "LBP": [], "HSV": [], "other": []}
    for col in df.columns:
        name = col.lower()
        if "glcm" in name or "contrast" in name or "correlation" in name or "energy" in name or "homogeneity" in name:
            groups["GLCM"].append(col)
        elif "lbp" in name or "lbp_" in name:
            groups["LBP"].append(col)
        elif "h_" in name or name.startswith("h_") or (name.startswith("h") and "hsv" in name):
            groups["HSV"].append(col)
        elif any(k in name for k in ["hsv", "s_", "v_"]):
            groups["HSV"].append(col)
        else:
            groups["other"].append(col)

    out["groups"] = {k: len(v) for k, v in groups.items()}
    out["groups_examples"] = {k: v[:10] for k, v in groups.items()}

    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect CSV feature specs")
    parser.add_argument("--csv", default="features_glcm_lbp_hsv.csv", help="Path to CSV file")
    parser.add_argument("--sample", type=int, default=5, help="Number of sample rows to show")
    parser.add_argument("--json", help="Write analysis JSON to given path")
    parser.add_argument("--low-var-threshold", type=float, default=1e-6, help="Threshold for low variance warning")
    args = parser.parse_args(argv)

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 2

    analysis = analyze(df, low_var_threshold=args.low_var_threshold)

    # Print concise human-readable report
    print(f"Rows: {analysis['n_rows']}, Columns: {analysis['n_cols']}")
    print("\nColumns summary:")
    for c in analysis["columns"]:
        line = f"- {c['name']}: {c.get('type','?')}, missing={c['n_missing']}, unique={c.get('unique', '?')}"
        if c.get("low_variance"):
            line += " [LOW_VARIANCE]"
        print(line)

    print("\nGroups (counts):")
    for k, v in analysis["groups"].items():
        print(f"- {k}: {v}")

    print(f"\nSample {args.sample} rows:")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.head(args.sample).to_string(index=False))

    if args.json:
        with open(args.json, "w", encoding="utf8") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
