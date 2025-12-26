# Features Summary

- **Total columns**: 63 (including `filename`)
- **Feature groups**: GLCM, LBP, HSV histograms, plus `filename` identifier

## GLCM (texture) — 12 features
- glcm_contrast_mean, glcm_contrast_std
- glcm_dissimilarity_mean, glcm_dissimilarity_std
- glcm_homogeneity_mean, glcm_homogeneity_std
- glcm_energy_mean, glcm_energy_std
- glcm_correlation_mean, glcm_correlation_std
- glcm_ASM_mean, glcm_ASM_std

## LBP (local binary pattern) — 10 features
- lbp_0 .. lbp_9 (10 histogram/bin features)

## HSV (color histograms) — 40 features
- Hue: h_0 .. h_15 (16 bins)
- Saturation: s_0 .. s_15 (16 bins)
- Value: v_0 .. v_7 (8 bins)

## Identifier / metadata
- filename (categorical)

## Counts & basic stats
- Dataset rows: 3277
- Numeric columns: 62 (all GLCM/LBP/HSV features)
- See generated analysis: [analysis.json](analysis.json)

## Source files & where to look
- Feature extraction notebook: [feature_extraction.ipynb](feature_extraction.ipynb)
- Inspector used to summarize features: [inspect_features.py](inspect_features.py)
- Preprocessing / helpers: [resize.py](resize.py), [scripts/separate_mask_cutout.py](scripts/separate_mask_cutout.py)
- Segmentation notebooks (related preprocessing/ROI): [segmentation_gmm.ipynb](segmentation_gmm.ipynb), [segmentation_hsv_grabcut.ipynb](segmentation_hsv_grabcut.ipynb)

## Quick reproduction
Regenerate the CSV/analysis by running your feature-extraction notebook or scripts. To re-run the quick analysis included here:

```bash
python inspect_features.py --csv features_glcm_lbp_hsv.csv --json analysis.json
```

## Notes & recommendations
- `v_0` and several high-index hue/saturation/value bins have very small means/std (near-zero) — consider removing very low-variance bins or applying thresholding/normalization.
- `filename` is an identifier and should be excluded for modeling.
- GLCM features include mean/std for common texture descriptors; LBP features are histogram bins — both are suitable for texture classification.
- If you plan further preprocessing, consider per-feature standardization and checking for multicollinearity among GLCM stats.


## Approach used for feature extraction

Overview: images were preprocessed and cropped to the region of interest (ROI), then three families of features were computed per-image and written to the CSV `features_glcm_lbp_hsv.csv`.

Pipeline (high-level):
- Input: original cutout images (see `dataset/` and `scripts/separate_mask_cutout.py`).
- Preprocessing: optional resizing (`resize.py`) and mask-based cutouts. Segmentation notebooks (`segmentation_gmm.ipynb`, `segmentation_hsv_grabcut.ipynb`) were used for ROI/mask generation when needed.
- ROI extraction: masks or cutouts are applied to isolate the object/region; features are computed on the masked area (or whole cutout if mask not present).
- Feature computation:
	- GLCM (Gray-Level Co-occurrence Matrix): computed texture descriptors — contrast, dissimilarity, homogeneity, energy, correlation, ASM — and saved as mean and standard-deviation (suffixes `_mean` and `_std`).
	- LBP (Local Binary Pattern): a set of histogram/bin features `lbp_0`..`lbp_9` representing LBP distribution across the ROI.
	- HSV histograms: color histograms split into Hue (`h_0`..`h_15`, 16 bins), Saturation (`s_0`..`s_15`, 16 bins), and Value (`v_0`..`v_7`, 8 bins). Histograms were normalized and stored as fractional/bin values.
- Aggregation & output: per-image features concatenated into one row with `filename` as identifier and exported to `features_glcm_lbp_hsv.csv`.

Reproducibility notes:
- To regenerate features, run the feature extraction notebook `feature_extraction.ipynb` (cells that perform ROI creation, then per-image feature computation). If you prefer a script, I can extract the notebook cells into a runnable script.
- Quick verify/analyze: `python inspect_features.py --csv features_glcm_lbp_hsv.csv --json analysis.json`.

Practical recommendations:
- Exclude `filename` before modeling; standardize numeric features per-column (z-score) for many ML algorithms.
- Remove or merge near-constant histogram bins (low-variance `v_*`, `h_*`, `s_*`) to reduce dimensionality.
- Consider computing GLCM on multiple distances/angles if more texture granularity is needed (current CSV stores aggregated mean/std).

