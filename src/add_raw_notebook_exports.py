"""
Script to add figure and metadata export functionality to pola_raw_svm_classification.ipynb
"""
import json

NOTEBOOK_PATH = '/mnt/libraries/pola/notebooks/pola_raw_svm_classification.ipynb'

# New cell to add at the end of the notebook for saving outputs
SAVE_OUTPUTS_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "id": "save_outputs_cell_raw",
    "metadata": {},
    "outputs": [],
    "source": [
        "# =======================================================================\n",
        "# SAVE FIGURES AND METADATA (RAW FEATURES)\n",
        "# =======================================================================\n",
        "import os\n",
        "import json\n",
        "from datetime import datetime\n",
        "\n",
        "# Create directories if they don't exist\n",
        "figures_dir = '../figures'\n",
        "reports_dir = '../reports'\n",
        "os.makedirs(figures_dir, exist_ok=True)\n",
        "os.makedirs(reports_dir, exist_ok=True)\n",
        "\n",
        "# --- Save Confusion Matrix ---\n",
        "plt.figure(figsize=(12, 10))\n",
        "cm = confusion_matrix(y_test, y_pred)\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', \n",
        "            xticklabels=species_names, yticklabels=species_names)\n",
        "plt.xlabel('Predicted')\n",
        "plt.ylabel('Actual')\n",
        "plt.title(f'Confusion Matrix - Raw Features (62D) + SVM')\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{figures_dir}/confusion_matrix_raw.png', dpi=150, bbox_inches='tight')\n",
        "plt.close()\n",
        "print(f'Saved: {figures_dir}/confusion_matrix_raw.png')\n",
        "\n",
        "# --- Save Cross-Validation Scores Plot ---\n",
        "plt.figure(figsize=(10, 6))\n",
        "plt.plot(range(1, 11), cv_scores, 'o-', linewidth=2, markersize=8, color='forestgreen')\n",
        "plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')\n",
        "plt.fill_between(range(1, 11), \n",
        "                 cv_scores.mean() - cv_scores.std(), \n",
        "                 cv_scores.mean() + cv_scores.std(), \n",
        "                 alpha=0.2, color='r')\n",
        "plt.xlabel('Fold Number')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.title('10-Fold Cross-Validation Scores (Raw Features)')\n",
        "plt.legend()\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{figures_dir}/cv_scores_raw.png', dpi=150, bbox_inches='tight')\n",
        "plt.close()\n",
        "print(f'Saved: {figures_dir}/cv_scores_raw.png')\n",
        "\n",
        "# --- Save Per-Class Metrics Bar Chart ---\n",
        "report_dict = classification_report(y_test, y_pred, target_names=species_names, output_dict=True)\n",
        "class_metrics = {k: v for k, v in report_dict.items() if k in species_names}\n",
        "metrics_df = pd.DataFrame(class_metrics).T\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(12, 6))\n",
        "x = np.arange(len(species_names))\n",
        "width = 0.25\n",
        "ax.bar(x - width, metrics_df['precision'], width, label='Precision', color='forestgreen')\n",
        "ax.bar(x, metrics_df['recall'], width, label='Recall', color='darkorange')\n",
        "ax.bar(x + width, metrics_df['f1-score'], width, label='F1-Score', color='steelblue')\n",
        "ax.set_ylabel('Score')\n",
        "ax.set_xlabel('Class (Ripening Day)')\n",
        "ax.set_title('Per-Class Classification Metrics (Raw Features)')\n",
        "ax.set_xticks(x)\n",
        "ax.set_xticklabels(species_names, rotation=45)\n",
        "ax.legend()\n",
        "ax.set_ylim([0, 1])\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{figures_dir}/per_class_metrics_raw.png', dpi=150, bbox_inches='tight')\n",
        "plt.close()\n",
        "print(f'Saved: {figures_dir}/per_class_metrics_raw.png')\n",
        "\n",
        "# --- Save Metadata JSON ---\n",
        "# Helper to convert numpy types to native Python\n",
        "def convert_to_native(obj):\n",
        "    if isinstance(obj, np.ndarray):\n",
        "        return obj.tolist()\n",
        "    elif isinstance(obj, (np.int64, np.int32)):\n",
        "        return int(obj)\n",
        "    elif isinstance(obj, (np.float64, np.float32)):\n",
        "        return float(obj)\n",
        "    elif isinstance(obj, dict):\n",
        "        return {k: convert_to_native(v) for k, v in obj.items()}\n",
        "    elif isinstance(obj, list):\n",
        "        return [convert_to_native(i) for i in obj]\n",
        "    return obj\n",
        "\n",
        "metadata = {\n",
        "    'experiment_date': datetime.now().isoformat(),\n",
        "    'dataset': 'features_glcm_lbp_hsv.csv',\n",
        "    'n_samples': int(len(df)),\n",
        "    'n_features': int(len(feature_cols)),\n",
        "    'n_classes': int(len(species_names)),\n",
        "    'class_names': list(species_names),\n",
        "    'train_size': int(len(X_train)),\n",
        "    'test_size': int(len(X_test)),\n",
        "    'best_params': convert_to_native(grid_search.best_params_),\n",
        "    'cv_scores': cv_scores.tolist(),\n",
        "    'cv_mean': float(cv_scores.mean()),\n",
        "    'cv_std': float(cv_scores.std()),\n",
        "    'test_accuracy': float(accuracy_score(y_test, y_pred)),\n",
        "    'test_precision_weighted': float(precision_score(y_test, y_pred, average='weighted')),\n",
        "    'test_recall_weighted': float(recall_score(y_test, y_pred, average='weighted')),\n",
        "    'test_f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),\n",
        "    'classification_report': convert_to_native(report_dict)\n",
        "}\n",
        "\n",
        "with open(f'{reports_dir}/raw_svm_metadata.json', 'w') as f:\n",
        "    json.dump(metadata, f, indent=2)\n",
        "print(f'Saved: {reports_dir}/raw_svm_metadata.json')\n",
        "\n",
        "print('\\\\n✓ All figures and metadata saved successfully!')\n"
    ]
}

def add_export_cell():
    # Load notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Check if export cell already exists
    for cell in nb['cells']:
        if cell.get('id') == 'save_outputs_cell_raw':
            print("Export cell already exists, skipping.")
            return
    
    # Add the new cell at the end
    nb['cells'].append(SAVE_OUTPUTS_CELL)
    
    # Save modified notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    
    print(f"Successfully added export cell to {NOTEBOOK_PATH}")

if __name__ == "__main__":
    add_export_cell()
