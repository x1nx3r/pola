"""
Remove existing export cell and re-add with fix
"""
import json

NOTEBOOK_PATH = '/mnt/libraries/pola/notebooks/pola_svm_classification.ipynb'

def remove_export_cell():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Filter out the export cell
    nb['cells'] = [c for c in nb['cells'] if c.get('id') != 'save_outputs_cell']
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    
    print("Removed old export cell")

if __name__ == "__main__":
    remove_export_cell()
