"""
Extract and run the SVM classification notebook to generate figures and metadata.
"""
import json
import os
import sys

NOTEBOOK_PATH = '/mnt/libraries/pola/notebooks/pola_svm_classification.ipynb'
OUTPUT_SCRIPT = '/mnt/libraries/pola/notebooks/run_svm_with_exports.py'

def extract_code():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            # Comment out magics and shell commands
            lines = []
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("!") or stripped.startswith("%"):
                    lines.append("# " + line)
                elif "display(" in line:
                    lines.append(line.replace("display(", "print("))
                else:
                    lines.append(line)
            code_cells.append("\n".join(lines))
    
    with open(OUTPUT_SCRIPT, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(code_cells))
    
    print(f"Extracted code to {OUTPUT_SCRIPT}")

if __name__ == "__main__":
    extract_code()
