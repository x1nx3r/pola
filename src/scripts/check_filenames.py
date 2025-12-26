#!/usr/bin/env python3
import csv, re, sys
fn='features_glcm_lbp_hsv.csv'
missing=[]
lower=[]
empty=[]
try:
    with open(fn,'r',encoding='utf-8') as f:
        r=csv.reader(f)
        header=next(r)
        try:
            idx=header.index('filename')
        except ValueError:
            idx=0
        for lineno,row in enumerate(r, start=2):
            fname = row[idx] if idx < len(row) else ''
            if fname is None or fname.strip()=='':
                empty.append(lineno)
                continue
            if re.search(r'h\d+', fname) and not re.search(r'H\d+', fname):
                lower.append((lineno,fname))
            if not re.search(r'H\d+', fname):
                missing.append((lineno,fname))
    print('total rows scanned:', lineno)
    print('missing H\\d+ count:', len(missing))
    if missing:
        print('sample missing (up to 20):')
        for ln,fnm in missing[:20]:
            print(ln, fnm)
    print('\nrows with lowercase h pattern count:', len(lower))
    if lower:
        print('sample lowercase-h (up to 20):')
        for ln,fnm in lower[:20]:
            print(ln, fnm)
    print('\nempty filename rows:', len(empty))
except FileNotFoundError:
    print('CSV not found:', fn)
    sys.exit(1)
