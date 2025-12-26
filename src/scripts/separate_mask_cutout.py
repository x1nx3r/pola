#!/usr/bin/env python3
"""
separate_mask_cutout.py

Scans a directory tree (default: `segmented_masks_gmm`) and copies files
ending with `_mask.png` into `masks/` and `_cutout.png` into `cutouts/`
inside the chosen output folder while preserving subfolder structure.

Usage:
    python3 scripts/separate_mask_cutout.py --input segmented_masks_gmm --output segmented_masks_gmm_separated

Options:
    --move      Move files instead of copying.
    --dry-run   Don't perform any filesystem changes; just print actions.
"""
from pathlib import Path
import shutil
import argparse
import sys


def is_mask(p: Path) -> bool:
    return p.name.lower().endswith('_mask.png') or p.stem.lower().endswith('mask')


def is_cutout(p: Path) -> bool:
    return p.name.lower().endswith('_cutout.png') or 'cutout' in p.stem.lower()


def process(input_dir: Path, output_dir: Path, move: bool = False, dry_run: bool = False):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    masks_root = output_dir / 'masks'
    cutouts_root = output_dir / 'cutouts'

    counts = {'mask': 0, 'cutout': 0, 'other': 0}

    for p in sorted(input_dir.rglob('*.png')):
        rel = p.relative_to(input_dir)
        if is_mask(p):
            dest = masks_root / rel
            dest_parent = dest.parent
            action = 'move' if move else 'copy'
            print(f"[MASK]  {p} -> {dest} ({action})")
            if not dry_run:
                dest_parent.mkdir(parents=True, exist_ok=True)
                if move:
                    shutil.move(str(p), str(dest))
                else:
                    shutil.copy2(str(p), str(dest))
            counts['mask'] += 1
        elif is_cutout(p):
            dest = cutouts_root / rel
            dest_parent = dest.parent
            action = 'move' if move else 'copy'
            print(f"[CUTOUT]{p} -> {dest} ({action})")
            if not dry_run:
                dest_parent.mkdir(parents=True, exist_ok=True)
                if move:
                    shutil.move(str(p), str(dest))
                else:
                    shutil.copy2(str(p), str(dest))
            counts['cutout'] += 1
        else:
            counts['other'] += 1

    print("\nSummary:")
    print(f"  Masks copied:   {counts['mask']}")
    print(f"  Cutouts copied: {counts['cutout']}")
    print(f"  Other PNGs seen:{counts['other']}")


def main(argv=sys.argv[1:]):
    p = argparse.ArgumentParser(description='Separate mask and cutout PNGs into folders')
    p.add_argument('--input', '-i', default='segmented_masks_gmm', help='Input root folder (default segmented_masks_gmm)')
    p.add_argument('--output', '-o', default='segmented_masks_gmm_separated', help='Output folder to create masks/ and cutouts/ inside')
    p.add_argument('--move', action='store_true', help='Move files instead of copying')
    p.add_argument('--dry-run', action='store_true', help="Don't copy/move, only print")
    args = p.parse_args(argv)

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}")
        return 2

    output_dir = Path(args.output)
    if args.dry_run:
        print("DRY RUN: no files will be copied/moved")

    process(input_dir, output_dir, move=args.move, dry_run=args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())
