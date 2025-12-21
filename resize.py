#!/usr/bin/env python3
"""Resize images in a folder (recursively) to a square size (default 512x512).

Usage:
	python resize.py --dir dataset --size 512

This script overwrites images in-place. Use `--dry-run` to see counts only.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

from PIL import Image
try:
	from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
	def _tqdm_fallback(iterable, desc=None, unit=None, **kwargs):
		items = list(iterable)
		total = len(items)
		if total == 0:
			return iter(())
		for i, item in enumerate(items, 1):
			pct = i * 100.0 / total
			label = (desc + ":") if desc else "Progress:"
			print(f"\r{label} {pct:5.1f}% ({i}/{total})", end="", flush=True)
			yield item
		print()

	tqdm = _tqdm_fallback

LOG = logging.getLogger(__name__)


def iter_image_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
	for p in root.rglob("*"):
		if p.is_file() and p.suffix.lower() in exts:
			yield p


def resize_image(path: Path, size: int) -> None:
	try:
		with Image.open(path) as im:
			im = im.convert("RGB")
			im = im.resize((size, size), Image.LANCZOS)
			im.save(path)
	except Exception as e:
		LOG.exception("Failed to resize %s: %s", path, e)


def main() -> int:
	parser = argparse.ArgumentParser(description="Resize images recursively to a square size.")
	parser.add_argument("--dir", default="dataset", help="Root directory to scan")
	parser.add_argument("--size", type=int, default=512, help="Target size (pixels) for width and height")
	parser.add_argument("--dry-run", action="store_true", help="Don't write files; only report what would be done")
	parser.add_argument("--ext", action="append", help="Additional extensions to include (e.g. .tif). Can be used multiple times")
	parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
	args = parser.parse_args()

	logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

	root = Path(args.dir)
	if not root.exists():
		LOG.error("Directory does not exist: %s", root)
		return 2

	default_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
	if args.ext:
		for e in args.ext:
			if not e.startswith("."):
				e = "." + e
			default_exts.add(e.lower())

	files = list(iter_image_files(root, default_exts))
	LOG.info("Found %d image(s) under %s", len(files), root)
	if args.dry_run:
		return 0

	count = 0
	for p in tqdm(files, desc="Resizing", unit="img"):
		LOG.debug("Resizing %s", p)
		resize_image(p, args.size)
		count += 1

	LOG.info("Resized %d image(s) to %dx%d", count, args.size, args.size)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

