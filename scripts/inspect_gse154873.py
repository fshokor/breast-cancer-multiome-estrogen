#!/usr/bin/env python3
"""Inspect six GEO files without constructing a dense matrix."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_inspection import inspect_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/GSE154873")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables")
    args = parser.parse_args()
    inspect_dataset(args.data_dir, args.output_dir)
