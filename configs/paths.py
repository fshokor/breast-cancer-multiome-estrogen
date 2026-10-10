"""Shared local/Colab paths; pass the Drive root explicitly in notebooks."""
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "raw" / "GSE154873"
PROCESSED_DIR = BASE / "data" / "processed"
RESULTS_DIR = BASE / "results"
