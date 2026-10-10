"""Run the paper-guided QC milestone locally."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.qc_preprocessing import run_preprocessing
if __name__ == "__main__":
    run_preprocessing(ROOT,json.loads((ROOT/"configs/qc_paper.json").read_text()))
