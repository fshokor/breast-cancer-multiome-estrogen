from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.motif_analysis import run_motifs
run_motifs(ROOT,*[json.loads((ROOT/"configs"/name).read_text()) for name in
                  ["motif_analysis.json","motif_references.json","peak_coordinate_audit.json"]])
