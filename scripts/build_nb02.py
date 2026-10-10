"""Build only nb02; preserve previously executed nb00/nb01."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def md(s):return {"cell_type":"markdown","metadata":{},"source":s.strip().splitlines(True)}
def code(s):return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip().splitlines(True)}
config=json.loads((ROOT/"configs/qc_paper.json").read_text())
cells=[
md("""# nb02 — Paper-guided QC and preprocessing
**Question:** Which paired cells and genes pass the RNA QC thresholds reported by Kupp et al.,
and how can we prepare RNA and ATAC while preserving counts and exact pairing?
Reference: [Kupp et al., Communications Biology (2025)](https://doi.org/10.1038/s42003-025-08533-7),
Methods, p.14, “Standard single-cell analysis workflow”.
This is the paper's standard-analysis QC branch, not a claim that its scMKL models used the same filtering.
No DE, GSEA, motif analysis, dimensionality reduction, clustering or models are run here."""),
code("""try:
    from google.colab import drive
except ImportError:
    IN_COLAB = False
else:
    IN_COLAB = True
    drive.mount('/content/drive')
from pathlib import Path
import os
BASE = Path("/content/drive/MyDrive/breast-cancer-multiome-estrogen") if IN_COLAB else Path(os.environ.get("MULTIOME_PROJECT_ROOT", Path.cwd())).resolve()
if BASE.name == "notebooks":
    BASE = BASE.parent
DATA_DIR = BASE / "data/raw/GSE154873"
RESULTS_DIR = BASE / "results"
assert (DATA_DIR / "download_manifest.json").is_file(), "Run nb00 first."
print("Project:", BASE)
print("Inputs:", DATA_DIR)
print("Processed:", BASE / "data/processed")
"""),
code("""import importlib.util, subprocess, sys
packages = {"anndata":"anndata", "numpy":"numpy", "pandas":"pandas", "scipy":"scipy", "matplotlib":"matplotlib"}
missing = [package for module,package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable,"-m","pip","install",*missing])
print("Python:",sys.version)
"""),
md("""## Parameters and interpretation
Keep genes detected in **at least 3 cells**; keep cells with **at least 100 detected genes**
and **mitochondrial counts ≤30%**. Exactly 30% passes. Human mitochondrial genes are recognized by MT-.
We implement one gene-first pass, then compute cell QC on the retained genes, separately per cell line.
Filter order, MT- recognition and scaling ddof=1 are explicit implementation choices, not fully specified paper parameters.
The paper supplies no library-size normalization target: use log1p(counts) and record sample-standard-deviation
scaling with no clipping. All choices are visible below.
No ATAC-specific cutoff is invented. Apply the identical RNA cell mask; retain all peaks.
Peak counts are not fragment counts. TSS enrichment, FRiP and nucleosome metrics need unavailable fragment-level inputs."""),
code("import json\nCONFIG = json.loads("+repr(json.dumps(config))+")\nprint(json.dumps(CONFIG,indent=2))"),
md("""## Sparse implementation
The next cell is generated from src/qc_preprocessing.py, so this notebook runs without source uploads.
Inputs must match nb00's SHA-256 manifest. Only one dense text row is parsed at a time.
RNA .X contains log1p counts; layers['counts'] contains integers.
Centered scaling is stored as a sparse scaled_no_center layer plus var['scale_center_offset']:
z = scaled_no_center - scale_center_offset. Use centered_scaled_block only on small requested blocks.
ATAC .X contains integer peak counts and layers['binary'] contains accessibility indicators.
The four .h5ad files are written under data/processed/ and overwritten deterministically when rerun."""),
code((ROOT/"src/qc_preprocessing.py").read_text()),
code("report = run_preprocessing(BASE, CONFIG)"),
code("""summary = pd.DataFrame({line: {k:v for k,v in values.items() if isinstance(v,(int,bool))}
                        for line,values in report["cell_lines"].items()}).T
display(summary)
for line, values in report["cell_lines"].items():
    print(line,"treatments before:",values["treatment_before"],"after:",values["treatment_after"])
    print("Exact RNA–ATAC order preserved:",values["exact_paired_order"])
"""),
code("""from IPython.display import Image, display
for line in ("MCF7","T47D"):
    display(Image(filename=str(RESULTS_DIR / "figures" / f"{line}_paper_qc.png")))
"""),
md("""## Provenance and limits
Table 1 matches the downloaded pre-QC dimensions and treatment counts exactly.
The scMKL paper describes 24-hour estrogen treatment (p.12), whereas the original Ors et al. paper
describes 48 hours for Multiome. Duration is unresolved across publications and is not changed in metadata.
The paper tests multiple scMKL representations. Its explicit ATAC TF-IDF uses count/binary values multiplied
by log(1 + (1+n_cells)/(1+n_detected_cells)), without the usual per-cell term-frequency division.
TF-IDF, feature selection, PCA/LSI and predictive scaling must be fitted inside training splits later;
this QC notebook does not fit them globally for prediction.
The descriptive RNA scaling here is not a reusable fitted transform for held-out classification.
Original archives remain untouched; they retain all original cells/features for a later scMKL-faithful branch.
See QC_PREPROCESSING.md for provenance, assumptions, outputs and validation.
Next: review QC results before nb03 RNA differential expression. Stop here.""")]
for i,c in enumerate(cells):c["id"]=f"nb02-{i:03d}"
nb={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},"language_info":{"name":"python","version":"3.11"},"colab":{"name":"nb02_qc_preprocessing.ipynb","provenance":[]}},"cells":cells}
(ROOT/"notebooks/nb02_qc_preprocessing.ipynb").write_text(json.dumps(nb,indent=2)+"\n")
print("Generated nb02.")
