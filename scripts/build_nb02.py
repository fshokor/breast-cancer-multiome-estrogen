"""Build the standalone Scanpy notebook; preserve nb00 and nb01."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def md(text):
    return {"cell_type":"markdown","metadata":{},"source":text.strip().splitlines(True)}
def code(text):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":text.strip().splitlines(True)}
config = json.loads((ROOT/"configs/qc_paper.json").read_text())
cells = [
md("""# nb02 — Scanpy QC, preprocessing, PCA, clustering and UMAP
**Questions:** Which cells and genes pass the published RNA QC thresholds?
How is expression variation organized within MCF-7 and T47D, and how does it relate
to treatment, sequencing depth and mitochondrial counts?

RNA exploration is performed independently for each cell line. ATAC is kept paired;
PCA, Leiden clusters and UMAP are **RNA-derived**, not ATAC or joint embeddings.
Clusters are exploratory groups, not cell-type or estrogen-response annotations.
Reference: [Kupp et al. (2025)](https://doi.org/10.1038/s42003-025-08533-7), Methods p.14.
This notebook does not perform differential expression, enrichment or prediction."""),
md("## 1. Mount Drive and select project paths"),
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
for folder in (BASE/"data/processed",RESULTS_DIR/"tables",RESULTS_DIR/"figures"):
    folder.mkdir(parents=True,exist_ok=True)
print("Project:", BASE)
print("Raw inputs:", DATA_DIR)
print("Processed outputs:", BASE/"data/processed")
"""),
md("## 2. Check the Scanpy environment"),
code("""import importlib.util, importlib.metadata, subprocess, sys
packages = {"scanpy":"scanpy==1.11.5", "anndata":"anndata", "numpy":"numpy",
            "pandas":"pandas", "scipy":"scipy", "matplotlib":"matplotlib", "igraph":"igraph"}
missing = [package for module,package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable,"-m","pip","install",*missing])
print("Python:",sys.version)
for package in packages:
    print(package,importlib.metadata.version(package))
"""),
md("""## 3. Inspect all parameters before running
**Paper thresholds:** retain genes detected in ≥3 cells and cells with ≥100 detected genes
and ≤30% mitochondrial counts. Exactly 30% passes. Apply one gene-first pass per cell line.
The exact filtering order and MT- prefix rule are documented project choices.

**Count-preserving branch:** use Scanpy calculate_qc_metrics, filter_genes, filter_cells,
log1p and scale. Preserve integer counts in layers['counts']; .X holds log1p raw counts.
All-gene scaling uses zero_center=False with stored mean/SD and offsets to stay sparse.

**Exploratory RNA branch:** copy counts, normalize_total(target_sum=10000), log1p,
highly_variable_genes(n_top_genes=2000, flavor='seurat'), center/scale only the HVGs
with clipping at 10, PCA(50), neighbors(15 neighbors, 30 PCs), Leiden(resolution=0.5),
UMAP(min_dist=0.3). Random seed=42. These choices are not asserted to reproduce the
paper's exact embeddings. Treatment labels enter plots/tables only, never the fit.
No cell-cycle regression, batch correction or cell-type annotation is applied."""),
code("import json\nCONFIG = json.loads("+repr(json.dumps(config))+")\nprint(json.dumps(CONFIG,indent=2))"),
md("""## 4. Load the reusable implementation
Generated from src/qc_preprocessing.py so this notebook works without source uploads.
The explicit Scanpy calls are visible in rna_qc(), sparse_log_and_scale(), and visualize_rna().
Custom code handles TSV loading, pairing, orchestration and output validation.
Input checksums must match nb00. Full count matrices remain sparse. Only the approximately
2,000-gene visualization subset becomes dense for centered scaling.

Saved RNA files include counts/log1p representations, X_pca, X_umap, Leiden labels,
PCA loadings and neighborhood graphs. A separate *_RNA_visualization.h5ad contains the
scaled HVG input plus raw-count and normalized-log layers.
ATAC retains all supplied peaks and the identical RNA-selected cells.
Re-running replaces derived files atomically per file; raw archives remain unchanged."""),
code((ROOT/"src/qc_preprocessing.py").read_text()),
code("""sc.settings.verbosity = 2
report = {"reference_doi": CONFIG["reference_doi"], "config": CONFIG,
          "run_at_utc": datetime.now(timezone.utc).isoformat(), "cell_lines": {}}
from IPython.display import Image, display
""")
]
for step, line in enumerate(("MCF7","T47D"),start=5):
    cells += [
md(f"""## {step}. {line}: QC → RNA preprocessing → PCA → neighbors → Leiden → UMAP
Checksums → load counts/metadata → Scanpy gene filter → Scanpy cell QC/filter →
mitochondrial cutoff → preserve counts/log1p → normalized RNA HVGs →
center/scale HVGs → PCA → neighbors → Leiden → UMAP → same cell mask for ATAC.
This cell streams the original TSVs and may take several minutes in Colab."""),
code(f'report["cell_lines"]["{line}"] = preprocess_line(BASE, "{line}", CONFIG)'),
code(f"""print(json.dumps(report["cell_lines"]["{line}"],indent=2))
for suffix in ("paper_qc","pca_variance","rna_pca","rna_umap"):
    display(Image(filename=str(RESULTS_DIR/"figures"/f"{line}_{{suffix}}.png")))
display(pd.read_csv(RESULTS_DIR/"tables"/"{line}_cluster_treatment_counts.tsv",sep="\\t"))
""")
]
cells += [
md("## 7. Save the combined report"),
code("""(RESULTS_DIR/"tables"/"paper_qc_summary.json").write_text(json.dumps(report,indent=2)+"\\n")
summary = pd.DataFrame({
    line: {"cells_before":v["cells_before"],"cells_after":v["cells_after"],
           "genes_after":v["genes_after"],"atac_peaks_after":v["atac_peaks_after"],
           "hvgs":v["visualization"]["n_hvgs"],"clusters":v["visualization"]["n_clusters"]}
    for line,v in report["cell_lines"].items()}).T
display(summary)
"""),
md("""## 8. Read the plots cautiously
- Compare treatment, clusters, RNA depth and mitochondrial fraction on the same embedding.
- Cluster numbers are arbitrary and are not comparable across independently fitted cell lines.
- UMAP separation is not statistical evidence of treatment effects or prediction accuracy.
- This fit uses the full retained cohort for description. Future prediction must refit
  feature selection, scaling and embeddings on training data.
- All 206,167 ATAC peaks are retained. No invented fragment/TSS/FRiP/doublet cutoffs.
- The scMKL paper states 24-hour treatment; the original study states 48 hours.
  Metadata do not resolve this. Treatment labels remain unchanged.
- This uses the stated QC thresholds plus exploratory visualization, not an exact
  reproduction of the paper's post-QC cells or figures.

See QC_PREPROCESSING.md for provenance and local validation. Review these outputs before
differential expression or any cell-state annotation.""")
]
for i,cell in enumerate(cells):
    cell["id"] = f"nb02-{i:03d}"
nb = {"nbformat":4,"nbformat_minor":5,"metadata":{
    "kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},
    "language_info":{"name":"python","version":"3.11"},
    "colab":{"name":"nb02_qc_preprocessing.ipynb","provenance":[]}},
    "cells":cells}
(ROOT/"notebooks/nb02_qc_preprocessing.ipynb").write_text(json.dumps(nb,indent=2)+"\n")
print("Generated Scanpy nb02.")
