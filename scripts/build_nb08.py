"""Generate nb08 without executing its cells."""
from pathlib import Path
import json
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/"src/unimodal_mlp.py").read_text()
config=json.loads((ROOT/"configs/unimodal_mlp.json").read_text())
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
cells=[
md("""# nb08 — RNA-only and ATAC-only MLP baselines

Train four independent models: **RNA-only and ATAC-only for MCF7**, then **RNA-only and ATAC-only
for T47D**. Predict treatment: **E2 = 1; Vehicle = 0**.
**nb09 will integrate RNA and ATAC using these saved cell splits.**

**Prepared without execution. You will run this notebook in Colab.**
Choose a GPU runtime (e.g. T4), then run all cells. CPU works but is slower.
The implementation is embedded: no source files or repository installation are needed.

Required nb02 files in your Drive project's data/processed/ directory:
- MCF7_RNA_paper_qc.h5ad and MCF7_ATAC_paired_paper_qc.h5ad
- T47D_RNA_paper_qc.h5ad and T47D_ATAC_paired_paper_qc.h5ad

Default project: /content/drive/MyDrive/breast-cancer-multiome-estrogen.
Change BASE below if your folder differs.
"""),
md("""## Design fixed before training

- Stratified splits, seed 42: approximately **72% train / 8% validation / 20% test** per cell line.
  Both modalities use identical cell IDs and labels in every split.
- Up to **5,000 variable features selected only on training cells**, detected in at least three
  training cells. No nb03–nb07 DEG, DA, pathways, motifs, PCA or LSI enter prediction.
- RNA: raw counts layer → log1p → training log-space variance selection → training centering/scaling.
  ATAC: raw peak counts → binary (>0) → training binary variance selection; no scaling or TF–IDF.
  Sparse inputs are densified only one minibatch at a time.
- Five 512-unit ReLU hidden layers, dropout 0.1 after each; one output logit.
  Adam learning rate 0.001, binary cross-entropy, batch size 256, maximum 100 epochs.
- Validation-loss early stopping: patience 5, minimum improvement 0.0001.
  Restore selected weights, then evaluate test cells. Threshold fixed at 0.5.
- Main metric **AUROC**; also average precision, accuracy, balanced accuracy, E2 F1/precision/recall,
  log loss, Brier score and confusion counts. A constant training-prevalence dummy baseline is saved.

### Relationship to the paper
Architecture/dropout follow the authors' [MLP implementation](
https://github.com/ohsu-cedar-comp-hub/scMKL_paper/blob/84d68eef6a0985ba757d4b8d4158bc765f48c1ac/Performance_Comparisons/mlp_classification.py)
for [Kupp et al. (2025)](https://doi.org/10.1038/s42003-025-08533-7).
This is a **paper-informed baseline, not an exact reproduction**:
PyTorch replaces Keras; one fixed split/batch size replaces repeated splits and four-fold batch-size
tuning. RNA selection/scaling use consistent log-space training statistics; the paper code uses
raw-count variance/SD. Early-stopping patience and best-weight restoration also differ.

### Interpretation
Available metadata do not identify independent biological replicates; treatment may be confounded
with library effects. These are held-out-cell scores within each cell line, not evidence of patient
or replicate generalization. Inputs inherit nb02 whole-cohort QC/gene filtering; learned model
preprocessing is refitted on training cells only. One split does not estimate split variability.
Keep the test set out of all later tuning; nb09 should select models on validation.
"""),
code("""# Keep Colab's GPU-enabled PyTorch; install only missing dependencies.
import importlib.util
import subprocess
import sys
packages = {"anndata":"anndata>=0.11,<0.13","h5py":"h5py","numpy":"numpy",
            "pandas":"pandas","scipy":"scipy","sklearn":"scikit-learn",
            "matplotlib":"matplotlib","torch":"torch"}
missing = [spec for module,spec in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable,"-m","pip","install","-q",*missing])
"""),
code("""from pathlib import Path
from datetime import datetime, timezone
import os
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/breast-cancer-multiome-estrogen")
else:
    current = Path.cwd()
    BASE = Path(os.environ.get("MULTIOME_PROJECT_ROOT",
                str(current.parent if current.name == "notebooks" else current)))
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
required = [BASE/"data"/"processed"/f"{line}_{suffix}.h5ad"
            for line in ("MCF7","T47D")
            for suffix in ("RNA_paper_qc","ATAC_paired_paper_qc")]
missing_files = [str(p) for p in required if not p.is_file()]
if missing_files:
    raise FileNotFoundError("Missing nb02 files:\\n"+"\\n".join(missing_files))
print("Project:",BASE)
print("Run:",RUN_ID)
"""),
md("## Configuration\nChange settings before starting the run; the same settings apply to both modalities."),
code("CONFIG = "+repr(config)+"\nCONFIG"),
md("""## Embedded implementation
The next cell defines functions only. The following setup cell saves split manifests,
environment versions and fingerprints of the four input files. Hashing can take time on Drive.
"""),
code(source),
code("""from IPython.display import display
OUTPUT, MODELS, MANIFESTS, split_counts = prepare(BASE,CONFIG,RUN_ID)
display(split_counts)
print("Results:",OUTPUT)
print("Models:",MODELS)
print("Device:","GPU" if torch.cuda.is_available() else "CPU (slower)")
""")
]
cells[-1].source+="\n_ = (OUTPUT/'unimodal_mlp_source.py').write_text("+repr(source)+")\n"
for line in config["cell_lines"]:
    for modality in ("RNA","ATAC"):
        cells.extend([
            md(f"## {line} — {modality}-only MLP\nRun once; the other modality is not an input."),
            code(f'display(train_one(BASE,OUTPUT,MODELS,MANIFESTS["{line}"],\n'
                 f'                  "{line}","{modality}",CONFIG))')])
cells.extend([
md("""## Compare and export all four models
The export requires all four training cells to finish and checks that test cells/labels are paired.

The ZIP contains metrics, settings, versions, source, input hashes, split manifests, selected
features, validation/test predictions, histories and diagnostic plots.
Weights and preprocessing arrays remain under data/processed/nb08/RUN_ID/.

**Send back the results ZIP**, or metrics_summary.csv and the diagnostic figures.
Preserve this run for nb09, which must reuse its exact split manifests.
"""),
code("""summary, RESULTS_ZIP = summarize(BASE,OUTPUT,CONFIG)
display(summary[["cell_line","modality","n_features","best_epoch","AUROC",
                 "average_precision","balanced_accuracy","F1_E2","log_loss"]])
print("Share this archive:",RESULTS_ZIP)
"""),
md("""### Optional download
The ZIP is already saved in Drive. Set DOWNLOAD_RESULTS to True to download through Colab.
Completed models cannot be overwritten in the same run. Changed settings need a new RUN_ID;
report reruns separately. Interrupted training epochs are not resumed automatically.
"""),
code("""DOWNLOAD_RESULTS = False
if DOWNLOAD_RESULTS and IN_COLAB:
    from google.colab import files
    files.download(str(RESULTS_ZIP))
""")])
nb=nbf.v4.new_notebook(cells=cells,metadata={
    "kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},
    "language_info":{"name":"python","version":"3.11"},
    "colab":{"name":"nb08_unimodal_mlp.ipynb","provenance":[]}})
nbf.write(nb,ROOT/"notebooks/nb08_unimodal_mlp.ipynb")
print("Built nb08. No notebook cells executed.")
