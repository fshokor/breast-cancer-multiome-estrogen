"""Build standalone nb09. Does not execute notebook cells."""
from pathlib import Path
import json
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/"src/scmkl_style.py").read_text()
config=json.loads((ROOT/"configs/scmkl_style.json").read_text())
hallmark=json.loads((ROOT/"configs/hallmark_v7.json").read_text())
refs=json.loads((ROOT/"configs/motif_references.json").read_text())
scan=json.loads((ROOT/"configs/motif_analysis.json").read_text())
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
cells=[
md("""# nb09 — scMKL-style RNA + ATAC integration

**Prepared for you to run in Colab. This notebook has not been executed.**

Compare **RNA-only, ATAC-only, and RNA + ATAC** grouped-kernel models separately in MCF7 and T47D,
then compare them with your nb08 MLP baselines. Reuse the exact nb08 training, validation and
test cells, input data and training-selected features. E2 is always the positive class.

This is an **scMKL-style implementation**, not the official scMKL package and not an exact
reproduction of the published benchmark. It retains the central design: biological feature
groups, Gaussian random Fourier feature maps, and group-lasso regression for binary prediction.
See the method and differences below.

**Colab runtime:** the celer solver and motif scanning use CPU. A GPU is not required and will
not accelerate this solver. A high-RAM runtime is helpful; do not keep other large notebooks
running in the same session. Model construction uses disk-backed arrays and runs one cell line
at a time. Dependencies, definitions and references are embedded; no repository upload is needed.

**Inputs on Drive**
- The four nb02 RNA/paired-ATAC QC H5AD files.
- The complete nb08 results folder for run **20261011T103750_408739Z**, including its config,
  split manifests, handoff, input fingerprints, selected features and test predictions.
- nb07 genome/JASPAR/blacklist references, if already present. Missing references download
  automatically with checksum verification; the hg38 genome is approximately 835 MB.

If you only have the nb08 results ZIP, extract its contents into
results/tables/nb08/20261011T103750_408739Z/ under the Drive project before starting.
Set NB08_RUN_ID explicitly below if you intend to use a different completed run.
"""),
md("""## Model and comparison plan

1. Verify nb08 input fingerprints, paired cell identities and immutable split manifests.
   Read its actual settings, including your MLP learning rate of 0.00001; do not retrain the MLP.
2. Recompute nb08's training-only selection of up to 5,000 features and require an exact
   match to its saved feature IDs. RNA uses log1p counts and training centering/scaling;
   ATAC uses binary peak accessibility.
3. **RNA kernels:** one per usable frozen MSigDB Hallmark v7 gene set.
   **ATAC kernels:** one per usable JASPAR2024 motif's peak set. Scan the training-selected peaks
   afresh; do not use DEG, DA, GSEA or motif-enrichment significance to choose groups.
4. Keep groups with at least five features. Collapse groups with identical feature membership;
   save aliases and all coverage exclusions. Blacklisted, nonstandard or ambiguous ATAC sequences
   are excluded from motif groups.
5. For each group, estimate a Gaussian bandwidth using the median positive pairwise distance
   among up to 256 **training cells**. Use 32 random frequencies, giving 64 cosine/sine features.
   Center and normalize each block using training cells to give comparable kernel scales.
6. Fit group lasso to targets Vehicle = -1, E2 = +1, separately for RNA, ATAC and joint input.
   Search five regularization ratios using **validation AUROC**, then validation average precision,
   fewer groups and stronger regularization as tie-breakers. Only converged fits are eligible.
7. Keep the chosen training fit without refitting on validation. Evaluate test cells only after
   selecting regularization. Report raw-score AUROC/AP and threshold-zero classification metrics.

The Gaussian kernel is k(x,z) = exp(-||x-z||² / (2σ²)). Each group's feature map is
[cos(xW), sin(xW)] / sqrt(D), with W sampled from N(0, 1/σ²).

The objective is squared error / (2n) + alpha × sum of coefficient-block L2 norms.
Alpha is a fraction of the training-only group-lasso alpha maximum.
**There is no MLP learning rate or neural-network epoch training in this notebook.**
Uncalibrated regression scores are saved as decision scores, not probabilities; therefore
log loss/Brier scores are not compared with nb08.
"""),
md("""## Provenance, adaptations and limits

The [authors' scMKL source](https://github.com/ohsu-cedar-comp-hub/scMKL/tree/d6448b73895ab62b31f78213f6e4fed910bce4c4)
uses random Fourier features and
[celer group lasso](https://mathurinm.github.io/celer/generated/celer.GroupLasso.html).
The [paper](https://doi.org/10.1038/s42003-025-08533-7) provides the scientific reference.

This notebook implements that core approach independently, with explicit adaptations:
nb08's existing feature set/preprocessing; fixed 32-frequency Gaussian approximation; positive-distance
median bandwidths; training-centered, unit-average-energy blocks; a group-norm alpha maximum;
validation selection rather than repeated nested cross-validation; and JASPAR2024 motif annotation.
It does not claim numerical equivalence with scMKL or reproduce its published scores.

Peak coordinate origin remains unresolved. A motif is admitted only when present under **both**
0-based half-open and 1-based closed interpretations; all boundary-sensitive hits are counted.
This conservative choice does not resolve the coordinate convention or demonstrate TF binding.

Group coefficient norms describe predictive contributions within correlated, overlapping groups.
They are not pathway activation scores, causal effects, or percentages of biological contribution.
Identical membership groups share a representative; no particular TF within a motif family is established.

The comparison is exploratory: biological replicate IDs are unavailable and library effects may
track treatment. nb08 test results have already been inspected. If the learning rate was chosen using
test performance, those cells are no longer an untouched final evaluation set; preserve this comparison
for continuity, but reserve independent data for a confirmatory performance claim.
The grouped-kernel controls help separate integration benefit from a change in model family.
"""),
code("""# Install missing runtime packages. No notebook model is fitted in this cell.
import importlib.util
import subprocess
import sys
packages = {
    "anndata":"anndata>=0.11,<0.13","h5py":"h5py","numpy":"numpy","pandas":"pandas",
    "scipy":"scipy","sklearn":"scikit-learn","matplotlib":"matplotlib",
    "celer":"celer>=0.7.4,<0.8","MOODS":"MOODS-python==1.9.4.1","py2bit":"py2bit==0.3.3",
}
missing=[spec for module,spec in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable,"-m","pip","install","-q",*missing])
"""),
code("""from pathlib import Path
from datetime import datetime, timezone
import os
try:
    import google.colab
    IN_COLAB=True
except ImportError:
    IN_COLAB=False
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE=Path("/content/drive/MyDrive/breast-cancer-multiome-estrogen")
else:
    here=Path.cwd()
    BASE=Path(os.environ.get("MULTIOME_PROJECT_ROOT",
              str(here.parent if here.name=="notebooks" else here)))

NB08_RUN_ID="20261011T103750_408739Z"
RUN_ID=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
print("Project:",BASE)
print("nb08 comparison run:",NB08_RUN_ID)
print("New nb09 run:",RUN_ID)
"""),
md("""## Settings and frozen references
Choose settings before setup. Do not change them after looking at test results.
The dictionaries below embed the existing project references; no label-based enrichment filtering
is applied. The number of retained groups is determined by training-selected feature coverage.
"""),
code("CONFIG = "+repr(config)+"\nCONFIG"),
code("HALLMARK = "+repr(hallmark)+"\nREFERENCES = "+repr(refs)+"\nSCAN_CONFIG = "+repr(scan)),
md("""## Implementation
Defines functions only. Runtime input verification, sequence scanning, kernel construction and fitting
are invoked explicitly in the cells below.
"""),
code(source),
md("""## Verify the baseline and prepare the run
This checks exact data checksums and cell splits against nb08 and records its actual configuration.
It will stop on mismatches rather than silently create a different comparison.
"""),
code("""from IPython.display import display
BASELINE, OUTPUT, ARTIFACTS, MANIFESTS, NB08_CONFIG = prepare_nb09(
    BASE,NB08_RUN_ID,RUN_ID,CONFIG,HALLMARK,REFERENCES,SCAN_CONFIG)
print("Results:",OUTPUT)
print("Models and feature maps:",ARTIFACTS)
print("Actual nb08 learning rate:",NB08_CONFIG["learning_rate"])
display(pd.concat([m.groupby(["cell_line","split","treatment"]).size().rename("cells").reset_index()
                   for m in MANIFESTS.values()],ignore_index=True))
"""),
md("""## Verify sequence resources
Reuse the existing reference files where possible. The hg38 genome is downloaded only if missing.
Reference checksums must match the frozen project versions.
"""),
code("ensure_references(BASE,REFERENCES,OUTPUT)"),
]
cells[11].source += "\n_ = (OUTPUT/'scmkl_style_source.py').write_text("+repr(source)+")\n"
for line in config["cell_lines"]:
    cells.extend([
        md(f"""## {line}: build biological feature groups and kernels
This step can take time: it annotates selected ATAC peaks and constructs one random-feature block per
retained pathway/motif group. It uses no test labels to fit preprocessing or bandwidths.
"""),
        code(f'{line}_PATHS, {line}_TABLES = construct_line(\n'
             f'    BASE,BASELINE,OUTPUT,ARTIFACTS,MANIFESTS["{line}"],"{line}",\n'
             f'    NB08_CONFIG,CONFIG,HALLMARK,REFERENCES,SCAN_CONFIG)\n'
             f'display(pd.DataFrame([{{"modality":m,"kernels":len(t)}} for m,t in {line}_TABLES.items()]))'),
        md(f"""## {line}: train RNA, ATAC and integrated models
Selection uses validation only. Test evaluation follows selection for each input mode.
Candidate convergence and selection metrics are saved; failed convergence is not silently accepted.
"""),
        code(f'display(fit_group_models({line}_PATHS,{line}_TABLES,MANIFESTS["{line}"],\n'
             f'                         "{line}",OUTPUT,ARTIFACTS,CONFIG))')
    ])
cells.extend([
md("""## Compare with nb08 and export
The table contains five models per cell line: two nb08 MLP baselines, two grouped-kernel controls,
and joint RNA + ATAC. Test IDs/labels must match exactly. Difference tables report joint-model
AUROC/AP minus each reference model; these are descriptive differences, not significance tests.

**Send back the generated results ZIP.** It includes selected regularization, solver status,
group coverage/weights, predictions, plots, input/split checksums and settings.
Large kernel arrays, random maps and coefficient checkpoints remain in data/processed/nb09/RUN_ID/.

A higher integrated score is not guaranteed, especially given nb08's near-ceiling RNA AUROC.
Use the controls and validation/test gap to interpret whether integration helps.
"""),
code("""comparison, RESULTS_ZIP = export_comparison(BASE,BASELINE,OUTPUT,MANIFESTS,CONFIG)
display(comparison[["cell_line","model","modality","AUROC","average_precision",
                    "balanced_accuracy","F1_E2"]])
print("Results archive:",RESULTS_ZIP)
"""),
md("""### Optional download
The ZIP is already saved on Drive. Turn on the switch below to download it through Colab.
A changed configuration requires a new run. Kernel arrays and models are not included in the ZIP.
"""),
code("""DOWNLOAD_RESULTS=False
if DOWNLOAD_RESULTS and IN_COLAB:
    from google.colab import files
    files.download(str(RESULTS_ZIP))
""")
])
nb=nbf.v4.new_notebook(cells=cells,metadata={
    "kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},
    "language_info":{"name":"python","version":"3.11"},
    "colab":{"name":"nb09_scmkl_style_integration.ipynb","provenance":[]}})
nbf.write(nb,ROOT/"notebooks/nb09_scmkl_style_integration.ipynb")
print("Built nb09; no notebook cells executed.")
