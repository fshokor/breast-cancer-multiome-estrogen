"""Generate nb10 without running notebook cells."""
from pathlib import Path
import nbformat as nbf
import textwrap
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/"src/model_comparison.py").read_text()
transfer_source=(ROOT/"src/cross_line_transfer.py").read_text()
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
cells=[
md("""# nb10 — Model comparison and biological interpretation

**Prepared in advance and intentionally unexecuted. Run this after nb09 finishes.**

This notebook reviews saved outputs and adds **frozen cross-cell-line treatment prediction**.\nIt never trains or tunes models. Transfer loads saved checkpoints and transforms raw target inputs\nusing source-model preprocessing. Compare nb08 MLPs, nb09 RNA-only/integrated kernels,\nand both completed nb11 pathway-model versions.

Questions:
- Does integration improve AUROC, average precision and classification errors?
- Does it improve over the saved RNA and ATAC MLP baselines?
- Were the selected fits converged, and how do validation/test scores compare?
- Which pathway and motif groups were selected? How do they relate to the earlier biological findings?

No conclusions about nb09 are assumed. Tables, figures and the report are created only when you run it.

### Required inputs
The complete nb09 results directory under results/tables/nb09/RUN_ID/.
It already includes nb08 test predictions, configuration and the split handoff, so model\nweights and raw inputs are unnecessary for the original review sections. The transfer section\nrequires full nb08/nb09/nb11 checkpoints, nb09 random-feature maps (including referenced older run\nfolders), nb11 structures/peak-gene links, all four nb02 H5AD files and the original results folders.\nResults ZIPs alone do not contain model checkpoints. If using the nb09 results ZIP, extract its contents into
that run directory first.

### Optional biological context
Existing nb04 Hallmark GSEA, nb06 pathway summaries and nb07 motif enrichment tables.
Missing context is reported explicitly and does not stop the performance comparison.
The original review is lightweight CPU work. Frozen neural inference can use a GPU; CPU also\nworks. Transfer needs more RAM/time. No sequence scan or reference download is required.
"""),
md("""## How to interpret the review

Use identical test cells, E2-positive labels and saved decision thresholds:
0.5 for MLP probabilities and 0 for group-lasso decision scores. Recompute metrics from the saved
predictions and verify them against nb09's summary before comparing models.

AUROC and average precision use continuous scores. Kernel scores are uncalibrated, so this notebook
does not compare their log loss/Brier scores with MLP probabilities. Error correction is paired by
cell; it reflects the fixed thresholds and is not a significance test.

Validation/test gaps and solver checks concern the integrated model for each cell line. nb09's portable
export does not include the MLP validation predictions, so this notebook does not invent them.

Group norms are **unsigned predictive weights**, not pathway activity, TF binding or causal effects.
Keep group-coverage exclusions and motif aliases; two names may represent the same kernel.
Cross-line overlap describes selection in two separately fitted models, not transfer performance or
repeated-split stability. Absence from a model's eligible groups is distinguished from a zero weight.

Earlier GSEA/ATAC/motif results use the same cohort and are descriptive context, not independent
validation. Both positive/negative NES and all motif directions, matching-QC flags and coordinate
scenarios are retained. Do not select the most favorable enrichment result.

These are single-split, cell-level results. Biological replicate IDs are unavailable, and
library/treatment confounding remains possible. Test results are already known; any test-guided
tuning needs independent confirmation. No confidence intervals or statistical significance claims
are produced here.
"""),
code("""import importlib.util
import subprocess
import sys
packages={"numpy":"numpy","pandas":"pandas","matplotlib":"matplotlib","sklearn":"scikit-learn"}
missing=[spec for name,spec in packages.items() if importlib.util.find_spec(name) is None]
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

# Leave blank only if exactly one completed nb09 results folder exists.
# If several runs exist, set the exact folder name to review.
NB09_RUN_ID="20261012T114858_335840Z"
RUN_ID=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
print("Project:",BASE)
print("Available nb09 folders:",[p.name for p in (BASE/"results"/"tables"/"nb09").glob("*") if p.is_dir()])
"""),
md("## Embedded review implementation\nDefines functions only; no model code or fitting library is imported."),
code(source),
md("""## Select and verify the completed run
An incomplete or ambiguous input selection stops with a clear message.
All outputs go into a new nb10 directory; earlier results are preserved.
"""),
code("""from IPython.display import display
SOURCE, OUTPUT, INPUTS, HANDOFF = prepare_review(BASE,RUN_ID,NB09_RUN_ID)
print("Output directory:",OUTPUT)
"""),
md("""## Verify predictions, performance and model selection
Check split checksums, exact cells/labels, saved classification thresholds, metric consistency,
fit convergence and the fixed-alpha validation record. Reorder predictions by cell ID
when needed; never compare unpaired rows.
"""),
code("""METRICS, GAPS, DIFFERENCES, PREDICTIONS = review_predictions(SOURCE,OUTPUT,INPUTS,HANDOFF)
display(METRICS[["cell_line","model","modality","AUROC","average_precision",
                 "balanced_accuracy","F1_E2","errors"]])
display(GAPS)
display(DIFFERENCES)
"""),
md("""## Performance figures
ROC and precision–recall curves compare the four models. Generalization panels show each line’s integrated model. Positive integration AUROC/AP differences mean a higher point estimate;
they do not establish statistical significance. Paired error tables identify corrections and new errors.
"""),
code("performance_plots(METRICS,GAPS,PREDICTIONS,OUTPUT)"),
md("""## Selected pathways and motifs
Review all model weights, inspect integrated-model selection, and retain aliases for identical kernel
memberships. Plots show representative groups only, avoiding duplicate motif labels.
Coefficient magnitudes are not percentages of biological contribution or direct evidence of regulation.
"""),
code("""WEIGHTS, ALIASES, CROSS_LINE = review_groups(SOURCE,OUTPUT,INPUTS)
display(WEIGHTS[(WEIGHTS.input_mode=="RNA_ATAC") & WEIGHTS.selected]
        .sort_values(["cell_line","modality","coefficient_group_norm"],ascending=[True,True,False]))
display(CROSS_LINE.head(20))
"""),
md("""## Relate predictive groups to earlier biology
Optional context is joined by exact Hallmark names and JASPAR motif IDs.
All scenarios are preserved in exported tables. A selected group with no matching context is not
evidence of biological absence; inspect coverage and resource versions. Earlier analyses do not
feed back into model fitting or selection.
"""),
code("""CONTEXT_STATUS = biological_context(BASE,OUTPUT,INPUTS,ALIASES)
display(CONTEXT_STATUS)
print("Context tables saved under:",OUTPUT)
"""),
md("""## Cross-cell-line treatment prediction — frozen models
**Question:** can an MCF7-trained model predict E2 versus Vehicle in T47D, and vice versa?
This predicts treatment across cell lines, not cell-line identity.

Default: **16 source→target cases**. nb08 RNA/ATAC; nb09 RNA/RNA+ATAC; nb11 v1 RNA+ATAC;
nb11 v2 RNA/ATAC/RNA+ATAC, each in both directions. Collapsed v2 ATAC models remain negative
controls. Explicit run IDs are below; remove a historical run only if deliberately excluding it.

Use the opposite line's existing nb08 **test cells only**, identical across model families.
Keep source feature order, centering/scaling, kernel frequencies/normalization, pathway masks,
peak-gene assignments and depth reference. No target statistics fit a transform, and no target
results select checkpoints or thresholds. Target-native scores use the separately trained model
from the same family/run on exactly those target test cells.

Each checkpoint must first reproduce its saved source-line test predictions within absolute/relative
tolerance 1e-4; otherwise transfer is withheld. Missing target RNA features use the source transformed
mean, with a predeclared 5% coverage limit and a full missing-feature list. Required ATAC peaks must
exist, and nb11 requires identical assay peak universes. No motif scan or target-specific mapping.

tqdm tracks evaluations and active kernels. Missing artifacts and verification failures are reported
separately, never assigned AUROC 0. The first pass hashes raw inputs and checkpoints and may take time.
Shared libraries, prior test inspection and absent biological replicates limit generalization claims.
"""),
code("""RUN_CROSS_LINE_TRANSFER = True
NB08_TRANSFER_RUN_ID = HANDOFF["run_id"]
NB11_TRANSFER_RUNS = [
    {"run_id": "20261012T173612_419824Z", "version": 1},
    {"run_id": "20261012T180159_920277Z", "version": 2},
]
MAX_MISSING_RNA_FRACTION = 0.05  # Fixed before inspecting transfer results.
if RUN_CROSS_LINE_TRANSFER:
    import importlib.util, subprocess, sys
    packages = {"torch":"torch>=2.6,<3", "anndata":"anndata>=0.10,<0.13",
                "h5py":"h5py>=3.10,<4", "scipy":"scipy>=1.11,<2", "tqdm":"tqdm>=4.66,<5"}
    missing = [spec for module,spec in packages.items() if importlib.util.find_spec(module) is None]
    if missing: subprocess.check_call([sys.executable,"-m","pip","install","-q",*missing])
"""),
md("### Embedded frozen-inference implementation\nArchitecture definitions match the completed notebook versions. No model fitting is implemented."),
code("if RUN_CROSS_LINE_TRANSFER:\n"+textwrap.indent(transfer_source,"    ")),
code("""if RUN_CROSS_LINE_TRANSFER:
    _ = (OUTPUT/"cross_line_transfer_source.py").write_text(TRANSFER_SOURCE_TEXT)
    TRANSFER_RESULTS = run_cross_line_transfer(
        BASE, OUTPUT, NB08_TRANSFER_RUN_ID, SOURCE.name, NB11_TRANSFER_RUNS,
        max_missing_fraction=MAX_MISSING_RNA_FRACTION)
    display(TRANSFER_RESULTS)
    failed = TRANSFER_RESULTS[TRANSFER_RESULTS["status"].ne("complete")]
    if len(failed):
        print("Some evaluations are incomplete. Restore the listed artifacts or resolve the diagnostic.")
        display(failed[["model","source","target","error"]])
else:
    print("Cross-cell-line transfer disabled; saved-output review only.")
""".replace("TRANSFER_SOURCE_TEXT",repr(transfer_source))),
md("""## Generate the review report and export
The report summarizes verified metrics, integration differences and selected groups, with limitations.
It does not automatically recommend tuning a model based on its test score.

**Send back the results ZIP** or comparison_report.md with the figures.
Cross-cell-line results, diagnostics and predictions are included in the archive when enabled.\nPredefined repeated splits and independent biological validation remain future work.
"""),
code("""RESULTS_ZIP = export_review(BASE,SOURCE,OUTPUT,INPUTS,METRICS,GAPS,DIFFERENCES,WEIGHTS,CONTEXT_STATUS)
from IPython.display import Markdown
display(Markdown((OUTPUT/"comparison_report.md").read_text()))
"""),
md("### Optional download\nThe archive is already saved in Drive. Set the switch below to download it."),
code("""DOWNLOAD_RESULTS=False
if DOWNLOAD_RESULTS and IN_COLAB:
    from google.colab import files
    files.download(str(RESULTS_ZIP))
""")
]
cells[7].source+="\n_ = (OUTPUT/'model_comparison_source.py').write_text("+repr(source)+")\n"
nb=nbf.v4.new_notebook(cells=cells,metadata={
    "kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},
    "language_info":{"name":"python","version":"3.11"},
    "colab":{"name":"nb10_model_comparison_interpretation.ipynb","provenance":[]}})
nbf.write(nb,ROOT/"notebooks/nb10_model_comparison_interpretation.ipynb")
print("Built nb10; no notebook cells executed.")
