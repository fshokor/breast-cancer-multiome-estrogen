# Pathway-structured multimodal MLP (nb11)

Notebook: [nb11](notebooks/nb11_pathway_multimodal_mlp.ipynb). Prepared without execution.
Source: src/pathway_mlp.py. Settings: configs/pathway_mlp.json.
Regenerate using scripts/build_nb11.py; statically validate using scripts/validate_nb11_static.py.

## Running and resuming

Use a Colab GPU and the same Google Drive project folder as nb08.
Supply all four nb02 processed H5AD files, the complete nb08 results directory and frozen
Hallmark/reference specifications embedded in the notebook. RefSeq and blacklist files are
downloaded if missing and accepted only when their hashes match the original frozen resources.
If a remote reference changed, copy the original reference file from earlier notebooks.

Check BASE and NB08_RUN_ID. The default is 20261011T103750_408739Z.
The printed nb11 RUN_ID identifies both results/tables/nb11/RUN_ID and
data/processed/nb11/RUN_ID. Put that value in RESUME_RUN_ID to continue after a disconnect,
then run from setup. Settings/source/reference/split signatures must match.

Completed preprocessing caches are reused. Interrupted preprocessing restarts that cell line.
Atomic last.pt saves model, optimizer, best model, history and RNG state after every completed
epoch. An interrupted epoch restarts from that checkpoint. No checkpoint exists before the first
epoch completes. Keep both results and artifacts folders on persistent Drive.
Changing devices/software can affect exact reproducibility even with restored random states.

## Model

Two independent models: MCF7 and T47D. Within each, paired RNA and ATAC cells use exactly nb08's
train/validation/test manifests. Raw input hashes and saved baseline predictions are verified.

Frozen Hallmark membership defines masked gene-to-pathway connectivity. Each modality encodes
four values per supported pathway. Pathway-specific fusion maps the eight RNA/ATAC values to
four fused values. A clipped noisy gate per pathway encourages sparsity via expected-L0
regularization. Evaluation uses deterministic gates. A 32-unit ReLU/dropout head predicts E2.

Masks are multiplied into weights on every forward pass. Layer normalization has no affine
parameters; the masked/fusion layers have no bias. Dense PyTorch matrix operations implement
the masks; this is biologically sparse connectivity, not a specialized sparse tensor backend.
Gates are not identifiable biological importance estimates; downstream weights can compensate
for gate magnitude. Validate dependence with ablations and attribution.

One fixed configuration per line, no search. AdamW model LR 0.0003, gate LR 0.003, batch 128,
max 100 epochs, min 20, patience 10. Expected-L0 penalty 0.001 ramps over epochs 5–15.
Validation BCE chooses checkpoints from epoch 15 onward. Test results never select checkpoints.
tqdm displays epoch and batch progress, AUROC, patience, best epoch and elapsed time.
ETA assumes the maximum epoch count; early stopping may finish sooner.
These are project defaults, not the paper's best parameters.

## Features and interpretation

RNA is log1p raw counts. ATAC is binary peak accessibility linked by fixed transcript-union
promoter (2 kb upstream/500 bp downstream of TSS) plus gene-body overlap. This overrides the
nearest-TSS analysis policy in nb06 while reusing its frozen reference annotation. Only links
present under both coordinate conventions are retained; blacklist and nonstandard intervals
are excluded. Multi-gene peaks receive fractional weights.

ATAC score before log1p:
sum(binary accessibility × fractional assignment) / sum(assignment weights)
× median training-cell accessible-peak count / cell accessible-peak count.
This custom coverage/depth-adjusted accessibility proxy is not measured expression or an exact
ArchR score. Distal enhancer links are not modeled. Peak and assignment tables are retained.

Training-only detection/variance filtering and standardization are applied in both modalities.
All 50 Hallmark sets are attempted; only pathways with >=3 eligible genes per modality survive.
No nb06 significance, DEG, GSEA or motif ranking selects predictive features.
The new feature pipeline differs from nb08, so comparisons do not isolate architecture effects.

Validation ablations remove each fused pathway, each pathway's RNA/ATAC branch, and each entire
modality branch. Positive AUROC drop means dependence of this fitted model, not causality.
Conditional integrated gradients target up to three pathways with positive validation ablation
effects, using <=16 validation cells per class and 32 integration intervals. Other pathway
latents stay fixed. Zero standardized input represents the training mean. Completeness errors
are exported and must be checked before interpreting gene rankings.
Candidate regions linked to attributed ATAC genes are annotations, not peak-level attribution.

Results include coverage, features, peak links, gates, epoch history, validation/test predictions,
baseline comparison, ablations, gene attribution, completeness errors and figures. The results ZIP
excludes checkpoints and prepared matrices, which stay in the artifact folder.

## Limits

No model or notebook cells were executed while authoring. Static syntax/schema checks cannot
establish runtime performance, memory use, convergence or biological validity.
Full-cohort QC is inherited; cell-level splits do not establish replicate/patient generalization.
Library/treatment confounding, prior test inspection, overlapping pathways, provisional peak
mapping and off-distribution ablations limit claims. Validation interpretation is exploratory.
Compare with both positive and negative nb06/GSEA findings as descriptive context, not independent
validation. Gate magnitudes do not indicate pathway activation or repression.

References: [P-NET](https://www.nature.com/articles/s41586-021-03922-4);
[stochastic gates](https://proceedings.mlr.press/v119/yamada20a.html).
