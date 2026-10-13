# nb10: model comparison and biological interpretation

Status: revised notebook intentionally unexecuted; earlier executed nb10 is archived.
It reviews saved scores and coefficients and adds frozen cross-cell-line prediction.
It never fits or tunes models. Transfer results will be generated only when you run it.

## Run after nb09 finishes

Open notebooks/nb10_model_comparison_interpretation.ipynb in Colab.
CPU is sufficient. It embeds its implementation and needs no repository upload.

Required: the complete results/tables/nb09/RUN_ID/ directory. A nb09 export ZIP
contains the required files; extract it into that directory if needed.
Those files suffice for the saved-output sections only. The enabled transfer section additionally
requires checkpoints, original maps, preprocessing metadata and raw-count H5AD files; see below.

Set NB09_RUN_ID to the exact run folder. Leaving it blank is accepted only when
exactly one completed run exists; it never silently chooses the newest run.
Incomplete results stop with a missing-file list.

Optional context:
- results/tables/nb04/MCF7_hallmark_gsea.tsv and T47D_hallmark_gsea.tsv
- results/tables/nb06/MCF7_pathway_summary.tsv and T47D_pathway_summary.tsv
- results/tables/nb07/motif_enrichment_all.tsv

Missing context is recorded, not treated as negative biological evidence.

## Checks and outputs

Verify split checksums against the nb08 handoff, paired cell IDs, E2-positive labels,
classification thresholds, and recomputed versus saved metrics. Reorder predictions
by cell ID if necessary; never compare unmatched rows.
Verify the selected regularization candidate converged and matches the fixed-alpha
configuration (one validation diagnostic record). Candidate warnings remain in the solver review table.

Compare three models in each cell line: saved RNA/ATAC MLPs and joint grouped kernels. Report AUROC, average precision, accuracy, balanced
accuracy, F1 and confusion counts. Compute paired error corrections/new errors and
descriptive joint-minus-reference metric differences. No significance tests or CIs.

Validation/test gaps cover grouped-kernel models only, because nb09's portable
export does not contain the MLP validation predictions. Raw kernel scores are not
probabilities, so log loss/Brier scores are not compared across model families.

Review representative group weights in integrated models and preserve aliases
for identical memberships. Join Hallmark names to GSEA and nb06 summaries and
JASPAR IDs to nb07 enrichment. Preserve every scenario, direction, coordinate
hypothesis and matching-QC flag. Do not select the most favorable enrichment.
Cross-line selection overlap is descriptive; unavailable groups remain distinct
from eligible groups receiving zero weight.

All outputs go under results/tables/nb10/RUN_ID/:
- verified_metrics.csv, integration_comparison.csv and per-line paired error audits
- validation_test_gaps.csv and solver_review.csv
- all_model_group_weights.csv, joint_group_aliases.csv, cross-line selection
- optional biological context tables and an availability report
- performance/group figures, source snapshot and input checksums
- comparison_report.md

A shareable results/exports/nb10_RUN_ID.zip contains the complete review.

## Interpretation

Weights are unsigned predictive contributions, not activity, repression, causal
effects, TF binding or percentages of biology. Motif aliases are not independent
TF evidence. Earlier biological analyses use the same cohort and are contextual,
not independent validation. Negative NES and failed matching flags must be retained.

Single cell-level splits cannot establish biological-replicate generalization.
Replicate IDs are unavailable; library/treatment confounding remains possible.
Test results are already known. If used for tuning, independent data are required
for confirmation. Repeated predefined splits and cross-line transfer are potential
follow-up experiments, not tasks executed by nb10.

## Maintenance

Canonical implementation: src/model_comparison.py.
Build notebook: python scripts/build_nb10.py.
Static-only checks: python scripts/validate_nb10_static.py.
Neither command executes notebook cells. Do not run the notebook locally.

## Frozen cross-cell-line evaluation (nb10 extension)

The revised nb10 also includes nb09's RNA-only model in the saved-output comparison and adds
bidirectional **treatment prediction**: source MCF7 → target T47D, then source T47D → target MCF7.
Cell-line identity is not the prediction target. No fitting, fine-tuning, calibration or threshold
selection occurs. The notebook is authored without executing any evaluation cells.

Default runs:
- nb08: 20261011T103750_408739Z — RNA-only and ATAC-only MLP.
- nb09: 20261012T114858_335840Z — RNA-only and RNA+ATAC kernels.
- nb11 v1: 20261012T173612_419824Z — original RNA+ATAC pathway MLP.
- nb11 v2: 20261012T180159_920277Z — RNA, ATAC, RNA+ATAC pathway MLPs.
These produce 16 directed evaluations. Collapsed v2 ATAC models are included. Historical partial
hyperparameter-search fits are not treated as additional final models.

The cross-line section requires saved weights/preprocessing/model specifications (nb08), coefficient
models and original random-feature maps including cache-origin folders (nb09), best checkpoints,
structure.npz, prepared.json and source peak-gene link tables (nb11), plus all four raw-count nb02
processed H5ADs and complete saved result folders. ZIP exports alone are insufficient.
The original executed nb10 was archived before revision.

Every model is reconstructed with its exact architecture and evaluated first on its source test
cells. Predictions must match the saved source scores within atol=rtol=1e-4. A mismatch blocks only
that transfer case and is recorded; no substitute checkpoint is selected.
The opposite line's existing nb08 test cells form the evaluation set for every model. Source
training transforms remain frozen. Target-native AUROC is computed from that model family's saved
target-line test predictions on exactly the same target cells; it is a descriptive comparator.

Features are aligned by exact IDs. Missing RNA features use the source mean in transformed space
(zero after source standardization), with a fixed default 5% maximum missing fraction. Lists and
coverage rates are exported. Missing required ATAC peaks fail the evaluation; nb11 additionally
requires identical assay peak universes because its depth denominator includes the full peak panel.
No target means, variance filters, pathway selection, motifs, bandwidths or random maps are fitted.
nb11 uses source peak-gene assignments/weights, source feature coverage and source training depth
reference, then source means/SDs. The target line's independently prepared nb11 matrix is never reused.

Outputs under results/tables/nb10/RUN_ID/cross_cell_line:
requested_models.csv, transfer_protocol.json, environment.json, input_artifact_hashes.csv,
feature_coverage.csv and missing-feature lists, per-model predictions, transfer_results.csv,
transfer_report.md and a bidirectional AUROC comparison figure.
Rows explicitly distinguish complete evaluations from missing/invalid artifacts; a failed evaluation
is not an AUROC of zero. Partial progress is saved after every model, and the report states coverage.
The final nb10 ZIP includes these outputs and the report includes a transfer section.

Colab: run from the beginning, verify all run IDs, leave RUN_CROSS_LINE_TRANSFER=True.
A GPU can accelerate neural inference; CPU works. Kernel inference uses saved maps, without celer
fitting, MOODS scanning or genome download. Memory is bounded by test-cell sparse matrices and
inference batches, although initial H5AD reads load one count matrix before test subsetting.

Interpretation: this measures transfer within a shared dataset, not independent biological
replicate/patient generalization. Shared libraries, treatment/library confounding, prior test
inspection and source-mean imputation constrain conclusions. Do not tune using transfer scores.
