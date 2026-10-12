# nb10: model comparison and biological interpretation

Status: prepared in advance and intentionally unexecuted. No results or conclusions
have been generated from nb09. This notebook does not fit, tune or run prediction
with any model. It reviews saved scores and coefficients.

## Run after nb09 finishes

Open notebooks/nb10_model_comparison_interpretation.ipynb in Colab.
CPU is sufficient. It embeds its implementation and needs no repository upload.

Required: the complete results/tables/nb09/RUN_ID/ directory. A nb09 export ZIP
contains the required files; extract it into that directory if needed.
No model weights, random-feature arrays, raw H5ADs or sequence resources are needed.

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
Verify the selected regularization candidate converged and matches the documented
validation selection rule. Candidate warnings remain in the solver review table.

Compare five models in each cell line: RNA/ATAC MLPs, RNA/ATAC grouped-kernel controls,
and joint grouped kernels. Report AUROC, average precision, accuracy, balanced
accuracy, F1 and confusion counts. Compute paired error corrections/new errors and
descriptive joint-minus-reference metric differences. No significance tests or CIs.

Validation/test gaps cover grouped-kernel models only, because nb09's portable
export does not contain the MLP validation predictions. Raw kernel scores are not
probabilities, so log loss/Brier scores are not compared across model families.

Review representative group weights in unimodal/joint models and preserve aliases
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
