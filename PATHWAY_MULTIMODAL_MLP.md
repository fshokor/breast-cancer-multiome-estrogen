# nb11 v2: matched sparse pathway models

[Notebook](notebooks/nb11_pathway_multimodal_mlp.ipynb).
The revised notebook is intentionally unexecuted. The completed v1 notebook is preserved in
notebooks/archive/nb11_completed_before_matched_models_12e4459137fd.ipynb.

## Models and comparisons
Six fits total: RNA-only, ATAC-only and RNA+ATAC separately for MCF7 and T47D.
They use nb08's exact paired-cell splits and the same training-only preprocessing and common
Hallmark pathway universe. Each active modality has the same masked gene-to-pathway encoder.
All models share the fusion output size, sparse pathway gates and classifier recipe.
The absent modality has no encoder; its fusion block receives constant zeros, never its data.
Parameter counts differ and are exported. A runtime input-isolation assertion checks unimodal predictions.

Compare structured RNA with nb08 RNA, structured ATAC with nb08 ATAC, and structured RNA+ATAC
with both new matched controls. nb08 is not retrained. Its different input preprocessing and
capacity mean performance differences do not isolate the effect of biological structure.

## Fixes after the first run
- Every epoch can become the best validation-BCE checkpoint, starting at epoch 1.
- Constant expected-L0 penalty 0.005 applies from the start, replacing the delayed 0.001 penalty.
- Gates start at 0.25 with Gaussian noise sigma 0.25.
- Smooth bounded tanh replaces repeated layer normalization, avoiding its large derivative at
  a zero reference. Pathway width remains 4; head width is 16, dropout 0.3, weight decay 0.02.
- AdamW model LR 0.0003, gate LR 0.003, batch 128; maximum 100 epochs, minimum 10 and patience 10.
- These are new starting settings, not demonstrated optimal settings. Stronger penalties can
  underfit or still fail to yield sparsity. Report exact-zero/nonzero gates and expected-open
  count separately, and inspect validation curves.
- Adaptive Gauss-Legendre conditional IG uses 32, 64, 128, then at most 256 points. Every sampled
  cell must pass completeness (absolute 0.01 + relative 0.05 times target logit change) and
  attribution-vector stability (absolute 0.01 + relative L1 0.10) across consecutive resolutions.
  If either fails, that pathway's gene rankings and candidate regions are withheld.
  Diagnostics are displayed and exported. Numerical checks do not establish causal validity.

## Colab and resuming
Use a GPU and the same Drive project folder. Supply the four nb02 H5AD files and complete nb08
results folder, including manifests, input hashes and test predictions. Frozen Hallmark membership
is embedded. Reference files are hash-checked; if a rolling download changed, supply the original
RefSeq/blacklist files from earlier notebooks.

Check BASE and NB08_RUN_ID. Start a NEW v2 run; v1 checkpoints are incompatible.
Set RESUME_RUN_ID only to an existing v2 run ID with identical settings/source/reference signature.
Results: results/tables/nb11/RUN_ID/CELL_LINE_MODE/.
Checkpoints: data/processed/nb11/RUN_ID/CELL_LINE/models/MODE/last.pt and best.pt.
Each model saves an atomic completed-epoch checkpoint including optimizer/RNG/history/best state.
An interrupted epoch restarts from the preceding completed epoch. Completed training is skipped
on resume, while evaluation/interpretation can regenerate. No checkpoint exists before epoch 1
finishes. Prepared matrices are shared across modes and cached per cell line in the new run.
Interrupted preprocessing restarts that cell line. No old-run preprocessing is silently reused.

tqdm reports epochs, batches, validation AUROC, best epoch, patience and elapsed time. ETA is to
the maximum epoch count, before early stopping. Completed-model statistics print the selected epoch.

## Features and explanation outputs
RNA is log1p raw counts. ATAC uses binary peak accessibility with coordinate-consensus RefSeq
transcript-union promoter/body links, blacklist exclusion and fractional multi-gene assignments.
Custom ATAC gene activity is weighted accessible fraction, multiplied by median training depth /
cell depth, then log1p. This is not measured expression or an exact ArchR gene score.
Train-only detection/variance filtering and standardization are shared across modes.
Pathways need at least three supported genes per modality; coverage is shared for matched comparison.
No GSEA/DEG/nb06 significance results select predictive features.

Validation ablations remove pathways, active modality branches within pathways, or entire active
branches. These perturbations are not separately trained unimodal baselines. Gene IG uses up to
16 validation cells per class and three pathways with positive validation ablation effects.
Other pathway latents stay fixed, with the training-mean input as reference.
Linked ATAC regions are candidates, never region-level attribution scores.

Final exports compare performance, selected-checkpoint sparsity, IG checks and validation pathway
ablation across all three modes. The ZIP excludes model checkpoints and prepared matrices.

## Interpretation limits
This architecture gives explicit biological organization and testable pathway dependence.
It does not automatically prove better explanation fidelity than nb08: nb08 has no equivalent
attribution/stability audit here. Numerical IG convergence does not establish biological truth.
A single seed cannot establish stable selection. A pathway gate is not activation/repression,
and raw gate magnitudes are not importance scores. Compare both positive/negative GSEA directions
and nb06 concordant/discordant evidence descriptively; they use the same cohort.
Existing full-cohort QC, previously inspected test results, unavailable replicate IDs and possible
library/treatment confounding limit generalization claims. Do not tune using test performance.

Authoring checks parse syntax and validate notebook structure without executing notebook cells,
preprocessing, model fitting, prediction or attribution. Runtime convergence/performance remain unverified.
