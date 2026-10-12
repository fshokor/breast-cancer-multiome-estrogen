# nb09: scMKL-style integration

Status: prepared for Colab; intentionally unexecuted. No motif scanning, kernel
construction, model training or data analysis has been run for nb09 locally.

## Run in Colab

Open notebooks/nb09_scmkl_style_integration.ipynb and run all cells.
This workflow uses the CPU celer solver and CPU motif scanning. A GPU does not
accelerate those steps; a high-RAM runtime is helpful. The notebook is standalone.

Use the four nb02 paired QC H5AD files and the complete nb08 results folder:
results/tables/nb08/20261011T103750_408739Z/. The default is the user's completed
MLP run with learning rate 0.00001; its actual config is read and copied, not
overwritten by the original source config. If restoring from the results ZIP,
extract its contents into this folder first. Change NB08_RUN_ID explicitly to
compare a different completed run.

The notebook verifies input SHA256 values and split checksums. No new splits
are generated. Existing nb07 hg38, blacklist and JASPAR resources are reused;
missing files download with frozen SHA256 verification. The genome is ~835 MB.

## Model

Fit exactly one joint RNA + ATAC model within each cell line: two fits total.
No unimodal kernel models are trained. Compare with the existing nb08 MLP baselines.
Retain the core scMKL design: biological feature groups, Gaussian random Fourier
features, and group-lasso regression with targets Vehicle=-1 and E2=+1.

Reuse nb08 train-only preprocessing and up-to-5,000 selected features per modality;
require the recomputed feature IDs to equal the saved baseline IDs. RNA uses
log1p counts and train centering/scaling; ATAC uses binary counts without scaling.

RNA groups are frozen Hallmark v7 gene sets. ATAC groups are JASPAR2024 motifs
annotated afresh on training-selected peaks, not a DEG/DA/enrichment-selected
universe. Group membership is sequence based and independent of treatment labels.
Exclude nonstandard/blacklisted/ambiguous sequences. Keep only motif hits present
under both unresolved coordinate interpretations. Retain groups with >=5 features
and collapse identical memberships while saving aliases/coverage exclusions.
Grouping omits uncovered features; comparisons therefore concern the complete
model pipelines, not architecture alone.

Each group gets 32 Gaussian frequencies (64 cosine/sine features); bandwidth is
the median positive distance among up to 256 training cells. Center each block
and normalize its mean squared row norm using training cells only. The kernel is
exp(-squared Euclidean distance/(2*sigma^2)); frequencies have SD 1/sigma.

Use celer GroupLasso with equal group penalties and an intercept:
squared error/(2n) + alpha * sum(group coefficient L2 norms).
The alpha maximum is max(group norms of Z_train.T @ centered_y / n).
Fix the ratio at 0.1 of the training-only maximum; there is no search.
This starting setting comes from the best recorded MCF7 ATAC validation result in
the interrupted run, not an established optimum for either integrated model.
Validation is diagnostic only. Require convergence and retain the training fit
without refitting on validation. Only then score test cells.
No MLP learning rate or neural-network epochs apply here.

Decision scores are not probabilities. Use raw-score AUROC/AP and a fixed zero
threshold for classification; do not compare uncalibrated log loss/Brier scores
to the MLP. Group norms describe predictive contributions, not pathway activity,
TF binding, causality or percentages of biological contribution.

## Relationship to scMKL

This is an independent, scMKL-style implementation, not a call to the official
package and not an exact reproduction of the paper:
- [scMKL source](https://github.com/ohsu-cedar-comp-hub/scMKL/tree/d6448b73895ab62b31f78213f6e4fed910bce4c4)
- [Paper](https://doi.org/10.1038/s42003-025-08533-7)
- [celer objective/API](https://mathurinm.github.io/celer/generated/celer.GroupLasso.html)

Adaptations include fixed nb08 splits/preprocessing/features, 32-frequency Gaussian
maps, median bandwidths with standard RBF scaling, train-centered unit-energy
blocks, group-norm alpha maximum, current frozen reference groups and a fixed
alpha rather than repeated nested cross-validation. Published scMKL scores
are not expected to be numerically reproduced.

## Outputs

results/tables/nb09/RUN_ID/ contains settings, provenance, kernel coverage,
group memberships/aliases, coordinate audit, solver convergence/selection tables,
predictions, group weights, metrics and plots.
data/processed/nb09/RUN_ID/ holds disk-backed feature arrays, random maps with
preprocessing parameters, model coefficients/intercepts and model specifications.
results/exports/nb09_RUN_ID.zip is the results bundle to return; large models/maps
are excluded. The final comparison verifies paired test IDs/labels and reports
integration minus each reference AUROC/AP, without claiming significance.

One split cannot quantify split uncertainty. Biological replicate IDs are
unavailable and library effects may track treatment. Prior whole-cohort nb02 QC
defines the cohort. nb08 test results are already known; if its learning rate
was tuned on test performance, this is no longer an untouched final holdout.
Continue comparison on matching cells, but use independent data for confirmation.

## Maintenance and static validation

Canonical code: src/scmkl_style.py.
Settings: configs/scmkl_style.json.
Build: python scripts/build_nb09.py.
Static check: python scripts/validate_nb09_static.py.
These commands create/check notebook files; they do not execute notebook cells.
Do not run this notebook locally: the user will run it in Colab.


## Restart after the interruption

Your partially executed notebook is preserved in
notebooks/archive/nb09_partial_20261012T085524_321769Z.ipynb.

Open the updated nb09 in Colab. For its first run, leave RESUME_RUN_ID empty and
keep CACHED_KERNEL_RUN_ID="20261012T085524_321769Z". Run all cells.
A new integrated-only output run is created. Completed RNA/ATAC kernels from the
old run are reused after configuration, reference, split and shape checks.
Missing kernels (for example T47D) are built normally. Old unimodal models are
not needed, and the interrupted ATAC search is not resumed or refitted.

Keep the old results and data/processed/nb09 folders: reused feature maps remain
there and the new model specification records their locations.

For another disconnect, set RESUME_RUN_ID to the NEW integrated run ID and run all.
A completed fit is checkpointed before evaluation/plotting. Completed models are
skipped. A solver call interrupted before returning must be fitted again; its
internal iterations are not checkpointed. Partial kernel construction restarts
that modality, while completed kernels are retained.

nb10 was updated to expect three models per cell line: two saved MLPs and one
integrated model. This comparison changes both model family and input modalities,
so it does not isolate the effect of adding ATAC alone.
