# nb08: unimodal MLP baseline

Open notebooks/nb08_unimodal_mlp.ipynb in Colab, select a GPU, and run all cells.
The standalone notebook embeds code/configuration. Required inputs are the four
nb02 paired QC H5AD files in the Drive project's data/processed/ directory.

**Status: prepared for user-run Colab training. No notebook execution or model training locally.**

## Design

Four models: RNA and ATAC separately within MCF7 and T47D. E2 is positive.
Identical stratified cells for both modalities: approximately 72% train, 8% validation,
20% test; seed 42. Select up to 5,000 variable features on training cells only,
requiring at least three detections. RNA uses log1p(raw counts), training log-space
variance selection, and training centering/scaling. ATAC uses binary peak counts,
training binary variance selection, and no scaling. No prior DEG, DA, motif,
pathway, PCA or LSI outputs enter prediction. Densify only minibatches.

Five 512-unit ReLU hidden layers, dropout 0.1, one output logit. Adam learning rate
0.001, binary cross-entropy, batch size 256, maximum 100 epochs. Validation-loss
early stopping: patience 5, minimum improvement 0.0001. Restore selected weights.
AUROC primary; also average precision, classification and calibration metrics.
Threshold fixed at 0.5. Save a constant training-prevalence dummy baseline.

Architecture/dropout follow the
[authors' MLP implementation](https://github.com/ohsu-cedar-comp-hub/scMKL_paper/blob/84d68eef6a0985ba757d4b8d4158bc765f48c1ac/Performance_Comparisons/mlp_classification.py).
This is not exact paper replication: PyTorch replaces Keras, a fixed split/batch
size replaces repeated splits/four-fold tuning, RNA uses consistent log-space
variance/SD rather than raw-count statistics, and stopping/restoration differ.
Settings: configs/unimodal_mlp.json.

## Outputs and nb09 handoff

- results/tables/nb08/RUN_ID/: metrics, plots, histories, predictions, selected features,
  splits, input fingerprints, environment/configuration and source snapshot.
- data/processed/nb08/RUN_ID/: selected weights, preprocessing arrays and model specs.
- results/exports/nb08_RUN_ID.zip: results bundle to share, excluding weights.

The summary verifies paired test cell IDs/labels. nb09 will integrate RNA/ATAC
using these exact splits. Fit on training and select on validation; do not tune
with test scores. Completed model cells refuse overwrites. Changed settings need
a new run; interrupted epochs are not resumed automatically.

Cell-level splits are exploratory: biological replicate IDs are unavailable and
library/treatment confounding is possible. Prior whole-cohort nb02 QC defines
the cohort. These scores do not establish patient, replicate or cross-line
generalization. One split does not estimate split-to-split variability.

## Maintenance

Canonical implementation: src/unimodal_mlp.py.
Regenerate: python scripts/build_nb08.py.
Static checks only: python scripts/validate_nb08_static.py.
Do not execute locally; the user will train in Colab and return results.
