# Paper-guided QC and preprocessing

## Reference and scope
Kupp et al. (2025), *Interpretable and integrative analysis of single-cell multiomics
with scMKL*, Communications Biology 8:1160.
[DOI: 10.1038/s42003-025-08533-7](https://doi.org/10.1038/s42003-025-08533-7).

The supplied PDF was read in full; Methods pp.12–14 were inspected, and p.14 was
visually checked. Table 1 exactly matches the six supplied GEO supplements:
MCF-7 6,438 cells and T47D 6,615 cells, each with 36,601 genes and 206,167 peaks.
This notebook implements the **standard Scanpy comparison workflow's QC thresholds**
from p.14. The paper also benchmarks scMKL with multiple representations, so these
filters must not be described as universally applied to all its scMKL experiments.

## Parameter provenance
| Step | Implemented setting | Evidence or explicit choice |
|---|---|---|
| RNA gene filter | detected in at least 3 cells | Methods p.14 |
| RNA cell filter | at least 100 detected genes | Methods p.14 |
| Mitochondrial filter | retain at most 30% of counts | Methods p.14 removes values above 30% |
| Mitochondrial annotation | gene symbol starts with MT- | Project implementation for human symbols |
| Order | gene filter, then cell QC; one pass per cell line | Explicit project choice; detailed order not supplied |
| RNA transform | natural log1p of counts | Log transformation in p.14; log1p supported by authors' code |
| Library-size normalization | not applied | No target sum specified for standard workflow |
| Scaling | mean/sample SD (ddof=1), no clipping | Standard scaling in p.14; ddof=1 is Scanpy-compatible project choice |
| ATAC cell selection | identical RNA QC mask | Project extension to preserve paired observations |
| ATAC peak filtering | none; all supplied peaks retained | No additional peak cutoff established for this step |
| ATAC representation | integer peak counts and binary accessibility | Paper compares binary/count-derived representations |
| Fragment QC/doublets | not computed or filtered | No established numeric cutoff; required fragment information absent |
| TF-IDF, PCA/LSI, HVGs | deferred | Must distinguish descriptive use from training-only fitting |

Cell QC is computed on the gene-filtered count matrix. Pre-gene-filter totals and
detected-gene counts are also reported. A cell at exactly 100 genes or 30% MT passes.
Gene prevalence is assessed on all supplied cells before the cell filter; it is not
iteratively recomputed. No treatment-specific threshold is used.

## RNA storage and scaling
RNA .X is sparse log1p counts. RNA layers['counts'] preserves integer counts for
retained genes/cells. Because centering would turn zeros into nonzero values,
centered z-scores are represented by two components:

    z = layers['scaled_no_center'] - var['scale_center_offset']

The stored mean and SD are also in var['log1p_mean'] and var['log1p_std'].
centered_scaled_block materializes only small requested blocks.
A test verifies numerical equivalence to dense centered scaling.
Constant features use SD=1. No full dense matrix is created.
These scaling statistics describe the retained cohort; later prediction must refit
feature selection and transformations using training data alone.

This is a transparent reproduction of the stated parameters, not a claim of
bit-for-bit reproduction of an unspecified normalization pipeline. The installed
Scanpy is 1.11.5; the paper mentions both 1.10.1 and 1.4.2 in the same paragraph.
Package versions remain recorded in the environment locks.

## ATAC storage
ATAC .X holds original integer peak counts for the retained RNA cells.
layers['binary'] holds count > 0. All 206,167 peaks, including non-chr contigs,
are retained. Peak sums and detected-peak counts are reported for every input cell,
but are not labeled as fragment counts, FRiP, TSS enrichment, or doublet scores.
Cells with zero ATAC peak counts are flagged; none are silently discarded.
Genome build and coordinate origin remain unresolved; supplied coordinates are
parsed without converting coordinate conventions.

## Authors' code review
Reviewed [scMKL_paper](https://github.com/ohsu-cedar-comp-hub/scMKL_paper/tree/84d68eef6a0985ba757d4b8d4158bc765f48c1ac)
and the [scMKL implementation](https://github.com/ohsu-cedar-comp-hub/scMKL/tree/d6448b73895ab62b31f78213f6e4fed910bce4c4).
These are pinned inspected revisions, not asserted to be the paper's archived v0.1.6.
The public comparison repository does not expose a standalone standard-QC script.
Its preprocessing functions support log1p without total-count normalization.
Its model-specific variance/scaling implementation is not substituted for
Scanpy-style descriptive scaling here.

The paper's TF-IDF formula and tfidf_train_test function weight each peak by:

    log(1 + (1 + number_of_training_cells) / (1 + training_cells_with_peak))

This is not the same as automatically inserting per-cell term-frequency normalization.
The source contains distinct helper implementations; future model work must pin the
chosen definition and apply training-derived statistics to held-out cells.
No scMKL package installation, fitting, prior construction, or prediction was performed.

## Treatment-duration discrepancy
The scMKL paper states 24 hours in Methods p.12.
The [original Ors et al. study](https://doi.org/10.1093/nar/gkac908), in its Multiome
results section, states 48 hours. The supplied metadata contain neither duration
nor dose. This disagreement is recorded; labels remain E2/Vehicle and no inferred
time column is introduced. The identity of the dataset is independently supported
by accession, dimensions and class counts.

## Run and outputs
Open notebooks/nb02_qc_preprocessing.ipynb in Colab and run after nb00/nb01.
It mounts the same Drive folder and embeds its source and configuration.
For local WSL execution:

    .conda/bin/python scripts/preprocess_gse154873.py

Regenerate only nb02 after source/configuration edits:

    .conda/bin/python scripts/build_nb02.py

Execute the notebook and validate saved objects:

    .conda/bin/python scripts/validate_qc.py

Outputs:
- data/processed/MCF7_RNA_paper_qc.h5ad
- data/processed/MCF7_ATAC_paired_paper_qc.h5ad
- data/processed/T47D_RNA_paper_qc.h5ad
- data/processed/T47D_ATAC_paired_paper_qc.h5ad
- results/tables/paper_qc_summary.json
- results/tables/{MCF7,T47D}_cell_qc.tsv and *_gene_qc.tsv
- results/figures/{MCF7,T47D}_paper_qc.png
- results/tables/qc_validation.json

Archives and processed matrices are Git-ignored. Re-running rebuilds the processed
outputs atomically per file. Original archives remain unchanged for later
scMKL-specific or count-based analysis. No DE, GSEA, differential ATAC, motifs,
clustering, embeddings, or classifiers are run by this notebook.

## Parameters reserved for later comparisons
The paper uses 2,000 variable RNA genes for a visualization comparison, 5,000
variable ATAC peaks for selected baselines, 50 PCA components, and 50 LSI
components with the first excluded. These are analysis-specific settings, not
universal QC filters, so nb02 retains all genes passing prevalence QC and all
ATAC peaks. Apply the relevant settings in the later comparison notebooks.

## Verified local QC results

| Cell line | Cells before | Cells retained | Genes retained | ATAC peaks retained | Vehicle / E2 retained |
|---|---:|---:|---:|---:|---:|
| MCF7 | 6,438 | 6,019 | 25,105 | 206,167 | 2,730 / 3,289 |
| T47D | 6,615 | 6,562 | 26,320 | 206,167 | 2,997 / 3,565 |

All excluded cells failed the mitochondrial threshold; none failed the minimum-gene threshold.
RNA and ATAC retained-cell identifiers match exactly, including order. No retained cell
has zero ATAC peak counts. Results are in results/tables/paper_qc_summary.json.
