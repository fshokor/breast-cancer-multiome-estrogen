# Scanpy QC, preprocessing and RNA visualization

## Scope and provenance
nb02 applies paper-guided RNA QC with **Scanpy**, preserves RNA–ATAC pairing,
and visualizes RNA separately for MCF-7 and T47D using PCA, Leiden and UMAP.
Custom code handles the unusual input TSVs, checksums, pairing and file outputs;
QC calculations and preprocessing use Scanpy APIs.

Reference: Kupp et al. (2025), *Interpretable and integrative analysis of single-cell
multiomics with scMKL*, [DOI 10.1038/s42003-025-08533-7](https://doi.org/10.1038/s42003-025-08533-7).
Table 1 matches our input dimensions and treatment counts exactly.
The QC thresholds come from the standard Scanpy comparison workflow on p.14.
This is not a claim that all scMKL experiments applied these filters, or that the
paper's exact post-QC objects and figures have been reproduced.

## Scanpy QC and count-preserving preprocessing
| Operation | Function or rule | Setting |
|---|---|---|
| QC before gene filtering | sc.pp.calculate_qc_metrics | MT- genes; no top-gene percentages |
| Gene prevalence | sc.pp.filter_genes | min_cells=3 |
| Recompute cell QC on retained genes | sc.pp.calculate_qc_metrics | counts, detected genes, mitochondrial % |
| Detected genes per cell | sc.pp.filter_cells | min_genes=100 |
| Mitochondrial cutoff | explicit obs mask | pct_counts_mt <= 30 |
| RNA log transformation | sc.pp.log1p | natural log of count + 1 |
| All-gene scaling | sc.pp.scale | zero_center=False, no clipping |

The three thresholds are stated in the paper. Gene-first order, a single pass per
cell line, and the MT- annotation convention are explicit project choices because
the paper does not fully specify these details. Cells at exactly 100 genes or
30% mitochondrial counts pass. The minimum-gene and MT masks are evaluated on
gene-filtered counts, without iterative filtering or treatment-specific thresholds.

Integer RNA counts are preserved in layers['counts']; .X contains log1p raw counts.
Because the paper gives no total-count normalization target for this standard-QC
branch, none is inserted into it. Scanpy log1p and scale use float64 internally
to avoid a float32 constant-gene variance issue exposed by a unit test. Saved sparse
values are float32, and every scaled entry must be finite.

Centered all-gene z-scores can be reconstructed from:

    z = layers['scaled_no_center'] - var['scale_center_offset']

Mean and SD are saved in var['log1p_mean'] and var['log1p_std'].
centered_scaled_block materializes only a requested small block. This is an affine
representation of centered scaling; .X is not silently replaced by z-scores.

## Exploratory RNA visualization
A **separate copy of integer counts** is used for visualization, with the following
explicit settings. These are reproducible project choices, not asserted to be the
paper's exact embedding recipe.

| Stage | Scanpy function | Setting |
|---|---|---|
| Library-size normalization | sc.pp.normalize_total | target_sum=10,000 |
| Log transformation | sc.pp.log1p | natural log1p |
| Variable genes | sc.pp.highly_variable_genes | n_top_genes=2,000, flavor='seurat' |
| Centering/scaling of HVGs | sc.pp.scale | zero_center=True, max_value=10 |
| PCA | sc.pp.pca | 50 components, ARPACK |
| Neighbors | sc.pp.neighbors | 15 neighbors, first 30 PCs, exact sklearn search |
| Clusters | sc.tl.leiden | igraph backend, resolution=0.5, 2 iterations, undirected |
| UMAP | sc.tl.umap | min_dist=0.3, spread=1, 500 iterations |
| Reproducibility | all stochastic steps | random_state=42 |

Only the approximately 2,000-gene visualization subset is centered and becomes
dense; full RNA and ATAC count matrices remain sparse. Seurat HVG selection may
include tied genes at the cutoff. The actual selected number is reported.

Each line is fitted independently. Treatment labels are used only for coloring
plots and making descriptive cluster counts; they do not enter the fit.
Figures show PCA and UMAP colored by treatment, Leiden cluster, RNA count depth
and mitochondrial fraction, plus a PCA variance plot and before/after QC histograms.

Clusters are not annotated cell types or estrogen-response states. Cluster IDs
cannot be compared directly across independent embeddings. UMAP separation is
not evidence of treatment significance, causal effects or held-out predictive
performance. No DE, GSEA, motif analysis, cell-cycle regression, batch correction
or classifier is run. Future predictive analysis must fit data-dependent
transforms on training data rather than reuse these whole-cohort embeddings.

## ATAC pairing and limitations
ATAC retains exactly the RNA-selected cells, in the same order, and all 206,167
supplied peaks. .X holds original integer peak counts; layers['binary'] is count > 0.
RNA-derived Leiden labels are retained in metadata for pairing convenience; they
are not ATAC-derived clusters. No ATAC PCA/LSI or joint RNA–ATAC embedding is fitted.

Peak sums are not fragment counts. No fragment, TSS, FRiP, nucleosome or doublet
cutoff is invented. Zero-count ATAC cells are flagged without silently dropping
them. Genome assembly and coordinate origin remain unresolved; coordinates are
stored as supplied.

## Results from the executed Scanpy notebook
| Cell line | Cells before QC | Cells retained | Genes retained | RNA Leiden clusters |
|---|---:|---:|---:|---:|
| MCF7 | 6,438 | 6,019 | 25,105 | 9 |
| T47D | 6,615 | 6,562 | 26,320 | 8 |

Every retained cell and gene identifier, including its order, matches the previous
custom QC implementation (commit 02c702d). This validates the Scanpy rewrite's
selection equivalence; it does not establish exact reproduction of the paper.
Cluster numbers refer to the stated exploratory resolution of 0.5.

## Saved objects
For each of MCF7 and T47D:
- *_RNA_paper_qc.h5ad: all QC-retained genes, sparse counts/log1p/scaled representations,
  HVG flag, X_pca, X_umap, PCA loadings, Leiden labels, neighborhood graphs and parameters.
  Embeddings derive from the separate normalized/HVG branch, not this object's .X.
- *_RNA_visualization.h5ad: the centered/scaled HVG input used for PCA, with raw counts,
  a normalized-log layer, embeddings, graph, cluster labels and settings.
- *_ATAC_paired_paper_qc.h5ad: paired integer counts and binary accessibility, all peaks.

All six biological objects are under data/processed/ and Git-ignored.
Tables under results/tables/ record QC masks, HVGs, PCA/UMAP coordinates, cluster
treatment counts and validation. Figures are under results/figures/.
Re-running replaces derived files atomically per file; original archives remain unchanged.

## Run and validate
Open notebooks/nb02_qc_preprocessing.ipynb in Colab and run top to bottom.
It mounts the same Drive project, embeds the implementation and parameters, and
installs missing dependencies including Scanpy and igraph. It displays actual package
versions; requirements-lock.txt records the tested local environment.

Local commands from the repository root:

    .conda/bin/python scripts/preprocess_gse154873.py
    .conda/bin/python scripts/build_nb02.py
    .conda/bin/python scripts/validate_qc.py

The validation runner executes the notebook, checks saved counts, finite embeddings,
graphs, labels, paired identifiers and HVG count preservation. Boundary tests verify
the exact QC inequalities. A small end-to-end test exercises real Scanpy PCA,
neighbors, Leiden and UMAP. When the previous local QC masks are available, validation
compares every retained cell and gene identifier to commit 02c702d's implementation.
That comparison measures implementation equivalence, not agreement with unpublished
paper outputs. Colab Drive authorization is not tested by local execution.

## Remaining provenance limits
The scMKL paper states 24-hour estrogen exposure on p.12, whereas the original
[Ors et al. paper](https://doi.org/10.1093/nar/gkac908) states 48 hours for Multiome.
Submitted metadata do not resolve this discrepancy; no inferred duration is added.
The supplied PDF fingerprint is saved in results/tables/qc_reference.json.

The paper mentions Scanpy 1.10.1 and 1.4.2 in the same methods paragraph; this
environment uses 1.11.5. The reviewed authors' repositories are pinned in
qc_reference.json; they are not asserted to be the archived scMKL v0.1.6.
The scMKL-specific TF-IDF/feature-group/model processing remains deferred.
