# nb05 — ATAC analysis

## Scope
Independently analyze MCF7 and T47D, using the exact RNA-QC-paired cells from nb02.
Keep every original peak and integer count in a new AnnData object; leave nb02 inputs
unchanged. Use Scanpy for count QC, neighbors, Leiden and UMAP. ATAC TF–IDF/LSI and
differential-accessibility testing use sparse SciPy, scikit-learn and statsmodels.

No biological replicate-to-cell mapping is available. All treatment-association
statistics are exploratory cell-level results. Library effects and biological
treatment effects cannot be separated with the available design.

## QC and embeddings
Count-based QC reports total peak counts and detected peaks. Peak counts are not
fragment counts. Without fragments, do not infer FRiP, TSS enrichment, nucleosome
signal or fragment-based doublets. No extra cells are removed; zero-accessibility
cells trigger an explicit error.

For LSI only, select peaks present in >=20 cells and <=95% of cells.
Use binary accessibility B, with:
- TF_ij = B_ij / sum_k B_ik, using all supplied peaks in the denominator.
- IDF_j = N / sum_i B_ij.
- TFIDF_ij = log1p(10000 * TF_ij * IDF_j).

This is an explicitly defined Signac-style binary variant, not an exact execution
of Signac and not the scMKL paper's feature transformation.
Fit randomized truncated SVD with 50 components, seven iterations, seed 42.
Save selected peak IDs, IDF, singular values and loadings for reproducibility.

Report each component's correlation with log peak-count depth. Omit LSI1 by a
predeclared choice, and standardize LSI2–31 across cells before neighbor finding.
This does not guarantee removal of depth effects; review the diagnostics.
Use Scanpy neighbors (15, sklearn exact search), Leiden (igraph, resolution 0.5,
two iterations, seed 42), and UMAP (min_dist 0.3, 500 iterations, seed 42).
RNA and ATAC cluster numbers have no shared meaning. Save cross-tabulations and
adjusted Rand index as descriptive agreement, not evidence of identical states.

## Depth-adjusted differential accessibility
The DA universe is independent of LSI filtering: a peak must be detected in >=10
cells and >=5% within either condition. Fit the depth-only null logistic model:

    treatment_E2 ~ intercept + standardized log1p(total supplied peak counts)

For each binary peak x, compute the efficient score for adding x to the model.
With null fitted probabilities mu, W=diag(mu*(1-mu)), nuisance design Z:

    U = x.T @ (y-mu)
    V = x.T @ W @ x - (x.T @ W @ Z) @ inv(Z.T @ W @ Z) @ (Z.T @ W @ x)
    signed_score = U / sqrt(V)
    p = chi2.sf(signed_score**2, df=1)

These calculations stay sparse. For binary x, x**2=x. Variance <=1e-8 is not
testable and is explicitly excluded. Apply BH correction across testable peaks
per cell line. Unit tests compare the calculation directly with statsmodels GLM
score_test. This tests treatment association conditional on depth, not a causal
effect of treatment on a peak.

Report raw per-condition accessibility fractions, their difference, and the
unadjusted log2 odds ratio with 0.5 added to each contingency-table cell. These raw
effects are not adjusted regression coefficients. A candidate requires BH q<=0.05,
absolute raw accessibility difference >=0.05, and matching adjusted/raw directions.
Depth adjustment does not eliminate all technical or composition confounding.
No permutations or pseudoreplicates are invented.

## Genome and provisional annotation
GEO GSM6418445/6 record hg38. The origin/convention of coordinates embedded in the
processed peak names remains unresolved. Preserve supplied names and boundaries.
The UCSC hg38 ncbiRefSeqCurated genePredExt reference is downloaded with a recorded
SHA-256. It uses zero-based starts and exclusive ends. Plus-strand TSS=start;
minus-strand TSS=end-1. Include all curated transcript TSSs; retain equally close
genes rather than selecting an arbitrary transcript.

Find nearest TSS to the peak's midpoint under two hypotheses:
- 0-based half-open: midpoint=(supplied_start+supplied_end-1)/2.
- 1-based closed: midpoint=(supplied_start+supplied_end-2)/2 in zero-based base positions.

Only report a provisional nearby-gene candidate if the nearest gene set agrees
under both hypotheses and both distances are <=100 kb. Save each hypothesis,
distance and status, including ties, distant matches and unmapped contigs.
This makes a limited coordinate-robust proximity annotation; it does not resolve
the coordinate convention, identify regulatory targets or justify promoter/motif
overlap claims. No BED export is made.

## Outputs
- notebooks/nb05_atac_analysis.ipynb: standalone Colab notebook with source/config embedded.
- data/processed/*_ATAC_nb05.h5ad: all original count/binary peaks and paired cells,
  RNA/ATAC labels, LSI, graph, UMAP, QC and parameters.
- data/processed/*_atac_lsi_model.npz: embedding peak IDs, IDF, SVD loadings and singular values.
- data/processed/*_nb05_da_all.tsv: full DA test results with provisional annotations.
- data/processed/*_nb05_da_candidates.tsv: complete exploratory candidate subset.
- data/processed/*_nb05_peak_annotations.tsv: all-peak coordinate sensitivity results.
- results/tables/nb05: DA universe, top 1,000 candidates per adjusted direction,
  per-cell QC/coordinates, cluster tables, LSI diagnostics, summary and validation.
- results/figures/nb05: count QC/LSI depth correlations, six-panel UMAP, DA volcano.

Large derived objects and complete peak-level DA/annotation files are Git-ignored.
Compact review outputs are tracked. The notebook downloads the reference if absent,
but refuses a changed reference checksum. Live Colab Drive authorization needs the user.

    .conda/bin/python scripts/build_nb05.py
    .conda/bin/python scripts/validate_nb05.py

Validation checks TF–IDF values, input preservation, agreement with statsmodels,
degenerate-peak handling, coordinate ties, saved counts and exact paired order.

## References
- [Signac TF–IDF](https://stuartlab.org/signac/reference/runtfidf)
- [Signac ATAC workflow](https://stuartlab.org/signac/articles/pbmc_vignette)
- [Statsmodels GLM score test](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.score_test.html)
- [UCSC coordinate formats](https://genome.ucsc.edu/FAQ/FAQformat.html)
- [GEO E2 ATAC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6418445)
- [GEO Vehicle ATAC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6418446)

## Initial run interpretation (10 October 2026)

| Cell line | Cells | ATAC clusters | Tested peaks | Exploratory DA candidates | Candidates detected in only one condition |
|---|---:|---:|---:|---:|---:|
| MCF7 | 6,019 | 6 | 30,678 | 1,400 | 573 |
| T47D | 6,562 | 9 | 44,380 | 2,367 | 998 |

Candidates satisfy cell-level BH q <= 0.05, an absolute accessibility-fraction difference >= 0.05, and agreement between adjusted and raw directions. These are exploratory associations, not replicate-supported differential accessibility.

LSI1 strongly correlates with depth (Spearman 0.998 / 0.999) and is excluded from the graph. LSI2 still correlates with depth (-0.613 / -0.605); excluding LSI1 does not eliminate depth effects. Interpret UMAP and clusters alongside the QC panels.

About 41% / 42% of candidate peaks are detected in only one condition. This may include biological differences, but condition-specific library effects or peak-calling differences cannot be separated with the available metadata. Depth adjustment cannot resolve this confounding.

Nearest-TSS gene annotations remain provisional: hg38 is documented, but the peak coordinate convention is unresolved. Agreement under both coordinate interpretations does not establish a regulatory target. No motif or causal regulatory conclusions are made.
