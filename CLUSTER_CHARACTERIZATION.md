# nb03 — Cluster characterization

nb03 characterizes the existing nb02 RNA clusters independently for MCF7 and T47D.
It does not perform treatment DEG, GSEA, ATAC analysis, or final cell-type annotation.

## Expression and marker discovery
Read the full retained-gene RNA objects, create a separate sparse copy from integer
counts, normalize each cell to 10,000, and apply log1p. Never rank markers on scaled
HVGs or on nb02's unnormalized log-count representation. Every retained gene is
tested, including genes no longer detected after cell filtering; these receive
uninformative rankings rather than an undocumented new filter.

Scanpy rank_genes_groups compares each original Leiden cluster against all other
cells in that cell line with Wilcoxon, tie correction, and Benjamini–Hochberg
adjustment. Full rankings are saved in ignored data/processed/*_nb03_all_cluster_markers.tsv.
Top 50 per cluster passing log2FC >=0.25, detection >=10%, detection difference >=0.10,
and adjusted p <=0.05 are tracked under results/tables/nb03.
The top three passing genes per cluster form the dot plot and cluster-mean heatmap.
The heatmap standardizes each displayed gene across cluster means, not across cells.

Markers and their p-values are exploratory: clusters were selected using these
data, cells are not independent biological replicates, and markers may reflect
unequal treatment composition. They are not treatment-DE evidence. Scanpy's
log fold changes are approximate, based on mean log expression.

## Cluster quality and sensitivity
Report cell counts, treatment counts/proportions, median RNA counts, detected
genes and mitochondrial percentage. Existing cells, labels, and UMAP remain intact.
Repeat Leiden on the existing graph at resolutions 0.3, 0.5, 0.8 and seeds 42, 7, 123,
using the same igraph backend and two iterations as nb02.
Report how closely resolution 0.5/seed 42 replays nb02's partition; do not assume exact
identity after graph serialization. Verify that two repeats on the saved graph agree.
Report adjusted Rand index versus nb02 and each original cluster's best Jaccard
overlap with a cluster in each alternative partition. These are parameter
sensitivity checks, not subsampling validation or a search for an optimal resolution.

## Descriptive programs
Frozen lists and download fingerprints are in configs/cluster_gene_sets.json:
- S and G2M genes: the Regev list mirrored in the Cook EMT study repository.
  The split follows the Scanpy tutorial: first 43 genes are S, remaining genes G2M.
  The mirror repeats HJURP; deduplication produces 43 S and 54 G2M genes.
- HALLMARK_ESTROGEN_RESPONSE_EARLY and HALLMARK_UNFOLDED_PROTEIN_RESPONSE:
  the explicitly versioned MSigDB v7.0 GMT distributed in Broad's ssGSEA2.0 repository.
  These frozen references are not presented as the latest MSigDB release.

Scanpy scores use log-normalized expression and matched control genes drawn from
the full retained-gene pool, seed 42, 25 expression bins, ctrl_as_ref=False.
For estrogen and UPR scores, ctrl_size=50; score_genes_cell_cycle uses the smaller
available phase-gene count for control size. Missing genes are listed for each line;
no symbol aliases are guessed. At least five matching genes per signature are required.

The notebook reports scores, estimated phase fractions, scores by cluster and
treatment, and score–QC Spearman correlations. UPR is only one stress-related
program. Scores can overlap and are not enrichment p-values. They should not be
compared directly across independent cell-line fits. G1 is an expression-based
phase estimate, not a quiescence diagnosis. No regression is performed.

For review, scores are standardized across cells within a cell line and averaged
by cluster. The largest relative program is a descriptive cue. Every cluster's
annotation status remains unassigned; corroborating markers and program patterns
must be reviewed before adopting biological names.

## Outputs and execution
Open notebooks/nb03_cluster_characterization.ipynb after running nb02.
It embeds both implementation and frozen lists, so no source-file upload or
gene-set download is required in Colab. Live Drive authorization requires the user.

Local generation and validation:

    .conda/bin/python scripts/build_nb03.py
    .conda/bin/python scripts/validate_nb03.py

Compact tables and five figures per line go under results/{tables,figures}/nb03.
The summary records parameters, gene coverage, package version, input fingerprints
and preservation checks. Full marker rankings go to ignored data/processed.
No original nb02 object is overwritten. The next notebook is nb04: treatment DEG + GSEA,
after establishing biological replicate availability.

## Sources
- [Scanpy marker ranking](https://scanpy.readthedocs.io/en/1.11.x/generated/scanpy.tl.rank_genes_groups.html)
- [Scanpy program scoring](https://scanpy.readthedocs.io/en/1.11.x/generated/scanpy.tl.score_genes.html)
- [Scanpy cell-cycle workflow](https://scanpy.readthedocs.io/en/1.11.x/how-to/cell-cycle.html)
- [Cook study cell-cycle list](https://github.com/dpcook/emt_dynamics/blob/master/regev_lab_cell_cycle_genes.txt)
- [Broad MSigDB v7.0 GMT](https://github.com/broadinstitute/ssGSEA2.0/blob/master/db/msigdb/h.all.v7.0.symbols.gmt)

## Executed run
See [initial interpretation](results/tables/nb03/initial_interpretation.md) for tentative biological patterns and the observed T47D clustering sensitivity.
Both cell lines completed; all nb02 input fingerprints, cell order and original cluster labels were preserved.
The full project suite passed 17 tests, including two new tests for count-derived normalization and known synthetic marker recovery. The notebook executed locally; live Colab Drive mounting was not tested.
