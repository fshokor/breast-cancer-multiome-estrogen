# nb04 — Exploratory treatment DEG and Hallmark GSEA

## Design and inferential scope
Analyze E2 versus Vehicle independently within MCF7 and T47D, using nb02's retained
cells and all retained RNA genes as the starting universe. nb03 clusters are not
used for treatment testing or filtering.

The GEO family SOFT audit identifies GSM6418443 (multiomic/GEX E2) and GSM6418444
(multiomic/GEX Vehicle), each jointly containing MCF7 and T47D. GSM6418445/6 are
paired ATAC, not biological replicates. The processed metadata provide condition
labels but no biological replicate mapping. We therefore cannot estimate
between-culture variance or run a valid replicated pseudobulk design from these
objects. Cells are not substituted for biological replicates, and no artificial
pseudoreplicates are constructed.

Outputs are exploratory cell-level differential expression and preranked GSEA,
not replicate-supported treatment inference. Concordance across cell lines is
descriptive because the two lines also share submitted condition libraries.
The audit URL, downloaded-record SHA-256 and selected sample fields are saved in
results/tables/nb04/geo_design_audit.json. Lack of replicate mapping is not a claim
that no cultures were pooled upstream; that information is unavailable here.

## Expression and gene universe
Start from integer layers['counts'], never scaled HVGs or unnormalized log counts.
Normalize using all QC-retained genes to 10,000 counts per cell, then log1p.
For testing, retain genes detected in at least 10 cells and at least 5% of cells
within either E2 or Vehicle. The same condition must meet both criteria.
This supports genes expressed primarily in one condition. Save every gene's
detection counts/fractions and test-inclusion mask. No cells are removed.

## Exploratory DEG
Scanpy rank_genes_groups uses Wilcoxon, tie correction and BH adjustment over
the tested genes within each cell line. E2 is the target; Vehicle is the reference.
Positive scores and approximate log2 fold changes point toward higher E2 expression.
The fold changes are Scanpy's approximate values based on mean log expression.
Cell-level p and BH q columns are named explicitly to prevent confusion with
biological-replicate inference.

The candidate display threshold is cell-level BH q<=0.05 and absolute approximate
log2FC>=0.5. Full results remain available. Volcano q values are clipped at 1e-300
for display only. Zero numerical p values do not represent exact zero probabilities.
No cell-cycle adjustment, cluster-specific DE or treatment-label permutation is used.
Composition and cell-state changes can contribute to these observed differences.

## Preranked GSEA
Use every tested gene's signed Wilcoxon score, including nonsignificant genes.
Do not use only the DEG candidate list. Sort descending, with alphabetical gene
symbols breaking exact ties, preserve that order in GSEApy via ascending=None,
and record the tied-score fraction. No artificial jitter or symbol renaming is used.

Use all 50 frozen MSigDB v7.0 Hallmarks from Broad's ssGSEA2.0 repository.
This is the same release used for nb03's scoring, not the latest release.
The complete collection and checksum are embedded in configs/hallmark_v7.json
and the standalone notebook; reruns require no online gene-set lookup.
Exact symbols are matched; each pathway's original and overlapping sizes are saved.
Retain pathways with 15–500 matching ranked genes.

GSEApy 1.3.1 uses classic gene-set permutations: 1,000 permutations, weight=1,
seed=42, one thread. Positive NES is enrichment toward the E2-ranked end;
negative NES points toward Vehicle. FDR is evaluated across the tested Hallmarks
separately for each cell line. Gene-set permutations do not recover biological
replicate uncertainty. Nominal p=0 means below sampled resolution, not zero probability.

Save all results and leading-edge genes. Pathway bar charts display the ten most
positive and ten most negative NES values regardless of FDR; consult the full
table rather than interpreting every bar as significant. Running curves display
estrogen-early, estrogen-late and the strongest positive/negative pathways, with
duplicate terms removed. The cross-line heatmap shows all tested Hallmarks.

## Files and validation
- notebooks/nb04_treatment_deg_gsea.ipynb: executed, standalone Colab notebook.
- results/tables/nb04: design audit, gene universes, all DEG results, exploratory
  candidates, full GSEA rankings, pathway coverage, GSEA tables, comparisons and summary.
- results/figures/nb04: volcanoes, NES plots, running enrichment curves and comparison.
- data/processed/*_nb04_gsea_curves.json: complete running curves, ignored by Git.
- src/treatment_deg_gsea.py: canonical implementation.
- configs/treatment_deg_gsea.json and configs/hallmark_v7.json: reproducible settings/reference.

The notebook hashes input RNA objects before and after analysis. It never overwrites
nb02 objects or edits their counts, cells, clusters or embeddings.
Synthetic tests verify expected up/down gene direction, expression filtering, count
preservation and signed GSEA enrichment. Real-data checks verify that every tested
gene is ranked once and all reported q values lie within [0,1].

    .conda/bin/python scripts/build_nb04.py
    .conda/bin/python scripts/validate_nb04.py

Live Colab Drive authorization is not tested by local notebook execution.

## Sources
- [GEO series](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE154873)
- [E2 RNA record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6418443)
- [Vehicle RNA record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6418444)
- [Scanpy rank_genes_groups](https://scanpy.readthedocs.io/en/1.11.x/generated/scanpy.tl.rank_genes_groups.html)
- [GSEApy prerank](https://gseapy.readthedocs.io/en/latest/run.html)
- [Frozen Hallmark GMT](https://github.com/broadinstitute/ssGSEA2.0/blob/master/db/msigdb/h.all.v7.0.symbols.gmt)

## Results from the executed notebook

| Cell line | Genes tested | Higher in E2 candidates | Higher in Vehicle candidates | Hallmarks tested |
|---|---:|---:|---:|---:|
| MCF7 | 11,650 | 1,584 | 1,457 | 49 |
| T47D | 11,918 | 1,243 | 1,138 | 49 |

Candidate counts use cell-level BH q<=0.05 and absolute approximate log2FC>=0.5.
They are exploratory, not replicate-supported DEG counts.
Estrogen-response early/late, E2F-target and G2M-checkpoint sets enrich toward E2
in both lines. Across 10,733 shared tested genes, approximate log2 fold changes
have Spearman correlation 0.707; this is descriptive concordance.
All 19 project tests passed and both original nb02 RNA files retained their hashes.
The notebook was executed locally; live Colab Drive mounting was not tested.
