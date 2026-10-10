# nb06 — RNA pathway and nearby ATAC comparison

This notebook examines **both positive and negative RNA GSEA NES** from nb04.
Positive NES means enrichment toward E2; negative NES means enrichment toward
Vehicle in the existing E2-versus-Vehicle ranking. This comparison asks whether
nearby accessibility changes agree with those RNA directions. It does not measure
pathway activity directly.

## Scope and methods
Reuse nb04 RNA statistics, the frozen MSigDB v7.0 Hallmarks and nb05 depth-adjusted
peak statistics. Scanpy produced the upstream RNA/ATAC analyses; this notebook
joins those outputs and summarizes them with pandas/NumPy. It does not rerun QC,
normalization, clustering or GSEA.

1. Use **all tested finite-score ATAC peaks**, not just significant peaks. Keep
   only a single nearest gene whose assignment agrees under both coordinate
   interpretations and whose distance is within the specified limit under both.
   Exclude tied nearest-gene assignments; use exact RNA gene symbols without alias
   guessing. Retain the full RNA-tested gene universe, including unmapped genes.
2. For each gene, take the **median signed depth-adjusted ATAC score** across
   eligible peaks. Also retain median raw accessibility-fraction change, peak
   counts, mixed directions, condition-exclusive counts and directional nb05
   candidate counts. A gene's median is a descriptive summary, not a statistical
   gene score or gene-level p-value. Missing peak evidence stays missing, not zero.
3. For every Hallmark, analyze all RNA-tested member genes and, separately, its
   existing RNA GSEA leading-edge genes. Give each mapped gene equal weight.
   Report the fraction of nonzero ATAC gene medians with the same sign as RNA NES,
   its denominator, coverage, opposite/zero counts and the pathway's median ATAC
   score. The ATAC score is **not an NES** and is not calibrated across genes.
   No new pathway p-values or FDR estimates are calculated.
4. Repeat for 100 kb and 10 kb distance limits, each with and without peaks
   detected in only one condition. "Shared detection" means detection in both
   conditions, not biological replication. These sensitivity settings are
   exploratory choices, not paper parameters; 10 kb is not a promoter definition.
5. Preserve all 50 Hallmarks. Untested RNA sets retain missing NES/FDR.
   RNA FDR <= 0.05 is a display flag, not an entry filter. Summaries with fewer
   than 10 mapped genes receive a limited-coverage label; 10 is a display rule,
   not a validated evidence threshold. A majority-direction label describes only
   the observed count and does not establish statistical support.
6. Compare MCF7 and T47D separately and side by side. Pathway memberships overlap,
   so rows do not represent independent evidence.

## Interpretation limits
- RNA GSEA and peak tests are exploratory without identifiable biological
  replicates. Cell counts do not replace independent cultures.
- Residual depth effects, condition/library confounding and cell-state composition
  can produce agreement or disagreement. Removing condition-exclusive peaks is
  a sensitivity analysis, not proof that those peaks are artifacts.
- hg38 is documented, but peak coordinate origin remains unresolved. Nearest-TSS
  assignments are provisional. Coordinate agreement does not validate a target;
  enhancers can act on other genes and accessibility can have complex effects.
- This is a proximity-based directional comparison, **not** paired-cell
  peak-expression correlation, motif enrichment, or validated regulatory linkage.
  Opposing directions need not mean either modality is wrong.
- GSEA permutation estimates reported as zero remain as supplied; zero is not a
  literal probability of no error. RNA FDR cannot be reused as an ATAC FDR.

## Outputs and rerun
Run notebooks/nb06_rna_atac_integration.ipynb after nb04 and nb05.
On Colab mount the project folder in Drive, including the full nb05 DA tables
under data/processed and upstream configs/tables. Code, settings and frozen
Hallmarks are embedded; input tables are required. Local execution does not
validate interactive Drive authorization.

Outputs in results/tables/nb06 include:
- gene evidence for all four mapping scenarios;
- provisional primary peak-to-gene links and pathway gene details;
- pathway summaries for both member subsets and all four scenarios;
- mapping coverage audits, cross-line comparisons, source hashes and validation.

Figures in results/figures/nb06 show both RNA NES directions, ATAC directional
agreement and gene coverage, plus leading-edge sensitivity plots.
Original input files are checked by SHA256 before and after analysis.

## References
- [GSEA User Guide](https://docs.gsea-msigdb.org/GSEA/GSEA_User_Guide/):
  interpretation of enrichment direction and leading-edge subsets.
- [Signac LinkPeaks](https://stuartlab.org/signac/reference/linkpeaks):
  an example of actual peak-expression correlation analysis, which is not
  performed by this proximity comparison.
