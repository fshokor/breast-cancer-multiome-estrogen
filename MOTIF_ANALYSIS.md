# nb07 — Motifs in E2-associated accessibility changes

## Question
Which sequence motifs are enriched in peaks more accessible under E2, and which
are enriched in peaks less accessible under E2, relative to comparable background
peaks? Compare MCF7 and T47D, and inspect the corresponding TF RNA and nb06 pathway
context. Motifs indicate sequence compatibility, not actual TF binding or activity.

This is an exploratory extension of nb05–nb06, not a reproduction of scMKL motif
group weights. Scanpy supplies the upstream QC, RNA statistics and ATAC workflow;
MOODS performs DNA motif scanning here, since Scanpy does not supply that step.

## Coordinate audit: unresolved, explicitly tested
GEO records hg38. The original authors' pinned code reads Cell Ranger peak matrices
into Signac and creates a separate MACS2 assay, but does not document the exact
export that generated our GEO TSV row names. Neither coordinate convention is
declared confirmed.

For a name chr-start-end, evaluate:
- zero-based half-open: hg38 bases [start, end);
- one-based closed: hg38 bases [start-1, end).

Scan the longer interval once, then retain only hits fully inside each interval.
The shorter interval is common to both hypotheses. Matching, peak identifiers and
foregrounds are held fixed between coordinate hypotheses; only motif presence
changes. GC and length matching use the common interval. This tests the remaining
one-base boundary ambiguity and does not resolve its historical provenance.
No definitive binding-site BED coordinates or validated peak-gene links are exported.

## Frozen resources and sequence QC
- UCSC hg38 2bit, verified against UCSC's published MD5 and recorded SHA256.
- JASPAR2024 CORE vertebrates, non-redundant PFMs (fixed release; not claimed latest).
- Boyle Lab hg38 ENCODE blacklist v2, frozen by SHA256.
- Preserve the resource URLs, checksums and retrieval date in configs/motif_references.json.

Scan the union of nb05 tested peaks. Keep chr1–22, chrX and chrY only; do not guess
aliases for other contigs. Exclude invalid intervals, any blacklist overlap under
either coordinate hypothesis, and sequences containing bases outside A/C/G/T.
Soft-masked repeat bases are uppercased and included. Sequence exclusion reasons
are saved for every input peak; original nb05 objects and counts are unchanged.

## Motif scan
Use MOODS-python 1.9.4.1, both strands, a uniform A/C/G/T background, total column
pseudocount 0.01 and per-position motif threshold p <= 1e-4. MOODS converts the
PFMs to natural-log odds matrices and computes score thresholds. This site
threshold is not a motif-enrichment FDR. Count each peak once per motif regardless
of the number of sites. Retain IDs, names, motif widths and thresholds.

Both coordinate presence matrices are cached with settings/reference/peak keys.
Changing the inputs invalidates the cache. Motifs can be highly similar; multiple
significant motifs do not imply distinct TF mechanisms.

## Foregrounds, backgrounds and enrichment
Reuse nb05 candidate definitions unchanged (cell-level BH q <= 0.05, absolute
accessibility-fraction difference >= 0.05, and adjusted/raw direction agreement).

Analyze E2-higher and Vehicle-higher peaks separately, per cell line. The latter
are peaks less accessible under E2; enriched motifs in those peaks are not called
“downregulated TFs.”

Controls are eligible tested peaks outside **both** candidate sets. They are
non-candidates, not proven invariant regions. Match each foreground to one unique
control within the same chromosome, without replacement within each comparison.
Use deterministic nearest-neighbor matching on:
- GC fraction (absolute difference <= 0.05);
- natural log length (absolute difference <= 0.25);
- logit accessibility (absolute difference <= 0.5), using the equal-condition mean
  of E2 and Vehicle detection fractions, clipped to [1e-6, 1-1e-6].

Search the nearest 200 controls in scaled feature space. Process foregrounds with
the fewest eligible matches first; select the nearest unused control, with peak
name tie breaking. Save exact pairs and unmatched foregrounds. Controls may be
reused across separate comparisons, never twice within one comparison.

Test motif enrichment using a one-sided exact matched-pair binomial test:
b = pairs with a foreground hit only; c = pairs with a background hit only;
P = Pr[Binomial(b+c, 0.5) >= b]. With no discordant pairs, P = 1.
Report motif fractions, b/c counts and smoothed log2 paired odds
log2[(b+0.5)/(c+0.5)]. With fewer than 20 matched pairs, P remains missing.
These are peak-pair sequence tests, not cell-level or biological-replicate tests.

Apply BH correction within each comparison and globally across all motifs,
two lines, two directions, two detection scenarios and two coordinate hypotheses.
The global q is the primary display value. Peak dependence, correlated motifs,
foreground selection and residual technical confounding limit formal inference.

Require >=80% foreground match coverage, >=20 pairs, and absolute standardized
mean differences <=0.1 for all three matching features before labeling a motif
a priority candidate at global q <=0.05. These are predeclared review thresholds,
not proof of adequate control of all confounders. All results remain available.

## Sensitivity and RNA/pathway context
Repeat the matching and testing after removing peaks detected in only one
condition from both foreground and control eligibility. This can remove genuine
biology as well as library-specific effects; neither result is privileged as truth.

Match motif labels to RNA-tested gene symbols after uppercasing and removing
terminal (var.N) suffixes; split :: dimers into components. This is candidate label
matching across a vertebrate motif collection, not validated orthology or alias
resolution. Retain unmatched labels. RNA fold change and RNA BH q are contextual;
TF RNA expression is not TF protein activity.

For the top 10 priority motifs per line/direction/detection scenario under the shorter coordinate hypothesis, list motif-bearing
matched foreground peaks that have provisional nb06 nearby-gene assignments.
Show their Hallmark membership and existing RNA NES/FDR and nb06 agreement.
These are traceable proximity intersections, not TF targets or pathway-enrichment
tests. Nearby-gene uncertainties from nb06 remain unchanged.

## Outputs and reproduction
Run notebooks/nb07_motif_analysis.ipynb after nb04–nb06. The notebook embeds its
implementation, settings and frozen reference manifest, installs missing scanner
dependencies, and downloads missing references with checksum checks. The hg38
reference is about 800 MB. Colab needs the project inputs in Drive; local execution
does not test interactive Drive authorization.

results/tables/nb07 contains sequence exclusions, all motif tests, background pairs
and balance, sensitivity/cross-line comparisons, TF RNA and pathway context,
reference provenance and validation. results/figures/nb07 contains enrichment and
matching-quality plots. Large presence matrices remain in ignored data/processed/nb07.
Upstream inputs are verified unchanged by SHA256.

## References
- [Original study analysis code](https://github.com/ohsu-cedar-comp-hub/BC_estrogen_TITAN_paper2022/blob/d8cbe099a81c7fb5bd148a845bacba17419feb8e/scripts/multiomics_analysis.md)
- [UCSC hg38 reference](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/)
- [JASPAR 2024 documentation](https://jaspar2024.elixir.no/docs/) and [downloads](https://jaspar2024.elixir.no/downloads/)
- [MOODS](https://github.com/jhkorhonen/MOODS)
- [ENCODE blacklist](https://github.com/Boyle-Lab/Blacklist)
- [Exact matched-pair testing](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)
