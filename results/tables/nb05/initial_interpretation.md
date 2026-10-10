## Initial run interpretation (10 October 2026)

| Cell line | Cells | ATAC clusters | Tested peaks | Exploratory DA candidates | Candidates detected in only one condition |
|---|---:|---:|---:|---:|---:|
| MCF7 | 6,019 | 6 | 30,678 | 1,400 | 573 |
| T47D | 6,562 | 9 | 44,380 | 2,367 | 998 |

Candidates satisfy cell-level BH q <= 0.05, an absolute accessibility-fraction difference >= 0.05, and agreement between adjusted and raw directions. These are exploratory associations, not replicate-supported differential accessibility.

LSI1 strongly correlates with depth (Spearman 0.998 / 0.999) and is excluded from the graph. LSI2 still correlates with depth (-0.613 / -0.605); excluding LSI1 does not eliminate depth effects. Interpret UMAP and clusters alongside the QC panels.

About 41% / 42% of candidate peaks are detected in only one condition. This may include biological differences, but condition-specific library effects or peak-calling differences cannot be separated with the available metadata. Depth adjustment cannot resolve this confounding.

Nearest-TSS gene annotations remain provisional: hg38 is documented, but the peak coordinate convention is unresolved. Agreement under both coordinate interpretations does not establish a regulatory target. No motif or causal regulatory conclusions are made.
