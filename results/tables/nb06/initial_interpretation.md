## Initial run findings (10 October 2026)

These are descriptive observations from the saved run; the tables above are the
authoritative outputs if the notebook is rerun with changed inputs.

| Cell line | Positive RNA NES | Negative RNA NES | Positive RNA FDR <= 0.05 | Negative RNA FDR <= 0.05 |
|---|---:|---:|---:|---:|
| MCF7 | 25 | 24 | 16 | 0 |
| T47D | 18 | 31 | 13 | 13 |

One of the 50 Hallmarks was not tested in each line by nb04 and retains an
undefined RNA direction here. Negative NES alone does not establish a reduced
expression signature at the chosen FDR threshold; in particular, **none of the
negative MCF7 pathways meets RNA FDR <= 0.05**.

With the primary 100 kb mapping, 9,316 of 11,650 RNA-tested MCF7 genes and 9,706 of
11,918 T47D genes have eligible nearby tested peaks. Unmapped genes remain missing
evidence, not evidence of unchanged accessibility.

For the RNA leading-edge genes of estrogen response early/late, the fractions of
mapped genes with ATAC medians agreeing with positive RNA NES are approximately
74% / 72% in MCF7 and 64% / 69% in T47D. Under the tighter 10 kb mapping with
condition-exclusive peaks removed, they are 74% / 75% and 52% / 63%, respectively.
Thus the early estrogen-response concordance in T47D is sensitive to mapping
choices and becomes close to evenly split. These fractions are not effect sizes,
ATAC enrichment statistics, or probabilities of pathway activation.

Inspect all pathways, including opposing directions, and the coverage and gene
tables before prioritization. No replicate-supported or causal regulatory claim
follows from these descriptive summaries.
