## Initial run interpretation (11 October 2026)

The analysis scanned **879 JASPAR motifs across 48,636 eligible peaks** from the
48,976-peak union of the two nb05 testing sets. Sequence QC excluded 324 blacklist
overlaps, 14 nonstandard-chromosome peaks and 2 ambiguous sequences.

Both coordinate hypotheses were tested. They differed for 7,971 peak–motif
presence entries across 4,660 peaks. This sensitivity check does not establish
which original coordinate convention was used.

| Cell line / peak direction | Matching passes, all candidates | Matching passes, shared detection | Priority motifs, all candidates | Priority motifs, shared detection |
|---|---|---|---:|---:|
| MCF7 / more accessible under E2 | No | Yes | 0 | 18 |
| MCF7 / less accessible under E2 | Yes | Yes | 1 | 0 |
| T47D / more accessible under E2 | No | No | 0 | 0 |
| T47D / less accessible under E2 | Yes | Yes | 4 | 4 |

Counts use the shorter coordinate hypothesis and require both acceptable matching
and global motif BH q <= 0.05. A zero caused by inadequate matching is **not**
evidence of no motif enrichment.

- **MCF7 opening peaks:** ESR1/ESR2-related motifs are enriched in the shared-
  detection analysis, where matching passes (global q approximately 6.6e-12
  and 3.7e-21). These related sequence motifs do not distinguish actual ESR1
  versus ESR2 binding or establish TF activity.
- **T47D closing peaks:** the TEAD1–4 motif family passes the matching and global-q
  criteria in both detection scenarios. These are related motif models, not four
  independent regulatory discoveries; closing-peak enrichment does not prove
  that TEAD activity is inhibited.
- **Matching limitations:** the all-candidate opening sets have residual length
  and accessibility imbalance (maximum absolute SMD approximately 0.129 in MCF7
  and 0.200 in T47D). T47D shared-detection opening peaks retain length imbalance
  (SMD approximately 0.122). Their enrichment statistics remain visible, but these
  groups are not promoted to priority findings.
- The single MCF7 closing-peak priority motif does not pass global q in the shared-
  detection analysis, so this result is sensitive to the foreground definition.

The coordinate and matching flags, no-biological-replicate limitation, and
provisional motif-label and nearby-gene assignments must accompany these findings.
