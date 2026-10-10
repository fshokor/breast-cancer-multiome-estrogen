## Initial interpretation of this run
These are tentative descriptions of the original nb02 clusters, not final cell-type labels.

| Cell line / cluster | Review hypothesis | Evidence in this run |
|---|---|---|
| MCF7 / 0 | Estrogen-response-associated expression | PGR, GREB1 and TFF1 among leading markers; relatively high estrogen score; 99.8% E2 cells |
| MCF7 / 2 | G2M-associated state | Highest cluster-mean G2M score; PIF1, KIF14 and HMMR among markers |
| MCF7 / 3 | S-phase-associated state | Highest cluster-mean S score; HELLS and ATAD2 among markers |
| MCF7 / 6 | UPR-associated state | Highest cluster-mean UPR score; GDF15 and DDIT3 among markers |
| T47D / 1 | Estrogen-response-associated expression | NRIP1, GREB1 and PGR among markers; highest cluster-mean estrogen score; 98.5% E2 cells |
| T47D / 6 | G2M-associated state | Highest cluster-mean G2M score; PIF1, KIF14 and CENPE among markers |
| T47D / 5 | Lower-depth cluster requiring QC review | Median 2,405.5 RNA counts and 1,398.5 detected genes; do not label it a biological state based only on its separation |

Treatment composition and cell-cycle variation both contribute to the observed structure.
These observations do not establish a causal treatment effect. MCF7 cluster 8 contains
only 27 cells and deserves separate scrutiny before annotation.

Replaying resolution 0.5 / seed 42 on the saved graph agrees with nb02 at ARI **0.9973
for MCF7** and **0.6534 for T47D**; a second repeat on the same saved graph is identical.
The reason for the difference from the original fit has not been established.
Additional resolution/seed results also show that exact boundaries are provisional,
especially in T47D. nb03 retains all original nb02 labels and makes no automatic
reassignment or cell exclusion. Review stability before downstream cluster-specific claims.
