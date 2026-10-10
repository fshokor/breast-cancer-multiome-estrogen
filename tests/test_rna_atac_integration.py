import unittest
import numpy as np
import pandas as pd
from src.rna_atac_integration import eligible_links,aggregate_genes,pathway_summary

class IntegrationTests(unittest.TestCase):
    def fixture(self):
        rna=pd.DataFrame({"gene":["A","B","C"],"approx_log2FC_E2_vs_Vehicle":[1,-1,1]})
        da=pd.DataFrame({"peak":["p1","p2","p3","tie","far","exclusive"],
            "candidate_nearby_genes":["A","A","B","A;B","C","C"],
            "testable":[True]*6,"coordinate_robust_nearest":[True]*6,
            "depth_adjusted_score":[2,-1,-4,9,2,8],
            "distance_bp_if_0based":[10,10,10,10,99999.5,10],
            "distance_bp_if_1based":[10,10,10,10,100000.5,10],
            "detected_E2":[5,5,5,5,5,5],"detected_Vehicle":[5,5,5,5,5,0],
            "fraction_difference_E2_minus_Vehicle":[.2,-.1,-.4,.9,.2,.8],
            "exploratory_candidate":[True]*6})
        return rna,da

    def test_mapping_uses_both_conventions_and_excludes_ties_and_exclusive(self):
        rna,da=self.fixture()
        x=eligible_links(da,set(rna.gene),100000,True)
        self.assertEqual(set(x.peak),{"p1","p2","p3"})
        self.assertEqual(len(da),6)

    def test_missing_evidence_is_not_zero_and_mixed_peaks_retained(self):
        rna,da=self.fixture()
        g=aggregate_genes(eligible_links(da,set(rna.gene),100000,True),rna).set_index("gene")
        self.assertTrue(np.isnan(g.loc["C","atac_median_score"]))
        self.assertFalse(g.loc["C","mapped"])
        self.assertTrue(g.loc["A","mixed_peak_directions"])
        self.assertEqual(g.loc["A","atac_median_score"],.5)

    def test_negative_nes_alignment_and_leading_edge(self):
        rna,da=self.fixture()
        genes=aggregate_genes(eligible_links(da,set(rna.gene),100000,True),rna)
        gsea=pd.DataFrame({"Term":["negative"],"NES":[-2],"FDR q-val":[.01],"Lead_genes":["B"]})
        s,_=pathway_summary(genes,gsea,{"negative":["A","B","C"],"untested":["A","C"]},1)
        full=s.loc[(s.pathway=="negative")&(s.subset=="all_members")].iloc[0]
        self.assertEqual(full.n_ATAC_mapped_genes,2)
        self.assertEqual(full.fraction_same_direction,.5)  # genes equally weighted, not three peaks
        lead=s.loc[(s.pathway=="negative")&(s.subset=="leading_edge")].iloc[0]
        self.assertEqual(lead.fraction_same_direction,1)
        self.assertEqual(lead.direction_summary,"majority_same_direction")
        self.assertTrue(s.loc[s.pathway=="untested","rna_NES"].isna().all())
        missing=s.loc[(s.pathway=="untested")&(s.subset=="all_members")].iloc[0]
        self.assertEqual(missing.n_undefined_RNA_direction,1)
        self.assertTrue(np.isnan(missing.fraction_same_direction))

    def test_positive_nes_and_low_coverage(self):
        rna,da=self.fixture()
        genes=aggregate_genes(eligible_links(da,set(rna.gene),100000,True),rna)
        gsea=pd.DataFrame({"Term":["positive"],"NES":[2],"FDR q-val":[.01],"Lead_genes":["A"]})
        s,_=pathway_summary(genes,gsea,{"positive":["A"]},10)
        self.assertEqual(s.iloc[0].fraction_same_direction,1)
        self.assertEqual(s.iloc[0].direction_summary,"limited_ATAC_coverage")

if __name__=="__main__":unittest.main()
