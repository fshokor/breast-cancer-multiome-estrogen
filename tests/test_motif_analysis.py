import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from src.motif_analysis import (paired_enrichment,matched_background,build_scanner,
    scan_union_sequence,motif_gene_labels,parse_jaspar)

class MotifTests(unittest.TestCase):
    def test_exact_paired_test_and_direction(self):
        fg=np.array([[1],[1],[1],[1],[0]],dtype=bool)
        bg=np.array([[0],[0],[0],[1],[1]],dtype=bool)
        r=paired_enrichment(fg,bg).iloc[0]
        self.assertEqual(r.foreground_only_pairs,3)
        self.assertEqual(r.background_only_pairs,1)
        self.assertAlmostEqual(r.peak_pair_p,binomtest(3,4,.5,alternative="greater").pvalue)
        self.assertGreater(paired_enrichment(bg,fg).iloc[0].peak_pair_p,r.peak_pair_p)

    def test_no_discordance_is_not_evidence(self):
        a=np.ones((10,2),dtype=bool)
        self.assertTrue(paired_enrichment(a,a).peak_pair_p.eq(1).all())

    def test_matching_unique_within_calipers_and_chromosome(self):
        fg=pd.DataFrame({"peak":["f1","f2","f3"],"chrom":["chr1","chr1","chr2"],
            "gc_fraction":[.4,.41,.5],"log_length":[6,6,6],"logit_accessibility":[-1,-1,-1]})
        bg=pd.DataFrame({"peak":["b1","b2","bad"],"chrom":["chr1","chr1","chr2"],
            "gc_fraction":[.4,.41,.9],"log_length":[6,6,6],"logit_accessibility":[-1,-1,-1]})
        cfg={"matching_calipers":{"gc_fraction":.05,"log_length":.25,"logit_accessibility":.5},
             "matching_nearest_candidates":200}
        pairs,missing=matched_background(fg,bg,cfg)
        self.assertEqual(len(pairs),2);self.assertTrue(pairs.background_peak.is_unique)
        self.assertEqual(missing,["f3"])
        self.assertTrue(pairs.foreground_peak.str.startswith("f").all())

    def test_coordinate_boundary_and_reverse_complement_scanning(self):
        # Strong ACGTAC motif; its reverse complement is GTACGT.
        motif={"matrix":[[100 if x==base else 0 for x in "ACGTAC"] for base in "ACGT"]}
        cfg={"sequence_background":[.25]*4,"pseudocount_total_per_column":.01,
             "motif_site_pvalue":.001,"site_scan_window":3}
        scanner,_=build_scanner([motif],cfg)
        a,b=scan_union_sequence(scanner,"ACGTACAAAA",1)
        self.assertFalse(a[0]);self.assertTrue(b[0]) # motif begins only at added leftmost base
        a,b=scan_union_sequence(scanner,"AAGTACGTAA",1)
        self.assertTrue(a[0]);self.assertTrue(b[0])  # reverse-complement match contained in both

    def test_motif_labels_preserve_dimer_components_and_missing(self):
        self.assertEqual(motif_gene_labels("Esr1(var.2)::FOXA1",{"ESR1"}),
                         [("ESR1",True),("FOXA1",False)])

    def test_jaspar_matrix_parser(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"test.jaspar"
            f.write_text(">MA0000.1 TEST\nA [ 1 2 ]\nC [ 3 4 ]\nG [ 5 6 ]\nT [ 7 8 ]\n")
            m=parse_jaspar(f)[0]
            self.assertEqual(m["matrix"],[[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
            self.assertEqual(m["motif_name"],"TEST")

if __name__=="__main__":unittest.main()
