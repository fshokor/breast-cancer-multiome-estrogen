import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,statsmodels.api as sm
from scipy import sparse
from src.atac_analysis import tfidf,depth_score_test,nearest_genes
class AtacTests(unittest.TestCase):
    def test_tfidf_formula_preserves_input(self):
        b=sparse.csr_matrix([[1,0,1],[1,1,0],[0,1,0]],dtype=np.uint8)
        original=b.copy();mask=np.array([True,False,True])
        x,idf=tfidf(b,mask,10000)
        dense=b.toarray().astype(float)
        expected=np.log1p(10000*dense[:,mask]/dense.sum(axis=1)[:,None]*(3/dense[:,mask].sum(axis=0)))
        np.testing.assert_allclose(x.toarray(),expected)
        self.assertEqual((b!=original).nnz,0)
    def test_score_matches_statsmodels(self):
        rng=np.random.default_rng(8);n=500
        depth=rng.integers(100,10000,n);z=np.log1p(depth);z=(z-z.mean())/z.std()
        y=rng.binomial(1,1/(1+np.exp(-(.3+.5*z))))
        b=sparse.csr_matrix(rng.binomial(1,.3,(n,4)))
        score,p,valid,_=depth_score_test(b,y,depth)
        model=sm.GLM(y,np.column_stack([np.ones(n),z]),family=sm.families.Binomial())
        fit=model.fit()
        for j in range(4):
            stat,expected,_=model.score_test(fit.params,exog_extra=b[:,j].toarray())
            np.testing.assert_allclose(score[j]**2,stat[0],rtol=1e-8)
            np.testing.assert_allclose(p[j],expected[0],rtol=1e-8)
        self.assertTrue(valid.all())
    def test_constant_peak_not_testable(self):
        b=sparse.csr_matrix(np.ones((100,1)));y=np.tile([0,1],50)
        _,p,valid,_=depth_score_test(b,y,np.arange(100)+10)
        self.assertFalse(valid[0]);self.assertTrue(np.isnan(p[0]))
    def test_coordinate_consensus_and_ties(self):
        var=pd.DataFrame({"contig":["chr1","chr1","missing"],"start_as_supplied":[1,5,1],
            "end_as_supplied":[2,6,2]},index=["near","tie","absent"])
        tss=pd.DataFrame({"contig":["chr1","chr1"],"tss0":[0,10],"gene":["A","B"]})
        out=nearest_genes(var,tss,100)
        self.assertEqual(out.loc["near","candidate_nearby_genes"],"A")
        self.assertEqual(out.loc["tie","candidate_nearby_genes"],"")
        self.assertFalse(out.loc["absent","coordinate_robust_nearest"])
if __name__=="__main__":unittest.main()
