import unittest,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,anndata as ad
from scipy import sparse
from src.treatment_deg_gsea import prepare_expression,differential_expression,enrichment
ROOT=Path(__file__).resolve().parents[1]
class TreatmentTests(unittest.TestCase):
    def test_direction_universe_and_counts(self):
        cfg=json.loads((ROOT/"configs/treatment_deg_gsea.json").read_text())
        rng=np.random.default_rng(4);x=rng.poisson(1,(80,60)).astype(np.int32)
        x[:40,:5]+=8;x[40:,5:10]+=8;x[:,59]=0
        a=ad.AnnData(sparse.csr_matrix(x),obs=pd.DataFrame({'treatment':['E2']*40+['Vehicle']*40},index=[f'c{i}' for i in range(80)]),
                    var=pd.DataFrame(index=[f'g{i}' for i in range(60)]))
        a.layers['counts']=a.X.copy();a.X.data[:]=99
        norm,audit=prepare_expression(a,cfg)
        self.assertFalse(audit.loc['g59','tested'])
        self.assertEqual(norm.n_vars,59)
        self.assertEqual((a.layers['counts']!=sparse.csr_matrix(x)).nnz,0)
        de,ranks=differential_expression(norm,cfg)
        self.assertEqual(len(ranks),norm.n_vars)
        d=de.set_index('gene')
        self.assertTrue((d.loc[['g0','g1','g2'],'wilcoxon_score']>0).all())
        self.assertTrue((d.loc[['g5','g6','g7'],'wilcoxon_score']<0).all())
        self.assertTrue(ranks.wilcoxon_score.is_monotonic_decreasing)
    def test_gsea_signed_direction_and_full_rank(self):
        cfg=json.loads((ROOT/"configs/treatment_deg_gsea.json").read_text())
        cfg.update(gsea_min_size=5,gsea_max_size=50,gsea_permutations=100)
        ranks=pd.DataFrame({'gene':[f'G{i}' for i in range(100)],'wilcoxon_score':np.linspace(5,-5,100)})
        ref={'gene_sets':{'UP':[f'G{i}' for i in range(15)],
                         'DOWN':[f'G{i}' for i in range(85,100)]}}
        result,frame,coverage=enrichment(ranks,ref,cfg)
        frame=frame.set_index('Term')
        self.assertGreater(frame.loc['UP','NES'],0)
        self.assertLess(frame.loc['DOWN','NES'],0)
        self.assertEqual(len(result.ranking),100)
        self.assertTrue(coverage.tested.all())
if __name__=='__main__':unittest.main()
