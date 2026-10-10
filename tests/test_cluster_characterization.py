import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from src.cluster_characterization import normalized_counts,rank_markers

class ClusterTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(12)
        x=rng.poisson(.5,(60,30)).astype(np.int32)
        x[:30,:3]+=6
        x[30:,3:6]+=6
        self.a=ad.AnnData(sparse.csr_matrix(x),
            obs=pd.DataFrame({'leiden':pd.Categorical(['0']*30+['1']*30)},index=[f'c{i}' for i in range(60)]),
            var=pd.DataFrame(index=[f'g{i}' for i in range(30)]))
        self.a.layers['counts']=self.a.X.copy()
        self.a.X.data[:]=99 # The helper must ignore .X.
        self.a.obsm['X_umap']=rng.normal(size=(60,2))
        self.a.obsp['connectivities']=sparse.eye(60,format='csr')
        self.a.uns['neighbors']={'connectivities_key':'connectivities'}
    def test_count_source_and_input_preservation(self):
        original=self.a.layers['counts'].copy()
        a=normalized_counts(self.a,10000)
        expected=original.toarray().astype(float)
        expected=np.log1p(expected/expected.sum(axis=1)[:,None]*10000)
        np.testing.assert_allclose(a.X.toarray(),expected)
        self.assertEqual((original!=self.a.layers['counts']).nnz,0)
        self.assertTrue(a.obs_names.equals(self.a.obs_names))
        np.testing.assert_array_equal(a.obsm['X_umap'],self.a.obsm['X_umap'])
    def test_known_markers_and_complete_ranking(self):
        a=normalized_counts(self.a,10000)
        cfg={'marker_method':'wilcoxon','tie_correct':True,'marker_min_log2fc':.25,
             'marker_min_pct':.1,'marker_min_pct_difference':.1,
             'marker_max_adjusted_p':.05,'top_markers_per_cluster':50}
        full,top=rank_markers(a,cfg)
        self.assertEqual(len(full),60)
        for group,genes in [('0',{'g0','g1','g2'}),('1',{'g3','g4','g5'})]:
            self.assertTrue(genes.issubset(set(top.loc[top.group==group,'names'])))
        self.assertTrue(full.pvals_adj.between(0,1).all())
if __name__=='__main__': unittest.main()
