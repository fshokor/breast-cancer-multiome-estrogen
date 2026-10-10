import gzip
import json
import tempfile
import unittest
import copy
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from src.qc_preprocessing import read_sparse_counts, rna_qc, sparse_log_and_scale, centered_scaled_block, read_metadata, visualize_rna

class PaperQCTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / "configs/qc_paper.json").read_text())

    def test_published_boundaries(self):
        x = np.ones((6,102), dtype=np.int32)
        x[:,100:] = 0
        x[1,99] = 0
        x[2,:100], x[2,100] = 7, 300
        x[3,:100], x[3,0], x[3,100] = 7, 6, 301
        x[4,100] = 1
        x[:2,101] = 1
        genes = pd.Index([f"G{i}" for i in range(100)] + ["MT-A","RARE"])
        keep, gene_keep, qc, var = rna_qc(sparse.csr_matrix(x), genes, self.config)
        self.assertEqual(keep.tolist(), [True,False,True,False,True,True])
        self.assertTrue(gene_keep[100])  # Exactly three cells retained.
        self.assertFalse(gene_keep[101])  # Only two cells detected.
        self.assertEqual(qc.loc[0,"rna_n_genes"], 100)
        self.assertAlmostEqual(qc.loc[2,"rna_pct_counts_mt"], 30.0)
        self.assertAlmostEqual(qc.loc[3,"rna_pct_counts_mt"], 30.1)

    def test_scaled_representation_matches_dense(self):
        x = sparse.csr_matrix([[0,2,3],[1,2,0],[5,2,2]], dtype=np.int32)
        log, scaled, mean, std = sparse_log_and_scale(x)
        expected = np.log1p(x.toarray().astype(np.float32))
        expected_std = expected.astype(np.float64).std(axis=0,ddof=1)
        expected_std[expected_std == 0] = 1
        np.testing.assert_allclose(std, expected_std, rtol=1e-6, atol=1e-8)
        a=ad.AnnData(log)
        a.layers["scaled_no_center"]=scaled
        a.var["scale_center_offset"]=mean/std
        np.testing.assert_allclose(centered_scaled_block(a,slice(None),slice(None)),
                                   (expected-expected.astype(np.float64).mean(axis=0))/expected_std,atol=2e-6)
        self.assertTrue(sparse.issparse(a.X))

    def test_streaming_loader_preserves_orientation_and_integers(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"counts.gz"
            with gzip.open(p,"wt") as f:f.write("cell1\tcell2\nG1\t0\t4\nMT-A\t3\t0\n")
            x,cells,genes=read_sparse_counts(p)
            np.testing.assert_array_equal(x.toarray(),[[0,3],[4,0]])
            self.assertEqual(x.dtype,np.int32)
            self.assertEqual(list(genes),["G1","MT-A"])
            self.assertEqual(list(cells),["cell1","cell2"])

    def test_fractional_counts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"counts.gz"
            with gzip.open(p,"wt") as f:f.write("cell1\tcell2\nG1\t0.5\t4\n")
            with self.assertRaisesRegex(ValueError,"Invalid integer"):
                read_sparse_counts(p)

    def test_metadata_reordering_not_silent(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"meta.gz"
            with gzip.open(p,"wt") as f:f.write("b\tYW_si_Control_plus_E2\na\tYW_si_Control_plus_Veh\n")
            with self.assertRaisesRegex(ValueError,"order"):
                read_metadata(p,["a","b"],"MCF7")

    def test_missing_mitochondrial_annotation_fails(self):
        with self.assertRaisesRegex(ValueError,"No mitochondrial"):
            rna_qc(sparse.csr_matrix(np.ones((4,3))),pd.Index(["A","B","C"]),self.config)


class RNAVisualizationTests(unittest.TestCase):
    def test_scanpy_embeddings_preserve_counts_and_cell_order(self):
        root = Path(__file__).resolve().parents[1]
        config = copy.deepcopy(json.loads((root/"configs/qc_paper.json").read_text()))
        config["visualization"].update(n_top_genes=30, pca_components=10,
                                       neighbors_n_pcs=8, n_neighbors=5, umap_maxiter=30)
        rng = np.random.default_rng(42)
        counts = sparse.csr_matrix(rng.poisson(2, (80, 120)).astype(np.int32))
        obs = pd.DataFrame({"treatment":["Vehicle"]*40+["E2"]*40,
                            "rna_total_counts":np.asarray(counts.sum(axis=1)).ravel(),
                            "rna_pct_counts_mt":rng.uniform(0,20,80)},
                           index=[f"cell{i}" for i in range(80)])
        a = ad.AnnData(X=counts.astype(np.float32), obs=obs,
                      var=pd.DataFrame(index=[f"gene{i}" for i in range(120)]))
        a.layers["counts"] = counts.copy()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for sub in ("results/tables","results/figures","data/processed"):
                (folder/sub).mkdir(parents=True)
            result = visualize_rna(a, folder, "TEST", config)
            self.assertEqual(a.obsm["X_pca"].shape,(80,10))
            self.assertEqual(a.obsm["X_umap"].shape,(80,2))
            self.assertTrue(np.isfinite(a.obsm["X_umap"]).all())
            self.assertTrue(a.obs["leiden"].notna().all())
            self.assertEqual(a.obsp["connectivities"].shape,(80,80))
            self.assertEqual((a.layers["counts"] != counts).nnz,0)
            self.assertEqual(list(a.obs_names),list(obs.index))
            self.assertGreaterEqual(result["n_hvgs"],30)
            v = ad.read_h5ad(folder/"data/processed/TEST_RNA_visualization.h5ad")
            self.assertEqual(list(v.obs_names),list(a.obs_names))
            self.assertEqual((v.layers["counts"] != counts[:,a.var.highly_variable.to_numpy()]).nnz,0)
            self.assertTrue((folder/"results/figures/TEST_rna_umap.png").is_file())
