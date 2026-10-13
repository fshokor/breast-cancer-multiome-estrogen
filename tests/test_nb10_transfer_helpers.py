"""Synthetic feature-alignment tests only; no checkpoint or notebook execution."""
import ast
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/"src/cross_line_transfer.py").read_text()
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="align_transfer_features")
namespace={"np":np,"pd":pd,"sp":sp}
exec(compile(ast.Module(body=[node],type_ignores=[]),"<alignment helper>","exec"),namespace)
align=namespace["align_transfer_features"]

class TransferFeatureTests(unittest.TestCase):
    def test_reorders_by_feature_identity_not_column_position(self):
        x=sp.csr_matrix([[1,2,3],[4,5,6]],dtype=np.float32)
        result,missing=align(x,["a","b","c"],["c","a"])
        np.testing.assert_array_equal(result.toarray(),[[3,1],[6,4]])
        self.assertFalse(missing.any())
    def test_missing_feature_is_explicit_not_another_gene(self):
        result,missing=align(sp.csr_matrix([[8,2]],dtype=np.float32),["a","b"],["b","absent","a"])
        np.testing.assert_array_equal(result.toarray(),[[2,0,8]])
        np.testing.assert_array_equal(missing,[False,True,False])
    def test_source_mean_imputation_gives_zero_standardized_input(self):
        result,missing=align(sp.csr_matrix([[8]],dtype=np.float32),["a"],["a","absent"])
        center=np.array([2,9],np.float32);scale=np.array([3,2],np.float32)
        dense=result.toarray();dense[:,missing]=center[missing]
        np.testing.assert_array_equal((dense-center)/scale,[[2,0]])
    def test_all_missing_preserves_shape_and_flags(self):
        result,missing=align(sp.csr_matrix([[1],[2]],dtype=np.float32),["a"],["x","y"])
        self.assertEqual(result.shape,(2,2));self.assertTrue(missing.all());self.assertEqual(result.nnz,0)
    def test_duplicate_source_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            align(sp.csr_matrix([[1,2]],dtype=np.float32),["a","a"],["a"])
    def test_duplicate_model_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            align(sp.csr_matrix([[1]],dtype=np.float32),["a"],["a","a"])

if __name__=="__main__":
    unittest.main()
