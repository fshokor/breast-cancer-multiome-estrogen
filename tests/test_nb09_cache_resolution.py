"""Filesystem-only regression tests; no model imports or notebook execution."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
tree=ast.parse((ROOT/"src/scmkl_style.py").read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="resolve_kernel_origin")
namespace={"Path":Path,"json":json}
exec(compile(ast.Module(body=[function],type_ignores=[]),"<cache resolver>","exec"),namespace)
resolve=namespace["resolve_kernel_origin"]

class CacheResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
    def run_paths(self,name,source=None):
        out=self.base/"results/tables/nb09"/name
        artifact=self.base/"data/processed/nb09"/name
        out.mkdir(parents=True,exist_ok=True)
        artifact.mkdir(parents=True,exist_ok=True)
        if source is not None:
            (out/"MCF7_RNA_cache.json").write_text(json.dumps({
                "source_run":source,"array":"/obsolete/drive/array.npy","maps":"/obsolete/maps"}))
        return out,artifact
    def test_external_cached_run_resolves_to_original(self):
        original=self.run_paths("original","original")
        cached=self.run_paths("cached","original")
        out,artifacts,chain=resolve(self.base,*cached,"MCF7","RNA")
        self.assertEqual((out,artifacts),original)
        self.assertEqual([p.name for p in chain],["cached","original"])
    def test_multiple_hops_from_current_run(self):
        original=self.run_paths("original","original")
        self.run_paths("intermediate","original")
        current=self.run_paths("current","intermediate")
        out,artifacts,chain=resolve(self.base,*current,"MCF7","RNA")
        self.assertEqual((out,artifacts),original)
        self.assertEqual(len(chain),3)
    def test_no_pointer_keeps_local_origin(self):
        current=self.run_paths("local")
        self.assertEqual(resolve(self.base,*current,"MCF7","RNA")[:2],current)
    def test_cycle_is_rejected(self):
        first=self.run_paths("first","second");self.run_paths("second","first")
        with self.assertRaisesRegex(ValueError,"Cyclic"):
            resolve(self.base,*first,"MCF7","RNA")
    def test_missing_referenced_run_is_actionable(self):
        current=self.run_paths("current","missing")
        with self.assertRaisesRegex(FileNotFoundError,"Restore this run"):
            resolve(self.base,*current,"MCF7","RNA")
    def test_path_traversal_is_rejected(self):
        current=self.run_paths("current","../outside")
        with self.assertRaisesRegex(ValueError,"Invalid"):
            resolve(self.base,*current,"MCF7","RNA")

if __name__=="__main__":
    unittest.main()
