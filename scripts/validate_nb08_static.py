"""Static syntax/schema checks only. Never import or execute model code."""
from pathlib import Path
import ast
import json
import nbformat
root=Path(__file__).resolve().parents[1]
for name in ("src/unimodal_mlp.py","scripts/build_nb08.py"):
    ast.parse((root/name).read_text(),filename=name)
nb=nbformat.read(root/"notebooks/nb08_unimodal_mlp.ipynb",as_version=4)
nbformat.validate(nb)
for i,c in enumerate(nb.cells):
    if c.cell_type=="code":
        ast.parse(c.source,filename=f"nb08 cell {i}")
        assert c.execution_count is None
        assert c.outputs==[]
source=(root/"src/unimodal_mlp.py").read_text()
assert any(c.cell_type=="code" and c.source==source for c in nb.cells)
config=json.loads((root/"configs/unimodal_mlp.json").read_text())
assert any(c.cell_type=="code" and c.source=="CONFIG = "+repr(config)+"\nCONFIG" for c in nb.cells)
for line in ("MCF7","T47D"):
    for modality in ("RNA","ATAC"):
        assert sum(f'"{line}","{modality}",CONFIG' in c.source for c in nb.cells if c.cell_type=="code")==1
assert config["hidden_sizes"]==[512]*5
print("PASS: Python syntax, notebook schema, source/config consistency, four model cells.")
print("PASS: all code cells unexecuted with no outputs. No training or data execution.")
