"""Validate nb10 statically only; do not import scientific implementation."""
from pathlib import Path
import ast
import nbformat
root=Path(__file__).resolve().parents[1]
source=(root/"src/model_comparison.py").read_text()
tree=ast.parse(source)
for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
        assert node.func.attr not in {"fit","fit_transform","predict","predict_proba","backward"}
    if isinstance(node,(ast.Import,ast.ImportFrom)):
        names=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names]
        assert not any(n and n.split(".")[0] in {"torch","celer","tensorflow","scmkl"} for n in names)
for name in ("scripts/build_nb10.py","scripts/validate_nb10_static.py"):
    ast.parse((root/name).read_text(),filename=name)
nb=nbformat.read(root/"notebooks/nb10_model_comparison_interpretation.ipynb",as_version=4)
nbformat.validate(nb)
for i,c in enumerate(nb.cells):
    if c.cell_type=="code":
        ast.parse(c.source,filename=f"nb10 cell {i}")
        assert c.execution_count is None and c.outputs==[]
assert sum(c.cell_type=="code" and c.source==source for c in nb.cells)==1
for name in ("prepare_review","review_predictions","performance_plots","review_groups",
             "biological_context","export_review"):
    assert any(c.cell_type=="code" and name+"(" in c.source and c.source!=source for c in nb.cells)
assert "model_comparison_source.py" in nb.cells[7].source
assert not any("model_comparison_source.py" in c.source for c in nb.cells if c.cell_type=="markdown")
assert 'NB09_RUN_ID=""' in nb.cells[3].source
assert 'matching_acceptable' in source and 'priority_candidate' in source
assert 'best_loss' not in source
print("PASS: source/notebook Python syntax, notebook schema, embedded source and review steps.")
print("PASS: no model-fitting/prediction calls or model-training library imports.")
print("PASS: all notebook cells unexecuted with empty outputs.")
print("No result metrics, plots, biological joins or model fitting were executed.")
