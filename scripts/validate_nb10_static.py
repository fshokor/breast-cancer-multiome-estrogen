"""Validate nb10 statically only; do not import scientific implementation."""
from pathlib import Path
import ast
import textwrap
import json
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
assert 'NB09_RUN_ID="20261012T114858_335840Z"' in nb.cells[3].source
assert 'matching_acceptable' in source and 'priority_candidate' in source
assert 'best_loss' not in source
assert 'MODES=("RNA_ATAC","RNA")' in source
print("PASS: source/notebook Python syntax, notebook schema, embedded source and review steps.")
print("PASS: original review has no fitting; transfer is checked separately.")
print("PASS: all notebook cells unexecuted with empty outputs.")
print("No result metrics, plots, biological joins or model fitting were executed.")

transfer=(root/"src/cross_line_transfer.py").read_text()
tree=ast.parse(transfer)
for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
        assert node.func.attr not in {"fit","fit_transform","backward","step"}
assert sum(c.cell_type=="code" and c.source=="if RUN_CROSS_LINE_TRANSFER:\n"+textwrap.indent(transfer,"    ") for c in nb.cells)==1
assert 'weights_only=True' in transfer
assert 'np.allclose(own,original,atol=1e-4,rtol=1e-4)' in transfer
assert transfer.index('scorer(job["source"])')<transfer.index('scorer(job["target"])')
assert 'm["split"].eq("test")' in transfer
assert 'atac_training_median_depth' in transfer and 'input_center' in transfer
assert 'PathwayMLPV1' in transfer and 'PathwayMLPV2' in transfer
assert 'transfer_results.csv' in transfer and 'input_artifact_hashes.csv' in transfer
assert any(c.cell_type=="code" and 'TRANSFER_RESULTS = run_cross_line_transfer(' in c.source for c in nb.cells)
print("PASS: frozen inference, source reproduction gate and held-out target selection.")

# The two frozen pathway architectures must exactly match the authored versions.
def definitions(text,names):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body
            if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names}
arch_names=("MaskedPathwayLayer","StochasticPathwayGate","PathwayMLP")
current=(root/"src/pathway_mlp.py").read_text()
archived=json.loads((root/"notebooks/archive/nb11_completed_before_matched_models_12e4459137fd.ipynb").read_text())
legacy=next("".join(c["source"]) for c in archived["cells"] if "".join(c["source"]).startswith('"""nb11:'))
for version,original in ((1,legacy),(2,current)):
    renamed=original
    for name in arch_names:renamed=renamed.replace(name,name+f"V{version}")
    names={name+f"V{version}" for name in arch_names}
    assert definitions(transfer,names)==definitions(renamed,names)
print("PASS: frozen nb11 v1/v2 architectures exactly match their source definitions.")
