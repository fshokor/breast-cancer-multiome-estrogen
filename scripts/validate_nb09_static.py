"""Static checks only. Never imports model code or executes a notebook cell."""
from pathlib import Path
import ast
import json
import nbformat

root=Path(__file__).resolve().parents[1]
source=(root/"src/scmkl_style.py").read_text()
tree=ast.parse(source)
assert all(isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.Expr)) for n in tree.body)
# There must be no top-level executable calls beyond the docstring.
for n in tree.body:
    if isinstance(n,ast.Expr):
        assert isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)
functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
assert len(functions)==sum(isinstance(n,ast.FunctionDef) for n in tree.body)
for filename in ("scripts/build_nb09.py","scripts/validate_nb09_static.py"):
    ast.parse((root/filename).read_text(),filename=filename)
nb=nbformat.read(root/"notebooks/nb09_scmkl_style_integration.ipynb",as_version=4)
nbformat.validate(nb)
for i,c in enumerate(nb.cells):
    if c.cell_type=="code":
        ast.parse(c.source,filename=f"nb09 cell {i}")
        assert c.execution_count is None
        assert c.outputs==[]
assert sum(c.cell_type=="code" and c.source==source for c in nb.cells)==1
config=json.loads((root/"configs/scmkl_style.json").read_text())
assert any(c.cell_type=="code" and c.source=="CONFIG = "+repr(config)+"\nCONFIG" for c in nb.cells)
assert config["alpha_ratios"]==[.1]
fit_source=ast.get_source_segment(source,functions["fit_group_models"])
assert 'for mode,modalities in (("RNA_ATAC",["RNA","ATAC"]),("RNA",["RNA"])):' in fit_source
assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="fit"
           for n in ast.walk(functions["fit_group_models"]))==1
assert "fit_checkpoint.npz" in fit_source
assert "verbose=True" in fit_source
assert "with solver_progress(out,tag,config):" in fit_source
assert "fit_seconds" in fit_source and "solver_outer_iterations" in fit_source
assert "solver_progress" in functions and "counted_progress" in functions
assert config["rff_frequencies"]==32
assert any(c.cell_type=="code" and 'NB08_RUN_ID="20261011T103750_408739Z"' in c.source for c in nb.cells)
setup=[c for c in nb.cells if c.cell_type=="code" and "BASELINE, OUTPUT, ARTIFACTS" in c.source]
assert len(setup)==1 and "scmkl_style_rna_extension_source.py" in setup[0].source
for line in ("MCF7","T47D"):
    assert sum(c.cell_type=="code" and c.source.startswith(f"{line}_PATHS, {line}_TABLES = construct_line")
               for c in nb.cells)==1
    assert sum(c.cell_type=="code" and c.source.startswith(f"display(fit_group_models({line}_PATHS")
               for c in nb.cells)==1
# Ensure reused preprocessing remains identical to nb08.
old=(root/"src/unimodal_mlp.py").read_text()
old_functions={n.name:n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
for name in ("sha256","input_path","annotations","load_features"):
    assert ast.dump(functions[name],include_attributes=False)==ast.dump(old_functions[name],include_attributes=False)
# Structural guards for scientific scope. These supplement review, not runtime tests.
assert 'train_test_split' not in source
assert '"layers/counts" if modality == "RNA" else "X"' in source
assert 'presence[i]=common & extended' in source
assert 'score_rows(paths,modalities,test,best["coef"],best["intercept"])' in source
assert 'best_loss' not in source
assert 'scmkl_style_source.py' not in ''.join(c.source for c in nb.cells if c.cell_type=="markdown")
print("PASS: syntax, schema, configuration, embedded source and reused preprocessing consistency.")
print("PASS: explicit nb08 run, both cell lines, RNA-only and integrated fits per line, cache/checkpoint entry points.")
print("PASS: all notebook code cells unexecuted and output-free.")
print("No scientific code, reference scan, feature construction or training was executed.")

assert "review_rna_pathways" in functions
review=ast.get_source_segment(source,functions["review_rna_pathways"])
assert '=="validation"' in review and '=="test"' not in review
assert "base-contribution" in review
assert "do not reconstruct model scores" in review
assert sum(c.cell_type=="code" and c.source.startswith("RNA_REVIEW = review_rna_pathways") for c in nb.cells)==2
assert 'RESUME_RUN_ID="20261012T184252_110749Z"' in "\n".join(c.source for c in nb.cells)
print("PASS: completed-run resume, validation-only pathway ablation and additive contribution reconstruction guard.")
