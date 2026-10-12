"""Static checks only: do not import project code or execute notebook cells."""
from pathlib import Path
import ast,hashlib,json
import nbformat
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/"src/pathway_mlp.py").read_text()
tree=ast.parse(source)
nb=nbformat.read(ROOT/"notebooks/nb11_pathway_multimodal_mlp.ipynb",as_version=4)
nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=="code"]
for cell in code:
    assert cell.execution_count is None and not cell.outputs
    lines=cell.source.splitlines()
    # The only permitted non-Python syntax is the dependency installation magic.
    assert not any(line.lstrip().startswith(("!","%%")) for line in lines)
    ast.parse("\n".join(line for line in lines if not line.startswith("%pip ")))
assert sum(c.source==source for c in code)==1
assert hashlib.sha256(source.encode()).hexdigest() in "\n".join(c.source for c in code)
names={n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert {"PathwayMLP","MaskedPathwayLayer","StochasticPathwayGate","train_model",
        "pathway_ablation","gene_attribution","start_run","prepare_line"}<=names
# Top level contains only imports, definitions, and docstring: no analysis on import.
assert all(isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef)) or
           isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) for n in tree.body)
assert 'self.weight*self.mask' in source
assert 'split=="val":split="validation"' in source
assert 'weights_only=True' in source and 'save_torch(checkpoint' in source
assert 'from tqdm.auto import tqdm' in source
assert 'best_state={k:v.detach().cpu().clone()' in source
config=json.loads((ROOT/"configs/pathway_mlp.json").read_text())
assert config["cell_lines"]==["MCF7","T47D"]
assert config["max_epochs"]>=config["min_epochs"]>=1
assert len(json.loads((ROOT/"configs/hallmark_v7.json").read_text())["gene_sets"])==50
calls=[ast.parse(c.source) for c in code if not c.source.startswith("# Installation")]
for function in ("prepare_line","train_model","evaluate_model","pathway_ablation","gene_attribution"):
    count=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==function
              for t in calls for n in ast.walk(t) if t.body)
    assert count==(2 if function=="prepare_line" else 6),(function,count)
print(f"PASS: notebook schema, {len(code)} unexecuted code cells, Python syntax, source synchronization and static safeguards.")
print("No project code, preprocessing, training, predictions or notebook cells executed.")

assert 'gate_ramp_end_epoch' not in source
assert 'strength=float(config["gate_lambda"])' in source
assert 'self.norm=nn.Tanh()' in source and 'LayerNorm' not in source
assert 'self.rna is not None else zero' in source
assert 'self.atac is not None else zero' in source
assert 'if not passed:' in source and 'gene rankings and region candidates withheld' in source
assert 'leggauss(steps)' in source and 'complete and stable' in source
assert 'folder=data["folder"]/"models"/mode' in source
assert config["model_modes"]==["RNA","ATAC","RNA_ATAC"]
print("PASS: six independent model calls, mode-specific checkpoints, early checkpoint eligibility and attribution fail-closed checks.")
