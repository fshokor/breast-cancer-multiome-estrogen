"""Generate the standalone Colab-compatible nb03."""
import json
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
config=json.loads((ROOT/"configs/cluster_characterization.json").read_text())
reference=json.loads((ROOT/"configs/cluster_gene_sets.json").read_text())
cells=[
md("""# nb03 — Cluster characterization and tentative cell-state review
**Question:** What distinguishes the RNA clusters within MCF7 and T47D?
Review cluster quality, markers, sensitivity and descriptive gene-program scores.
This is exploratory characterization of the existing nb02 clusters.
Cluster-marker p-values are descriptive: clusters and markers use the same data,
cells are not independent biological replicates, and treatment composition may drive markers.
No E2-versus-Vehicle DEG analysis or GSEA is performed here; those belong in nb04."""),
md("## 1. Mount Drive and locate nb02 outputs"),
code("""try:
    from google.colab import drive
except ImportError:
    IN_COLAB=False
else:
    IN_COLAB=True
    drive.mount('/content/drive')
import os
from pathlib import Path
BASE=Path('/content/drive/MyDrive/breast-cancer-multiome-estrogen') if IN_COLAB else Path(os.environ.get('MULTIOME_PROJECT_ROOT',Path.cwd())).resolve()
if BASE.name=='notebooks': BASE=BASE.parent
for line in ('MCF7','T47D'):
    assert (BASE/f'data/processed/{line}_RNA_paper_qc.h5ad').is_file(), 'Run nb02 first.'
print(BASE)
"""),
md("## 2. Check dependencies"),
code("""import importlib.util,importlib.metadata,subprocess,sys
packages={'scanpy':'scanpy==1.11.5','anndata':'anndata','numpy':'numpy','pandas':'pandas',
          'scipy':'scipy','sklearn':'scikit-learn','matplotlib':'matplotlib','igraph':'igraph'}
missing=[v for k,v in packages.items() if importlib.util.find_spec(k) is None]
if missing: subprocess.check_call([sys.executable,'-m','pip','install',*missing])
for name in packages.values():
    name=name.split('==')[0]
    print(name,importlib.metadata.version(name))
"""),
md("""## 3. Settings and frozen reference signatures
Markers: Wilcoxon, tie correction, BH adjustment, each cluster versus all remaining
cells within the same cell line. Use **all retained genes**, normalized to 10,000
counts per cell and log1p transformed. Do not use scaled HVGs or nb02's unnormalized .X.
Keep the complete ranking; descriptive marker plots require log2FC ≥0.25,
detection in ≥10% of cluster cells, ≥10 percentage-point detection difference,
and adjusted p≤0.05. Scanpy log fold changes are approximate.

Sensitivity: fixed nb02 neighbor graph, resolutions 0.3/0.5/0.8, seeds 42/7/123.
ARI compares whole partitions; best Jaccard overlap compares individual clusters.
This measures parameter sensitivity, not independent validation or an optimal resolution.

Program scoring: expression-matched controls, full retained-gene pool, seed 42.
Use the published S/G2M list and frozen MSigDB v7.0 estrogen-early and unfolded-protein
response sets. This is descriptive Scanpy scoring, **not GSEA**. Source URLs and
SHA-256 fingerprints are embedded below. Duplicate HJURP is removed. Missing symbols
are reported explicitly; no unverified alias substitutions.
[Scanpy marker documentation](https://scanpy.readthedocs.io/en/1.11.x/generated/scanpy.tl.rank_genes_groups.html)
and [cell-cycle tutorial](https://scanpy.readthedocs.io/en/1.11.x/how-to/cell-cycle.html)."""),
code("import json\nCONFIG=json.loads("+repr(json.dumps(config))+")\nREFERENCE=json.loads("+repr(json.dumps(reference))+")\nprint(json.dumps(CONFIG,indent=2))\nprint(json.dumps(REFERENCE['sources'],indent=2))"),
md("""## 4. Reusable implementation
Embedded from src/cluster_characterization.py for standalone Colab execution.
Original nb02 objects are read only. Existing cell IDs, cluster labels and UMAP are
preserved. Large full marker rankings go to data/processed; compact results and
figures go to results/tables/nb03 and results/figures/nb03.
No cell-cycle regression or new filtering is applied."""),
code((ROOT/"src/cluster_characterization.py").read_text()),
code("""from IPython.display import Image,display
from datetime import datetime,timezone
sc.settings.verbosity=2
report={'run_at_utc':datetime.now(timezone.utc).isoformat(),'config':CONFIG,
        'reference':REFERENCE,'cell_lines':{}}
""")]
for line in ("MCF7","T47D"):
    cells.extend([
md(f"## {line}: markers, quality, sensitivity and program scores"),
code(f"report['cell_lines']['{line}']=characterize_line(BASE,'{line}',CONFIG,REFERENCE)"),
code(f"""tables=BASE/'results/tables/nb03'
review=pd.read_csv(tables/'{line}_cluster_review.tsv',sep='\\t')
display(review)
display(pd.read_csv(tables/'{line}_signature_coverage.tsv',sep='\\t'))
display(pd.read_csv(tables/'{line}_stability.tsv',sep='\\t'))
for suffix in ('cluster_quality','marker_dotplot','marker_heatmap','state_umap','cluster_sensitivity'):
    display(Image(filename=str(BASE/'results/figures/nb03'/f'{line}_{{suffix}}.png')))
""")])
cells.extend([
md("""## Save the validation and provenance record"""),
code("""(BASE/'results/tables/nb03/characterization_summary.json').write_text(json.dumps(report,indent=2)+'\\n')
display(pd.DataFrame({k:{x:v[x] for x in ('n_cells','n_genes_tested','n_clusters','n_filtered_markers','input_unchanged')} for k,v in report['cell_lines'].items()}).T)
"""),
md("""## Interpretation and next step
Read the marker dot plot and heatmap together with cluster QC, treatment fractions,
program scores and stability. A cluster with different E2/Vehicle proportions can
have treatment-associated markers without representing a distinct cell type.

S/G2M/phase are expression-based estimates; a G1 call is not evidence of quiescence.
UPR is one stress-related program, not a general stress diagnosis. Estrogen and UPR
signatures can overlap; scores are not mutually exclusive or directly comparable
between cell lines. Gene-set scores are not enrichment p-values.

The review table's highest relative program is the largest cluster mean after
standardizing each score within that cell line. It is a review cue, not an automatic
annotation. All clusters remain explicitly **unassigned** until marker and program
evidence supports a label. Small or sensitive clusters deserve extra review.

nb04 will perform E2-versus-Vehicle DEG and GSEA together, after checking biological
replicate availability. Do not treat cluster-marker significance as treatment-DE evidence.
""")])
interpretation=ROOT/"results/tables/nb03/initial_interpretation.md"
if interpretation.exists():
    cells.insert(-1,md(interpretation.read_text()))
nb=nbf.v4.new_notebook(cells=cells,metadata={
"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},
"colab":{"name":"nb03_cluster_characterization.ipynb","provenance":[]}})
nbf.validate(nb)
nbf.write(nb,ROOT/"notebooks/nb03_cluster_characterization.ipynb")
print("Generated nb03.")
