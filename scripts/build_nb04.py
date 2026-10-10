"""Build standalone nb04: exploratory treatment DEG and GSEA."""
from pathlib import Path
import json,nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
cfg=json.loads((ROOT/"configs/treatment_deg_gsea.json").read_text())
ref=json.loads((ROOT/"configs/hallmark_v7.json").read_text())
audit=json.loads((ROOT/"results/tables/nb04/geo_design_audit.json").read_text())
cells=[md("""# nb04 — E2 versus Vehicle DEG and GSEA
**Question:** Which genes and pathways differ between the observed E2 and Vehicle
cells, and which patterns are shared between MCF7 and T47D?

**Exploratory inference:** GEO provides one submitted RNA/GEX library per condition,
jointly containing both lines (GSM6418443 E2; GSM6418444 Vehicle). The processed
cell metadata have no biological replicate mapping. The paired ATAC libraries are
not replicates. Cells cannot substitute for independent cultures. Therefore this
notebook reports cell-level Wilcoxon rankings and gene-set-permutation GSEA;
their p/q values do not establish reproducible population-level treatment effects.
No artificial pseudoreplicates or replicate-based DESeq2 analysis are created.

Analyze each cell line independently, using all cells retained by nb02.
Cluster labels do not enter testing; cell-state composition can contribute to the
observed response. This is not an exact reproduction of either paper's DE recipe."""),
md("## 1. Mount Drive and locate RNA count objects"),
code("""try:
    from google.colab import drive
except ImportError:
    IN_COLAB=False
else:
    IN_COLAB=True
    drive.mount('/content/drive')
from pathlib import Path
import os
BASE=Path('/content/drive/MyDrive/breast-cancer-multiome-estrogen') if IN_COLAB else Path(os.environ.get('MULTIOME_PROJECT_ROOT',Path.cwd())).resolve()
if BASE.name=='notebooks': BASE=BASE.parent
for line in ('MCF7','T47D'):
    assert (BASE/f'data/processed/{line}_RNA_paper_qc.h5ad').exists(),'Run nb02 first.'
print(BASE)
"""),
md("## 2. Environment and recorded design"),
code("""import importlib.util,importlib.metadata,subprocess,sys
packages={'scanpy':'scanpy==1.11.5','anndata':'anndata','numpy':'numpy','pandas':'pandas',
          'scipy':'scipy','matplotlib':'matplotlib','gseapy':'gseapy==1.3.1'}
missing=[v for k,v in packages.items() if importlib.util.find_spec(k) is None]
if missing: subprocess.check_call([sys.executable,'-m','pip','install',*missing])
VERSIONS={v.split('==')[0]:importlib.metadata.version(v.split('==')[0]) for v in packages.values()}
print(VERSIONS)
"""),
code("import json\nDESIGN=json.loads("+repr(json.dumps(audit))+")\nprint(json.dumps(DESIGN,indent=2))"),
md("""## 3. Preprocessing, testing universe and GSEA settings
Start from sparse integer RNA counts; normalize all QC-retained genes to 10,000
counts per cell, then log1p. Test genes detected in at least 10 cells and at least
5% of cells in **either** condition. Save the inclusion mask for every retained gene.

Scanpy Wilcoxon with tie correction compares E2 against Vehicle. BH correction
covers the tested genes within each cell line. Positive scores/fold changes mean
higher expression in E2. Scanpy log2 fold changes are approximate. The illustrative
candidate list uses cell-level BH q≤0.05 and absolute approximate log2FC≥0.5;
it is not a list of replicate-validated DEGs.

Preranked GSEA uses **every tested gene**, ranked by signed Wilcoxon score.
Ties are ordered alphabetically without jitter; the tied fraction is reported.
Use all 50 frozen MSigDB v7.0 Hallmarks, exact gene-symbol matching, overlap sizes
15–500, 1,000 gene-set permutations, weight 1, seed 42 and one thread.
This is the same reference release as nb03, not the latest MSigDB release.
Positive NES means enrichment at the E2-ranked end. Nominal permutation p=0
means below the sampled resolution, not a true probability of zero.
GSEA FDR is computed across the tested Hallmarks separately within each line.
It is a gene-set-permutation quantity, not biological replicate uncertainty.

[Scanpy DE documentation](https://scanpy.readthedocs.io/en/1.11.x/generated/scanpy.tl.rank_genes_groups.html)
and [GSEApy documentation](https://gseapy.readthedocs.io/en/latest/run.html).
"""),
code("CONFIG=json.loads("+repr(json.dumps(cfg))+")\nREFERENCE=json.loads("+repr(json.dumps(ref))+")\nprint(json.dumps(CONFIG,indent=2))\nprint(REFERENCE['version'],REFERENCE['sha256'])"),
md("## 4. Embedded reusable implementation\nGenerated from src/treatment_deg_gsea.py. Original nb02 files are read only."),
code((ROOT/"src/treatment_deg_gsea.py").read_text()),
code("""from IPython.display import Image,display
from datetime import datetime,timezone
sc.settings.verbosity=2
report={'run_at_utc':datetime.now(timezone.utc).isoformat(),'config':CONFIG,
        'versions':VERSIONS,'reference':{k:v for k,v in REFERENCE.items() if k!='gene_sets'},
        'design':DESIGN,'cell_lines':{}}
""")]
for line in ("MCF7","T47D"):
    cells += [md(f"## {line}: E2 versus Vehicle DEG → complete ranking → Hallmark GSEA"),
code(f"report['cell_lines']['{line}']=analyze_line(BASE,'{line}',CONFIG,REFERENCE)"),
code(f"""tables=BASE/'results/tables/nb04'
display(pd.read_csv(tables/'{line}_design.tsv',sep='\\t'))
de=pd.read_csv(tables/'{line}_deg_all.tsv',sep='\\t')
display(pd.concat([de.nlargest(10,'wilcoxon_score'),de.nsmallest(10,'wilcoxon_score')]))
gsea=pd.read_csv(tables/'{line}_hallmark_gsea.tsv',sep='\\t')
display(gsea.sort_values('FDR q-val'))
for suffix in ('deg_volcano','gsea_nes','gsea_curves'):
    display(Image(filename=str(BASE/'results/figures/nb04'/f'{line}_{{suffix}}.png')))
""")]
cells += [md("## Compare the two cell lines"),
code("""report['cross_line']=compare_lines(BASE)
display(pd.read_csv(BASE/'results/tables/nb04/hallmark_cross_line_comparison.tsv',sep='\\t'))
display(Image(filename=str(BASE/'results/figures/nb04/hallmark_cross_line_heatmap.png')))
(BASE/'results/tables/nb04/deg_gsea_summary.json').write_text(json.dumps(report,indent=2)+'\\n')
display(pd.DataFrame(report['cell_lines']).T)
print(report['cross_line'])
"""),md("""## Interpretation
Use direction, effect size, detection fractions and pathway leading-edge genes
together. Review whether patterns are driven by cell-cycle or composition changes
identified in nb03. No cluster-specific testing or cell-cycle adjustment was applied.

The volcano shows cell-level q values; many cells can produce very small values
without biological replication. Pathway bars show extremes whether or not FDR<0.05;
consult the full result table. Pathway overlap makes findings correlated.

Both cell lines come from the same submitted condition libraries. Agreement is
descriptive cross-line concordance, not independent replication. Replicate-aware
confirmation needs a verified replicate-to-cell mapping or additional experiments.
Raw counts, original QC masks, clusters and embeddings are preserved.
""")]
nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},
"colab":{"name":"nb04_treatment_deg_gsea.ipynb","provenance":[]}})
nbf.validate(nb);nbf.write(nb,ROOT/"notebooks/nb04_treatment_deg_gsea.ipynb")
print("Generated nb04.")
