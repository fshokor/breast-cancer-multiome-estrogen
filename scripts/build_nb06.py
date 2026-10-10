"""Build self-contained nb06 with the frozen gene sets and implementation."""
from pathlib import Path
import json,nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
config=json.loads((ROOT/"configs/rna_atac_integration.json").read_text())
hallmarks=json.loads((ROOT/"configs/hallmark_v7.json").read_text())
cells=[md((ROOT/"RNA_ATAC_INTEGRATION.md").read_text()),
md("## 1. Locate project and record environment"),
code("""try:
    from google.colab import drive
except ImportError:
    IN_COLAB=False
else:
    IN_COLAB=True
    drive.mount('/content/drive')
from pathlib import Path
import os,importlib.util,importlib.metadata,subprocess,sys,json
BASE=Path('/content/drive/MyDrive/breast-cancer-multiome-estrogen') if IN_COLAB else Path(os.environ.get('MULTIOME_PROJECT_ROOT',Path.cwd())).resolve()
if BASE.name=='notebooks': BASE=BASE.parent
packages=['numpy','pandas','matplotlib']
missing=[x for x in packages if importlib.util.find_spec(x) is None]
if missing: subprocess.check_call([sys.executable,'-m','pip','install',*missing])
VERSIONS={x:importlib.metadata.version(x) for x in packages}
for line in ('MCF7','T47D'):
    for relative in [f'results/tables/nb04/{line}_deg_all.tsv',f'results/tables/nb04/{line}_hallmark_gsea.tsv',
                     f'data/processed/{line}_nb05_da_all.tsv']:
        assert (BASE/relative).is_file(),f'Missing {relative}; run nb04/nb05 first.'
print(BASE, VERSIONS)
"""),
md("## 2. Frozen settings and implementation"),
code("CONFIG=json.loads("+repr(json.dumps(config))+")\nHALLMARKS=json.loads("+repr(json.dumps(hallmarks))+")\nprint(CONFIG)"),
code((ROOT/"src/rna_atac_integration.py").read_text()),
md("## 3. Run both cell lines and all sensitivity scenarios"),
code("""summary,cross_line,report=run_integration(BASE,CONFIG,HALLMARKS)
report['versions']=VERSIONS
report['executed_in_colab']=IN_COLAB
(BASE/'results/tables/nb06/integration_summary.json').write_text(json.dumps(report,indent=2)+'\\n')
from IPython.display import display,Image,Markdown
display(pd.concat([pd.DataFrame(v['mapping_scenarios']).assign(cell_line=k) for k,v in report['cell_lines'].items()],ignore_index=True))
"""),
md("""## 4. Examine both RNA enrichment directions
The following tables select up to eight strongest positive and eight strongest
negative RNA NES values per line. The complete output includes every Hallmark.
RNA FDR is shown explicitly. ATAC fractions are descriptive, not significance
estimates; inspect coverage before interpreting them.""")]
for line in ("MCF7","T47D"):
    cells += [md(f"### {line}"),
code(f"""primary=summary.loc[(summary.cell_line=='{line}')&(summary.scenario=='100kb_all')&(summary.subset=='all_members')]
columns=['pathway','rna_NES','rna_GSEA_FDR','n_RNA_tested_genes','n_ATAC_mapped_genes',
         'n_same_direction','n_opposite_direction','fraction_same_direction','direction_summary']
display(Markdown('**Positive RNA NES**'))
display(primary.loc[primary.rna_NES.gt(0)].nlargest(8,'rna_NES')[columns])
display(Markdown('**Negative RNA NES**'))
display(primary.loc[primary.rna_NES.lt(0)].nsmallest(8,'rna_NES')[columns])
for suffix in ('pathway_concordance','leading_edge_sensitivity'):
    display(Image(filename=str(BASE/'results/figures/nb06'/f'{line}_{{suffix}}.png')))
""")]
cells += [md("""## 5. Cross-line comparison and targeted gene evidence
Estrogen-response and cell-cycle rows are shown alongside both directions across
the full set. Inspect RNA fold changes and individual peak evidence before making
a biological claim. Shared directions do not imply independent replication."""),
code("""selected=['HALLMARK_ESTROGEN_RESPONSE_EARLY','HALLMARK_ESTROGEN_RESPONSE_LATE',
          'HALLMARK_E2F_TARGETS','HALLMARK_G2M_CHECKPOINT']
comparison=cross_line.loc[(cross_line.scenario=='100kb_all')&(cross_line.subset=='all_members')]
display(comparison[['pathway','rna_NES_MCF7','rna_NES_T47D','fraction_same_direction_MCF7',
                    'fraction_same_direction_T47D','n_ATAC_mapped_genes_MCF7','n_ATAC_mapped_genes_T47D']])
display(summary.loc[summary.pathway.isin(selected),['cell_line','pathway','subset','scenario','rna_NES',
                    'n_ATAC_mapped_genes','fraction_same_direction','direction_summary']])
# Change these genes for focused inspection; an absent row means RNA was not tested.
for line in ('MCF7','T47D'):
    gene_table=pd.read_csv(BASE/f'results/tables/nb06/{line}_100kb_all_gene_evidence.tsv',sep='\\t')
    display(gene_table.loc[gene_table.gene.isin(['ESR1','TFF1','GREB1','PGR','MYC','CCND1','ELOVL5'])])
"""),
md("""## 6. What can we conclude?
Agreement is a candidate coordinated RNA/accessibility response. Disagreement
can reflect different regulatory mechanisms, timing, imperfect target assignment,
or technical effects. Neither outcome establishes pathway activation or inhibition.

Before prioritizing a pathway, check both signs of RNA NES, the RNA FDR, mapped
gene counts, the leading-edge subset, mixed peak directions, and all four sensitivity
scenarios. Use the full peak/gene evidence tables to trace each summary back to
upstream statistics. Biological replication and validated peak-gene links remain
needed for stronger conclusions.
""")]
interpretation=ROOT/'results/tables/nb06/initial_interpretation.md'
if interpretation.exists(): cells.append(md(interpretation.read_text()))
nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},
"colab":{"name":"nb06_rna_atac_integration.ipynb","provenance":[]}})
nbf.validate(nb);nbf.write(nb,ROOT/"notebooks/nb06_rna_atac_integration.ipynb")
print("Generated nb06.")
