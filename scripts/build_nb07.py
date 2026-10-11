"""Build standalone nb07 motif notebook."""
from pathlib import Path
import json,nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
cfg=json.loads((ROOT/"configs/motif_analysis.json").read_text())
refs=json.loads((ROOT/"configs/motif_references.json").read_text())
audit=json.loads((ROOT/"configs/peak_coordinate_audit.json").read_text())
cells=[md((ROOT/"MOTIF_ANALYSIS.md").read_text()),
md("## 1. Locate project and prepare the environment"),
code("""try:
    from google.colab import drive
except ImportError:
    IN_COLAB=False
else:
    IN_COLAB=True
    drive.mount('/content/drive')
from pathlib import Path
import os,sys,json,subprocess,importlib.util,importlib.metadata
BASE=Path('/content/drive/MyDrive/breast-cancer-multiome-estrogen') if IN_COLAB else Path(os.environ.get('MULTIOME_PROJECT_ROOT',Path.cwd())).resolve()
if BASE.name=='notebooks': BASE=BASE.parent
packages={'numpy':'numpy','pandas':'pandas','scipy':'scipy','statsmodels':'statsmodels',
          'matplotlib':'matplotlib','MOODS':'MOODS-python==1.9.4.1','py2bit':'py2bit==0.3.3'}
missing=[v for k,v in packages.items() if importlib.util.find_spec(k) is None]
if missing: subprocess.check_call([sys.executable,'-m','pip','install',*missing])
VERSIONS={x.split('==')[0]:importlib.metadata.version(x.split('==')[0]) for x in packages.values()}
for line in ('MCF7','T47D'):
    assert (BASE/f'data/processed/{line}_nb05_da_all.tsv').exists(),'Run nb05 first.'
    assert (BASE/f'results/tables/nb06/{line}_provisional_peak_gene_links.tsv').exists(),'Run nb06 first.'
print(BASE, VERSIONS)
"""),
md("## 2. Coordinate uncertainty and frozen resources"),
code("CONFIG=json.loads("+repr(json.dumps(cfg))+")\nREFERENCES=json.loads("+repr(json.dumps(refs))+")\nCOORDINATE_AUDIT=json.loads("+repr(json.dumps(audit))+")\nprint(json.dumps(COORDINATE_AUDIT,indent=2))"),
code("""import urllib.request,hashlib
for name,ref in REFERENCES['resources'].items():
    path=BASE/ref['path'];path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        print('Downloading',name,flush=True)
        temporary=path.with_name(path.name+'.partial')
        with urllib.request.urlopen(ref['url'],timeout=120) as response,temporary.open('wb') as out:
            while chunk:=response.read(1048576):out.write(chunk)
        h=hashlib.sha256()
        with temporary.open('rb') as f:
            while chunk:=f.read(1048576):h.update(chunk)
        assert h.hexdigest()==ref['sha256'],'Reference changed: review before continuing.'
        temporary.replace(path)
print('References available; checksums verified again by the analysis.')
"""),
md("## 3. Embedded implementation"),
code((ROOT/"src/motif_analysis.py").read_text()),
md("## 4. Sequence QC, motif scanning and matched enrichment"),
code("""enrichment,matching,report=run_motifs(BASE,CONFIG,REFERENCES,COORDINATE_AUDIT)
report['versions']=VERSIONS
report['executed_in_colab']=IN_COLAB
(BASE/'results/tables/nb07/motif_summary.json').write_text(json.dumps(report,indent=2)+'\\n')
from IPython.display import display,Image,Markdown
display(pd.Series(report['sequence_qc_counts'],name='peaks'))
display(matching)
print('Peaks with at least one motif-presence change between coordinate hypotheses:',report['coordinate_changed_peaks'])
display(Image(filename=str(BASE/'results/figures/nb07/background_balance.png')))
"""),
md("""## 5. More-accessible and less-accessible peak sets
Both directions are tested for motif enrichment. A motif in closing peaks does not
mean the TF itself is downregulated. Gray bars flag results that do not meet the
combined matching and global-q criteria.""")]
for line in ("MCF7","T47D"):
    cells += [md(f"### {line}"),
code(f"""primary=enrichment.loc[(enrichment.cell_line=='{line}')&(enrichment.scenario=='all_candidates')&
                         (enrichment.coordinate_hypothesis=='zero_based_half_open')]
cols=['motif_id','motif_name','direction','n_matched_pairs','foreground_hit_fraction',
      'background_hit_fraction','paired_log2_odds_smoothed','global_BH_q','matching_acceptable','priority_candidate']
for direction,title in [('E2','More accessible under E2'),('Vehicle','Less accessible under E2')]:
    display(Markdown('**'+title+'**'))
    display(primary.loc[primary.direction==direction].sort_values(['global_BH_q','peak_pair_p']).head(20)[cols])
display(Image(filename=str(BASE/'results/figures/nb07/{line}_motif_enrichment.png')))
display(Image(filename=str(BASE/'results/figures/nb07/{line}_motif_enrichment_shared_detection.png')))
""")]
cells += [md("## 6. Sensitivity to coordinates and condition-exclusive peaks"),
code("""sensitivity=pd.read_csv(BASE/'results/tables/nb07/motif_sensitivity.tsv',sep='\\t')
primary=enrichment.loc[(enrichment.scenario=='all_candidates')&(enrichment.coordinate_hypothesis=='zero_based_half_open')]
selected=primary.loc[primary.priority_candidate,['cell_line','direction','motif_id']].drop_duplicates()
display(sensitivity.merge(selected,on=['cell_line','direction','motif_id'],how='inner'))
display(enrichment.groupby(['cell_line','direction','scenario','coordinate_hypothesis']).priority_candidate.sum().rename('priority_motifs').reset_index())
"""),
md("""## 7. TF RNA expression and provisional pathway context
The complete tables include all motifs. The focus below is a contextual view of
estrogen-associated TF labels, not a result-selected confirmatory test. Motif
families can share sequences; RNA expression does not identify which family member
bound a site. Pathway intersections below are proximity evidence only."""),
code("""rna_context=pd.read_csv(BASE/'results/tables/nb07/motif_TF_RNA_context.tsv',sep='\\t')
focus=['ESR1','ESR2','FOXA1','GATA3','FOXM1']
display(rna_context.loc[rna_context.candidate_TF_gene_label.isin(focus)])
focus_ids=rna_context.loc[rna_context.candidate_TF_gene_label.isin(focus),'motif_id'].unique()
display(enrichment.loc[enrichment.motif_id.isin(focus_ids),['cell_line','direction','scenario',
      'coordinate_hypothesis','motif_id','motif_name','global_BH_q','paired_log2_odds_smoothed','priority_candidate']])
pathway_context=pd.read_csv(BASE/'results/tables/nb07/top_motif_pathway_context.tsv',sep='\\t')
display(pathway_context.head(40))
print('Full traceable gene/peak and pathway context is saved in top_motif_pathway_context.tsv.')
"""),
md("""## 8. Interpretation
Prioritize motifs that have adequate matching, survive global correction, and
remain informative across coordinate and condition-exclusive-peak sensitivity
analyses. Compare the two cell lines without treating them as biological
replicates. Interpret related motif families together.

The experiment still lacks identifiable biological replicates. This analysis
does not establish TF binding, TF activation/inhibition, or causal pathway
regulation. The coordinate origin remains historically unresolved even if the
two boundary hypotheses give nearly identical results.
""")]
note=ROOT/"results/tables/nb07/initial_interpretation.md"
if note.exists():cells.append(md(note.read_text()))
nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},
"colab":{"name":"nb07_motif_analysis.ipynb","provenance":[]}})
nbf.validate(nb);nbf.write(nb,ROOT/"notebooks/nb07_motif_analysis.ipynb")
print("Generated nb07.")
