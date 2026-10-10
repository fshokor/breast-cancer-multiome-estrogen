"""Build a standalone Colab ATAC notebook."""
from pathlib import Path
import json,nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
cfg=json.loads((ROOT/"configs/atac_analysis.json").read_text())
ref=json.loads((ROOT/"configs/atac_reference.json").read_text())
cells=[md("""# nb05 — ATAC QC, LSI, visualization and differential accessibility
**Question:** How does chromatin accessibility differ between observed E2 and Vehicle
cells, and how does ATAC structure relate to the existing RNA clusters?
Analyze MCF7 and T47D independently using exactly the RNA-QC-paired cells from nb02.

**Scope:** Scanpy handles available QC, neighbors, Leiden and UMAP. ATAC uses sparse
binary TF–IDF and truncated SVD/LSI, not RNA normalization/PCA.
Differential accessibility uses a depth-adjusted logistic score test implemented
with SciPy/statsmodels, not a Scanpy RNA DEG test.
No biological replicate mapping exists; every cell-level p/q value is exploratory.
Condition, submitted library and possible batch effects cannot be separated here.
"""),md("## 1. Mount Drive and locate paired ATAC"),
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
    assert (BASE/f'data/processed/{line}_ATAC_paired_paper_qc.h5ad').exists(),'Run nb02 first.'
print(BASE)
"""),md("## 2. Dependencies and explicit choices"),
code("""import importlib.util,importlib.metadata,sys,subprocess
packages={'scanpy':'scanpy==1.11.5','anndata':'anndata','numpy':'numpy','pandas':'pandas',
          'scipy':'scipy','sklearn':'scikit-learn','statsmodels':'statsmodels','igraph':'igraph','matplotlib':'matplotlib'}
missing=[v for k,v in packages.items() if importlib.util.find_spec(k) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
VERSIONS={v.split('==')[0]:importlib.metadata.version(v.split('==')[0]) for v in packages.values()}
print(VERSIONS)
"""),md("""## 3. Methods and limits
**QC:** report total supplied peak counts and detected peaks. These are not fragment
counts. No fragments means no valid TSS enrichment, FRiP or nucleosome-signal calculation.
Do not invent cutoffs or automatically remove further cells. Stop for zero-accessibility
cells rather than silently breaking RNA pairing.

**Embedding:** binarize accessibility; include peaks detected in at least 20 cells
and at most 95% of cells for LSI only. TF is binary accessibility divided by the
cell's total detected peaks across all supplied peaks; IDF=N/detected-cell count.
Use log1p(10,000 × TF × IDF), then 50-component randomized truncated SVD, seven
iterations, seed 42. This explicit Signac-style binary variant is a project choice,
not the scMKL paper's TF-IDF transformation. Plot depth correlation for every LSI
component. Predefine omission of component 1 and standardize components 2–31 for
neighbors (15), Leiden (resolution 0.5, igraph, two iterations), and UMAP (min_dist
0.3, 500 iterations). Omitting LSI1 does not guarantee removal of all depth effects.
RNA and ATAC cluster numbers are unrelated; inspect the overlap table.

**Differential accessibility:** independent of LSI peak selection, test a peak if
detected in >=10 cells and >=5% of either condition. Fit the null logistic model
treatment_E2 ~ standardized log1p(total supplied peak counts), and use an efficient
score test for adding each binary peak. This asks whether a peak is associated
with treatment after adjustment for depth; it is not a causal model.
The null model is shared across peaks. Variance accounts for fitted nuisance
parameters. Exclude degenerate peaks, apply BH across testable peaks within each line.
Report signed adjusted score, raw accessibility fractions/difference and unadjusted
log2 odds ratio (0.5 correction). Do not interpret that raw odds ratio as an adjusted
coefficient. Candidates require cell-level BH q<=0.05, absolute raw fraction
difference >=5 percentage points, and agreement of adjusted/raw directions.
No treatment permutation, independent-culture inference or motif analysis.

**Annotation:** GEO ATAC records state hg38. Peak coordinate origin remains unresolved.
Download the pinned UCSC curated RefSeq reference (checksum verified). Find nearest
TSS to peak midpoint under both 0-based half-open and 1-based closed interpretations.
Only label a provisional nearby-gene candidate when the nearest gene set agrees
and distance is <=100 kb under both. Keep ties and unmapped contigs explicit.
Nearest genes are not established regulatory targets. No BED export or
promoter-overlap claims are made until the coordinate convention is confirmed.

[Signac TF–IDF](https://stuartlab.org/signac/reference/runtfidf),
[statsmodels score test](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.score_test.html),
[UCSC coordinates](https://genome.ucsc.edu/FAQ/FAQformat.html).
"""),
code("import json\nCONFIG=json.loads("+repr(json.dumps(cfg))+")\nREFERENCE=json.loads("+repr(json.dumps(ref))+")\nprint(json.dumps(CONFIG,indent=2))\nprint(json.dumps(REFERENCE,indent=2))"),
md("## 4. Embedded implementation\nOriginal nb02 objects remain unchanged; new outputs use nb05 paths."),
code((ROOT/"src/atac_analysis.py").read_text()),
code("""from IPython.display import Image,display
from datetime import datetime,timezone
sc.settings.verbosity=2
report={'run_at_utc':datetime.now(timezone.utc).isoformat(),'config':CONFIG,'reference':REFERENCE,
        'versions':VERSIONS,'cell_lines':{}}
""")]
for line in ("MCF7","T47D"):
    cells += [md(f"## {line}: QC → TF–IDF/LSI → Scanpy graph/UMAP → exploratory DA → provisional annotation"),
code(f"report['cell_lines']['{line}']=analyze_atac(BASE,'{line}',CONFIG,REFERENCE)"),
code(f"""print(json.dumps(report['cell_lines']['{line}'],indent=2))
tables=BASE/'results/tables/nb05'
display(pd.read_csv(tables/'{line}_top_da_candidates.tsv',sep='\\t').head(20))
display(pd.read_csv(tables/'{line}_atac_cluster_treatment.tsv',sep='\\t'))
display(pd.read_csv(tables/'{line}_atac_rna_cluster_overlap.tsv',sep='\\t'))
for suffix in ('atac_qc_lsi','atac_umap','da_volcano'):
    display(Image(filename=str(BASE/'results/figures/nb05'/f'{line}_{{suffix}}.png')))
""")]
cells += [md("## Save report and interpret cautiously"),
code("""(BASE/'results/tables/nb05/atac_summary.json').write_text(json.dumps(report,indent=2)+'\\n')
display(pd.DataFrame(report['cell_lines']).T)
"""),md("""All supplied peaks and paired cells remain in the new AnnData object; only
the LSI fitting subset and DA testing universe differ. Full DA and annotation
tables are under data/processed; compact review tables are under results/tables/nb05.
No motif enrichment or RNA–ATAC regulatory linkage is inferred in this notebook.
Inspect depth, treatment imbalance, cluster composition and annotation uncertainty
before taking candidates into motif analysis. Full objects and LSI models are
descriptive whole-cohort fits, unsuitable as pretrained transforms for held-out prediction.
""")]
interpretation=ROOT/'results/tables/nb05/initial_interpretation.md'
if interpretation.exists():
    cells.append(md(interpretation.read_text()))
nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},
"colab":{"name":"nb05_atac_analysis.ipynb","provenance":[]}})
nbf.validate(nb);nbf.write(nb,ROOT/"notebooks/nb05_atac_analysis.ipynb")
print("Generated nb05.")
