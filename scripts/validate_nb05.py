import os,sys,json,subprocess,gc
from pathlib import Path
import nbformat,anndata as ad,numpy as np,pandas as pd
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
os.environ["MULTIOME_PROJECT_ROOT"]=str(ROOT)
subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-p","test_atac_analysis.py"],check=True)
p=ROOT/"notebooks/nb05_atac_analysis.ipynb";nb=nbformat.read(p,as_version=4)
NotebookClient(nb,timeout=3600,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}}).execute()
nbformat.validate(nb);nbformat.write(nb,p)
report=json.loads((ROOT/"results/tables/nb05/atac_summary.json").read_text())
for line,r in report["cell_lines"].items():
    original=ad.read_h5ad(ROOT/f"data/processed/{line}_ATAC_paired_paper_qc.h5ad")
    a=ad.read_h5ad(ROOT/f"data/processed/{line}_ATAC_nb05.h5ad")
    assert a.obs_names.equals(original.obs_names) and a.var_names.equals(original.var_names)
    assert (a.X!=original.X).nnz==0 and (a.layers["binary"]!=original.layers["binary"]).nnz==0
    assert a.obs.rna_leiden.equals(original.obs.leiden)
    assert np.isfinite(a.obsm["X_umap"]).all() and np.isfinite(a.obsm["X_lsi"]).all()
    da=pd.read_csv(ROOT/f"data/processed/{line}_nb05_da_all.tsv",sep="\t")
    assert da.loc[da.testable,"cell_level_BH_q"].between(0,1).all()
    assert r["nb02_input_unchanged"] and r["cells_and_rna_labels_preserved"]
    print(line,"ATAC round-trip, pairing and DA checks passed.",flush=True)
    del a,original;gc.collect()
(ROOT/"results/tables/nb05/validation.json").write_text(json.dumps({"executed_locally":True,
"colab_drive_mount_tested":False,"new_tests_passed":4,"real_data_validation":"passed",
"nb02_inputs_unchanged":True,"all_cells_peaks_counts_preserved":True},indent=2)+"\n")
print("nb05 validation passed.",flush=True)
