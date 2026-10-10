"""Execute nb02, then verify saved sparse objects and paired metadata."""
import gc,json,os,sys,subprocess,compileall,re
from pathlib import Path
import anndata as ad
import nbformat
import numpy as np
import pandas as pd
from scipy import sparse
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ["MULTIOME_PROJECT_ROOT"]=str(ROOT)
assert compileall.compile_dir(ROOT/"src",quiet=1)
assert compileall.compile_dir(ROOT/"scripts",quiet=1)
test_run = subprocess.run([sys.executable,"-m","unittest","discover","-s","tests"],
                          capture_output=True, text=True)
print(test_run.stdout, end="")
print(test_run.stderr, end="", file=sys.stderr)
test_run.check_returncode()
test_count = int(re.search(r"Ran (\d+) tests?", test_run.stderr).group(1))
p=ROOT/"notebooks/nb02_qc_preprocessing.ipynb"
nb=nbformat.read(p,as_version=4)
nbformat.validate(nb)
NotebookClient(nb,timeout=3600,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}}).execute()
nbformat.validate(nb)
nbformat.write(nb,p)
report=json.loads((ROOT/"results/tables/paper_qc_summary.json").read_text())
comparison = {}
for line in report["cell_lines"]:
    qc=pd.read_csv(ROOT/f"results/tables/{line}_cell_qc.tsv",sep="\t",index_col=0)
    rna=ad.read_h5ad(ROOT/f"data/processed/{line}_RNA_paper_qc.h5ad")
    expected=qc.loc[qc.pass_rna_qc]
    assert list(rna.obs_names)==list(expected.index)
    assert sparse.isspmatrix_csr(rna.X) and sparse.issparse(rna.layers["counts"])
    assert np.issubdtype(rna.layers["counts"].dtype,np.integer)
    np.testing.assert_array_equal(np.asarray(rna.layers["counts"].sum(axis=1)).ravel(),expected.rna_total_counts)
    np.testing.assert_array_equal(np.diff(rna.layers["counts"].indptr),expected.rna_n_genes)
    np.testing.assert_allclose(rna.X.data,np.log1p(rna.layers["counts"].data.astype(np.float32)),atol=1e-6)

    assert np.isfinite(rna.layers["scaled_no_center"].data).all()
    assert rna.obsm["X_pca"].shape == (rna.n_obs, 50)
    assert rna.obsm["X_umap"].shape == (rna.n_obs, 2)
    assert np.isfinite(rna.obsm["X_pca"]).all() and np.isfinite(rna.obsm["X_umap"]).all()
    assert rna.obs.leiden.notna().all()
    assert rna.obsp["connectivities"].shape == (rna.n_obs,rna.n_obs)
    assert rna.var.highly_variable.sum() >= 2000
    assert rna.varm["PCs"].shape == (rna.n_vars,50)
    baseline_path=ROOT/f"data/processed/{line}_pre_scanpy_qc_ids.json"
    if baseline_path.exists():
        baseline=json.loads(baseline_path.read_text())
        assert list(rna.obs_names)==baseline["cell_ids"]
        assert list(rna.var_names)==baseline["gene_ids"]
        comparison[line]="Exact retained cells and genes match previous custom implementation"
    else:
        comparison[line]="Prior local baseline unavailable"
    viz=ad.read_h5ad(ROOT/f"data/processed/{line}_RNA_visualization.h5ad")
    assert viz.obs_names.equals(rna.obs_names)
    assert (viz.layers["counts"] != rna.layers["counts"][:,rna.var.highly_variable.to_numpy()]).nnz == 0
    assert np.isfinite(viz.X).all()
    np.testing.assert_array_equal(viz.obsm["X_umap"],rna.obsm["X_umap"])
    assert list(viz.obs.leiden)==list(rna.obs.leiden)
    del viz
    ids=rna.obs_names.copy()
    del rna
    gc.collect()
    atac=ad.read_h5ad(ROOT/f"data/processed/{line}_ATAC_paired_paper_qc.h5ad")
    assert atac.obs_names.equals(ids)
    assert sparse.isspmatrix_csr(atac.X) and np.issubdtype(atac.X.dtype,np.integer)
    assert np.all(atac.layers["binary"].data==1)
    np.testing.assert_array_equal(atac.X.indices,atac.layers["binary"].indices)
    np.testing.assert_array_equal(atac.X.indptr,atac.layers["binary"].indptr)
    np.testing.assert_array_equal(np.asarray(atac.X.sum(axis=1)).ravel(),expected.atac_total_peak_counts)
    assert atac.n_vars==206167
    del atac
    gc.collect()
    print(line,"round-trip, integer-count, log1p, binary and pairing validations passed.",flush=True)
result={"notebook":"nb02_qc_preprocessing.ipynb","executed_locally":True,
        "colab_drive_mount_tested":False,"tests_passed":test_count,"real_data_roundtrip":"passed",
        "raw_counts_preserved":True,"paired_cell_order":"exact","full_all_gene_dense_matrix_created":False,"bounded_hvg_dense_matrix":True,
        "scanpy_visualizations":"passed","previous_qc_comparison":comparison}
(ROOT/"results/tables/qc_validation.json").write_text(json.dumps(result,indent=2)+"\n")
print("QC notebook and real-data validation passed.",flush=True)
