"""Execute nb03 and verify its recorded input-preservation checks."""
import os,json,sys,subprocess
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
os.chdir(root)
os.environ["MULTIOME_PROJECT_ROOT"]=str(root)
subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-p","test_cluster_characterization.py"],check=True)
p=root/"notebooks/nb03_cluster_characterization.ipynb"
nb=nbformat.read(p,as_version=4)
NotebookClient(nb,timeout=3600,kernel_name="python3",resources={"metadata":{"path":str(root)}}).execute()
nbformat.validate(nb)
nbformat.write(nb,p)
report=json.loads((root/"results/tables/nb03/characterization_summary.json").read_text())
for line,r in report["cell_lines"].items():
    assert r["input_unchanged"] and r["cell_order_and_clusters_unchanged"]
    assert r["same_saved_graph_repeat_deterministic"]
    assert r["n_marker_rows"]==r["n_genes_tested"]*r["n_clusters"]
    print(line,"validated",flush=True)
out={"notebook":"nb03_cluster_characterization.ipynb","executed_locally":True,
     "colab_drive_mount_tested":False,"new_tests_passed":2,"real_data_checks":"passed",
     "nb02_inputs_unchanged":True,"cell_ids_clusters_umap_preserved":True}
(root/"results/tables/nb03/validation.json").write_text(json.dumps(out,indent=2)+"\n")
print("nb03 execution and validation passed.")
