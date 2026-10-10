"""Run nb04 on real data and check complete rankings and input preservation."""
from pathlib import Path
import os,sys,json,subprocess
import nbformat,pandas as pd,numpy as np
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1];os.chdir(root)
os.environ["MULTIOME_PROJECT_ROOT"]=str(root)
subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-p","test_treatment_deg_gsea.py"],check=True)
p=root/"notebooks/nb04_treatment_deg_gsea.ipynb"
nb=nbformat.read(p,as_version=4)
NotebookClient(nb,timeout=3600,kernel_name="python3",resources={"metadata":{"path":str(root)}}).execute()
nbformat.validate(nb);nbformat.write(nb,p)
tables=root/"results/tables/nb04"
report=json.loads((tables/"deg_gsea_summary.json").read_text())
for line,r in report["cell_lines"].items():
    de=pd.read_csv(tables/f"{line}_deg_all.tsv",sep="\t")
    ranks=pd.read_csv(tables/f"{line}_gsea_ranked_genes.tsv",sep="\t")
    gsea=pd.read_csv(tables/f"{line}_hallmark_gsea.tsv",sep="\t")
    assert r["nb02_input_unchanged"] and r["gsea_rank_contains_all_tested_genes"]
    assert len(ranks)==len(de)==r["genes_tested"]
    assert set(ranks.gene)==set(de.gene) and ranks.gene.is_unique
    assert ranks.wilcoxon_score.is_monotonic_decreasing
    assert de.cell_level_BH_q.between(0,1).all()
    assert gsea["FDR q-val"].between(0,1).all()
    print(line,"DEG/GSEA checks passed.",flush=True)
(tables/"validation.json").write_text(json.dumps({"executed_locally":True,"colab_drive_mount_tested":False,
"new_tests_passed":2,"all_tested_genes_in_GSEA":True,"nb02_inputs_unchanged":True,
"replicate_supported_inference":False,"real_data_validation":"passed"},indent=2)+"\n")
print("nb04 validation passed.")
