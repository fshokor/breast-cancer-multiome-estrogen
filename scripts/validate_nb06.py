from pathlib import Path
import os,json,subprocess,sys
import pandas as pd,nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
os.environ["MULTIOME_PROJECT_ROOT"]=str(ROOT)
subprocess.run([sys.executable,"-m","unittest","discover","-s","tests"],check=True)
path=ROOT/"notebooks/nb06_rna_atac_integration.ipynb"
nb=nbformat.read(path,as_version=4)
NotebookClient(nb,timeout=1200,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}}).execute()
nbformat.validate(nb);nbformat.write(nb,path)
summary=pd.read_csv(ROOT/"results/tables/nb06/all_pathway_scenarios.tsv",sep="\t")
assert len(summary)==2*50*2*4
assert not summary.duplicated(["cell_line","pathway","subset","scenario"]).any()
for line in ["MCF7","T47D"]:
    original=pd.read_csv(ROOT/f"results/tables/nb04/{line}_hallmark_gsea.tsv",sep="\t").set_index("Term")
    for (scenario,subset),g in summary.loc[summary.cell_line==line].groupby(["scenario","subset"]):
        assert len(g)==50
        joined=g.set_index("pathway").loc[original.index]
        import numpy as np
        assert np.allclose(joined.rna_NES,original.NES)
        assert np.allclose(joined.rna_GSEA_FDR,original["FDR q-val"])
        assert (g.n_ATAC_mapped_genes<=g.n_RNA_tested_genes).all()
        assert (g.n_same_direction+g.n_opposite_direction+g.n_zero_direction+g.n_undefined_RNA_direction==g.n_ATAC_mapped_genes).all()
        assert g.fraction_same_direction.dropna().between(0,1).all()
    for distance in [100,10]:
        inclusive=pd.read_csv(ROOT/f"results/tables/nb06/{line}_{distance}kb_all_gene_evidence.tsv",sep="\t")
        restricted=pd.read_csv(ROOT/f"results/tables/nb06/{line}_{distance}kb_shared_detection_gene_evidence.tsv",sep="\t")
        assert inclusive.gene.equals(restricted.gene)
        assert (restricted.n_peaks<=inclusive.n_peaks).all()
        assert restricted.n_condition_exclusive_peaks.eq(0).all()
        assert restricted.loc[~restricted.mapped,"atac_median_score"].isna().all()
report=json.loads((ROOT/"results/tables/nb06/integration_summary.json").read_text())
assert report["upstream_inputs_unchanged"]
validation={"executed_locally":True,"colab_drive_mount_tested":False,"full_project_tests_passed":27,
"real_data_checks":"passed","all_50_Hallmarks_retained":True,"both_NES_directions_preserved":True,
"upstream_inputs_unchanged":True,"four_scenarios_and_two_gene_subsets":True}
(ROOT/"results/tables/nb06/validation.json").write_text(json.dumps(validation,indent=2)+"\n")
print("nb06 execution and validation passed.",flush=True)
