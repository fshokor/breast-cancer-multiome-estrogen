"""Exploratory E2 versus Vehicle DEG and preranked Hallmark GSEA."""
from pathlib import Path
import gc,json,hashlib,importlib.metadata
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import gseapy as gp
from scipy import sparse
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def fingerprint(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def table(df,path,index=False):
    df.to_csv(path,sep="\t",index=index,float_format="%.9g",lineterminator="\n")

def prepare_expression(source,cfg):
    if set(source.obs.treatment.astype(str))!={"E2","Vehicle"}:
        raise ValueError("Expected exactly E2 and Vehicle.")
    if not source.var_names.is_unique:raise ValueError("Duplicate gene symbols.")
    counts=source.layers["counts"]
    if not sparse.issparse(counts) or not np.issubdtype(counts.dtype,np.integer):
        raise ValueError("Expected sparse integer counts.")
    obs=source.obs.copy()
    obs["treatment"]=pd.Categorical(obs.treatment,categories=["Vehicle","E2"])
    a=ad.AnnData(X=counts.astype(np.float64),obs=obs,
                var=pd.DataFrame(index=source.var_names.copy()))
    audit=pd.DataFrame(index=source.var_names)
    masks=[]
    for condition in ("E2","Vehicle"):
        cells=obs.treatment.eq(condition).to_numpy()
        if cells.sum()<2:raise ValueError("Insufficient cells for exploratory ranking.")
        detected=np.asarray((counts[cells]>0).sum(axis=0)).ravel()
        audit[f"n_detected_{condition}"]=detected
        audit[f"pct_detected_{condition}"]=detected/cells.sum()
        masks.append((detected>=cfg["min_detected_cells"]) &
            (detected/cells.sum()>=cfg["min_detection_fraction_either_condition"]))
    audit["tested"]=masks[0]|masks[1]
    sc.pp.normalize_total(a,target_sum=cfg["normalization_target"])
    sc.pp.log1p(a)
    # Normalize with every QC-retained gene before selecting the testing universe.
    a=a[:,audit.tested.to_numpy()].copy()
    if a.n_vars<2:raise ValueError("Insufficient expressed genes.")
    return a,audit

def differential_expression(a,cfg):
    sc.tl.rank_genes_groups(a,groupby="treatment",groups=["E2"],reference="Vehicle",
        method=cfg["de_method"],tie_correct=cfg["tie_correct"],
        corr_method="benjamini-hochberg",use_raw=False,pts=True,n_genes=a.n_vars)
    de=sc.get.rank_genes_groups_df(a,group="E2")
    de=de.rename(columns={"names":"gene","scores":"wilcoxon_score",
        "logfoldchanges":"approx_log2FC_E2_vs_Vehicle","pvals":"cell_level_p",
        "pvals_adj":"cell_level_BH_q","pct_nz_group":"pct_E2","pct_nz_reference":"pct_Vehicle"})
    de["exploratory_candidate"]=(de.cell_level_BH_q<=cfg["exploratory_adjusted_p"]) & (
        de.approx_log2FC_E2_vs_Vehicle.abs()>=cfg["exploratory_abs_log2fc"])
    de["direction"]=np.where(de.approx_log2FC_E2_vs_Vehicle>=0,"higher_in_E2","higher_in_Vehicle")
    assert len(de)==a.n_vars and de.gene.is_unique
    assert np.isfinite(de.select_dtypes(include=np.number).to_numpy()).all()
    # Full testing universe, independent of significance or fold-change thresholds.
    ranks=de[["gene","wilcoxon_score"]].sort_values(
        ["wilcoxon_score","gene"],ascending=[False,True],kind="mergesort").reset_index(drop=True)
    return de,ranks

def enrichment(ranks,reference,cfg):
    genes=set(ranks.gene)
    coverage=pd.DataFrame([{"pathway":k,"original_size":len(v),"overlap_size":len(set(v)&genes)}
                          for k,v in reference["gene_sets"].items()])
    coverage["tested"]=coverage.overlap_size.between(cfg["gsea_min_size"],cfg["gsea_max_size"])
    if not coverage.tested.any():raise ValueError("No gene sets pass overlap-size limits.")
    result=gp.prerank(rnk=ranks.copy(),gene_sets=reference["gene_sets"],outdir=None,
        min_size=cfg["gsea_min_size"],max_size=cfg["gsea_max_size"],
        permutation_num=cfg["gsea_permutations"],weight=cfg["gsea_weight"],
        seed=cfg["gsea_seed"],threads=cfg["gsea_threads"],ascending=None,
        no_plot=True,verbose=False,method="permutation")
    assert list(result.ranking.index)==ranks.gene.tolist()
    frame=result.res2d.copy()
    for c in ["ES","NES","NOM p-val","FDR q-val","FWER p-val"]:
        frame[c]=pd.to_numeric(frame[c])
    assert np.isfinite(frame[["ES","NES","NOM p-val","FDR q-val"]].to_numpy()).all()
    assert set(frame.Term)==set(coverage.loc[coverage.tested,"pathway"])
    frame["direction"]=np.where(frame.NES>0,"E2_ranked_end","Vehicle_ranked_end")
    return result,frame,coverage

def line_figures(de,gsea,result,figures,line,cfg):
    colors=np.where(de.exploratory_candidate,
                    np.where(de.approx_log2FC_E2_vs_Vehicle>0,"#c44e52","#2878b5"),"#bbbbbb")
    fig,ax=plt.subplots(figsize=(8,6),layout="constrained")
    y=-np.log10(de.cell_level_BH_q.clip(lower=1e-300))
    ax.scatter(de.approx_log2FC_E2_vs_Vehicle,y,c=colors,s=5,alpha=.6)
    for sign in [-1,1]:ax.axvline(sign*cfg["exploratory_abs_log2fc"],color="gray",ls="--",lw=.7)
    ax.axhline(-np.log10(cfg["exploratory_adjusted_p"]),color="gray",ls="--",lw=.7)
    ax.set(xlabel="Approximate log2 fold change: E2 / Vehicle",
           ylabel="-log10 cell-level BH q (capped at 300)",
           title=f"{line}: exploratory differential expression")
    for positive,xpos in ((False,.06),(True,.94)):
        subset=de.loc[de.wilcoxon_score.gt(0) if positive else de.wilcoxon_score.lt(0)]
        selected=subset.reindex(subset.wilcoxon_score.abs().sort_values(ascending=False).index).head(3)
        for j,(_,r) in enumerate(selected.iterrows()):
            ax.annotate(r.gene,(r.approx_log2FC_E2_vs_Vehicle,-np.log10(max(r.cell_level_BH_q,1e-300))),
                xytext=(xpos,.94-j*.085),textcoords="axes fraction",fontsize=8,
                ha="right" if positive else "left",va="center",
                arrowprops={"arrowstyle":"-","color":"#555555","lw":.6},
                bbox={"facecolor":"white","edgecolor":"none","alpha":.8})
    fig.savefig(figures/f"{line}_deg_volcano.png",dpi=150);plt.close(fig)
    show=pd.concat([gsea.nsmallest(10,"NES"),gsea.nlargest(10,"NES")]).drop_duplicates("Term").sort_values("NES")
    fig,ax=plt.subplots(figsize=(10,8),layout="constrained")
    ax.barh(show.Term.str.replace("HALLMARK_","").str.replace("_"," "),show.NES,
            color=np.where(show.NES>0,"#c44e52","#2878b5"),
            alpha=.8)
    ax.axvline(0,color="black",lw=.7)
    ax.set(xlabel="Normalized enrichment score (positive = E2-ranked end)",
           title=f"{line}: exploratory preranked GSEA\n10 pathways at each end, regardless of FDR")
    fig.savefig(figures/f"{line}_gsea_nes.png",dpi=150);plt.close(fig)
    terms=list(dict.fromkeys(["HALLMARK_ESTROGEN_RESPONSE_EARLY","HALLMARK_ESTROGEN_RESPONSE_LATE",
                              gsea.loc[gsea.NES.idxmax(),"Term"],gsea.loc[gsea.NES.idxmin(),"Term"]]))
    terms=[t for t in terms if t in result.results]
    fig,axes=plt.subplots(len(terms),1,figsize=(10,3*len(terms)),layout="constrained",squeeze=False)
    for ax,term in zip(axes.flat,terms):
        r=result.results[term];curve=np.asarray(r["RES"])
        ax.plot(np.arange(len(curve)),curve,color="#2878b5")
        lo,hi=curve.min(),curve.max();span=max(hi-lo,.01)
        ax.vlines(r["hits"],lo-.12*span,lo-.04*span,color="black",alpha=.35,lw=.4)
        ax.axhline(0,color="gray",lw=.6)
        fdr_label=f"{r['fdr']:.3g}" if r["fdr"]>0 else "0 (permutation estimate)"
        ax.set(title=f"{term.replace('HALLMARK_','')} | NES {r['nes']:.2f}, gene-set FDR {fdr_label}",
               xlabel="Gene rank: E2 end → Vehicle end",ylabel="Running enrichment")
    fig.savefig(figures/f"{line}_gsea_curves.png",dpi=150);plt.close(fig)

def analyze_line(root,line,cfg,reference):
    root=Path(root);tables=root/"results/tables/nb04";figures=root/"results/figures/nb04"
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    path=root/f"data/processed/{line}_RNA_paper_qc.h5ad";before=fingerprint(path)
    source=ad.read_h5ad(path)
    samples=source.obs.groupby(["sample_label","treatment"],observed=True).size().reset_index(name="n_cells")
    table(samples,tables/f"{line}_design.tsv")
    a,audit=prepare_expression(source,cfg)
    assert a.obs_names.equals(source.obs_names)
    del source;gc.collect()
    table(audit.reset_index(names="gene"),tables/f"{line}_gene_universe.tsv")
    print(f"{line}: exploratory E2 vs Vehicle ranking of {a.n_vars} genes",flush=True)
    de,ranks=differential_expression(a,cfg)
    table(de,tables/f"{line}_deg_all.tsv")
    table(de.loc[de.exploratory_candidate],tables/f"{line}_deg_exploratory_candidates.tsv")
    table(ranks,tables/f"{line}_gsea_ranked_genes.tsv")
    print(f"{line}: Hallmark GSEA, {cfg['gsea_permutations']} gene-set permutations",flush=True)
    result,gsea,coverage=enrichment(ranks,reference,cfg)
    table(gsea,tables/f"{line}_hallmark_gsea.tsv")
    table(coverage,tables/f"{line}_hallmark_coverage.tsv")
    curves={k:{"RES":v["RES"],"hits":v["hits"]} for k,v in result.results.items()}
    (root/f"data/processed/{line}_nb04_gsea_curves.json").write_text(json.dumps(curves))
    line_figures(de,gsea,result,figures,line,cfg)
    assert fingerprint(path)==before
    report={"cells":a.n_obs,"cells_by_condition":{k:int(v) for k,v in a.obs.treatment.value_counts().items()},
        "genes_before_expression_filter":len(audit),"genes_tested":a.n_vars,
        "exploratory_higher_E2":int((de.exploratory_candidate & (de.approx_log2FC_E2_vs_Vehicle>0)).sum()),
        "exploratory_higher_Vehicle":int((de.exploratory_candidate & (de.approx_log2FC_E2_vs_Vehicle<0)).sum()),
        "gsea_pathways_tested":len(gsea),"gsea_FDR_below_005":int((gsea["FDR q-val"]<.05).sum()),
        "tied_score_fraction":float(ranks.wilcoxon_score.duplicated().mean()),
        "gsea_rank_contains_all_tested_genes":len(ranks)==a.n_vars,
        "source_sha256":before,"nb02_input_unchanged":True,
        "replicate_supported_inference":False,
        "inference":"Exploratory cell-level DEG and gene-set-permutation GSEA; no biological replicate mapping"}
    print(json.dumps(report,indent=2),flush=True)
    del a,result;gc.collect()
    return report

def compare_lines(root):
    root=Path(root);tables=root/"results/tables/nb04";figures=root/"results/figures/nb04"
    dfs={line:pd.read_csv(tables/f"{line}_deg_all.tsv",sep="\t").set_index("gene") for line in ("MCF7","T47D")}
    shared=dfs["MCF7"][["approx_log2FC_E2_vs_Vehicle","wilcoxon_score"]].join(
        dfs["T47D"][["approx_log2FC_E2_vs_Vehicle","wilcoxon_score"]],how="inner",lsuffix="_MCF7",rsuffix="_T47D")
    table(shared,tables/"shared_gene_effects.tsv",True)
    gsea={line:pd.read_csv(tables/f"{line}_hallmark_gsea.tsv",sep="\t").set_index("Term") for line in dfs}
    comparison=gsea["MCF7"][["NES","FDR q-val"]].join(gsea["T47D"][["NES","FDR q-val"]],
        how="outer",lsuffix="_MCF7",rsuffix="_T47D")
    table(comparison,tables/"hallmark_cross_line_comparison.tsv",True)
    show=comparison.sort_index()
    fig,ax=plt.subplots(figsize=(9,14),layout="constrained")
    x=show[["NES_MCF7","NES_T47D"]].to_numpy();limit=np.nanmax(np.abs(x))
    im=ax.imshow(x,aspect="auto",cmap="RdBu_r",vmin=-limit,vmax=limit)
    ax.set_xticks([0,1],["MCF7","T47D"]);ax.set_yticks(range(len(show)),show.index.str.replace("HALLMARK_","").str.replace("_"," "),fontsize=8)
    ax.set_title("Exploratory Hallmark GSEA across cell lines")
    fig.colorbar(im,ax=ax,label="NES: positive = E2-ranked end")
    fig.savefig(figures/"hallmark_cross_line_heatmap.png",dpi=150);plt.close(fig)
    return {"shared_tested_genes":len(shared),
        "log2FC_spearman":float(spearmanr(shared.iloc[:,0],shared.iloc[:,2]).statistic),
        "note":"Descriptive concordance; MCF7 and T47D are not biological replicates of one cell line."}
