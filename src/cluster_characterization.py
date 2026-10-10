"""Descriptive RNA cluster characterization; no treatment DE or GSEA."""
from pathlib import Path
import copy, gc, hashlib, json, importlib.metadata
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCORE_MAP = {"S_phase":"S_score","G2M_phase":"G2M_score",
            "HALLMARK_ESTROGEN_RESPONSE_EARLY":"estrogen_score",
            "HALLMARK_UNFOLDED_PROTEIN_RESPONSE":"upr_score"}

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def write_table(df,path,index=True):
    df.to_csv(path,sep="\t",index=index,float_format="%.7g",lineterminator="\n")

def normalized_counts(source,target):
    """New sparse object from integer counts, never from scaled or log1p .X."""
    counts=source.layers["counts"]
    if not sparse.issparse(counts) or not np.issubdtype(counts.dtype,np.integer):
        raise ValueError("Expected sparse integer RNA counts.")
    if not source.obs_names.is_unique or not source.var_names.is_unique:
        raise ValueError("Identifiers must be unique.")
    a=ad.AnnData(X=counts.astype(np.float64),obs=source.obs.copy(),
                 var=pd.DataFrame(index=source.var_names.copy()))
    a.obsm["X_umap"]=source.obsm["X_umap"].copy()
    a.obsp["connectivities"]=source.obsp["connectivities"].copy()
    a.uns["neighbors"]=copy.deepcopy(source.uns["neighbors"])
    sc.pp.normalize_total(a,target_sum=target)
    sc.pp.log1p(a)
    assert sparse.issparse(a.X) and np.isfinite(a.X.data).all()
    return a

def rank_markers(a,config):
    """All retained genes, each existing cluster versus the remaining cells."""
    sc.tl.rank_genes_groups(a,groupby="leiden",reference="rest",
        method=config["marker_method"],tie_correct=config["tie_correct"],
        corr_method="benjamini-hochberg",use_raw=False,pts=True,
        n_genes=a.n_vars,key_added="cluster_markers")
    markers=sc.get.rank_genes_groups_df(a,group=None,key="cluster_markers")
    markers["group"]=markers["group"].astype(str)
    markers["pct_difference"]=markers["pct_nz_group"]-markers["pct_nz_reference"]
    markers["passes_descriptive_filter"]=(
        (markers["logfoldchanges"]>=config["marker_min_log2fc"]) &
        (markers["pct_nz_group"]>=config["marker_min_pct"]) &
        (markers["pct_difference"]>=config["marker_min_pct_difference"]) &
        (markers["pvals_adj"]<=config["marker_max_adjusted_p"]))
    selected=markers.loc[markers.passes_descriptive_filter].copy()
    top=selected.groupby("group",observed=True,sort=False).head(config["top_markers_per_cluster"])
    return markers,top

def stability(a,config,tables,line):
    baseline=a.obs.leiden.astype(str)
    labels={"nb02":baseline.to_numpy()}
    rows=[]
    for resolution in config["resolutions"]:
        for seed in config["seeds"]:
            name=f"r{resolution}_seed{seed}"
            sc.tl.leiden(a,adjacency=a.obsp["connectivities"],resolution=resolution,
                random_state=seed,flavor="igraph",n_iterations=config["leiden_iterations"],
                directed=False,key_added=name)
            v=a.obs.pop(name).astype(str).to_numpy()
            labels[name]=v
            rows.append({"run":name,"resolution":resolution,"seed":seed,
                "n_clusters":len(np.unique(v)),"ARI_vs_nb02":adjusted_rand_score(baseline,v)})
    # Serialization/traversal may affect seeded graph algorithms; report actual
    # agreement with nb02 instead of silently replacing its cluster labels.
    sc.tl.leiden(a,adjacency=a.obsp["connectivities"],resolution=0.5,
        random_state=42,flavor="igraph",n_iterations=config["leiden_iterations"],
        directed=False,key_added="repeat_check")
    assert adjusted_rand_score(labels["r0.5_seed42"],a.obs.pop("repeat_check"))==1
    labels=pd.DataFrame(labels,index=a.obs_names)
    jaccard=[]
    for cluster in a.obs.leiden.cat.categories:
        original=baseline.eq(str(cluster)).to_numpy()
        for name in labels.columns[1:]:
            best=max(np.count_nonzero(original & (labels[name].to_numpy()==v))/
                     np.count_nonzero(original | (labels[name].to_numpy()==v))
                     for v in labels[name].unique())
            jaccard.append({"cluster":str(cluster),"run":name,"best_jaccard":best})
    write_table(labels,tables/f"{line}_stability_labels.tsv")
    write_table(pd.DataFrame(rows),tables/f"{line}_stability.tsv",False)
    jaccard=pd.DataFrame(jaccard)
    write_table(jaccard,tables/f"{line}_cluster_stability.tsv",False)
    return pd.DataFrame(rows),jaccard

def score_programs(a,config,reference,tables,line):
    present={}
    coverage=[]
    for name,genes in reference["gene_sets"].items():
        found=[g for g in genes if g in a.var_names]
        missing=[g for g in genes if g not in a.var_names]
        coverage.append({"program":name,"requested":len(genes),"present":len(found),
                         "fraction_present":len(found)/len(genes),"missing_genes":";".join(missing)})
        if len(found)<5: raise ValueError(f"Insufficient genes for {name}")
        present[name]=found
    # Controls are expression matched, using the full retained-gene pool.
    sc.tl.score_genes_cell_cycle(a,s_genes=present["S_phase"],g2m_genes=present["G2M_phase"],
        use_raw=False,random_state=config["random_state"],ctrl_as_ref=False,
        n_bins=config["signature_n_bins"])
    for name in ("HALLMARK_ESTROGEN_RESPONSE_EARLY","HALLMARK_UNFOLDED_PROTEIN_RESPONSE"):
        sc.tl.score_genes(a,present[name],score_name=SCORE_MAP[name],use_raw=False,
            random_state=config["random_state"],ctrl_as_ref=False,
            ctrl_size=config["signature_ctrl_size"],n_bins=config["signature_n_bins"])
    assert np.isfinite(a.obs[list(SCORE_MAP.values())].to_numpy()).all()
    write_table(pd.DataFrame(coverage),tables/f"{line}_signature_coverage.tsv",False)
    return coverage

def plot_review(a,summary,top,stability_df,figures,line,config):
    clusters=list(a.obs.leiden.cat.categories)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    summary[["E2","Vehicle"]].plot.bar(stacked=True,ax=axes[0,0],color=["#2878b5","#ed8b23"])
    axes[0,0].set_title(f"{line}: cells by treatment"); axes[0,0].set_ylabel("Cells")
    for ax,col,title in zip(axes.flat[1:],["rna_total_counts","rna_n_genes","rna_pct_counts_mt"],
            ["RNA count depth","Detected genes","Mitochondrial RNA (%)"]):
        ax.boxplot([a.obs.loc[a.obs.leiden==c,col] for c in clusters],
                   tick_labels=clusters,showfliers=False)
        ax.set_title(title);ax.set_xlabel("Cluster")
    fig.savefig(figures/f"{line}_cluster_quality.png",dpi=150,bbox_inches="tight");plt.close(fig)
    chosen=top.groupby("group",observed=True,sort=False).head(config["plot_markers_per_cluster"])
    genes=list(dict.fromkeys(chosen.names))
    if not genes: raise ValueError("No markers pass descriptive filters.")
    p=sc.pl.dotplot(a,genes,groupby="leiden",use_raw=False,return_fig=True,show=False,
                   figsize=(max(10,len(genes)*0.36),5),title=f"{line}: selected cluster markers")
    p.savefig(figures/f"{line}_marker_dotplot.png",dpi=150,bbox_inches="tight");plt.close("all")
    # Dense conversion is limited to cluster means for a few displayed genes.
    means=np.vstack([np.asarray(a[a.obs.leiden==c,genes].X.mean(axis=0)).ravel() for c in clusters])
    z=(means-means.mean(axis=0))/np.maximum(means.std(axis=0),1e-9)
    fig,ax=plt.subplots(figsize=(max(10,len(genes)*0.36),5),layout="constrained")
    im=ax.imshow(z,aspect="auto",cmap="RdBu_r",vmin=-2,vmax=2)
    ax.set_xticks(range(len(genes)),genes,rotation=90);ax.set_yticks(range(len(clusters)),clusters)
    ax.set_ylabel("Cluster");ax.set_title(f"{line}: marker expression across clusters")
    fig.colorbar(im,ax=ax,label="Cluster-mean log expression (gene-wise z-score)")
    fig.savefig(figures/f"{line}_marker_heatmap.png",dpi=150,bbox_inches="tight");plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout="constrained")
    for ax,col in zip(axes.flat,["leiden","phase",*SCORE_MAP.values()]):
        sc.pl.umap(a,color=col,ax=ax,show=False,title=f"{line}: {col}",size=7,frameon=False)
    fig.savefig(figures/f"{line}_state_umap.png",dpi=150,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    for seed,group in stability_df.groupby("seed"):
        ax.plot(group.resolution,group.ARI_vs_nb02,"o-",label=f"Seed {seed}")
    ax.set(xlabel="Leiden resolution",ylabel="Adjusted Rand index vs nb02",ylim=(-0.02,1.03),
           title=f"{line}: sensitivity on the fixed neighbor graph")
    ax.legend();fig.savefig(figures/f"{line}_cluster_sensitivity.png",dpi=150);plt.close(fig)

def characterize_line(root,line,config,reference):
    root=Path(root)
    tables=root/"results/tables/nb03";figures=root/"results/figures/nb03"
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    source_path=root/f"data/processed/{line}_RNA_paper_qc.h5ad"
    before=sha256(source_path)
    print(f"{line}: load full retained-gene counts",flush=True)
    source=ad.read_h5ad(source_path)
    a=normalized_counts(source,config["normalize_total"])
    ids=source.obs_names.copy();original_labels=source.obs.leiden.copy()
    del source;gc.collect()
    print(f"{line}: marker ranking across {a.n_vars} retained genes",flush=True)
    markers,top=rank_markers(a,config)
    full_path=root/f"data/processed/{line}_nb03_all_cluster_markers.tsv"
    write_table(markers,full_path,False)
    write_table(top,tables/f"{line}_top_cluster_markers.tsv",False)
    print(f"{line}: cluster sensitivity and program scoring",flush=True)
    stability_df,jaccard=stability(a,config,tables,line)
    coverage=score_programs(a,config,reference,tables,line)
    group=a.obs.groupby("leiden",observed=True)
    summary=group[["rna_total_counts","rna_n_genes","rna_pct_counts_mt"]].median()
    summary.columns=["median_"+c for c in summary.columns]
    summary["n_cells"]=group.size()
    counts=pd.crosstab(a.obs.leiden,a.obs.treatment).reindex(columns=["E2","Vehicle"],fill_value=0)
    summary=summary.join(counts)
    summary["fraction_E2"]=summary.E2/summary.n_cells
    for col in SCORE_MAP.values(): summary["mean_"+col]=group[col].mean()
    # These are descriptive relative scores, not definitive cluster annotations.
    standardized=(a.obs[list(SCORE_MAP.values())]-a.obs[list(SCORE_MAP.values())].mean())/a.obs[list(SCORE_MAP.values())].std()
    relative=standardized.groupby(a.obs.leiden,observed=True).mean()
    summary["highest_relative_program"]=relative.idxmax(axis=1)
    summary["highest_relative_program_z"]=relative.max(axis=1)
    summary["annotation_status"]="unassigned; review markers and programs"
    summary["top_markers"]=[";".join(top.loc[top.group==str(c),"names"].head(5)) for c in summary.index]
    summary["min_best_jaccard"]=jaccard.groupby("cluster").best_jaccard.min().reindex(summary.index.astype(str)).to_numpy()
    write_table(summary,tables/f"{line}_cluster_review.tsv")
    phase=pd.crosstab(a.obs.leiden,a.obs.phase,normalize="index")
    write_table(phase,tables/f"{line}_phase_fractions.tsv")
    stratified=a.obs.groupby(["leiden","treatment"],observed=True)[list(SCORE_MAP.values())].agg(["count","mean","median"])
    write_table(stratified,tables/f"{line}_programs_by_cluster_treatment.tsv")
    correlations=[]
    for score in SCORE_MAP.values():
        for metric in ["rna_total_counts","rna_n_genes","rna_pct_counts_mt"]:
            correlations.append({"score":score,"qc_metric":metric,
                                 "spearman_rho":float(spearmanr(a.obs[score],a.obs[metric]).statistic)})
    write_table(pd.DataFrame(correlations),tables/f"{line}_score_qc_correlations.tsv",False)
    write_table(a.obs,tables/f"{line}_cell_states.tsv")
    plot_review(a,summary,top,stability_df,figures,line,config)
    assert a.obs_names.equals(ids) and a.obs.leiden.equals(original_labels)
    assert sha256(source_path)==before,"nb02 input must not change"
    assert len(markers)==a.n_vars*a.obs.leiden.nunique()
    assert markers[["scores","logfoldchanges","pvals","pvals_adj","pct_nz_group","pct_nz_reference"]].notna().all().all()
    assert summary.n_cells.sum()==a.n_obs and (summary.E2+summary.Vehicle==summary.n_cells).all()
    result={"n_cells":a.n_obs,"n_genes_tested":a.n_vars,"n_clusters":int(a.obs.leiden.nunique()),
        "n_marker_rows":len(markers),"n_filtered_markers":int(markers.passes_descriptive_filter.sum()),
        "signature_coverage":coverage,"source_sha256":before,"input_unchanged":True,
        "cell_order_and_clusters_unchanged":True,"same_saved_graph_repeat_deterministic":True,
        "nb02_replay_ARI":float(stability_df.loc[stability_df.run=="r0.5_seed42","ARI_vs_nb02"].iloc[0]),
        "scanpy_version":importlib.metadata.version("scanpy"),
        "full_marker_table":str(full_path.relative_to(root))}
    print(json.dumps(result,indent=2),flush=True)
    del a,markers;gc.collect()
    return result
