"""ATAC-specific sparse LSI, Scanpy visualization and exploratory depth-adjusted DA."""
from pathlib import Path
import gc,gzip,hashlib,json,urllib.request
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import sparse
from scipy.stats import chi2,spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def save_table(df,path,index=False):
    df.to_csv(path,sep="\t",index=index,float_format="%.8g",lineterminator="\n")

def get_reference(root,ref):
    path=Path(root)/ref["annotation_file"]
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(".partial")
        with urllib.request.urlopen(ref["annotation_url"],timeout=120) as response:
            tmp.write_bytes(response.read())
        if digest(tmp)!=ref["annotation_sha256"]:
            raise ValueError("Reference download changed; review provenance before using it.")
        tmp.replace(path)
    if digest(path)!=ref["annotation_sha256"]:raise ValueError("Reference checksum mismatch")
    rows=[]
    with gzip.open(path,"rt") as f:
        for line in f:
            v=line.rstrip().split("\t")
            rows.append((v[2],int(v[4]) if v[3]=="+" else int(v[5])-1,v[12]))
    return pd.DataFrame(rows,columns=["contig","tss0","gene"]).drop_duplicates()

def nearest_genes(var,tss,max_distance):
    lookup={}
    for chrom,g in tss.groupby("contig"):
        grouped=g.groupby("tss0").gene.agg(lambda v:";".join(sorted(set(v))))
        lookup[chrom]=(grouped.index.to_numpy(),grouped.to_numpy())
    def nearest(chrom,mid):
        if chrom not in lookup:return "",np.nan
        positions,names=lookup[chrom]
        i=np.searchsorted(positions,mid)
        candidates=[j for j in (i-1,i) if 0<=j<len(positions)]
        d=min(abs(positions[j]-mid) for j in candidates)
        genes=sorted({gene for j in candidates if abs(positions[j]-mid)==d for gene in names[j].split(";")})
        return ";".join(genes),d
    records=[]
    for peak,row in var.iterrows():
        # Midpoint of covered zero-based base positions under the two hypotheses.
        mid0=(row.start_as_supplied+row.end_as_supplied-1)/2
        mid1=(row.start_as_supplied+row.end_as_supplied-2)/2
        g0,d0=nearest(row.contig,mid0);g1,d1=nearest(row.contig,mid1)
        stable=bool(g0) and g0==g1
        near=stable and max(d0,d1)<=max_distance
        records.append((peak,g0,g1,d0,d1,stable,g0 if near else "",
                        "provisional_consensus" if near else "distant_or_ambiguous_or_unmapped"))
    return pd.DataFrame(records,columns=["peak","nearest_genes_if_0based","nearest_genes_if_1based",
        "distance_bp_if_0based","distance_bp_if_1based","coordinate_robust_nearest",
        "candidate_nearby_genes","annotation_status"]).set_index("peak")

def tfidf(binary,mask,scale):
    """Binary TF over all supplied peaks; IDF=N/detection count; log1p(scale*TF*IDF)."""
    totals=np.asarray(binary.sum(axis=1)).ravel()
    if np.any(totals<=0):raise ValueError("Zero-accessibility cells require explicit handling.")
    selected=binary[:,mask].astype(np.float64).tocsr()
    prevalence=np.asarray(selected.sum(axis=0)).ravel()
    idf=binary.shape[0]/prevalence
    selected=selected.multiply(1/totals[:,None]).multiply(idf[None,:]).tocsr()
    selected.data=np.log1p(scale*selected.data)
    assert np.isfinite(selected.data).all()
    return selected,idf

def depth_score_test(binary,y,depth):
    """Efficient score test of a binary peak added to treatment~log-depth."""
    z=np.log1p(np.asarray(depth,dtype=float))
    z=(z-z.mean())/max(z.std(),1e-12)
    design=np.column_stack([np.ones(len(y)),z]) if z.std()>0 else np.ones((len(y),1))
    fit=sm.GLM(y,design,family=sm.families.Binomial()).fit()
    if not fit.converged:raise ValueError("Depth-only logistic model did not converge.")
    mu=fit.predict();w=mu*(1-mu)
    information=design.T@(w[:,None]*design)
    if np.linalg.cond(information)>1e12:raise ValueError("Depth model is ill-conditioned.")
    nuisance=np.asarray(binary.T@(w[:,None]*design))
    variance=np.asarray(binary.T@w).ravel()-np.einsum(
        "ij,jk,ik->i",nuisance,np.linalg.inv(information),nuisance)
    score=np.asarray(binary.T@(y-mu)).ravel()
    valid=variance>1e-8
    signed=np.full(binary.shape[1],np.nan)
    signed[valid]=score[valid]/np.sqrt(variance[valid])
    p=np.full_like(signed,np.nan);p[valid]=chi2.sf(signed[valid]**2,1)
    return signed,p,valid,{"coefficients":fit.params.tolist(),"converged":bool(fit.converged),
                         "design":"treatment_E2 ~ intercept + standardized log1p(total peak counts)"}

def differential_accessibility(a,cfg):
    b=a.layers["binary"];y=a.obs.treatment.eq("E2").to_numpy().astype(float)
    n1=int(y.sum());n0=len(y)-n1
    if min(n1,n0)<2:raise ValueError("Both conditions are required.")
    c1=np.asarray(b[y==1].sum(axis=0)).ravel();c0=np.asarray(b[y==0].sum(axis=0)).ravel()
    p1=c1/n1;p0=c0/n0
    mask=((c1>=cfg["da_min_detected_cells"])&(p1>=cfg["da_min_fraction_either"]))|(
        (c0>=cfg["da_min_detected_cells"])&(p0>=cfg["da_min_fraction_either"]))
    universe=pd.DataFrame({"peak":a.var_names,"detected_E2":c1,"detected_Vehicle":c0,
        "fraction_E2":p1,"fraction_Vehicle":p0,"passes_prevalence":mask})
    score,p,valid,null=depth_score_test(b[:,mask],y,a.obs.atac_total_peak_counts.to_numpy())
    df=universe.loc[mask].copy()
    df["depth_adjusted_score"]=score;df["cell_level_p"]=p;df["testable"]=valid
    df["cell_level_BH_q"]=np.nan
    df.loc[valid,"cell_level_BH_q"]=multipletests(p[valid],method="fdr_bh")[1]
    df["fraction_difference_E2_minus_Vehicle"]=df.fraction_E2-df.fraction_Vehicle
    df["unadjusted_log2_odds_ratio"]=np.log2(((df.detected_E2+.5)/(n1-df.detected_E2+.5))/
        ((df.detected_Vehicle+.5)/(n0-df.detected_Vehicle+.5)))
    df["adjusted_direction"]=np.where(df.depth_adjusted_score>0,"E2","Vehicle")
    df["direction_agrees_with_raw_fraction"]=np.sign(df.depth_adjusted_score)==np.sign(df.fraction_difference_E2_minus_Vehicle)
    df["exploratory_candidate"]=(df.cell_level_BH_q<=cfg["da_max_bh_q"]) & (
        df.fraction_difference_E2_minus_Vehicle.abs()>=cfg["da_min_abs_fraction_difference"]) & df.direction_agrees_with_raw_fraction
    return df,universe,null

def figures_for_line(a,da,diagnostics,figures,line):
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for condition,color in [("E2","#c44e52"),("Vehicle","#2878b5")]:
        obs=a.obs.loc[a.obs.treatment==condition]
        axs[0].hist(np.log10(obs.atac_total_peak_counts),bins=40,histtype="step",label=condition,color=color)
        axs[1].hist(obs.atac_n_peaks,bins=40,histtype="step",label=condition,color=color)
    axs[0].set(xlabel="log10 total peak counts",ylabel="Cells",title=f"{line}: available ATAC QC")
    axs[1].set(xlabel="Detected peaks",ylabel="Cells");axs[0].legend()
    axs[2].plot(diagnostics.component,diagnostics.depth_spearman,"o-",ms=3)
    axs[2].axhline(0,color="gray",lw=.6);axs[2].set(xlabel="LSI component",ylabel="Spearman r with log depth",title="Depth association")
    fig.savefig(figures/f"{line}_atac_qc_lsi.png",dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(15,9),layout="constrained")
    for ax,col in zip(axs.flat,["treatment","atac_leiden","rna_leiden","log1p_atac_counts","rna_pct_counts_mt","LSI1"]):
        sc.pl.umap(a,color=col,ax=ax,show=False,size=7,frameon=False,title=f"{line}: {col}")
    fig.savefig(figures/f"{line}_atac_umap.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6),layout="constrained")
    valid=da.testable
    color=np.where(da.exploratory_candidate,np.where(da.depth_adjusted_score>0,"#c44e52","#2878b5"),"#bbbbbb")
    ax.scatter(100*da.loc[valid,"fraction_difference_E2_minus_Vehicle"],
               -np.log10(da.loc[valid,"cell_level_BH_q"].clip(lower=1e-300)),
               c=color[valid],s=4,alpha=.6)
    ax.set(xlabel="Unadjusted accessibility difference: E2 − Vehicle (percentage points)",
           ylabel="-log10 depth-adjusted cell-level BH q (capped at 300)",
           title=f"{line}: exploratory differential accessibility")
    fig.savefig(figures/f"{line}_da_volcano.png",dpi=150);plt.close(fig)

def analyze_atac(root,line,cfg,ref):
    root=Path(root);tables=root/"results/tables/nb05";figures=root/"results/figures/nb05"
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    path=root/f"data/processed/{line}_ATAC_paired_paper_qc.h5ad";before=digest(path)
    a=ad.read_h5ad(path);ids=a.obs_names.copy()
    assert sparse.issparse(a.X) and np.issubdtype(a.X.dtype,np.integer)
    assert np.all(a.layers["binary"].data==1)
    np.testing.assert_array_equal(a.X.indices,a.layers["binary"].indices)
    np.testing.assert_array_equal(a.X.indptr,a.layers["binary"].indptr)
    sc.pp.calculate_qc_metrics(a,percent_top=None,log1p=False,inplace=True)
    np.testing.assert_array_equal(a.obs.total_counts,a.obs.atac_total_peak_counts)
    a.obs["rna_leiden"]=a.obs.pop("leiden")
    original_rna_labels=a.obs.rna_leiden.copy()
    a.obs["log1p_atac_counts"]=np.log1p(a.obs.atac_total_peak_counts)
    detected=np.asarray(a.layers["binary"].sum(axis=0)).ravel()
    mask=(detected>=cfg["embedding_min_cells"])&(detected/a.n_obs<=cfg["embedding_max_fraction"])
    a.var["used_for_lsi"]=mask
    print(f"{line}: TF-IDF/LSI on {mask.sum()} peaks",flush=True)
    x,idf=tfidf(a.layers["binary"],mask,cfg["tfidf_scale"])
    svd=TruncatedSVD(n_components=cfg["lsi_components"],n_iter=cfg["lsi_iterations"],random_state=cfg["seed"])
    lsi=svd.fit_transform(x)
    a.obsm["X_lsi"]=lsi.astype(np.float32)
    diagnostics=pd.DataFrame({"component":np.arange(1,lsi.shape[1]+1),
        "depth_spearman":[spearmanr(lsi[:,i],a.obs.log1p_atac_counts).statistic for i in range(lsi.shape[1])],
        "explained_variance_ratio":svd.explained_variance_ratio_})
    save_table(diagnostics,tables/f"{line}_lsi_diagnostics.tsv")
    np.savez_compressed(root/f"data/processed/{line}_atac_lsi_model.npz",peak_ids=a.var_names[mask].to_numpy(dtype=str),
                        idf=idf,components=svd.components_,singular_values=svd.singular_values_)
    start=1 if cfg["exclude_lsi1"] else 0
    selected=lsi[:,start:start+cfg["neighbor_lsi_components"]]
    a.obsm["X_lsi_neighbors"]=((selected-selected.mean(axis=0))/selected.std(axis=0)).astype(np.float32)
    a.obs["LSI1"]=lsi[:,0]
    del x,lsi,svd;gc.collect()
    sc.pp.neighbors(a,use_rep="X_lsi_neighbors",n_neighbors=cfg["neighbors"],transformer="sklearn",random_state=cfg["seed"])
    sc.tl.leiden(a,resolution=cfg["leiden_resolution"],flavor="igraph",n_iterations=cfg["leiden_iterations"],
                 directed=False,random_state=cfg["seed"],key_added="atac_leiden")
    sc.tl.umap(a,min_dist=cfg["umap_min_dist"],maxiter=cfg["umap_iterations"],random_state=cfg["seed"])
    print(f"{line}: depth-adjusted differential accessibility",flush=True)
    da,universe,null=differential_accessibility(a,cfg)
    print(f"{line}: coordinate-sensitive nearest-TSS annotation",flush=True)
    annotations=nearest_genes(a.var,get_reference(root,ref),cfg["annotation_max_distance"])
    da=da.join(annotations,on="peak")
    save_table(da,root/f"data/processed/{line}_nb05_da_all.tsv")
    save_table(annotations,root/f"data/processed/{line}_nb05_peak_annotations.tsv",True)
    save_table(universe,tables/f"{line}_da_peak_universe.tsv")
    candidates=da.loc[da.exploratory_candidate].sort_values("cell_level_BH_q")
    save_table(candidates,root/f"data/processed/{line}_nb05_da_candidates.tsv")
    save_table(candidates.groupby("adjusted_direction").head(1000),tables/f"{line}_top_da_candidates.tsv")
    save_table(pd.crosstab(a.obs.atac_leiden,a.obs.treatment),tables/f"{line}_atac_cluster_treatment.tsv",True)
    save_table(pd.crosstab(a.obs.atac_leiden,a.obs.rna_leiden),tables/f"{line}_atac_rna_cluster_overlap.tsv",True)
    save_table(a.obs,tables/f"{line}_atac_cell_qc.tsv",True)
    coords=pd.DataFrame(a.obsm["X_umap"],index=a.obs_names,columns=["UMAP1","UMAP2"])
    save_table(coords.join(a.obs[["treatment","atac_leiden","rna_leiden"]]),tables/f"{line}_atac_embeddings.tsv",True)
    figures_for_line(a,da,diagnostics,figures,line)
    a.uns["nb05"]={"config_json":json.dumps(cfg),"reference_json":json.dumps(ref),
                  "X":"original integer peak counts; LSI from separate binary TF-IDF","source_sha256":before}
    output=root/f"data/processed/{line}_ATAC_nb05.h5ad"
    tmp=output.with_suffix(".partial.h5ad");a.write_h5ad(tmp,compression="gzip");tmp.replace(output)
    assert a.obs_names.equals(ids) and a.obs.rna_leiden.equals(original_rna_labels)
    assert np.isfinite(a.obsm["X_lsi"]).all() and np.isfinite(a.obsm["X_umap"]).all()
    assert digest(path)==before
    result={"cells":a.n_obs,"peaks":a.n_vars,"lsi_peaks":int(mask.sum()),"atac_clusters":int(a.obs.atac_leiden.nunique()),
        "da_prevalence_pass":len(da),"da_testable":int(da.testable.sum()),"da_candidates":len(candidates),
        "candidate_direction_counts":candidates.adjusted_direction.value_counts().to_dict(),
        "candidate_with_provisional_gene":int(candidates.candidate_nearby_genes.ne("").sum()),
        "RNA_ATAC_cluster_ARI":float(adjusted_rand_score(a.obs.rna_leiden,a.obs.atac_leiden)),
        "lsi1_depth_spearman":float(diagnostics.depth_spearman.iloc[0]),
        "null_depth_model":null,"source_sha256":before,"nb02_input_unchanged":True,
        "cells_and_rna_labels_preserved":True,"biological_replicates_identifiable":False,
        "coordinate_origin_confirmed":False,"annotation":"provisional nearest TSS, agreement under two coordinate conventions"}
    print(json.dumps(result,indent=2),flush=True)
    del a,da;gc.collect()
    return result
