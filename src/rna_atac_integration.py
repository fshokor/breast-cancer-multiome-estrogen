"""Descriptive RNA pathway / provisional nearby ATAC integration."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1048576),b""): h.update(block)
    return h.hexdigest()

def write_tsv(df,path):
    df.to_csv(path,sep="\t",index=False,float_format="%.9g")

def eligible_links(da,rna_genes,distance,exclude_exclusive):
    """Exclude tied/unknown gene assignments; missing evidence is not zero."""
    x=da.copy()
    names=x.candidate_nearby_genes.fillna("").astype(str)
    valid=(x.testable & x.coordinate_robust_nearest &
           np.isfinite(x.depth_adjusted_score) &
           x.distance_bp_if_0based.le(distance) & x.distance_bp_if_1based.le(distance) &
           names.ne("") & ~names.str.contains(";",regex=False) & names.isin(rna_genes))
    x=x.loc[valid].copy()
    x["gene"]=names.loc[valid]
    x["condition_exclusive"]=x.detected_E2.eq(0)|x.detected_Vehicle.eq(0)
    if exclude_exclusive: x=x.loc[~x.condition_exclusive].copy()
    return x

def aggregate_genes(links,rna):
    """One row per RNA-tested gene, including genes without eligible peaks."""
    columns=["gene","n_peaks","atac_median_score","atac_median_fraction_difference",
             "n_positive_peaks","n_negative_peaks","n_condition_exclusive_peaks",
             "n_candidate_positive","n_candidate_negative"]
    rows=[]
    for gene,g in links.groupby("gene",sort=True):
        z=g.depth_adjusted_score
        candidate=g.exploratory_candidate
        rows.append([gene,len(g),z.median(),g.fraction_difference_E2_minus_Vehicle.median(),
                     int(z.gt(0).sum()),int(z.lt(0).sum()),int(g.condition_exclusive.sum()),
                     int((candidate & z.gt(0)).sum()),int((candidate & z.lt(0)).sum())])
    a=pd.DataFrame(rows,columns=columns)
    out=rna.merge(a,on="gene",how="left",validate="one_to_one")
    for col in ["n_peaks","n_positive_peaks","n_negative_peaks","n_condition_exclusive_peaks",
                "n_candidate_positive","n_candidate_negative"]:
        out[col]=out[col].fillna(0).astype(int)
    out["mapped"]=out.n_peaks.gt(0)
    out["mixed_peak_directions"]=out.n_positive_peaks.gt(0)&out.n_negative_peaks.gt(0)
    out["rna_atac_direction_agrees"]=pd.array([pd.NA]*len(out),dtype="boolean")
    nz=out.mapped & out.atac_median_score.ne(0) & out.approx_log2FC_E2_vs_Vehicle.ne(0)
    out.loc[nz,"rna_atac_direction_agrees"]=(np.sign(out.loc[nz,"atac_median_score"]) ==
                    np.sign(out.loc[nz,"approx_log2FC_E2_vs_Vehicle"])).values
    return out

def pathway_summary(genes,gsea,sets,minimum):
    indexed=genes.set_index("gene")
    rows=[];members=[]
    # Include all frozen Hallmarks, including sets not tested by nb04.
    gsea=gsea.set_index("Term")
    for term,reference_genes in sets.items():
        present=term in gsea.index
        gr=gsea.loc[term] if present else None
        nes=float(gr["NES"]) if present else np.nan
        q=float(gr["FDR q-val"]) if present else np.nan
        lead=set(str(gr["Lead_genes"]).split(";")) if present and pd.notna(gr["Lead_genes"]) else set()
        for subset,requested in [("all_members",set(reference_genes)),("leading_edge",lead)]:
            matched=sorted(requested & set(indexed.index))
            g=indexed.loc[matched].copy()
            m=g.loc[g.mapped]
            aligned=m.atac_median_score*np.sign(nes)
            same=int(aligned.gt(0).sum());opposite=int(aligned.lt(0).sum())
            nonzero=same+opposite
            fraction=same/nonzero if nonzero else np.nan
            median=m.atac_median_score.median()
            raw=m.atac_median_fraction_difference.median()
            if not present: status="RNA_GSEA_not_tested"
            elif len(m)<minimum: status="limited_ATAC_coverage"
            elif not nonzero or same==opposite: status="balanced_or_zero"
            elif same>opposite: status="majority_same_direction"
            else: status="majority_opposite_direction"
            rows.append(dict(pathway=term,subset=subset,rna_NES=nes,rna_GSEA_FDR=q,
                rna_direction=("E2_positive" if nes>0 else "Vehicle_negative") if np.isfinite(nes) else "not_tested",
                n_reference_genes=len(reference_genes),n_subset_genes=len(requested),
                n_RNA_tested_genes=len(g),n_ATAC_mapped_genes=len(m),
                mapped_fraction_of_RNA_tested=len(m)/len(g) if len(g) else np.nan,
                n_linked_peaks=int(m.n_peaks.sum()),n_same_direction=same,n_opposite_direction=opposite,
                n_zero_direction=int(m.atac_median_score.eq(0).sum()) if np.isfinite(nes) and nes!=0 else 0,
                n_undefined_RNA_direction=0 if np.isfinite(nes) and nes!=0 else len(m),
                fraction_same_direction=fraction,median_ATAC_score=median,
                median_ATAC_fraction_difference=raw,n_mixed_peak_genes=int(m.mixed_peak_directions.sum()),
                direction_summary=status))
            if len(g):
                g=g.reset_index()
                g.insert(0,"subset",subset);g.insert(0,"pathway",term)
                g["rna_NES"]=nes
                g["ATAC_score_aligned_to_RNA_NES"]=g.atac_median_score*np.sign(nes)
                members.append(g)
    return pd.DataFrame(rows),pd.concat(members,ignore_index=True) if members else pd.DataFrame()

def plot_pathways(summary,line,directory):
    s=summary.loc[(summary.scenario=="100kb_all") & (summary.subset=="all_members")].copy()
    s=s.sort_values("rna_NES",na_position="last")
    fig,axes=plt.subplots(1,3,figsize=(17,15),sharey=True,gridspec_kw={"width_ratios":[1,1,0.7]})
    y=np.arange(len(s))
    axes[0].barh(y,s.rna_NES,color=np.where(s.rna_NES.ge(0),"#bb4d39","#3979ac"))
    sig=s.rna_GSEA_FDR.le(.05)
    axes[0].scatter(s.loc[sig,"rna_NES"],y[sig],s=10,color="black",zorder=3)
    axes[0].set_yticks(y,s.pathway.str.replace("HALLMARK_","",regex=False).str.replace("_"," ",regex=False),fontsize=8)
    axes[0].set_xlabel("RNA GSEA NES\nblack dot: RNA FDR ≤ 0.05")
    axes[0].axvline(0,color="gray",lw=.7)
    valid=s.fraction_same_direction.notna()
    axes[1].scatter(s.loc[valid,"fraction_same_direction"],y[valid],
                    c=np.where(s.loc[valid,"n_ATAC_mapped_genes"].ge(10),"#315b53","#b8b8b8"),s=22)
    axes[1].axvline(.5,color="gray",ls="--",lw=.8);axes[1].set_xlim(-.03,1.03)
    axes[1].set_xlabel("Fraction of mapped genes whose ATAC\nscore agrees with the RNA NES sign")
    axes[2].barh(y,s.n_ATAC_mapped_genes,color="#8599a8")
    axes[2].set_xlabel("Genes with eligible\nnearby tested peaks")
    axes[0].set_title("Both RNA enrichment directions")
    axes[1].set_title("Descriptive agreement; not ATAC activity")
    axes[2].set_title("Coverage")
    fig.suptitle(f"{line}: RNA pathways and provisional nearest-gene ATAC evidence\n100 kb, all eligible tested peaks; gray points: fewer than 10 mapped genes",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(directory/f"{line}_pathway_concordance.png",dpi=160);plt.close(fig)
    leading=summary.loc[summary.subset=="leading_edge"]
    pivot=leading.pivot(index="pathway",columns="scenario",values="fraction_same_direction").reindex(s.pathway)
    count=leading.pivot(index="pathway",columns="scenario",values="n_ATAC_mapped_genes").reindex(s.pathway)
    shown=pivot.where(count.ge(10))
    fig,ax=plt.subplots(figsize=(11,14))
    im=ax.imshow(np.ma.masked_invalid(shown.to_numpy()),aspect="auto",vmin=0,vmax=1,cmap=plt.get_cmap("RdBu_r").with_extremes(bad="#c7c7c7"),origin="lower")
    labels=[]
    primary=s.set_index("pathway")
    for term in shown.index:
        row=primary.loc[term]
        direction="+" if row.rna_NES>0 else "-" if row.rna_NES<0 else "NA"
        star=" *" if row.rna_GSEA_FDR<=.05 else ""
        labels.append(term.replace("HALLMARK_","").replace("_"," ")+f" [{direction}]{star}")
    ax.set_yticks(np.arange(len(shown)),labels,fontsize=8)
    ax.set_xticks(np.arange(len(shown.columns)),shown.columns,rotation=20,ha="right")
    ax.set_title(f"{line}: RNA leading-edge genes, ATAC direction agreement\nGray: fewer than 10 mapped genes or no direction estimate\n[+/-]: RNA NES sign; *: RNA FDR ≤ 0.05")
    fig.colorbar(im,ax=ax,label="Fraction of nonzero gene ATAC medians agreeing with RNA NES")
    fig.tight_layout();fig.savefig(directory/f"{line}_leading_edge_sensitivity.png",dpi=160);plt.close(fig)

def run_integration(root,config,hallmarks):
    root=Path(root);tables=root/"results/tables/nb06";figs=root/"results/figures/nb06"
    tables.mkdir(parents=True,exist_ok=True);figs.mkdir(parents=True,exist_ok=True)
    all_summaries=[];report={"run_at_utc":datetime.now(timezone.utc).isoformat(),"config":config,"cell_lines":{}}
    sources=[root/"configs/hallmark_v7.json",root/"configs/atac_reference.json"]
    for line in ("MCF7","T47D"):
        sources.extend([root/f"results/tables/nb04/{line}_deg_all.tsv",
                        root/f"results/tables/nb04/{line}_hallmark_gsea.tsv",
                        root/f"data/processed/{line}_nb05_da_all.tsv"])
    before={str(x.relative_to(root)):file_hash(x) for x in sources}
    assert json.loads((root/"configs/hallmark_v7.json").read_text())==hallmarks,"Frozen Hallmarks differ from embedded copy."
    for line in ("MCF7","T47D"):
        rna=pd.read_csv(root/f"results/tables/nb04/{line}_deg_all.tsv",sep="\t")
        gsea=pd.read_csv(root/f"results/tables/nb04/{line}_hallmark_gsea.tsv",sep="\t")
        da=pd.read_csv(root/f"data/processed/{line}_nb05_da_all.tsv",sep="\t")
        assert rna.gene.is_unique and da.peak.is_unique and gsea.Term.is_unique
        summaries=[];audits=[]
        for distance in config["distance_cutoffs_bp"]:
            for exclude in config["exclude_condition_exclusive"]:
                scenario=f"{distance//1000}kb_"+("shared_detection" if exclude else "all")
                links=eligible_links(da,set(rna.gene),distance,exclude)
                genes=aggregate_genes(links,rna)
                summary,details=pathway_summary(genes,gsea,hallmarks["gene_sets"],
                           config["min_mapped_genes_for_direction_summary"])
                for frame in (links,genes,summary,details):
                    frame.insert(0,"scenario",scenario);frame.insert(0,"cell_line",line)
                write_tsv(genes,tables/f"{line}_{scenario}_gene_evidence.tsv")
                if scenario=="100kb_all":
                    write_tsv(links,tables/f"{line}_provisional_peak_gene_links.tsv")
                    write_tsv(details,tables/f"{line}_pathway_gene_evidence.tsv")
                summaries.append(summary)
                audits.append(dict(scenario=scenario,tested_ATAC_peaks=len(da),eligible_peaks=len(links),
                    RNA_tested_genes=len(rna),mapped_RNA_genes=int(genes.mapped.sum()),
                    condition_exclusive_eligible_peaks=int(links.condition_exclusive.sum())))
        summary=pd.concat(summaries,ignore_index=True)
        write_tsv(summary,tables/f"{line}_pathway_summary.tsv")
        write_tsv(pd.DataFrame(audits),tables/f"{line}_mapping_audit.tsv")
        plot_pathways(summary,line,figs)
        all_summaries.append(summary)
        report["cell_lines"][line]={"mapping_scenarios":audits,
            "RNA_positive_NES":int(gsea.NES.gt(0).sum()),"RNA_negative_NES":int(gsea.NES.lt(0).sum()),
            "RNA_positive_FDR_005":int((gsea.NES.gt(0)&gsea["FDR q-val"].le(.05)).sum()),
            "RNA_negative_FDR_005":int((gsea.NES.lt(0)&gsea["FDR q-val"].le(.05)).sum())}
    combined=pd.concat(all_summaries,ignore_index=True)
    write_tsv(combined,tables/"all_pathway_scenarios.tsv")
    a=all_summaries[0];b=all_summaries[1]
    cross=a.merge(b,on=["pathway","subset","scenario"],suffixes=("_MCF7","_T47D"),validate="one_to_one")
    cross["RNA_NES_sign_agrees"]=(np.sign(cross.rna_NES_MCF7)==np.sign(cross.rna_NES_T47D)).astype("boolean")
    invalid=cross.rna_NES_MCF7.isna()|cross.rna_NES_T47D.isna()
    cross.loc[invalid,"RNA_NES_sign_agrees"]=pd.NA
    write_tsv(cross,tables/"cross_line_pathway_comparison.tsv")
    after={str(x.relative_to(root)):file_hash(x) for x in sources}
    assert before==after,"Upstream files changed."
    report["source_sha256"]=before;report["upstream_inputs_unchanged"]=True
    report["interpretation"]="Descriptive agreement only; no ATAC pathway p-values, activity labels, or validated regulatory links."
    (tables/"integration_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    return combined,cross,report
