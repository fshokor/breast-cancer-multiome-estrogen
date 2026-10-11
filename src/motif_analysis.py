"""Sequence motif enrichment with auditable matched backgrounds and coordinate sensitivity."""
from pathlib import Path
import gzip,hashlib,json,re,time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import py2bit
import MOODS.scan,MOODS.tools
from scipy.spatial import cKDTree
from scipy.special import logit
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def save_tsv(df,path):
    df.to_csv(path,sep="\t",index=False,float_format="%.9g")

def parse_jaspar(path):
    entries=[]
    lines=Path(path).read_text().splitlines()
    for i in range(0,len(lines),5):
        motif_id,name=lines[i][1:].split(None,1)
        matrix=[]
        for base,line in zip("ACGT",lines[i+1:i+5]):
            assert line.startswith(base)
            matrix.append([float(x) for x in re.findall(r"[\d.]+",line)])
        assert len({len(v) for v in matrix})==1
        entries.append({"motif_id":motif_id,"motif_name":name,"matrix":matrix})
    assert len({m["motif_id"] for m in entries})==len(entries)
    return entries

def build_scanner(motifs,config):
    bg=config["sequence_background"]
    forward=[MOODS.tools.log_odds(m["matrix"],bg,config["pseudocount_total_per_column"]) for m in motifs]
    matrices=forward+[MOODS.tools.reverse_complement(m,4) for m in forward]
    thresholds=[MOODS.tools.threshold_from_p(m,bg,config["motif_site_pvalue"]) for m in matrices]
    scanner=MOODS.scan.Scanner(config["site_scan_window"])
    scanner.set_motifs(matrices,bg,thresholds)
    return scanner,thresholds[:len(motifs)]

def scan_union_sequence(scanner,seq,n_motifs):
    """Union sequence is [s-1,e); both hypotheses differ only at leftmost base."""
    common=np.zeros(n_motifs,dtype=bool);extended=common.copy()
    results=scanner.scan(seq)
    for j,matches in enumerate(results):
        if matches:
            idx=j%n_motifs
            extended[idx]=True
            if any(hit.pos>=1 for hit in matches):common[idx]=True
    return common,extended

def sequence_qc(root,da_tables,refs,config):
    peaks=sorted(set().union(*(set(x.peak) for x in da_tables.values())))
    genome=py2bit.open(str(root/refs["resources"]["hg38.2bit"]["path"]))
    sizes=genome.chroms()
    blacklist=pd.read_csv(root/refs["resources"]["hg38-blacklist.v2.bed.gz"]["path"],
                          sep="\t",header=None,usecols=[0,1,2],names=["chrom","start","end"])
    lookup={}
    for chrom,g in blacklist.groupby("chrom"):
        g=g.sort_values("start")
        lookup[chrom]=(g.start.to_numpy(),np.maximum.accumulate(g.end.to_numpy()))
    records=[];sequences={}
    for peak in peaks:
        chrom,start,end=peak.rsplit("-",2);start=int(start);end=int(end)
        reason="eligible";seq=""
        if chrom not in config["standard_chromosomes"]:reason="nonstandard_chromosome"
        elif chrom not in sizes:reason="missing_reference_chromosome"
        elif start<1 or end<=start or end>sizes[chrom]:reason="invalid_interval_under_one_or_both_hypotheses"
        else:
            if chrom in lookup:
                starts,ends=lookup[chrom];i=np.searchsorted(starts,end,side="left")-1
                if i>=0 and ends[i]>start-1:reason="overlaps_blacklist_under_at_least_one_hypothesis"
            if reason=="eligible":
                seq=genome.sequence(chrom,start-1,end).upper()
                if len(seq)!=end-start+1:raise ValueError("Genome extraction length mismatch")
                if set(seq)-set("ACGT"):reason="ambiguous_sequence"
        common=seq[1:] if seq else ""
        gc=(common.count("G")+common.count("C"))/len(common) if common else np.nan
        records.append((peak,chrom,start,end,end-start,gc,reason))
        if reason=="eligible":sequences[peak]=seq
    genome.close()
    return pd.DataFrame(records,columns=["peak","chrom","start_as_supplied","end_as_supplied",
             "length_common_interval","gc_fraction","sequence_status"]),sequences

def matched_background(foreground,controls,config):
    """Deterministic 1:1 matching without replacement within chromosome and calipers."""
    features=["gc_fraction","log_length","logit_accessibility"]
    scales=np.array([config["matching_calipers"][x] for x in features])
    pairs=[];unmatched=[]
    for chrom,fg in foreground.groupby("chrom",sort=True):
        fg=fg.sort_values("peak");bg=controls.loc[controls.chrom==chrom].sort_values("peak")
        if bg.empty:
            unmatched.extend(fg.peak.tolist());continue
        f=fg[features].to_numpy();b=bg[features].to_numpy()
        k=min(config["matching_nearest_candidates"],len(bg))
        distances,indices=cKDTree(b/scales).query(f/scales,k=k)
        if k==1:distances=distances[:,None];indices=indices[:,None]
        choices=[]
        for i in range(len(fg)):
            valid=np.all(np.abs((b[indices[i]]-f[i])/scales)<=1+1e-12,axis=1)
            choices.append([(float(d),int(j)) for d,j in zip(distances[i][valid],indices[i][valid])])
        used=set()
        for i in sorted(range(len(fg)),key=lambda i:(len(choices[i]),fg.iloc[i].peak)):
            options=[(d,j) for d,j in choices[i] if j not in used]
            if not options:unmatched.append(fg.iloc[i].peak);continue
            distance,j=min(options,key=lambda x:(x[0],bg.iloc[x[1]].peak));used.add(j)
            record={"foreground_peak":fg.iloc[i].peak,"background_peak":bg.iloc[j].peak,
                    "chrom":chrom,"scaled_euclidean_distance":distance}
            for z,feature in enumerate(features):
                record["foreground_"+feature]=f[i,z];record["background_"+feature]=b[j,z]
            pairs.append(record)
    columns=["foreground_peak","background_peak","chrom","scaled_euclidean_distance"]+[
        side+"_"+f for f in features for side in ["foreground","background"]]
    return pd.DataFrame(pairs,columns=columns),unmatched

def balance_report(pairs):
    rows=[]
    for feature in ["gc_fraction","log_length","logit_accessibility"]:
        a=pairs["foreground_"+feature].to_numpy();b=pairs["background_"+feature].to_numpy()
        if len(a)<2:smd=np.nan
        else:
            denom=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
            smd=(a.mean()-b.mean())/denom if denom>0 else 0.0 if a.mean()==b.mean() else np.inf
        rows.append(dict(feature=feature,foreground_mean=a.mean() if len(a) else np.nan,
                         background_mean=b.mean() if len(b) else np.nan,
                         standardized_mean_difference=smd,
                         maximum_absolute_pair_difference=float(np.abs(a-b).max()) if len(a) else np.nan))
    return pd.DataFrame(rows)

def paired_enrichment(foreground_hits,background_hits):
    assert foreground_hits.shape==background_hits.shape
    fg_only=(foreground_hits & ~background_hits).sum(axis=0)
    bg_only=(~foreground_hits & background_hits).sum(axis=0)
    discordant=fg_only+bg_only
    p=binom.sf(fg_only-1,discordant,.5)
    p=np.where(discordant==0,1.0,p)
    return pd.DataFrame({"foreground_hits":foreground_hits.sum(axis=0),
          "background_hits":background_hits.sum(axis=0),"foreground_only_pairs":fg_only,
          "background_only_pairs":bg_only,"paired_log2_odds_smoothed":np.log2((fg_only+.5)/(bg_only+.5)),
          "peak_pair_p":p})

def motif_gene_labels(name,rna_genes):
    # Case-normalized motif label candidates; not an orthology or TF-target mapping.
    parts=name.split("::")
    candidates=[re.sub(r"\(var\.\d+\)$","",x).upper() for x in parts]
    return [(x,x in rna_genes) for x in candidates]

def plots(enrichment,balance,root):
    directory=root/"results/figures/nb07";directory.mkdir(parents=True,exist_ok=True)
    for line in ["MCF7","T47D"]:
        for scenario in ["all_candidates","shared_detection"]:
            primary=enrichment.loc[(enrichment.cell_line==line)&
              (enrichment.scenario==scenario)&
              (enrichment.coordinate_hypothesis=="zero_based_half_open")]
            fig,axes=plt.subplots(1,2,figsize=(15,8))
            for ax,direction,title in zip(axes,["E2","Vehicle"],["More accessible under E2","Less accessible under E2"]):
                d=primary.loc[primary.direction==direction].sort_values(["global_BH_q","peak_pair_p","motif_id"]).head(15).iloc[::-1]
                x=-np.log10(d.global_BH_q.clip(lower=1e-300))
                colors=np.where(d.priority_candidate,"#bd5945","#9ca3af")
                ax.barh(np.arange(len(d)),x,color=colors)
                ax.set_yticks(np.arange(len(d)),d.motif_name+" "+d.motif_id,fontsize=9)
                ax.axvline(-np.log10(.05),color="black",ls="--",lw=.8)
                ax.set_xlabel("−log10 global motif BH q");ax.set_title(title)
            fig.suptitle(f"{line}: matched-background motif enrichment ({scenario})\nPrimary coordinate hypothesis; orange requires acceptable matching and q ≤ 0.05")
            fig.tight_layout(rect=(0,0,1,.93));fig.savefig(directory/(f"{line}_motif_enrichment.png" if scenario=="all_candidates" else f"{line}_motif_enrichment_shared_detection.png"),dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6))
    for feature,g in balance.groupby("feature"):
        labels=g.cell_line+" / "+g.direction+" / "+g.scenario
        ax.scatter(g.standardized_mean_difference,labels,label=feature,s=45)
    ax.axvline(0,color="gray");ax.axvline(.1,color="gray",ls="--");ax.axvline(-.1,color="gray",ls="--")
    ax.set_xlabel("Standardized mean difference after matching")
    ax.set_title("GC, peak length and accessibility balance\nDashed lines: ±0.1 review threshold")
    ax.legend(loc="best");fig.tight_layout();fig.savefig(directory/"background_balance.png",dpi=160);plt.close(fig)

def run_motifs(root,config,refs,coordinate_audit):
    root=Path(root);tables=root/"results/tables/nb07";tables.mkdir(parents=True,exist_ok=True)
    cache=root/"data/processed/nb07";cache.mkdir(parents=True,exist_ok=True)
    sources=[root/f"data/processed/{line}_nb05_da_all.tsv" for line in ["MCF7","T47D"]]+[
        root/f"results/tables/nb04/{line}_deg_all.tsv" for line in ["MCF7","T47D"]]+[
        root/f"results/tables/nb06/{line}_provisional_peak_gene_links.tsv" for line in ["MCF7","T47D"]]+[
        root/"results/tables/nb06/all_pathway_scenarios.tsv",root/"configs/hallmark_v7.json"]
    before={str(x.relative_to(root)):sha256(x) for x in sources}
    for name,ref in refs["resources"].items():
        assert sha256(root/ref["path"])==ref["sha256"],f"Reference changed: {name}"
    data={line:pd.read_csv(root/f"data/processed/{line}_nb05_da_all.tsv",sep="\t") for line in ["MCF7","T47D"]}
    motifs=parse_jaspar(root/refs["resources"]["JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt"]["path"])
    motif_table=pd.DataFrame([{k:v for k,v in m.items() if k!="matrix"} for m in motifs])
    motif_table["length"]=[len(m["matrix"][0]) for m in motifs]
    print("Sequence QC and reference verification...",flush=True)
    qc,sequences=sequence_qc(root,data,refs,config)
    save_tsv(qc,tables/"peak_sequence_qc.tsv")
    eligible=qc.loc[qc.sequence_status=="eligible"].copy().sort_values("peak").reset_index(drop=True)
    peak_names=eligible.peak.tolist();peak_index={p:i for i,p in enumerate(peak_names)}
    key=hashlib.sha256(json.dumps({"refs":refs,"config":config,"peaks":peak_names,
                     "implementation":"nb07-v1"},sort_keys=True).encode()).hexdigest()
    hit_file=cache/"motif_presence.npz";keyfile=cache/"motif_presence_key.json"
    if hit_file.exists() and keyfile.exists() and json.loads(keyfile.read_text())["key"]==key:
        arrays=np.load(hit_file,allow_pickle=False);common=arrays["common"];extended=arrays["extended"]
        assert arrays["peaks"].tolist()==peak_names
        assert arrays["motifs"].tolist()==motif_table.motif_id.tolist()
        thresholds=json.loads(keyfile.read_text())["thresholds"];print("Reused verified motif presence cache.",flush=True)
    else:
        scanner,thresholds=build_scanner(motifs,config)
        common=np.zeros((len(peak_names),len(motifs)),dtype=bool);extended=common.copy()
        for i,peak in enumerate(peak_names):
            common[i],extended[i]=scan_union_sequence(scanner,sequences[peak],len(motifs))
            if (i+1)%2500==0:print(f"Scanned {i+1:,}/{len(peak_names):,} peaks",flush=True)
        np.savez_compressed(hit_file,common=common,extended=extended,peaks=np.array(peak_names),
                            motifs=motif_table.motif_id.to_numpy(dtype=str))
        keyfile.write_text(json.dumps({"key":key,"thresholds":thresholds},indent=2)+"\n")
    assert common.shape==extended.shape==(len(peak_names),len(motifs))
    assert np.all(~common|extended)
    motif_table["site_threshold_natural_log_odds"]=thresholds
    save_tsv(motif_table,tables/"motif_catalog.tsv")
    reports=[];all_results=[];balances=[];pair_lookup={}
    for line,da in data.items():
        universe=da.loc[da.testable].merge(eligible,on="peak",validate="one_to_one")
        universe["condition_exclusive"]=universe.detected_E2.eq(0)|universe.detected_Vehicle.eq(0)
        universe["log_length"]=np.log(universe.length_common_interval)
        universe["balanced_accessibility"]=(universe.fraction_E2+universe.fraction_Vehicle)/2
        universe["logit_accessibility"]=logit(universe.balanced_accessibility.clip(1e-6,1-1e-6))
        for scenario in config["sensitivity_scenarios"]:
            allowed=universe if scenario=="all_candidates" else universe.loc[~universe.condition_exclusive]
            controls=allowed.loc[~allowed.exploratory_candidate]
            for direction in ["E2","Vehicle"]:
                foreground=allowed.loc[allowed.exploratory_candidate & (allowed.adjusted_direction==direction)]
                pairs,unmatched=matched_background(foreground,controls,config)
                tag=f"{line}_{direction}_{scenario}"
                save_tsv(pairs,tables/f"{tag}_matched_pairs.tsv")
                save_tsv(pd.DataFrame({"peak":unmatched}),tables/f"{tag}_unmatched_foreground.tsv")
                balance=balance_report(pairs)
                balance["cell_line"]=line;balance["scenario"]=scenario;balance["direction"]=direction
                balances.append(balance)
                fraction=len(pairs)/len(foreground) if len(foreground) else 0
                good=(len(pairs)>=config["minimum_matched_pairs"] and
                      fraction>=config["minimum_foreground_match_fraction"] and
                      balance.standardized_mean_difference.abs().le(config["maximum_absolute_standardized_mean_difference"]).all())
                reports.append(dict(cell_line=line,scenario=scenario,direction=direction,
                    eligible_foreground=len(foreground),available_controls=len(controls),matched_pairs=len(pairs),
                    foreground_match_fraction=fraction,matching_acceptable=bool(good)))
                pair_lookup[(line,scenario,direction)]=pairs
                fidx=np.array([peak_index[x] for x in pairs.foreground_peak],dtype=int)
                bidx=np.array([peak_index[x] for x in pairs.background_peak],dtype=int)
                for hypothesis,hits in [("zero_based_half_open",common),("one_based_closed",extended)]:
                    out=paired_enrichment(hits[fidx],hits[bidx])
                    if len(pairs)<config["minimum_matched_pairs"]:out["peak_pair_p"]=np.nan
                    out=pd.concat([motif_table.reset_index(drop=True),out],axis=1)
                    out["cell_line"]=line;out["scenario"]=scenario;out["direction"]=direction
                    out["coordinate_hypothesis"]=hypothesis;out["n_matched_pairs"]=len(pairs)
                    out["foreground_match_fraction"]=fraction;out["matching_acceptable"]=bool(good)
                    out["foreground_hit_fraction"]=out.foreground_hits/len(pairs) if len(pairs) else np.nan
                    out["background_hit_fraction"]=out.background_hits/len(pairs) if len(pairs) else np.nan
                    out["within_comparison_BH_q"]=np.nan
                    valid=out.peak_pair_p.notna()
                    if valid.any():out.loc[valid,"within_comparison_BH_q"]=multipletests(out.loc[valid,"peak_pair_p"],method="fdr_bh")[1]
                    all_results.append(out)
                print(tag,f"{len(pairs)}/{len(foreground)} foreground peaks matched; balance acceptable={good}",flush=True)
    results=pd.concat(all_results,ignore_index=True);results["global_BH_q"]=np.nan
    valid=results.peak_pair_p.notna()
    results.loc[valid,"global_BH_q"]=multipletests(results.loc[valid,"peak_pair_p"],method="fdr_bh")[1]
    results["priority_candidate"]=results.global_BH_q.le(config["motif_enrichment_fdr"]) & results.matching_acceptable
    save_tsv(results,tables/"motif_enrichment_all.tsv")
    match_report=pd.DataFrame(reports);save_tsv(match_report,tables/"matching_summary.tsv")
    balance=pd.concat(balances,ignore_index=True);save_tsv(balance,tables/"matching_balance.tsv")
    sensitivity=results.pivot(index=["cell_line","direction","motif_id","motif_name"],
                         columns=["scenario","coordinate_hypothesis"],values="global_BH_q")
    sensitivity.columns=["q__"+"__".join(x) for x in sensitivity.columns]
    sensitivity=sensitivity.reset_index()
    qcols=[x for x in sensitivity if x.startswith("q__")]
    sensitivity["q_le_005_all_four_scenarios"]=sensitivity[qcols].le(config["motif_enrichment_fdr"]).all(axis=1)
    save_tsv(sensitivity,tables/"motif_sensitivity.tsv")
    cross=results.loc[results.cell_line=="MCF7"].merge(results.loc[results.cell_line=="T47D"],
          on=["motif_id","scenario","direction","coordinate_hypothesis"],suffixes=("_MCF7","_T47D"),validate="one_to_one")
    save_tsv(cross,tables/"cross_line_motifs.tsv")
    rna_rows=[]
    for line in ["MCF7","T47D"]:
        rna=pd.read_csv(root/f"results/tables/nb04/{line}_deg_all.tsv",sep="\t").set_index("gene")
        for m in motifs:
            for label,found in motif_gene_labels(m["motif_name"],set(rna.index)):
                record={"cell_line":line,"motif_id":m["motif_id"],"motif_name":m["motif_name"],
                        "candidate_TF_gene_label":label,"RNA_gene_tested":found,
                        "mapping_policy":"case-normalized label; variant suffix removed; not validated orthology"}
                if found:
                    record.update({x:rna.loc[label,x] for x in ["approx_log2FC_E2_vs_Vehicle","cell_level_BH_q","pct_E2"]})
                rna_rows.append(record)
    rna_table=pd.DataFrame(rna_rows);save_tsv(rna_table,tables/"motif_TF_RNA_context.tsv")
    save_tsv(results.merge(rna_table,on=["cell_line","motif_id","motif_name"],how="left"),
             tables/"motif_enrichment_with_TF_RNA.tsv")
    # Trace top candidates in each detection scenario to nb06 proximity context; no target tests.
    hallmark=json.loads((root/"configs/hallmark_v7.json").read_text())["gene_sets"]
    pathways=pd.read_csv(root/"results/tables/nb06/all_pathway_scenarios.tsv",sep="\t")
    evidence=[];selected=[]
    for line in ["MCF7","T47D"]:
        links=pd.read_csv(root/f"results/tables/nb06/{line}_provisional_peak_gene_links.tsv",sep="\t").set_index("peak")
        for direction in ["E2","Vehicle"]:
            for scenario in ["all_candidates","shared_detection"]:
                top=results.loc[(results.cell_line==line)&(results.direction==direction)&
                   (results.scenario==scenario)&(results.coordinate_hypothesis=="zero_based_half_open")&
                   results.priority_candidate].sort_values(["global_BH_q","peak_pair_p"]).head(10)
                pairs=pair_lookup[(line,scenario,direction)]
                for _,m in top.iterrows():
                    selected.append({"cell_line":line,"direction":direction,"motif_id":m.motif_id,"motif_name":m.motif_name})
                    j=motif_table.index[motif_table.motif_id==m.motif_id][0]
                    hit_peaks=[x for x in pairs.foreground_peak if common[peak_index[x],j] and x in links.index]
                    genes=set(links.loc[hit_peaks,"gene"]) if hit_peaks else set()
                    for term,members in hallmark.items():
                        overlap=genes & set(members)
                        if not overlap:continue
                        context=pathways.loc[(pathways.cell_line==line)&(pathways.pathway==term)&
                                    (pathways.scenario=="100kb_all")&(pathways.subset=="all_members")].iloc[0]
                        evidence.append(dict(cell_line=line,scenario=scenario,direction=direction,motif_id=m.motif_id,motif_name=m.motif_name,
                           pathway=term,n_provisional_genes=len(overlap),genes=";".join(sorted(overlap)),
                           peaks=";".join(x for x in hit_peaks if links.loc[x,"gene"] in overlap),
                           rna_NES=context.rna_NES,rna_GSEA_FDR=context.rna_GSEA_FDR,
                           nb06_fraction_same_direction=context.fraction_same_direction,
                           interpretation="motif-bearing peak proximity only; not TF targets or pathway enrichment"))
    save_tsv(pd.DataFrame(evidence,columns=["cell_line","scenario","direction","motif_id","motif_name","pathway",
       "n_provisional_genes","genes","peaks","rna_NES","rna_GSEA_FDR","nb06_fraction_same_direction","interpretation"]),
       tables/"top_motif_pathway_context.tsv")
    plots(results,balance,root)
    assert before=={str(x.relative_to(root)):sha256(x) for x in sources},"Upstream files changed"
    report={"run_at_utc":datetime.now(timezone.utc).isoformat(),"config":config,"references":refs,
       "coordinate_audit":coordinate_audit,"n_motifs":len(motifs),"n_union_tested_peaks":len(qc),
       "n_sequence_eligible_peaks":len(eligible),"sequence_qc_counts":qc.sequence_status.value_counts().to_dict(),
       "coordinate_changed_presence_entries":int(np.count_nonzero(common!=extended)),
       "coordinate_changed_peaks":int(np.any(common!=extended,axis=1).sum()),
       "source_sha256":before,"upstream_inputs_unchanged":True,
       "matching":reports,"matching_acceptability_required_for_priority":True,
       "primary_priority_counts":results.loc[(results.scenario=="all_candidates")&
           (results.coordinate_hypothesis=="zero_based_half_open")].groupby(["cell_line","direction"]).priority_candidate.sum().astype(int).to_dict()}
    report["primary_priority_counts"]={"/".join(k):v for k,v in report["primary_priority_counts"].items()}
    (tables/"motif_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    return results,match_report,report
