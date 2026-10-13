"""nb10: review saved predictions and group weights; no model fitting."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve)

LINES=("MCF7","T47D")
MODES=("RNA_ATAC","RNA")

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def json_save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")

def read_file(path,inputs,sep=","):
    path=Path(path)
    inputs.append({"path":str(path),"sha256":sha256(path),"bytes":path.stat().st_size})
    if path.suffix==".json":
        return json.loads(path.read_text())
    return pd.read_csv(path,sep=sep)

def as_bool(series):
    mapping={"true":True,"false":False,"1":True,"0":False}
    text=series.astype(str).str.lower()
    if not text.isin(mapping).all():
        raise ValueError(f"Invalid boolean values in {series.name}.")
    return text.map(mapping)

def choose_run(base,run_id):
    root=Path(base)/"results"/"tables"/"nb09"
    if run_id:
        chosen=root/run_id
    else:
        candidates=sorted(p for p in root.glob("*") if (p/"metrics_comparison.csv").is_file())
        if len(candidates)!=1:
            names=[p.name for p in candidates]
            raise ValueError(f"Set NB09_RUN_ID to one completed run. Found: {names}. "
                             "Wait for nb09 to finish or extract its results ZIP into its run folder.")
        chosen=candidates[0]
    required=["config.json","nb08_config.json","nb08_handoff.json","metrics_comparison.csv"]
    for line in LINES:
        required.append(f"{line}_splits.csv")
        for mode in MODES:
            required.extend(f"{line}_{mode}_{suffix}" for suffix in
                ("metrics.json","validation_search.csv","validation_predictions.csv",
                 "test_predictions.csv","group_weights.csv"))
        for mode in ("RNA","ATAC"):
            required.extend([f"nb08_{line}_{mode}_predictions.csv",
                             f"{line}_{mode}_group_coverage.csv"])
    missing=[name for name in required if not (chosen/name).is_file()]
    if missing:
        raise FileNotFoundError("nb09 results incomplete:\n"+"\n".join(missing))
    return chosen

def prepare_review(base,run_id,nb09_run_id):
    source=choose_run(base,nb09_run_id)
    out=Path(base)/"results"/"tables"/"nb10"/run_id
    if out.exists():
        raise FileExistsError("Choose a new RUN_ID; existing review will not be overwritten.")
    out.mkdir(parents=True)
    inputs=[]
    handoff=read_file(source/"nb08_handoff.json",inputs)
    if handoff["positive_class"]!="E2":
        raise ValueError("Unexpected positive class.")
    config=read_file(source/"config.json",inputs)
    baseline=read_file(source/"nb08_config.json",inputs)
    json_save(out/"review_config.json",{
        "nb09_run_id":source.name,"nb08_run_id":handoff["run_id"],
        "nb08_learning_rate":baseline["learning_rate"],"nb09_config":config,
        "positive_class":"E2","confidence_intervals":"not estimated",
        "scope":"Saved-output kernel-model review plus separately configured frozen cross-line inference. No fitting or tuning."})
    json_save(out/"environment.json",{"python":sys.version,
        "packages":{p:importlib.metadata.version(p) for p in
                    ("numpy","pandas","matplotlib","scikit-learn")}})
    print("Reviewing nb09:",source.name)
    print("nb08 run:",handoff["run_id"],"; learning rate:",baseline["learning_rate"])
    return source,out,inputs,handoff

def prediction_frame(path,expected,inputs,score_column,threshold):
    p=read_file(path,inputs)
    required={"cell_id","label_E2",score_column,"predicted_label_E2"}
    if not required.issubset(p.columns) or not p.cell_id.is_unique:
        raise ValueError(f"Invalid prediction table: {path}")
    p["cell_id"]=p.cell_id.astype(str)
    if not set(p.cell_id)==set(expected.cell_id):
        raise ValueError(f"Missing/extra test or validation cells: {path}")
    p=p.set_index("cell_id").loc[expected.cell_id].reset_index()
    if not np.array_equal(p.label_E2.to_numpy(),expected.label_E2.to_numpy()):
        raise ValueError(f"Label mismatch: {path}")
    score=p[score_column].to_numpy(dtype=float)
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite predictions.")
    if score_column=="probability_E2" and ((score<0)|(score>1)).any():
        raise ValueError("MLP probabilities outside [0,1].")
    predicted=(score>=threshold).astype(int)
    if not np.array_equal(predicted,p.predicted_label_E2):
        raise ValueError("Saved classification differs from its stated threshold.")
    p["score"]=score
    p["predicted"]=predicted
    return p

def calculate_metrics(p):
    y=p.label_E2.to_numpy()
    if set(y)!={0,1}:
        raise ValueError("Both classes required.")
    tn,fp,fn,tp=confusion_matrix(y,p.predicted,labels=[0,1]).ravel()
    return {"n_cells":len(p),"E2_prevalence":float(y.mean()),
        "AUROC":float(roc_auc_score(y,p.score)),
        "average_precision":float(average_precision_score(y,p.score)),
        "accuracy":float(accuracy_score(y,p.predicted)),
        "balanced_accuracy":float(balanced_accuracy_score(y,p.predicted)),
        "F1_E2":float(f1_score(y,p.predicted,zero_division=0)),
        "TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"errors":int(fp+fn)}

def review_predictions(source,out,inputs,handoff):
    saved=read_file(source/"metrics_comparison.csv",inputs)
    predictions,gaps,solver_rows,split_counts={},[],[],[]
    metric_rows=[]
    for line in LINES:
        path=source/f"{line}_splits.csv"
        if sha256(path)!=handoff["manifests"][line]["sha256"]:
            raise ValueError("Split hash differs from nb08 handoff.")
        m=read_file(path,inputs)
        m["cell_id"]=m.cell_id.astype(str)
        if not m.cell_id.is_unique or set(m["split"])!={"train","validation","test"}:
            raise ValueError("Invalid split manifest.")
        if set(m.cell_line)!={line} or not np.array_equal(m.label_E2,(m.treatment=="E2").astype(int)):
            raise ValueError("Invalid split labels.")
        expected=m.loc[m["split"]=="test",["cell_id","label_E2"]].reset_index(drop=True)
        val=m.loc[m["split"]=="validation",["cell_id","label_E2"]].reset_index(drop=True)
        split_counts.append(m.groupby(["cell_line","split","treatment"]).size().rename("cells").reset_index())
        specifications=[("MLP_nb08",mode,f"nb08_{line}_{mode}_predictions.csv","probability_E2",.5)
                        for mode in ("RNA","ATAC")]
        specifications += [("scMKL_style",mode,f"{line}_{mode}_test_predictions.csv","decision_score_E2",0)
                           for mode in MODES]
        for family,mode,filename,column,threshold in specifications:
            p=prediction_frame(source/filename,expected,inputs,column,threshold)
            met=calculate_metrics(p)
            previous=saved[(saved.cell_line==line)&(saved.model==family)&(saved.modality==mode)]
            if len(previous)!=1:
                raise ValueError("Missing/duplicate model in nb09 summary.")
            for key in ("AUROC","average_precision","accuracy","balanced_accuracy","F1_E2"):
                if not np.isclose(met[key],float(previous.iloc[0][key]),atol=1e-7,rtol=0):
                    raise ValueError(f"Saved metric mismatch: {line} {family} {mode} {key}")
            predictions[(line,family,mode)]=p
            metric_rows.append({"cell_line":line,"model":family,"modality":mode,**met})
            if family=="scMKL_style":
                meta=read_file(source/f"{line}_{mode}_metrics.json",inputs)
                vp=prediction_frame(source/f"{line}_{mode}_validation_predictions.csv",val,inputs,column,0)
                vm=calculate_metrics(vp)
                if not np.isclose(vm["AUROC"],meta["validation_AUROC"],atol=1e-7,rtol=0):
                    raise ValueError("Validation AUROC mismatch.")
                search=read_file(source/f"{line}_{mode}_validation_search.csv",inputs)
                search["converged"]=as_bool(search.converged)
                selected=search[np.isclose(search.alpha_ratio,meta["selected_alpha_ratio"])]
                if len(selected)!=1 or not selected.iloc[0]["converged"]:
                    raise ValueError("Selected model was not a unique converged candidate.")
                if not np.isclose(selected.iloc[0]["validation_AUROC"],vm["AUROC"],atol=1e-7,rtol=0):
                    raise ValueError("Selected candidate/validation predictions disagree.")
                winner=search[search.converged].sort_values(
                    ["validation_AUROC","validation_average_precision","selected_groups","alpha_ratio"],
                    ascending=[False,False,True,False]).iloc[0]
                if not np.isclose(winner.alpha_ratio,meta["selected_alpha_ratio"]):
                    raise ValueError("Selected model does not match the recorded validation selection rule.")
                for _,candidate in search.iterrows():
                    solver_rows.append({"cell_line":line,"modality":mode,**candidate.to_dict(),
                        "chosen":bool(np.isclose(candidate.alpha_ratio,meta["selected_alpha_ratio"]))})
                gaps.append({"cell_line":line,"modality":mode,
                    "training_AUROC":meta["training_AUROC"],"validation_AUROC":vm["AUROC"],
                    "test_AUROC":met["AUROC"],
                    "validation_minus_test_AUROC":vm["AUROC"]-met["AUROC"],
                    "training_minus_test_AUROC":meta["training_AUROC"]-met["AUROC"]})
    metrics=pd.DataFrame(metric_rows)
    metrics.to_csv(out/"verified_metrics.csv",index=False)
    pd.concat(split_counts).to_csv(out/"split_counts.csv",index=False)
    gaps=pd.DataFrame(gaps)
    gaps.to_csv(out/"validation_test_gaps.csv",index=False)
    pd.DataFrame(solver_rows).to_csv(out/"solver_review.csv",index=False)
    comparisons=[]
    for line in LINES:
        joint=predictions[(line,"scMKL_style","RNA_ATAC")]
        joint_met=calculate_metrics(joint)
        for family,mode in (("MLP_nb08","RNA"),("MLP_nb08","ATAC"),("scMKL_style","RNA")):
            other=predictions[(line,family,mode)]
            other_met=calculate_metrics(other)
            jc=joint.predicted.to_numpy()==joint.label_E2.to_numpy()
            oc=other.predicted.to_numpy()==other.label_E2.to_numpy()
            comparisons.append({"cell_line":line,"reference":family+"_"+mode,
                "AUROC_joint_minus_reference":joint_met["AUROC"]-other_met["AUROC"],
                "AP_joint_minus_reference":joint_met["average_precision"]-other_met["average_precision"],
                "errors_reference_minus_joint":other_met["errors"]-joint_met["errors"],
                "corrected_by_joint":int((jc&~oc).sum()),"new_errors_from_joint":int((~jc&oc).sum()),
                "both_correct":int((jc&oc).sum()),"both_wrong":int((~jc&~oc).sum())})
        errors=pd.DataFrame({"cell_id":joint.cell_id,"label_E2":joint.label_E2})
        for (cl,family,mode),p in predictions.items():
            if cl==line:
                errors[family+"_"+mode+"_correct"]=p.predicted.to_numpy()==p.label_E2.to_numpy()
        errors.to_csv(out/f"{line}_paired_error_audit.csv",index=False)
    differences=pd.DataFrame(comparisons)
    differences.to_csv(out/"integration_comparison.csv",index=False)
    return metrics,gaps,differences,predictions

def performance_plots(metrics,gaps,predictions,out):
    for line in LINES:
        fig,axes=plt.subplots(1,3,figsize=(17,4))
        for (cl,family,mode),p in predictions.items():
            if cl!=line:continue
            label=family+" "+mode
            fpr,tpr,_=roc_curve(p.label_E2,p.score)
            precision,recall,_=precision_recall_curve(p.label_E2,p.score)
            axes[0].plot(fpr,tpr,label=label)
            axes[1].plot(recall,precision,label=label)
        axes[0].plot([0,1],[0,1],"--",color="grey")
        axes[0].set(xlabel="False positive rate",ylabel="True positive rate",title=line+" test ROC")
        axes[1].axhline(float(metrics.loc[metrics.cell_line==line,"E2_prevalence"].iloc[0]),
                       linestyle="--",color="grey")
        axes[1].set(xlabel="Recall",ylabel="Precision",title="Test precision–recall")
        axes[0].legend(fontsize=7)
        g=gaps[gaps.cell_line==line].set_index("modality")
        g[["training_AUROC","validation_AUROC","test_AUROC"]].plot.bar(ax=axes[2])
        axes[2].set(ylim=(0,1.02),ylabel="AUROC",title="Kernel-model generalization gaps")
        axes[2].tick_params(axis="x",rotation=30)
        fig.tight_layout()
        fig.savefig(out/f"{line}_performance.png",dpi=160)
        plt.show()
        plt.close(fig)

def review_groups(source,out,inputs):
    frames,alias_frames=[],[]
    for line in LINES:
        for mode in MODES:
            w=read_file(source/f"{line}_{mode}_group_weights.csv",inputs)
            if not w.group.is_unique:
                raise ValueError("Duplicate group weights.")
            norms=w.coefficient_group_norm.to_numpy(dtype=float)
            if not np.isfinite(norms).all() or (norms<0).any():
                raise ValueError("Invalid group norms.")
            w["selected"]=as_bool(w.selected)
            if not np.array_equal(w.selected,norms>1e-8):
                raise ValueError("Inconsistent group selection flags.")
            w["cell_line"]=line
            w["input_mode"]=mode
            frames.append(w)
            if mode=="RNA_ATAC":
                joint=w.copy()
        coverage=pd.concat([read_file(source/f"{line}_{mode}_group_coverage.csv",inputs)
                            for mode in ("RNA","ATAC")],ignore_index=True)
        coverage["cell_line"]=line
        coverage.to_csv(out/f"{line}_group_coverage.csv",index=False)
        available=coverage[coverage.status.isin(["retained","identical_feature_membership"])].copy()
        aliases=available.merge(joint[["group","coefficient_group_norm","selected"]].rename(
            columns={"group":"representative"}),on="representative",how="left",validate="many_to_one")
        if aliases.coefficient_group_norm.isna().any():
            raise ValueError("Coverage alias points to a missing joint-model group.")
        alias_frames.append(aliases)
        # Display distinct kernel groups, not duplicate motif aliases.
        for modality in ("RNA","ATAC"):
            top=joint[(joint.modality==modality)&joint.selected].nlargest(15,"coefficient_group_norm")
            fig,ax=plt.subplots(figsize=(10,6))
            if len(top):
                top=top.sort_values("coefficient_group_norm")
                labels=top.group.str.replace("HALLMARK_","",regex=False)
                ax.barh(labels,top.coefficient_group_norm)
            else:
                ax.text(.5,.5,"No selected groups",ha="center",va="center",transform=ax.transAxes)
            ax.set(title=f"{line}: joint-model {modality} groups",xlabel="Coefficient block L2 norm")
            fig.tight_layout()
            fig.savefig(out/f"{line}_{modality}_selected_groups.png",dpi=150)
            plt.show()
            plt.close(fig)
    weights=pd.concat(frames,ignore_index=True)
    weights.to_csv(out/"all_model_group_weights.csv",index=False)
    aliases=pd.concat(alias_frames,ignore_index=True)
    aliases.to_csv(out/"joint_group_aliases.csv",index=False)
    weights.pivot(index=["cell_line","group","modality"],columns="input_mode",
                  values="coefficient_group_norm").to_csv(out/"unimodal_joint_group_comparison.csv")
    a,b=[aliases[aliases.cell_line==line][["group","coefficient_group_norm","selected"]].rename(
        columns={"coefficient_group_norm":f"norm_{line}","selected":f"selected_{line}"}) for line in LINES]
    cross=a.merge(b,on="group",how="outer",validate="one_to_one")
    for line in LINES:
        cross[f"represented_{line}"]=cross[f"norm_{line}"].notna()
    cross.to_csv(out/"cross_line_group_selection.csv",index=False)
    return weights,aliases,cross

def biological_context(base,out,inputs,aliases):
    """Attach earlier full-cohort findings descriptively; never use them to select models."""
    statuses=[]
    def optional(path):
        if not path.is_file():
            statuses.append({"file":str(path),"status":"missing_optional_not_interpreted"})
            return None
        result=read_file(path,inputs,sep="\t")
        statuses.append({"file":str(path),"status":"loaded_descriptive_context"})
        return result
    for line in LINES:
        rna=aliases[(aliases.cell_line==line)&(aliases.modality=="RNA")].copy()
        rna["pathway"]=rna.group.str.replace("^RNA::","",regex=True)
        gsea=optional(Path(base)/"results"/"tables"/"nb04"/f"{line}_hallmark_gsea.tsv")
        if gsea is not None:
            gsea=gsea.rename(columns={"Term":"pathway"})
            gsea=gsea.rename(columns={c:"gsea_"+c for c in gsea.columns if c!="pathway"})
            rna=rna.merge(gsea,on="pathway",how="left",validate="many_to_one",indicator="gsea_match")
        rna.to_csv(out/f"{line}_RNA_groups_GSEA_context.csv",index=False)
        pathway=optional(Path(base)/"results"/"tables"/"nb06"/f"{line}_pathway_summary.tsv")
        if pathway is not None:
            # Preserve every scenario and both all-member/leading-edge summaries.
            joined=rna.merge(pathway,on=["cell_line","pathway"],how="left",
                             suffixes=("","_nb06"),validate="one_to_many")
            joined.to_csv(out/f"{line}_RNA_groups_ATAC_pathway_context.csv",index=False)
    motif=optional(Path(base)/"results"/"tables"/"nb07"/"motif_enrichment_all.tsv")
    atac=aliases[aliases.modality=="ATAC"].copy()
    atac["motif_id"]=atac.group.str.split("::",n=1).str[-1].str.split("__",n=1).str[0]
    if motif is not None:
        for column in ("matching_acceptable","priority_candidate"):
            motif[column]=as_bool(motif[column])
        # Keep direction, coordinate hypothesis, scenario and matching QC; no best-q cherry-picking.
        atac=atac.merge(motif,on=["cell_line","motif_id"],how="left",
                        validate="one_to_many",indicator="enrichment_match")
    atac.to_csv(out/"ATAC_groups_motif_enrichment_context.csv",index=False)
    status=pd.DataFrame(statuses)
    status.to_csv(out/"biological_context_availability.csv",index=False)
    return status

def export_review(base,source,out,inputs,metrics,gaps,differences,weights,status):
    provenance=pd.DataFrame(inputs).drop_duplicates("path")
    provenance.to_csv(out/"input_checksums.csv",index=False)
    lines=[
        "# nb10 model comparison and interpretation",
        "",
        f"Reviewed nb09 run: {source.name}",
        "",
        "This report uses saved predictions and coefficients only. No models were fitted or tuned.",
        "",
        "## Test performance",
        "",
        "| Cell line | Model | Inputs | AUROC | Average precision | Balanced accuracy | Errors |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _,row in metrics.iterrows():
        lines.append(f"| {row.cell_line} | {row.model} | {row.modality} | {row.AUROC:.4f} | "
                     f"{row.average_precision:.4f} | {row.balanced_accuracy:.4f} | {int(row.errors)} |")
    lines+=["","## Descriptive integration differences",""]
    for _,row in differences.iterrows():
        lines.append(f"- {row.cell_line}, joint versus {row.reference}: "
                     f"AUROC difference {row.AUROC_joint_minus_reference:+.5f}; "
                     f"AP difference {row.AP_joint_minus_reference:+.5f}; "
                     f"{int(row.corrected_by_joint)} errors corrected and "
                     f"{int(row.new_errors_from_joint)} new errors.")
    lines+=["","These are single-split point differences, not significance tests or estimates of "
            "biological-replicate uncertainty.","","## Selected groups",""]
    for line in LINES:
        for modality in ("RNA","ATAC"):
            selected=weights[(weights.cell_line==line)&(weights.input_mode=="RNA_ATAC")&
                             (weights.modality==modality)&weights.selected]
            names=selected.nlargest(5,"coefficient_group_norm").group.tolist()
            lines.append(f"- {line} {modality}: {len(selected)} selected representative groups. "
                         f"Highest coefficient norms: {', '.join(names) if names else 'none'}.")
    lines+=["","## Interpretation limits","",
        "- Group norms are unsigned predictive weights, not activation, repression or causal importance.",
        "- Aliases sharing a kernel are not independent selected TFs/pathways.",
        "- Earlier GSEA/ATAC/motif outputs use this same cohort and provide descriptive context, not independent validation.",
        "- Missing groups reflect coverage or filtering; missing context files are not negative biological evidence.",
        "- All nb07 matching-QC flags, directions and coordinate scenarios are retained.",
        "- Cross-line overlap is descriptive and does not measure repeat-split stability or transfer performance.",
        "- Test scores have already been inspected; any test-guided tuning requires independent confirmation.",
        "- Biological replicate IDs are unavailable, and treatment/library confounding remains possible.",
        "- No confidence intervals, significance tests or retraining were performed. Cross-line transfer, when enabled, is reported separately below.",
        "",
        "## Next decisions",
        "",
        "Review whether joint kernels improve over the RNA and ATAC MLP baselines; inspect solver "
        "status and validation/test gaps before interpreting small differences. Compare selected groups "
        "with all available biological context, preserving negative NES and failed matching flags. "
        "Use predefined additional splits or independent data for robustness; do not tune on this report.",
        "",
        f"Optional biological-context files available: {int((status.status=='loaded_descriptive_context').sum())}"
        f"/{len(status)}.",
    ]
    transfer_report=out/"cross_cell_line"/"transfer_report.md"
    if transfer_report.exists():
        lines+=["","---","",transfer_report.read_text()]
    else:
        lines+=["","Cross-cell-line transfer was not run in this review."]
    (out/"comparison_report.md").write_text("\n".join(lines)+"\n")
    exports=Path(base)/"results"/"exports"
    exports.mkdir(parents=True,exist_ok=True)
    archive=shutil.make_archive(str(exports/f"nb10_{out.name}"),"zip",root_dir=out)
    print("Report:",out/"comparison_report.md")
    print("Results archive:",archive)
    return Path(archive)
