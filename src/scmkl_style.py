"""nb09: paper-inspired grouped Gaussian kernels and group-lasso integration.
No training or downloads occur at import. This is not the official scMKL package.
"""
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import re
import shutil
import sys
import time
import urllib.request
import warnings
import threading
from contextlib import contextmanager, redirect_stdout
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import py2bit
import MOODS.scan
import MOODS.tools
from celer import GroupLasso
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve)

def save_json(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+".partial")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def input_path(base, line, modality):
    suffix = "RNA_paper_qc" if modality == "RNA" else "ATAC_paired_paper_qc"
    return Path(base)/"data"/"processed"/f"{line}_{suffix}.h5ad"

def annotations(path):
    with h5py.File(path, "r") as f:
        obs, var = ad.io.read_elem(f["obs"]), ad.io.read_elem(f["var"])
    if not obs.index.is_unique or not var.index.is_unique:
        raise ValueError("Duplicate cell or feature IDs.")
    return obs, var

def load_features(base,line,modality,manifest,config):
    path = input_path(base,line,modality)
    obs,var = annotations(path)
    if list(obs.index.astype(str)) != manifest["cell_id"].tolist():
        raise ValueError("Input cell order changed.")
    with h5py.File(path,"r") as f:
        x = ad.io.read_elem(f["layers/counts" if modality == "RNA" else "X"])
    if not sp.issparse(x):
        raise TypeError("Expected sparse raw counts.")
    x = x.tocsr().astype(np.float32)
    x.sum_duplicates()
    x.eliminate_zeros()
    if x.shape != (len(obs),len(var)):
        raise ValueError("Matrix/metadata dimensions differ.")
    if not np.isfinite(x.data).all() or np.any(x.data<0):
        raise ValueError("Invalid counts.")
    if not np.allclose(x.data,np.rint(x.data),atol=1e-5,rtol=0):
        raise ValueError("Expected raw integer counts.")
    if modality == "RNA":
        x.data = np.log1p(x.data)
    else:
        x.data.fill(1)
    train = x[np.flatnonzero(manifest["split"].to_numpy()=="train")].astype(np.float64)
    mean = np.asarray(train.mean(axis=0)).ravel()
    variance = np.maximum(np.asarray(train.power(2).mean(axis=0)).ravel()-mean**2,0)
    detected = np.asarray(train.getnnz(axis=0)).ravel()
    eligible = np.flatnonzero((detected>=config["min_train_detected"]) & (variance>1e-8))
    if not len(eligible):
        raise ValueError("No eligible training features.")
    selected = eligible[np.argsort(-variance[eligible],kind="stable")[:config["max_features"]]]
    center = mean[selected].astype(np.float32) if modality=="RNA" else np.zeros(len(selected),np.float32)
    scale = np.sqrt(variance[selected]).astype(np.float32) if modality=="RNA" else np.ones(len(selected),np.float32)
    table = pd.DataFrame({"input_column":selected,"feature_id":var.index.astype(str).to_numpy()[selected],
        "training_detected_cells":detected[selected],"training_mean":mean[selected],
        "training_variance":variance[selected],"model_center":center,"model_scale":scale})
    return x[:,selected].tocsr(),center,scale,table

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


def prepare_nb09(base, nb08_run_id, run_id, config, hallmark, refs, scan_config, resume=False):
    baseline=Path(base)/"results"/"tables"/"nb08"/nb08_run_id
    required=["config.json","input_provenance.csv","integration_handoff.json","metrics_summary.csv"]
    required += [f"{line}_{suffix}" for line in config["cell_lines"]
                 for suffix in ("splits.csv","RNA_features.csv","ATAC_features.csv",
                                "RNA_test_predictions.csv","ATAC_test_predictions.csv")]
    missing=[str(baseline/name) for name in required if not (baseline/name).is_file()]
    if missing:
        raise FileNotFoundError("Restore the complete nb08 results folder before proceeding:\n"+"\n".join(missing))
    baseline_config=json.loads((baseline/"config.json").read_text())
    handoff=json.loads((baseline/"integration_handoff.json").read_text())
    if handoff["positive_class"]!="E2" or handoff["run_id"]!=nb08_run_id:
        raise ValueError("Incorrect nb08 handoff or positive class.")
    manifests={}
    provenance=pd.read_csv(baseline/"input_provenance.csv")
    for line in config["cell_lines"]:
        path=baseline/handoff["manifests"][line]["file"]
        if sha256(path)!=handoff["manifests"][line]["sha256"]:
            raise ValueError("nb08 split checksum mismatch.")
        m=pd.read_csv(path,dtype={"cell_id":str})
        if not m["cell_id"].is_unique or set(m["split"])!={"train","validation","test"}:
            raise ValueError("Invalid nb08 split manifest.")
        if set(m["cell_line"])!={line} or set(m["treatment"])!={"E2","Vehicle"}:
            raise ValueError("Unexpected labels.")
        if not np.array_equal(m["label_E2"],(m["treatment"]=="E2").astype(int)):
            raise ValueError("Positive-class mapping mismatch.")
        for part in ("train","validation","test"):
            if set(m.loc[m["split"]==part,"label_E2"])!={0,1}:
                raise ValueError("Each split must contain both classes.")
        for modality in ("RNA","ATAC"):
            path=input_path(base,line,modality)
            expected=provenance[(provenance.cell_line==line)&(provenance.modality==modality)]
            if len(expected)!=1 or sha256(path)!=expected.iloc[0]["sha256"]:
                raise ValueError(f"{line} {modality}: input differs from nb08.")
            obs,_=annotations(path)
            if list(obs.index.astype(str))!=m["cell_id"].tolist():
                raise ValueError("Cell order differs from nb08.")
            if not np.array_equal(obs["treatment"].astype(str),m["treatment"]):
                raise ValueError("Treatment differs from nb08.")
        manifests[line]=m
    out=Path(base)/"results"/"tables"/"nb09"/run_id
    artifacts=Path(base)/"data"/"processed"/"nb09"/run_id
    if out.exists() or artifacts.exists():
        if not resume or not out.is_dir() or not artifacts.is_dir():
            raise FileExistsError("Choose a new RUN_ID, or explicitly set RESUME_RUN_ID.")
        for filename,value in (("config.json",config),("hallmark_reference.json",hallmark),
                               ("motif_references.json",refs),("motif_scan_config.json",scan_config),
                               ("nb08_config.json",baseline_config),("nb08_handoff.json",handoff)):
            if json.loads((out/filename).read_text()) != value:
                raise ValueError(f"Cannot resume: {filename} changed.")
        for line in config["cell_lines"]:
            if sha256(out/f"{line}_splits.csv") != handoff["manifests"][line]["sha256"]:
                raise ValueError("Resume split mismatch.")
        print("Resuming existing integrated-model run:",run_id)
        return baseline,out,artifacts,manifests,baseline_config
    out.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    save_json(out/"config.json",config)
    save_json(out/"hallmark_reference.json",hallmark)
    save_json(out/"motif_references.json",refs)
    save_json(out/"motif_scan_config.json",scan_config)
    shutil.copy2(baseline/"config.json",out/"nb08_config.json")
    shutil.copy2(baseline/"input_provenance.csv",out/"input_provenance.csv")
    shutil.copy2(baseline/"integration_handoff.json",out/"nb08_handoff.json")
    for line,m in manifests.items():
        m.to_csv(out/f"{line}_splits.csv",index=False)
    packages=["anndata","numpy","pandas","scipy","scikit-learn","celer","h5py","MOODS-python","py2bit"]
    save_json(out/"environment.json",{"python":sys.version,
        "packages":{name:importlib.metadata.version(name) for name in packages}})
    print("nb08 learning rate recorded:",baseline_config["learning_rate"])
    print("nb09 fits RNA-only and RNA+ATAC per line at fixed alpha ratio 0.1.")
    return baseline,out,artifacts,manifests,baseline_config

def ensure_references(base,refs,out):
    records=[]
    for name in ("hg38.2bit","JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
                 "hg38-blacklist.v2.bed.gz"):
        spec=refs["resources"][name]
        path=Path(base)/spec["path"]
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_name(path.name+".download")
            print(f"Downloading {name} ({spec['size']/1e6:.1f} MB)...",flush=True)
            urllib.request.urlretrieve(spec["url"],temp)
            if sha256(temp)!=spec["sha256"]:
                raise ValueError(f"Reference checksum mismatch: {temp}")
            temp.replace(path)
        if sha256(path)!=spec["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {path}")
        records.append({"reference":name,"sha256":spec["sha256"],"path":spec["path"]})
    pd.DataFrame(records).to_csv(out/"reference_checksums.csv",index=False)

def define_groups(base,line,modality,features,hallmark,refs,scan_config,out,config):
    names=features["feature_id"].astype(str).tolist()
    if modality=="RNA":
        lookup={name:i for i,name in enumerate(names)}
        groups={name:np.array(sorted({lookup[g] for g in genes if g in lookup}),dtype=int)
                for name,genes in hallmark["gene_sets"].items()}
    else:
        # Fresh sequence-only annotation of TRAIN-selected peaks. No DA/GSEA-selected universe.
        qc,sequences=sequence_qc(Path(base),{"selected":pd.DataFrame({"peak":names})},refs,scan_config)
        qc.to_csv(out/f"{line}_ATAC_sequence_qc.csv",index=False)
        motifs=parse_jaspar(Path(base)/refs["resources"][
            "JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt"]["path"])
        scanner,_=build_scanner(motifs,scan_config)
        presence=np.zeros((len(names),len(motifs)),dtype=bool)
        boundary=[]
        scan_started=scan_update=time.monotonic()
        for i,peak in enumerate(names):
            if peak in sequences:
                common,extended=scan_union_sequence(scanner,sequences[peak],len(motifs))
                presence[i]=common & extended
                boundary.append({"peak":peak,"stable_motif_count":int(presence[i].sum()),
                                 "boundary_sensitive_motif_count":int((extended & ~common).sum())})
            if time.monotonic()-scan_update>=30 or i+1==len(names):
                counted_progress(out,f"{line}_motif_scan",scan_started,i+1,len(names))
                scan_update=time.monotonic()
        pd.DataFrame(boundary).to_csv(out/f"{line}_motif_coordinate_audit.csv",index=False)
        groups={m["motif_id"]+"__"+m["motif_name"]:np.flatnonzero(presence[:,j])
                for j,m in enumerate(motifs)}
    kept,records,seen={},[],{}
    for name,indices in groups.items():
        key=tuple(indices.tolist())
        status="retained"
        if len(indices)<config["min_group_features"]:
            status="too_few_training_selected_features"
        elif key in seen:
            status="identical_feature_membership"
        else:
            seen[key]=name
            kept[modality+"::"+name]=indices
        records.append({"group":modality+"::"+name,"modality":modality,"feature_count":len(indices),
                        "status":status,"representative":modality+"::"+seen[key] if key in seen else ""})
    pd.DataFrame(records).to_csv(out/f"{line}_{modality}_group_coverage.csv",index=False)
    if not kept:
        raise ValueError(f"No usable {modality} groups.")
    save_json(out/f"{line}_{modality}_group_membership.json",
              {name:[names[i] for i in ix] for name,ix in kept.items()})
    return kept

def rff_design(x,center,scale,groups,manifest,line,modality,config,artifacts,out):
    train=np.flatnonzero(manifest["split"].to_numpy()=="train")
    d=config["rff_frequencies"]
    estimate=x.shape[0]*len(groups)*2*d*4/1024**3
    if estimate>config["max_design_gib"]:
        raise MemoryError(f"Design needs {estimate:.2f} GiB; lower rff_frequencies in a NEW run.")
    # Disk-backed maps permit sequential modality construction.
    path=artifacts/f"{line}_{modality}_rff.npy"
    z=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(x.shape[0],len(groups)*2*d))
    directory=artifacts/f"{line}_{modality}_maps"
    directory.mkdir(exist_ok=True)
    report=[]
    kernel_started=last_update=time.monotonic()
    print(f"{line} {modality}: constructing {len(groups)} kernels.",flush=True)
    for j,(name,ix) in enumerate(groups.items()):
        seed=int.from_bytes(hashlib.sha256(f"{config['seed']}|{line}|{name}".encode()).digest()[:4],"little")
        rng=np.random.default_rng(seed)
        sample=rng.choice(train,min(config["bandwidth_sample_cells"],len(train)),replace=False)
        sampled=(x[sample][:,ix].toarray()-center[ix])/scale[ix]
        distances=pdist(sampled.astype(np.float64),"euclidean")
        positive=distances[distances>0]
        if not len(positive):
            raise ValueError(f"No positive training distances for {name}")
        sigma=max(float(np.median(positive)),1e-6)
        # Standard Gaussian RBF: k=exp(-||a-b||^2/(2*sigma^2)).
        w=rng.normal(0,1/sigma,size=(len(ix),d)).astype(np.float32)
        block=np.empty((x.shape[0],2*d),dtype=np.float32)
        for start in range(0,x.shape[0],512):
            stop=min(start+512,x.shape[0])
            dense=(x[start:stop,ix].toarray()-center[ix])/scale[ix]
            angle=dense@w
            block[start:stop,:d]=np.cos(angle)/np.sqrt(d)
            block[start:stop,d:]=np.sin(angle)/np.sqrt(d)
        mu=block[train].mean(axis=0,dtype=np.float64).astype(np.float32)
        norm=float(np.sqrt(np.mean(np.sum((block[train]-mu)**2,axis=1),dtype=np.float64)))
        if norm<1e-8:
            raise ValueError(f"Degenerate training kernel: {name}")
        z[:,j*2*d:(j+1)*2*d]=(block-mu)/norm
        np.savez_compressed(directory/f"group_{j:04d}.npz",feature_columns=ix,
            input_center=center[ix],input_scale=scale[ix],frequencies=w,rff_center=mu,
            rff_scale=np.array(norm),sigma=np.array(sigma))
        report.append({"group":name,"modality":modality,"feature_count":len(ix),
                       "sigma":sigma,"rff_scale":norm,"map_file":f"group_{j:04d}.npz"})
        if time.monotonic()-last_update>=30 or j+1==len(groups):
            counted_progress(out,f"{line}_{modality}_kernels",kernel_started,j+1,len(groups))
            last_update=time.monotonic()
    z.flush()
    table=pd.DataFrame(report)
    table.to_csv(out/f"{line}_{modality}_kernels.csv",index=False)
    del z
    return path,table

def construct_line(base,baseline,out,artifacts,manifest,line,baseline_config,config,hallmark,refs,scan_config,
                   cached_run_id=""):
    if config != json.loads((out/"config.json").read_text()):
        raise ValueError("Configuration changed after setup. Start a new run.")
    if baseline_config != json.loads((baseline/"config.json").read_text()):
        raise ValueError("nb08 configuration changed after setup.")
    paths,tables={},{}
    for modality in ("RNA","ATAC"):
        reused=reuse_kernel(base,out,artifacts,manifest,line,modality,config,cached_run_id)
        if reused is not None:
            paths[modality],tables[modality]=reused
            continue
        x,center,scale,features=load_features(base,line,modality,manifest,baseline_config)
        expected=pd.read_csv(baseline/f"{line}_{modality}_features.csv",dtype={"feature_id":str})
        if features["feature_id"].tolist()!=expected["feature_id"].tolist():
            raise ValueError("Feature selection differs from nb08. Check versions/config/data.")
        features.to_csv(out/f"{line}_{modality}_features.csv",index=False)
        groups=define_groups(base,line,modality,features,hallmark,refs,scan_config,out,config)
        paths[modality],tables[modality]=rff_design(x,center,scale,groups,manifest,line,modality,
                                                  config,artifacts,out)
        del x
        gc.collect()
    return paths,tables

def design_rows(paths,modalities,indices):
    arrays=[np.load(paths[m],mmap_mode="r") for m in modalities]
    result=np.empty((len(indices),sum(a.shape[1] for a in arrays)),dtype=np.float32,order="F")
    offset=0
    for a in arrays:
        result[:,offset:offset+a.shape[1]]=a[indices]
        offset+=a.shape[1]
    return result

def score_rows(paths,modalities,indices,coef,intercept):
    score=np.full(len(indices),intercept,dtype=np.float64)
    offset=0
    for modality in modalities:
        a=np.load(paths[modality],mmap_mode="r")
        w=coef[offset:offset+a.shape[1]]
        for start in range(0,len(indices),256):
            stop=min(start+256,len(indices))
            score[start:stop]+=a[indices[start:stop]]@w
        offset+=a.shape[1]
    assert offset==len(coef)
    return score

def score_metrics(y,score,threshold=0):
    predicted=score>=threshold
    tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel()
    return {"n_cells":len(y),"AUROC":float(roc_auc_score(y,score)),
            "average_precision":float(average_precision_score(y,score)),
            "accuracy":float(accuracy_score(y,predicted)),
            "balanced_accuracy":float(balanced_accuracy_score(y,predicted)),
            "F1_E2":float(f1_score(y,predicted,zero_division=0)),
            "TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)}

def plot_model(y,score,search,weights,path,title):
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    fpr,tpr,_=roc_curve(y,score)
    axes[0].plot(fpr,tpr,label=f"AUROC={roc_auc_score(y,score):.4f}")
    axes[0].plot([0,1],[0,1],"--",color="grey")
    axes[0].set(xlabel="False positive rate",ylabel="True positive rate",title="Test ROC")
    axes[0].legend()
    good=search[search.converged]
    axes[1].semilogx(good["alpha_ratio"],good["validation_AUROC"],"o-")
    axes[1].set(xlabel="Regularization / alpha max",ylabel="Validation AUROC",title="Fixed alpha; validation diagnostic")
    top=weights.sort_values("coefficient_group_norm").tail(12)
    axes[2].barh(top["group"].str.replace("HALLMARK_","",regex=False),top["coefficient_group_norm"])
    axes[2].set(xlabel="Coefficient group norm",title="Predictive groups (not activity)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path,dpi=150,bbox_inches="tight")
    plt.show()
    plt.close(fig)


def duration_text(seconds):
    seconds=max(0,int(seconds))
    hours,remainder=divmod(seconds,3600)
    minutes,seconds=divmod(remainder,60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def counted_progress(out,tag,started,done,total):
    elapsed=time.monotonic()-started
    remaining=elapsed*(total-done)/done if done else None
    message=f"{tag}: {done}/{total} ({100*done/total:.1f}%), elapsed {duration_text(elapsed)}"
    if remaining is not None:
        message+=f", rough remaining {duration_text(remaining)}"
    print(message,flush=True)
    save_json(out/f"{tag}_progress.json",{"stage":tag,"completed":done,"total":total,
        "elapsed_seconds":elapsed,"rough_remaining_seconds":remaining,
        "note":"Rate-based estimate; groups/peaks vary in cost."})

@contextmanager
def solver_progress(out,tag,config,every_seconds=30):
    """Capture native iteration logs; timing is observational and never changes fitting."""
    started=time.monotonic()
    original=sys.stdout
    stop=threading.Event()
    lock=threading.RLock()
    state={"stage":"fitting","outer_iteration_0based":None,"duality_gap":None,
           "inner_epoch_if_reported":None,"outer_iteration_limit":config["solver_max_iter"],
           "inner_epochs_per_subproblem_limit":config["solver_max_epochs"],
           "eta_seconds":None,"attempt":str(time.time_ns())}
    log=(out/f"{tag}_solver_progress.jsonl").open("a",buffering=1)
    def emit(message):
        with lock:
            event={**state,"elapsed_seconds":time.monotonic()-started,
                   "unix_time":time.time(),"message":message}
            original.write(f"[{tag} | {duration_text(event['elapsed_seconds'])}] {message}\n")
            original.flush()
            log.write(json.dumps(event,allow_nan=False)+"\n")
            log.flush()
            save_json(out/f"{tag}_live_status.json",event)
    class SolverOutput:
        def __init__(self):
            self.pending=""
        def write(self,text):
            with lock:
                self.pending+=text
                while "\n" in self.pending:
                    line,self.pending=self.pending.split("\n",1)
                    if not line.strip():continue
                    found=re.search(r"Iter\s+(\d+)",line)
                    if found:state["outer_iteration_0based"]=int(found.group(1))
                    found=re.search(r"(?:Exit )?Epoch\s+(\d+)",line,re.I)
                    if found:state["inner_epoch_if_reported"]=int(found.group(1))
                    found=re.search(r"gap:?\s+([0-9.eE+-]+)",line)
                    if found:
                        value=float(found.group(1))
                        if np.isfinite(value):state["duality_gap"]=value
                    emit(line)
            return len(text)
        def flush(self):
            original.flush()
        def finish(self):
            if self.pending:
                self.write("\n")
    def heartbeat():
        while not stop.wait(every_seconds):
            iteration=state["outer_iteration_0based"]
            label="waiting for first solver update" if iteration is None else f"last reported outer iteration {iteration} (0-based)"
            emit(f"Still fitting; {label}. Remaining time unknown.")
    emit(f"Fit started: max {config['solver_max_iter']} outer iterations; "
         f"up to {config['solver_max_epochs']} inner epochs per subproblem. "
         "These are limits, not a completion percentage. ETA unavailable.")
    worker=threading.Thread(target=heartbeat,daemon=True)
    sink=SolverOutput()
    worker.start()
    try:
        with redirect_stdout(sink):
            yield state
    except BaseException:
        state["stage"]="interrupted_or_failed"
        raise
    else:
        state["stage"]="fit_returned"
    finally:
        stop.set()
        worker.join()
        sink.finish()
        emit("Solver call ended: "+state["stage"])
        log.close()

def fit_group_models(paths,tables,manifest,line,out,artifacts,config):
    if config != json.loads((out/"config.json").read_text()):
        raise ValueError("Configuration changed after setup. Start a new run.")
    if config["alpha_ratios"] != [0.1]:
        raise ValueError("This notebook is configured for one fixed-alpha fit per input mode and cell line.")
    y=manifest["label_E2"].to_numpy(dtype=int)
    train,val,test=[np.flatnonzero(manifest["split"].to_numpy()==part)
                    for part in ("train","validation","test")]
    block=2*config["rff_frequencies"]
    results=[]
    for mode,modalities in (("RNA_ATAC",["RNA","ATAC"]),("RNA",["RNA"])):
        tag=f"{line}_{mode}"
        if (out/f"{tag}_metrics.json").exists():
            print(f"{tag}: completed model already saved; skipping fit.")
            results.append(json.loads((out/f"{tag}_metrics.json").read_text()))
            continue
        stage_started=time.monotonic()
        print(f"{tag}: stage 1/4 — loading saved features for the training cells.",flush=True)
        xtrain=design_rows(paths,modalities,train)
        print(f"{tag}: feature loading took {duration_text(time.monotonic()-stage_started)}.",flush=True)
        target=(2*y[train]-1).astype(np.float32)
        correlation=xtrain.T@(target-target.mean())/len(train)
        alpha_max=float(np.linalg.norm(correlation.reshape(-1,block),axis=1).max())
        if alpha_max<=0 or not np.isfinite(alpha_max):
            raise ValueError("No finite label-associated kernel variation in training.")
        best=None
        records=[]
        started=time.monotonic()
        for ratio in config["alpha_ratios"]:
            checkpoint=artifacts/f"{tag}_fit_checkpoint.npz"
            checkpoint_meta=artifacts/f"{tag}_fit_checkpoint.json"
            fit_seconds=None
            solver_iterations=None
            if checkpoint.exists() and checkpoint_meta.exists():
                saved=json.loads(checkpoint_meta.read_text())
                if (saved["config_sha256"] != sha256(out/"config.json") or
                    saved["split_sha256"] != sha256(out/f"{line}_splits.csv") or
                    not np.isclose(saved["alpha"],alpha_max*ratio)):
                    raise ValueError("Checkpoint configuration/split/alpha mismatch.")
                with np.load(checkpoint,allow_pickle=False) as ck:
                    coefficients=ck["coef"].copy()
                    intercept=float(ck["intercept"])
                converged=bool(saved["converged"])
                warning_text=saved["solver_warnings"]
                fit_seconds=saved.get("fit_seconds")
                solver_iterations=saved.get("solver_outer_iterations")
                print(f"{tag}: reusing saved fit checkpoint; no retraining.")
            else:
                model=GroupLasso(groups=block,alpha=alpha_max*ratio,weights=None,
                    fit_intercept=True,tol=config["solver_tolerance"],
                    max_iter=config["solver_max_iter"],max_epochs=config["solver_max_epochs"],verbose=True)
                print(f"{tag}: stage 2/4 — fitting one grouped-kernel model, alpha ratio {ratio}.",flush=True)
                fit_started=time.monotonic()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with solver_progress(out,tag,config):
                        model.fit(xtrain,target)
                fit_seconds=time.monotonic()-fit_started
                reported=getattr(model,"n_iter_",None)
                solver_iterations=int(np.asarray(reported).ravel()[0]) if reported is not None else None
                print(f"{tag}: fit returned in {duration_text(fit_seconds)}; "
                      f"reported outer iterations={solver_iterations}. Saving checkpoint.",flush=True)
                converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
                coefficients=np.asarray(model.coef_).copy()
                intercept=float(model.intercept_)
                warning_text=" | ".join(str(w.message) for w in caught)
                temp=checkpoint.with_name(checkpoint.stem+".partial.npz")
                np.savez_compressed(temp,coef=coefficients,intercept=np.array(intercept))
                temp.replace(checkpoint)
                save_json(checkpoint_meta,{"converged":converged,"solver_warnings":warning_text,
                    "alpha":alpha_max*ratio,"config_sha256":sha256(out/"config.json"),
                    "split_sha256":sha256(out/f"{line}_splits.csv"),
                    "fit_seconds":fit_seconds,"solver_outer_iterations":solver_iterations})
                del model
            if not np.isfinite(coefficients).all() or not np.isfinite(intercept):
                raise FloatingPointError("Invalid group-lasso coefficients.")
            print(f"{tag}: stage 3/4 — validation and test evaluation (no additional fit).",flush=True)
            pv=score_rows(paths,modalities,val,coefficients,intercept)
            vm=score_metrics(y[val],pv)
            selected=int(np.sum(np.linalg.norm(coefficients.reshape(-1,block),axis=1)>1e-8))
            record={"alpha_ratio":ratio,"alpha":alpha_max*ratio,"alpha_max":alpha_max,
                "validation_AUROC":vm["AUROC"],"validation_average_precision":vm["average_precision"],
                "selected_groups":selected,"converged":converged,
                "solver_warnings":warning_text,"fit_seconds":fit_seconds,
                "solver_outer_iterations":solver_iterations}
            records.append(record)
            pd.DataFrame(records).to_csv(out/f"{tag}_validation_search.csv",index=False)
            print(f"{tag}: alpha ratio={ratio:g}, val AUROC={vm['AUROC']:.5f}, "
                  f"groups={selected}, converged={converged}",flush=True)
            # Tie-break by validation AP, fewer groups, then stronger regularization.
            key=(vm["AUROC"],vm["average_precision"],-selected,ratio)
            if converged and (best is None or key>best["key"]):
                best={"key":key,"coef":coefficients,"intercept":intercept,
                      "record":record,"validation_score":pv}
        if best is None:
            raise RuntimeError("No converged model. Inspect solver_warnings before changing settings.")
        # Test predictions only after all candidates have been selected using validation.
        pt=score_rows(paths,modalities,test,best["coef"],best["intercept"])
        train_score=score_rows(paths,modalities,train,best["coef"],best["intercept"])
        kernel_table=pd.concat([tables[m] for m in modalities],ignore_index=True)
        kernel_table["coefficient_group_norm"]=np.linalg.norm(best["coef"].reshape(-1,block),axis=1)
        kernel_table["selected"]=kernel_table["coefficient_group_norm"]>1e-8
        kernel_table.to_csv(out/f"{tag}_group_weights.csv",index=False)
        np.savez_compressed(artifacts/f"{tag}_model.npz",coef=best["coef"],
                           intercept=np.array(best["intercept"]),modalities=np.array(modalities))
        save_json(artifacts/f"{tag}_model_spec.json",{
            "modalities":modalities,"group_order":kernel_table["group"].tolist(),
            "rff_frequencies":config["rff_frequencies"],"target_encoding":{"E2":1,"Vehicle":-1},
            "classification_threshold":0,"selection":best["record"],
            "maps":[str(Path(paths[m]).parent/f"{line}_{m}_maps") for m in modalities],
            "score_interpretation":"Uncalibrated regression decision score; not a probability."})
        for part,ix,pred in (("validation",val,best["validation_score"]),("test",test,pt)):
            frame=manifest.iloc[ix].copy()
            frame["decision_score_E2"]=pred
            frame["predicted_label_E2"]=(pred>=0).astype(int)
            frame.to_csv(out/f"{tag}_{part}_predictions.csv",index=False)
        row={"cell_line":line,"model":"scMKL_style","modality":mode,
             "selected_alpha_ratio":best["record"]["alpha_ratio"],
             "selected_groups":best["record"]["selected_groups"],
             "training_AUROC":float(roc_auc_score(y[train],train_score)),
             "validation_AUROC":best["record"]["validation_AUROC"],
             "seconds":time.monotonic()-started,**score_metrics(y[test],pt)}
        print(f"{tag}: stage 4/4 — saving diagnostics and metrics.",flush=True)
        row["fit_seconds"]=fit_seconds
        row["solver_outer_iterations"]=solver_iterations
        plot_model(y[test],pt,pd.DataFrame(records),kernel_table,out/f"{tag}_diagnostics.png",tag)
        save_json(out/f"{tag}_metrics.json",row)
        save_json(out/f"{tag}_live_status.json",{"stage":"completed",
                  "elapsed_seconds":time.monotonic()-stage_started,"fit_seconds":fit_seconds,
                  "solver_outer_iterations":solver_iterations,"eta_seconds":0})
        finished=sum((out/f"{cell}_{mode}_metrics.json").exists() for cell in config["cell_lines"] for mode in ("RNA_ATAC","RNA"))
        print(f"{tag}: finished in {duration_text(time.monotonic()-stage_started)}. "
              f"Models completed: {finished}/{2*len(config['cell_lines'])}.",flush=True)
        results.append(row)
        del xtrain,best
        gc.collect()
    return pd.DataFrame(results)

def export_comparison(base,baseline,out,manifests,config):
    rows=[]
    for line in config["cell_lines"]:
        for mode in ("RNA_ATAC","RNA"):
            rows.append(json.loads((out/f"{line}_{mode}_metrics.json").read_text()))
        expected=manifests[line].loc[manifests[line]["split"]=="test",["cell_id","label_E2"]].reset_index(drop=True)
        for mode in ("RNA","ATAC"):
            pred=pd.read_csv(baseline/f"{line}_{mode}_test_predictions.csv",dtype={"cell_id":str})
            if not pred[["cell_id","label_E2"]].equals(expected):
                raise ValueError("nb08 test order/labels mismatch.")
            shutil.copy2(baseline/f"{line}_{mode}_test_predictions.csv",out/f"nb08_{line}_{mode}_predictions.csv")
            rows.append({"cell_line":line,"model":"MLP_nb08","modality":mode,
                **score_metrics(pred["label_E2"].to_numpy(),pred["probability_E2"].to_numpy(),.5)})
        for mode in ("RNA_ATAC","RNA"):
            pred=pd.read_csv(out/f"{line}_{mode}_test_predictions.csv",dtype={"cell_id":str})
            if not pred[["cell_id","label_E2"]].equals(expected):
                raise ValueError("nb09 test order/labels mismatch.")
    summary=pd.DataFrame(rows)
    summary.to_csv(out/"metrics_comparison.csv",index=False)
    deltas=[]
    for line in config["cell_lines"]:
        subset=summary[summary.cell_line==line]
        joint=subset[(subset.model=="scMKL_style")&(subset.modality=="RNA_ATAC")].iloc[0]
        for _,row in subset.iterrows():
            if row.model=="scMKL_style" and row.modality=="RNA_ATAC":continue
            deltas.append({"cell_line":line,"reference":row.model+"_"+row.modality,
                "AUROC_difference_joint_minus_reference":joint.AUROC-row.AUROC,
                "AP_difference_joint_minus_reference":joint.average_precision-row.average_precision})
    pd.DataFrame(deltas).to_csv(out/"integration_differences.csv",index=False)
    rna_differences=[]
    for line in config["cell_lines"]:
        subset=summary[summary.cell_line==line]
        structured=subset[(subset.model=="scMKL_style")&(subset.modality=="RNA")].iloc[0]
        baseline_row=subset[(subset.model=="MLP_nb08")&(subset.modality=="RNA")].iloc[0]
        rna_differences.append(dict(cell_line=line,
            AUROC_RNA_kernel_minus_nb08_RNA=float(structured.AUROC-baseline_row.AUROC),
            AP_RNA_kernel_minus_nb08_RNA=float(structured.average_precision-baseline_row.average_precision)))
    pd.DataFrame(rna_differences).to_csv(out/"rna_baseline_differences.csv",index=False)
    summary["label"]=summary["model"]+" "+summary["modality"]
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,line in zip(axes,config["cell_lines"]):
        subset=summary[summary.cell_line==line]
        ax.bar(subset["label"],subset.AUROC)
        ax.set(ylim=(0,1.02),ylabel="Test AUROC",title=line)
        ax.tick_params(axis="x",rotation=35)
    fig.tight_layout()
    fig.savefig(out/"model_comparison.png",dpi=150)
    plt.show()
    plt.close(fig)
    exports=Path(base)/"results"/"exports"
    exports.mkdir(parents=True,exist_ok=True)
    archive=shutil.make_archive(str(exports/f"nb09_{out.name}"),"zip",root_dir=out)
    print("Share:",archive)
    return summary,Path(archive)

def reuse_kernel(base,out,artifacts,manifest,line,modality,config,cached_run_id):
    """Reuse complete kernels only when preprocessing, references and splits match."""
    pointer=out/f"{line}_{modality}_cache.json"
    candidates=[(out,artifacts)]
    if pointer.exists():
        origin=json.loads(pointer.read_text())["source_run"]
        if origin != out.name:
            candidates=[(Path(base)/"results"/"tables"/"nb09"/origin,
                         Path(base)/"data"/"processed"/"nb09"/origin)]
            if not (candidates[0][0]/f"{line}_{modality}_kernels.csv").exists():
                raise FileNotFoundError("Previously reused kernel source is missing; preserve the old run.")
    if cached_run_id and cached_run_id != out.name:
        candidates.append((Path(base)/"results"/"tables"/"nb09"/cached_run_id,
                           Path(base)/"data"/"processed"/"nb09"/cached_run_id))
    for old_out,old_artifacts in candidates:
        path=old_artifacts/f"{line}_{modality}_rff.npy"
        table_path=old_out/f"{line}_{modality}_kernels.csv"
        if not table_path.exists():
            continue
        if not path.exists():
            raise FileNotFoundError(f"Kernel completion table exists but array is missing: {path}")
        old_config=json.loads((old_out/"config.json").read_text())
        if {k:v for k,v in old_config.items() if k!="alpha_ratios"} != {k:v for k,v in config.items() if k!="alpha_ratios"}:
            raise ValueError("Cached kernel construction settings differ.")
        for name in ("hallmark_reference.json","motif_references.json","motif_scan_config.json",
                     "nb08_config.json","nb08_handoff.json"):
            if json.loads((old_out/name).read_text()) != json.loads((out/name).read_text()):
                raise ValueError(f"Cached reference/baseline mismatch: {name}")
        if sha256(old_out/f"{line}_splits.csv") != sha256(out/f"{line}_splits.csv"):
            raise ValueError("Cached cell split mismatch.")
        table=pd.read_csv(table_path)
        array=np.load(path,mmap_mode="r")
        if array.shape != (len(manifest),len(table)*2*config["rff_frequencies"]):
            raise ValueError("Incomplete or mismatched kernel array.")
        for start in range(0,len(manifest),256):
            if not np.isfinite(array[start:start+256]).all():
                raise ValueError("Nonfinite cached kernels.")
        maps=old_artifacts/f"{line}_{modality}_maps"
        if any(not (maps/name).is_file() for name in table.map_file):
            raise FileNotFoundError("Cached random-feature maps are incomplete.")
        if old_out != out:
            suffixes=["kernels.csv","features.csv","group_coverage.csv","group_membership.json"]
            if modality=="ATAC":suffixes+=["sequence_qc.csv"]
            for suffix in suffixes:
                original=old_out/f"{line}_{modality}_{suffix}"
                if original.exists():shutil.copy2(original,out/original.name)
            audit=old_out/f"{line}_motif_coordinate_audit.csv"
            if modality=="ATAC" and audit.exists():shutil.copy2(audit,out/audit.name)
        save_json(out/f"{line}_{modality}_cache.json",{"source_run":old_out.name,
                  "array":str(path),"maps":str(maps),"array_bytes":path.stat().st_size})
        print(f"{line} {modality}: reusing completed kernels from {old_out.name}.")
        return path,table
    return None


def review_rna_pathways(paths,tables,manifest,line,out,artifacts,config):
    """Exact additive RNA kernel contributions and validation-only leave-pathway-out analysis."""
    val=np.flatnonzero(manifest["split"].to_numpy()=="validation")
    y=manifest.iloc[val]["label_E2"].to_numpy()
    block=2*config["rff_frequencies"]
    array=np.load(paths["RNA"],mmap_mode="r")
    table=tables["RNA"].reset_index(drop=True)
    reviews=[]
    for mode in ("RNA","RNA_ATAC"):
        tag=f"{line}_{mode}"
        with np.load(artifacts/f"{tag}_model.npz",allow_pickle=False) as ck:
            coef=ck["coef"].copy();intercept=float(ck["intercept"])
            modalities=ck["modalities"].tolist()
        expected=["RNA"] if mode=="RNA" else ["RNA","ATAC"]
        if modalities!=expected:raise ValueError("Unexpected model modality order.")
        spec=json.loads((artifacts/f"{tag}_model_spec.json").read_text())
        if spec["group_order"][:len(table)]!=table["group"].tolist():
            raise ValueError("RNA group order does not match the saved model.")
        base=score_rows(paths,modalities,val,coef,intercept)
        saved=pd.read_csv(out/f"{tag}_validation_predictions.csv",dtype={"cell_id":str})
        if list(saved.cell_id)!=manifest.iloc[val].cell_id.tolist():
            raise ValueError("Validation cell order mismatch.")
        if not np.allclose(base,saved.decision_score_E2,atol=1e-5,rtol=1e-5):
            raise ValueError("Saved model/kernel predictions do not match.")
        base_auc=roc_auc_score(y,base);base_ap=average_precision_score(y,base)
        contributions=np.empty((len(val),len(table)),dtype=np.float64)
        for gi,row in table.iterrows():
            begin=gi*block;end=begin+block
            contribution=np.asarray(array[val,begin:end])@coef[begin:end]
            contributions[:,gi]=contribution
            removed=base-contribution
            norm=float(np.linalg.norm(coef[begin:end]))
            reviews.append(dict(cell_line=line,mode=mode,pathway=row["group"],
                selected=norm>1e-8,coefficient_group_norm=norm,
                validation_AUROC=float(base_auc),
                ablated_AUROC=float(roc_auc_score(y,removed)),
                AUROC_drop=float(base_auc-roc_auc_score(y,removed)),
                AP_drop=float(base_ap-average_precision_score(y,removed)),
                mean_absolute_score_contribution=float(np.abs(contribution).mean()),
                mean_score_contribution_E2=float(contribution[y==1].mean()),
                mean_score_contribution_Vehicle=float(contribution[y==0].mean())))
        if mode=="RNA" and not np.allclose(intercept+contributions.sum(1),base,atol=1e-5,rtol=1e-5):
            raise ValueError("RNA pathway contributions do not reconstruct model scores.")
        frame=pd.DataFrame(contributions,columns=table["group"].tolist())
        frame.insert(0,"label_E2",y);frame.insert(0,"cell_id",manifest.iloc[val].cell_id.to_numpy())
        frame.to_csv(out/f"{tag}_validation_pathway_contributions.csv",index=False)
        print(f"{tag}: {sum(np.linalg.norm(coef[:len(table)*block].reshape(-1,block),axis=1)>1e-8)}/{len(table)} RNA pathways selected.")
    result=pd.DataFrame(reviews)
    result.to_csv(out/f"{line}_RNA_pathway_ablation.csv",index=False)
    pivot=result.pivot(index="pathway",columns="mode",values="AUROC_drop")
    pivot.to_csv(out/f"{line}_RNA_pathway_ablation_comparison.csv")
    names=pivot.max(axis=1).nlargest(15).index
    ax=pivot.loc[names].sort_values("RNA").plot.barh(figsize=(10,7),title=f"{line}: validation RNA pathway ablation")
    ax.set_xlabel("AUROC drop after removing a pathway contribution")
    ax.figure.tight_layout();ax.figure.savefig(out/f"{line}_RNA_pathway_ablation.png",dpi=150)
    plt.show();plt.close(ax.figure)
    return result
