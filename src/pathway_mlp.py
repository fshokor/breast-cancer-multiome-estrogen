"""nb11: pathway-masked RNA/ATAC MLP. No training at import."""
from pathlib import Path
import gc
import gzip
import hashlib
import json
import math
import shutil
import time
import urllib.request
import importlib.metadata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import h5py
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.auto import tqdm
from sklearn.metrics import (roc_auc_score,average_precision_score,accuracy_score,
    balanced_accuracy_score,f1_score,log_loss,brier_score_loss,confusion_matrix)

def save_json(path,value):
    path=Path(path)
    tmp=path.with_name(path.name+".partial")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)

def save_torch(path,value):
    path=Path(path)
    tmp=path.with_name(path.name+".partial")
    torch.save(value,tmp)
    tmp.replace(path)

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


def start_run(base,nb08_run_id,run_id,config,hallmark,annotation,blacklist,source_sha,resume=False):
    if config["max_epochs"]<config["min_epochs"]:
        raise ValueError("max_epochs must be >= min_epochs.")
    baseline=Path(base)/"results"/"tables"/"nb08"/nb08_run_id
    handoff=json.loads((baseline/"integration_handoff.json").read_text())
    if handoff["positive_class"]!="E2" or handoff["run_id"]!=nb08_run_id:
        raise ValueError("Unexpected baseline handoff.")
    if config["pathway_width"]<2 or config["integrated_gradients_steps"]<1:
        raise ValueError("Pathway width must be >=2 and attribution steps >=1.")
    if config["gate_sigma"]<=0 or config["batch_size"]<1:
        raise ValueError("Gate sigma and batch size must be positive.")
    if config["ig_max_steps"]<2*config["integrated_gradients_steps"]:
        raise ValueError("IG requires at least two integration resolutions for stability checks.")
    manifests={}
    provenance=pd.read_csv(baseline/"input_provenance.csv")
    for line in config["cell_lines"]:
        path=baseline/handoff["manifests"][line]["file"]
        if sha256(path)!=handoff["manifests"][line]["sha256"]:
            raise ValueError("Baseline split checksum mismatch.")
        m=pd.read_csv(path,dtype={"cell_id":str})
        if not m.cell_id.is_unique or set(m["split"])!={"train","validation","test"}:
            raise ValueError("Invalid baseline splits.")
        if not np.array_equal(m.label_E2,(m.treatment=="E2").astype(int)):
            raise ValueError("Baseline label mapping mismatch.")
        for part in ("train","validation","test"):
            if set(m.loc[m["split"]==part,"label_E2"])!={0,1}:
                raise ValueError("Each split requires both classes.")
        for modality in ("RNA","ATAC"):
            path=input_path(base,line,modality)
            expected=provenance[(provenance.cell_line==line)&(provenance.modality==modality)]
            print(f"Verifying {line} {modality} input...",flush=True)
            if len(expected)!=1 or sha256(path)!=expected.iloc[0]["sha256"]:
                raise ValueError("Input differs from nb08.")
            obs,_=annotations(path)
            if list(obs.index.astype(str))!=m.cell_id.tolist() or not np.array_equal(
                obs.treatment.astype(str),m.treatment):
                raise ValueError("Paired cell order/treatment mismatch.")
            if set(obs.cell_line.astype(str))!={line}:raise ValueError("Wrong cell line.")
        for modality in ("RNA","ATAC"):
            predictions=pd.read_csv(baseline/f"{line}_{modality}_test_predictions.csv",dtype={"cell_id":str})
            test=m[m["split"].eq("test")]
            if not predictions.cell_id.is_unique or set(predictions.cell_id)!=set(test.cell_id):
                raise ValueError("Missing or mismatched nb08 test predictions.")
            aligned=predictions.set_index("cell_id").loc[test.cell_id]
            if not np.array_equal(aligned.label_E2,test.label_E2):
                raise ValueError("nb08 test labels differ.")
            probabilities=aligned["probability_E2"].to_numpy()
            if not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any():
                raise ValueError("Invalid nb08 probabilities.")
        manifests[line]=m
    out=Path(base)/"results"/"tables"/"nb11"/run_id
    artifacts=Path(base)/"data"/"processed"/"nb11"/run_id
    signature={"config":config,"hallmark":hallmark,"annotation":annotation,"blacklist":blacklist,
               "source_sha256":source_sha,"nb08_handoff":handoff}
    if out.exists() or artifacts.exists():
        if not resume or not out.is_dir() or not artifacts.is_dir():
            raise FileExistsError("Use a new RUN_ID or explicitly set RESUME_RUN_ID.")
        if json.loads((out/"run_signature.json").read_text())!=signature:
            raise ValueError("Run settings, implementation or reference changed; use a new run.")
    else:
        out.mkdir(parents=True);artifacts.mkdir(parents=True)
        save_json(out/"run_signature.json",signature)
        save_json(out/"config.json",config)
        shutil.copy2(baseline/"config.json",out/"nb08_config.json")
        shutil.copy2(baseline/"input_provenance.csv",out/"input_provenance.csv")
        for line,m in manifests.items():m.to_csv(out/f"{line}_splits.csv",index=False)
    packages=["torch","numpy","pandas","scipy","anndata","h5py","tqdm","scikit-learn"]
    save_json(out/f"environment_{time.time_ns()}.json",{
        "versions":{p:importlib.metadata.version(p) for p in packages},
        "device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})
    return baseline,out,artifacts,manifests

def frozen_reference(base,spec,path_key,url_key,hash_key):
    path=Path(base)/spec[path_key]
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.name+".partial")
        urllib.request.urlretrieve(spec[url_key],temp)
        if sha256(temp)!=spec[hash_key]:
            raise ValueError("Reference URL has changed. Copy the original frozen reference from the prior notebooks.")
        temp.replace(path)
    if sha256(path)!=spec[hash_key]:raise ValueError(f"Reference hash mismatch: {path}")
    return path

def raw_matrix(base,line,modality):
    path=input_path(base,line,modality)
    _,var=annotations(path)
    with h5py.File(path,"r") as f:
        x=ad.io.read_elem(f["layers/counts" if modality=="RNA" else "X"])
    if not sp.issparse(x):raise TypeError("Expected sparse raw counts.")
    x=x.tocsr().astype(np.float32);x.sum_duplicates();x.eliminate_zeros()
    if not np.isfinite(x.data).all() or (x.data<0).any() or not np.allclose(x.data,np.rint(x.data)):
        raise ValueError("Expected nonnegative integer raw counts.")
    return x,var.index.astype(str).to_numpy()

def peak_gene_map(peak_names,genes,annotation_path,blacklist_path,config,out,line):
    """Fixed transcript-union promoter/body links, robust to both coordinate conventions."""
    lookup={g:i for i,g in enumerate(genes)}
    peak_rows=[]
    for i,name in enumerate(peak_names):
        chrom,s,e=name.rsplit("-",2)
        peak_rows.append((i,chrom,int(s),int(e),name))
    peaks=pd.DataFrame(peak_rows,columns=["index","chrom","start","end","peak"])
    standard={"chr"+str(i) for i in range(1,23)}|{"chrX","chrY"}
    eligible=peaks.chrom.isin(standard)&(peaks.start>=1)&(peaks.end>peaks.start)
    bl=pd.read_csv(blacklist_path,sep="\t",header=None,usecols=[0,1,2],names=["chrom","start","end"])
    for chrom,g in bl.groupby("chrom"):
        g=g.sort_values("start");bs=g.start.to_numpy();be=np.maximum.accumulate(g.end.to_numpy())
        ix=peaks.index[peaks.chrom==chrom]
        pos=np.searchsorted(bs,peaks.loc[ix,"end"],side="left")-1
        overlap=(pos>=0)&(be[np.maximum(pos,0)]>peaks.loc[ix,"start"].to_numpy()-1)
        eligible.loc[ix[overlap]]=False
    peaks["eligible_standard_interval_not_blacklisted"]=eligible
    peaks.to_csv(out/f"{line}_peak_mapping_qc.csv",index=False)
    by_chrom={chrom:g.sort_values("start") for chrom,g in peaks[eligible].groupby("chrom")}
    transcripts=[]
    with gzip.open(annotation_path,"rt") as f:
        for row in f:
            v=row.rstrip().split("\t")
            if v[12] not in lookup or v[2] not in by_chrom:continue
            start,end=int(v[4]),int(v[5]);tss=start if v[3]=="+" else end-1
            lo=tss-config["promoter_upstream_bp"] if v[3]=="+" else tss-config["promoter_downstream_bp"]
            hi=tss+config["promoter_downstream_bp"]+1 if v[3]=="+" else tss+config["promoter_upstream_bp"]+1
            if config["include_gene_body"]:lo,hi=min(lo,start),max(hi,end)
            transcripts.append((v[2],max(0,lo),hi,lookup[v[12]]))
    common,extended=set(),set()
    for chrom,lo,hi,gene in tqdm(sorted(set(transcripts)),desc=f"{line}: peak-to-gene annotation",unit="transcript"):
        p=by_chrom[chrom];starts=p.start.to_numpy();ends=p.end.to_numpy();ids=p["index"].to_numpy()
        stop=np.searchsorted(starts,hi+1,side="left")
        possible=np.flatnonzero(ends[:stop]>lo)
        for j in possible:
            if starts[j]-1<hi:extended.add((int(ids[j]),gene))
            if starts[j]<hi:common.add((int(ids[j]),gene))
    links=sorted(common & extended)
    if not links:raise ValueError("No coordinate-robust peak-to-gene links.")
    rr=np.array([p for p,g in links]);cc=np.array([g for p,g in links])
    assignments=np.bincount(rr,minlength=len(peak_names))
    values=1/assignments[rr].astype(np.float32)
    mapping=sp.csr_matrix((values,(rr,cc)),shape=(len(peak_names),len(genes)),dtype=np.float32)
    table=pd.DataFrame({"peak_index":rr,"peak":peak_names[rr],"gene":[genes[g] for g in cc],
                        "assignment_weight":values})
    table.to_csv(out/f"{line}_peak_gene_links.csv",index=False)
    save_json(out/f"{line}_mapping_summary.json",{
        "robust_links":len(common),"coordinate_sensitive_links_excluded":len(extended-common),
        "linked_peaks":int((assignments>0).sum()),"genes_with_links":int((mapping.getnnz(axis=0)>0).sum()),
        "method":"Transcript-union promoter/body overlap, fractional multi-gene assignment; provisional, not distal enhancer validation."})
    return mapping

def training_scale(x,train,config):
    tr=x[train].astype(np.float64)
    mean=np.asarray(tr.mean(axis=0)).ravel()
    variance=np.maximum(np.asarray(tr.power(2).mean(axis=0)).ravel()-mean**2,0)
    detected=np.asarray(tr.getnnz(axis=0)).ravel()
    keep=np.flatnonzero((detected>=config["min_train_detected"])&(variance>config["minimum_variance"]))
    return keep,mean[keep].astype(np.float32),np.sqrt(variance[keep]).astype(np.float32),detected[keep]

def prepare_line(base,line,manifest,config,hallmark,annotation,blacklist,out,artifacts):
    folder=artifacts/line;folder.mkdir(exist_ok=True)
    ready=folder/"prepared.json"
    if ready.exists():
        meta=json.loads(ready.read_text())
        for name,digest in meta["checksums"].items():
            if sha256(folder/name)!=digest:raise ValueError("Prepared data cache changed.")
        print(f"{line}: loading prepared RNA/accessibility matrices.")
        return load_prepared(folder,manifest)
    reference=frozen_reference(base,annotation,"annotation_file","annotation_url","annotation_sha256")
    bl=frozen_reference(base,blacklist,"path","url","sha256")
    genes=sorted(set().union(*map(set,hallmark["gene_sets"].values())))
    train=np.flatnonzero(manifest["split"].to_numpy()=="train")
    r,rg=raw_matrix(base,line,"RNA")
    rlookup={g:i for i,g in enumerate(rg)}
    rgenes=[g for g in genes if g in rlookup]
    r=r[:,[rlookup[g] for g in rgenes]].tocsr();r.data=np.log1p(r.data)
    rk,rm,rs,rd=training_scale(r,train,config);r=r[:,rk];rgenes=np.array(rgenes)[rk]
    b,peaks=raw_matrix(base,line,"ATAC");b.data.fill(1)
    depth=np.asarray(b.getnnz(axis=1)).ravel().astype(np.float32)
    if (depth<=0).any():raise ValueError("Zero-accessibility cells need explicit handling.")
    mapping=peak_gene_map(peaks,genes,reference,bl,config,out,line)
    coverage=np.asarray(mapping.sum(axis=0)).ravel()
    depth_reference=float(np.median(depth[train]))
    a=(b@mapping).multiply(1/np.maximum(coverage,1e-12)).multiply((depth_reference/depth)[:,None]).tocsr()
    a.eliminate_zeros();a.data=np.log1p(a.data)
    ak,am,ass,adetected=training_scale(a,train,config);a=a[:,ak];agenes=np.array(genes)[ak]
    del b,mapping;gc.collect()
    rlookup={g:i for i,g in enumerate(rgenes)};alookup={g:i for i,g in enumerate(agenes)}
    paths=[];mr=[];ma=[];coverage_rows=[]
    for name,members in hallmark["gene_sets"].items():
        ri=[rlookup[g] for g in members if g in rlookup];ai=[alookup[g] for g in members if g in alookup]
        usable=min(len(ri),len(ai))>=config["min_genes_per_pathway_per_modality"]
        coverage_rows.append({"pathway":name,"RNA_genes":len(ri),"ATAC_genes":len(ai),"retained":usable})
        if usable:
            paths.append(name)
            row=np.zeros(len(rgenes),np.float32);row[ri]=1;mr.append(row)
            row=np.zeros(len(agenes),np.float32);row[ai]=1;ma.append(row)
    if not paths:raise ValueError("No pathways supported in both modalities.")
    pd.DataFrame(coverage_rows).to_csv(out/f"{line}_pathway_coverage.csv",index=False)
    # Drop features that have no connections to any retained pathway.
    mr=np.array(mr);ma=np.array(ma);kr=np.flatnonzero(mr.sum(0));ka=np.flatnonzero(ma.sum(0))
    r=r[:,kr].tocsr();a=a[:,ka].tocsr();mr=mr[:,kr];ma=ma[:,ka]
    rgenes=rgenes[kr];agenes=agenes[ka];rm=rm[kr];rs=rs[kr];am=am[ka];ass=ass[ka]
    for modality,names,mean,scale in (("RNA",rgenes,rm,rs),("ATAC",agenes,am,ass)):
        pd.DataFrame({"gene":names,"training_mean":mean,"training_sd":scale}).to_csv(
            out/f"{line}_{modality}_features.csv",index=False)
    sp.save_npz(folder/"rna.npz",r);sp.save_npz(folder/"atac.npz",a)
    np.savez_compressed(folder/"structure.npz",rna_genes=rgenes.astype(str),atac_genes=agenes.astype(str),
        pathways=np.array(paths,dtype=str),rna_mask=mr,atac_mask=ma,rna_mean=rm,rna_sd=rs,
        atac_mean=am,atac_sd=ass,atac_training_median_depth=np.array(depth_reference))
    names=["rna.npz","atac.npz","structure.npz"]
    save_json(ready,{"checksums":{n:sha256(folder/n) for n in names}})
    return load_prepared(folder,manifest)

def load_prepared(folder,manifest):
    with np.load(folder/"structure.npz",allow_pickle=False) as f:structure={k:f[k] for k in f.files}
    r=sp.load_npz(folder/"rna.npz");a=sp.load_npz(folder/"atac.npz")
    if r.shape[0]!=len(manifest) or a.shape[0]!=len(manifest):raise ValueError("Prepared row mismatch.")
    return {"rna":r,"atac":a,"structure":structure,"manifest":manifest,"folder":folder}

class MaskedPathwayLayer(nn.Module):
    def __init__(self,mask,width):
        super().__init__()
        p,g=mask.shape
        expanded=torch.as_tensor(mask,dtype=torch.float32).repeat_interleave(width,dim=0)
        self.register_buffer("mask",expanded)
        self.weight=nn.Parameter(torch.empty(p*width,g))
        with torch.no_grad():
            self.weight.normal_()
            self.weight.mul_(torch.sqrt(2/expanded.sum(1,keepdim=True).clamp_min(1)))
            self.weight.mul_(expanded)
        self.p=p;self.width=width
        self.norm=nn.Tanh()
    def forward(self,x):
        h=nn.functional.linear(x,self.weight*self.mask).reshape(-1,self.p,self.width)
        return self.norm(h)

class StochasticPathwayGate(nn.Module):
    def __init__(self,pathways,sigma,initial_mu):
        super().__init__()
        self.mu=nn.Parameter(torch.full((pathways,),float(initial_mu)));self.sigma=sigma
    def values(self):
        noise=torch.randn_like(self.mu)*self.sigma if self.training else 0
        return (self.mu+.5+noise).clamp(0,1)
    def expected_open(self):
        return .5*(1+torch.erf((self.mu+.5)/(self.sigma*math.sqrt(2))))

class PathwayMLP(nn.Module):
    def __init__(self,structure,config,mode):
        super().__init__()
        if mode not in ("RNA","ATAC","RNA_ATAC"):raise ValueError("Unknown input mode.")
        self.mode=mode
        p=len(structure["pathways"]);d=config["pathway_width"]
        self.pathways=p;self.width=d
        self.rna=MaskedPathwayLayer(structure["rna_mask"],d) if mode!="ATAC" else None
        self.atac=MaskedPathwayLayer(structure["atac_mask"],d) if mode!="RNA" else None
        self.fusion_weight=nn.Parameter(torch.empty(p,2*d,d))
        nn.init.normal_(self.fusion_weight,std=math.sqrt(1/d))
        self.fusion_norm=nn.Tanh()
        self.gates=StochasticPathwayGate(p,config["gate_sigma"],config["gate_initial_mu"])
        self.head=nn.Sequential(nn.Linear(p*d,config["head_width"]),nn.ReLU(),
            nn.Dropout(config["dropout"]),nn.Linear(config["head_width"],1))
    def fused(self,r,a,ablate=None):
        zero=r.new_zeros((len(r),self.pathways,self.width))
        hr=self.rna(r) if self.rna is not None else zero
        ha=self.atac(a) if self.atac is not None else zero
        if ablate is not None:
            branch,index=ablate
            keep=torch.ones(hr.shape[1],device=hr.device)
            if index is None:keep.zero_()
            else:keep[index]=0
            if branch=="RNA":hr=hr*keep[None,:,None]
            elif branch=="ATAC":ha=ha*keep[None,:,None]
        joined=torch.cat([hr,ha],dim=-1)
        return self.fusion_norm(torch.einsum("bpi,pid->bpd",joined,self.fusion_weight))
    def classify(self,fused,ablate=None):
        z=fused*self.gates.values()[None,:,None]
        if ablate is not None and ablate[0]=="PATHWAY":
            keep=torch.ones(z.shape[1],device=z.device);keep[ablate[1]]=0
            z=z*keep[None,:,None]
        return self.head(z.flatten(1)).ravel()
    def forward(self,r,a,ablate=None):
        return self.classify(self.fused(r,a,ablate),ablate)


def batch(data,indices,device):
    st=data["structure"]
    values=[]
    for key in ("rna","atac"):
        x=data[key][indices].toarray()
        x=(x-st[key+"_mean"])/st[key+"_sd"]
        values.append(torch.as_tensor(x,dtype=torch.float32,device=device))
    y=torch.as_tensor(data["manifest"].iloc[indices]["label_E2"].to_numpy(),dtype=torch.float32,device=device)
    return *values,y

def split_indices(data,split):
    if split=="val":split="validation"
    return np.flatnonzero(data["manifest"]["split"].eq(split).to_numpy())

@torch.no_grad()
def predict(model,data,indices,config,ablate=None):
    model.eval(); device=next(model.parameters()).device; logits=[]
    for start in range(0,len(indices),config["batch_size"]):
        r,a,y=batch(data,indices[start:start+config["batch_size"]],device)
        logits.append(model(r,a,ablate).cpu())
    z=torch.cat(logits)
    y=torch.tensor(data["manifest"].iloc[indices]["label_E2"].to_numpy(),dtype=torch.float32)
    return z.sigmoid().numpy(),float(nn.functional.binary_cross_entropy_with_logits(z,y)),z.numpy()

def scores(y,p):
    pred=p>=.5
    return dict(AUROC=float(roc_auc_score(y,p)),AUPRC=float(average_precision_score(y,p)),
        accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        F1=float(f1_score(y,pred)),log_loss=float(log_loss(y,p,labels=[0,1])),
        Brier=float(brier_score_loss(y,p)),confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())

def train_model(data,line,config,out,mode):
    """Resume only completed epochs; validation chooses epoch; test never enters training."""
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(config["seed"])
    rng=np.random.default_rng(config["seed"])
    model=PathwayMLP(data["structure"],config,mode).to(device)
    optimizer=torch.optim.AdamW([
        {"params":[p for n,p in model.named_parameters() if n!="gates.mu"],"lr":config["learning_rate"],"weight_decay":config["weight_decay"]},
        {"params":[model.gates.mu],"lr":config["gate_learning_rate"],"weight_decay":0.}])
    folder=data["folder"]/"models"/mode;folder.mkdir(parents=True,exist_ok=True)
    out=out/f"{line}_{mode}";out.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/"last.pt"
    train=split_indices(data,"train"); val=split_indices(data,"val")
    history=[]; best=float("inf"); best_epoch=0; stale=0; best_state=None; done=False
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location=device,weights_only=True)
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"])
        history=ck["history"];best=ck["best"];best_epoch=ck["best_epoch"]
        stale=ck["stale"];best_state=ck["best_state"];done=ck["done"]
        rng.bit_generator.state=ck["numpy_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if device.type=="cuda" and ck["cuda_rng"]:
            torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
        print(f"{line}: resumed after epoch {len(history)}, device={device}")
    else:print(f"{line}: training on {device}; tqdm ETA is to the maximum epoch count, before early stopping.")
    assert config["max_epochs"]>=config["min_epochs"]>=1
    elapsed=sum(row["epoch_seconds"] for row in history)
    bar=tqdm(range(len(history)+1,config["max_epochs"]+1),desc=f"{line} {mode} epochs",initial=len(history),total=config["max_epochs"],disable=done)
    if not done:
        for epoch in bar:
            began=time.perf_counter();model.train();total_bce=0.
            strength=float(config["gate_lambda"])
            order=rng.permutation(train)
            batches=tqdm(range(0,len(order),config["batch_size"]),desc=f"epoch {epoch} batches",leave=False)
            for start in batches:
                r,a,y=batch(data,order[start:start+config["batch_size"]],device)
                optimizer.zero_grad(set_to_none=True)
                bce=nn.functional.binary_cross_entropy_with_logits(model(r,a),y)
                loss=bce+strength*model.gates.expected_open().sum()
                if not torch.isfinite(loss):raise RuntimeError("Non-finite training loss; last completed epoch is retained.")
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),config["gradient_clip"]);optimizer.step()
                total_bce+=float(bce.detach())*len(y)
                batches.set_postfix(BCE=f"{float(bce.detach()):.4f}")
            prob,val_bce,_=predict(model,data,val,config)
            if not math.isfinite(val_bce):raise RuntimeError("Non-finite validation loss.")
            auc=float(roc_auc_score(data["manifest"].iloc[val]["label_E2"],prob))
            if val_bce<best-config["min_delta"]:
                best=val_bce;best_epoch=epoch;stale=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            duration=time.perf_counter()-began;elapsed+=duration
            row=dict(epoch=epoch,train_BCE=total_bce/len(train),val_BCE=val_bce,val_AUROC=auc,
                gate_penalty=strength,expected_open=float(model.gates.expected_open().sum().detach()),
                gates_above_half=int((model.gates.values()>.5).sum()),
                gates_exactly_zero=int((model.gates.values()==0).sum()),epoch_seconds=duration,
                cumulative_seconds=elapsed,best_epoch=best_epoch,patience_used=stale)
            history.append(row)
            done=(epoch>=config["min_epochs"] and stale>=config["patience"]) or epoch==config["max_epochs"]
            save_torch(checkpoint,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                history=history,best=best,best_epoch=best_epoch,stale=stale,best_state=best_state,
                done=done,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else []))
            pd.DataFrame(history).to_csv(out/f"{line}_training_history.csv",index=False)
            save_json(out/f"{line}_progress.json",row)
            bar.set_postfix(val_AUROC=f"{auc:.4f}",best_epoch=best_epoch,patience=f"{stale}/{config['patience']}",minutes=f"{elapsed/60:.1f}")
            if done:break
    bar.close()
    if best_state is None:raise RuntimeError("No eligible validation checkpoint.")
    model.load_state_dict(best_state);model.eval()
    # Verify the excluded modality cannot influence unimodal predictions.
    if mode!="RNA_ATAC":
        r,a,_=batch(data,val[:8],device)
        with torch.no_grad():
            original=model(r,a)
            altered=model(r,torch.zeros_like(a)) if mode=="RNA" else model(torch.zeros_like(r),a)
        if not torch.equal(original,altered):raise RuntimeError("Unimodal input isolation failed.")
    print(f"{line} {mode}: selected epoch {best_epoch}, validation BCE={best:.6f}, completed epochs={len(history)}.")
    save_torch(folder/"best.pt",best_state)
    save_json(out/f"{line}_training_complete.json",dict(mode=mode,best_epoch=best_epoch,best_validation_BCE=best,epochs=len(history),training_seconds=elapsed,device=str(device),parameters=sum(p.numel() for p in model.parameters())))
    fig,axes=plt.subplots(1,2,figsize=(11,3))
    frame=pd.DataFrame(history)
    frame.plot(x="epoch",y=["train_BCE","val_BCE"],ax=axes[0])
    frame.plot(x="epoch",y="expected_open",ax=axes[1])
    axes[0].axvline(best_epoch,color="black",linestyle="--",label="selected epoch");axes[0].legend()
    fig.tight_layout();fig.savefig(out/f"{line}_training.png",dpi=160);plt.show()
    return model

def evaluate_model(model,data,line,config,out,baseline):
    out=out/f"{line}_{model.mode}"
    records=[]
    for split in ("val","test"):
        idx=split_indices(data,split);p,bce,z=predict(model,data,idx,config)
        frame=data["manifest"].iloc[idx].copy()
        frame["probability_E2"]=p;frame["logit_E2"]=z;frame["predicted_E2"]=(p>=.5).astype(int)
        frame.to_csv(out/f"{line}_{split}_predictions.csv",index=False)
        metric=scores(frame["label_E2"],p)
        save_json(out/f"{line}_{split}_metrics.json",metric)
        records.append(dict(cell_line=line,model="pathway_"+model.mode,split=split,**{k:v for k,v in metric.items() if k!="confusion_matrix"}))
        if split=="test":
            for modality in ("RNA","ATAC"):
                old=pd.read_csv(baseline/f"{line}_{modality}_test_predictions.csv",dtype={"cell_id":str})
                if not old["cell_id"].is_unique or set(old["cell_id"])!=set(frame["cell_id"]):
                    raise ValueError("Baseline test IDs differ.")
                old=old.set_index("cell_id").loc[frame["cell_id"]]
                if not np.array_equal(old["label_E2"],frame["label_E2"]):raise ValueError("Baseline labels differ.")
                bm=scores(old["label_E2"],old["probability_E2"])
                records.append(dict(cell_line=line,model="nb08_"+modality,split="test",
                    **{k:v for k,v in bm.items() if k!="confusion_matrix"}))
    st=data["structure"]
    pd.DataFrame({"pathway":st["pathways"],"deterministic_gate":model.gates.values().detach().cpu().numpy(),
        "expected_open_probability":model.gates.expected_open().detach().cpu().numpy()}).to_csv(out/f"{line}_gates.csv",index=False)
    return pd.DataFrame(records)

def pathway_ablation(model,data,line,config,out):
    out=out/f"{line}_{model.mode}"
    idx=split_indices(data,"val"); y=data["manifest"].iloc[idx]["label_E2"].to_numpy()
    base,bce,logit=predict(model,data,idx,config);base_auc=roc_auc_score(y,base)
    modalities=("RNA","ATAC") if model.mode=="RNA_ATAC" else (model.mode,)
    jobs=[(branch,i,str(p)) for i,p in enumerate(data["structure"]["pathways"]) for branch in ("PATHWAY",*modalities)]
    jobs += [(branch,None,"ALL_PATHWAYS") for branch in modalities]
    rows=[]
    for branch,index,name in tqdm(jobs,desc=f"{line} validation ablations"):
        prob,loss,z=predict(model,data,idx,config,(branch,index))
        rows.append(dict(pathway=name,branch=branch,baseline_AUROC=float(base_auc),
            ablated_AUROC=float(roc_auc_score(y,prob)),AUROC_drop=float(base_auc-roc_auc_score(y,prob)),
            BCE_increase=float(loss-bce),mean_absolute_logit_change=float(np.abs(logit-z).mean()),
            mean_signed_logit_change=float((logit-z).mean())))
    result=pd.DataFrame(rows);result.to_csv(out/f"{line}_validation_ablation.csv",index=False)
    plot=result[result.branch.eq("PATHWAY")].nlargest(15,"AUROC_drop").sort_values("AUROC_drop")
    ax=plot.plot.barh(x="pathway",y="AUROC_drop",figsize=(9,6),legend=False,title=f"{line}: validation pathway ablation")
    ax.figure.tight_layout();ax.figure.savefig(out/f"{line}_ablation.png",dpi=160);plt.show()
    return result


def gene_attribution(model,data,line,config,out,ablations):
    """Adaptive Gauss-Legendre conditional IG; withhold rankings unless checks pass."""
    root_out=out;out=out/f"{line}_{model.mode}"
    selected=ablations[ablations.branch.eq("PATHWAY") & ablations.AUROC_drop.gt(0)].nlargest(config["attribution_top_pathways"],"AUROC_drop")
    rows=[];checks=[]
    rng=np.random.default_rng(config["seed"]);val=split_indices(data,"val")
    labels=data["manifest"]["label_E2"].to_numpy()
    idx=np.concatenate([rng.choice(val[labels[val]==c],size=min(config["attribution_cells_per_class"],sum(labels[val]==c)),replace=False) for c in (0,1)])
    device=next(model.parameters()).device;model.eval()
    r,a,y=batch(data,idx,device);context=model.fused(r,a).detach()
    st=data["structure"]
    modalities=("RNA","ATAC") if model.mode=="RNA_ATAC" else (model.mode,)
    for name in tqdm(selected.pathway,desc=f"{line} {model.mode} checked IG"):
        pi=list(st["pathways"]).index(name);mask=torch.zeros_like(context);mask[:,pi,:]=1
        with torch.no_grad():
            reference=context*(1-mask)+model.fused(torch.zeros_like(r),torch.zeros_like(a))*mask
            delta=(model.classify(context)-model.classify(reference)).detach()
        steps=config["integrated_gradients_steps"];previous=None;passed=False
        while steps<=config["ig_max_steps"]:
            nodes,weights=np.polynomial.legendre.leggauss(steps)
            accum=[torch.zeros_like(r),torch.zeros_like(a)]
            for alpha,weight in zip((nodes+1)/2,weights/2):
                rr=(r*float(alpha)).detach().requires_grad_()
                aa=(a*float(alpha)).detach().requires_grad_()
                mixed=context*(1-mask)+model.fused(rr,aa)*mask
                grads=torch.autograd.grad(model.classify(mixed).sum(),(rr,aa),allow_unused=True)
                for k,grad in enumerate(grads):
                    if grad is not None:accum[k]+=grad.detach()*float(weight)
            attrs=[r*accum[0],a*accum[1]]
            error=attrs[0].sum(1)+attrs[1].sum(1)-delta
            complete=bool(torch.all(error.abs()<=config["ig_absolute_tolerance"]+config["ig_relative_tolerance"]*delta.abs()))
            finite=all(bool(torch.isfinite(x).all()) for x in attrs)
            stable=False;max_change=None
            if previous is not None:
                change=sum((x-old).abs().sum(1) for x,old in zip(attrs,previous))
                scale=sum(x.abs().sum(1) for x in attrs)
                stable=bool(torch.all(change<=config["ig_absolute_tolerance"]+config["ig_stability_relative_tolerance"]*scale))
                max_change=float(change.max())
            passed=finite and complete and stable
            checks.append(dict(pathway=name,steps=steps,passed=passed,complete=complete,stable=stable,finite=finite,
                max_absolute_completeness_error=float(error.abs().max()),
                mean_absolute_completeness_error=float(error.abs().mean()),
                max_L1_attribution_change=max_change,max_absolute_target_logit_change=float(delta.abs().max())))
            if passed:break
            previous=[x.detach().clone() for x in attrs];steps*=2
        if not passed:
            print(f"{name}: IG checks failed; gene rankings and region candidates withheld.")
            continue
        for modality,attr in zip(("RNA","ATAC"),attrs):
            if modality not in modalities:continue
            values=attr.detach().cpu().numpy();key=modality.lower()
            connected=st[key+"_mask"][pi].astype(bool)
            if np.any(~connected):assert np.max(np.abs(values[:,~connected]))<1e-5
            for gi in np.flatnonzero(connected):
                rows.append(dict(pathway=name,modality=modality,gene=str(st[key+"_genes"][gi]),
                    mean_absolute_IG=float(np.abs(values[:,gi]).mean()),mean_signed_IG=float(values[:,gi].mean()),
                    mean_IG_E2=float(values[labels[idx]==1,gi].mean()),mean_IG_Vehicle=float(values[labels[idx]==0,gi].mean())))
    columns=["pathway","modality","gene","mean_absolute_IG","mean_signed_IG","mean_IG_E2","mean_IG_Vehicle"]
    frame=pd.DataFrame(rows,columns=columns)
    frame.to_csv(out/f"{line}_validation_gene_attribution.csv",index=False)
    diagnostics=pd.DataFrame(checks,columns=["pathway","steps","passed","complete","stable","finite","max_absolute_completeness_error","mean_absolute_completeness_error","max_L1_attribution_change","max_absolute_target_logit_change"])
    diagnostics.to_csv(out/f"{line}_IG_completeness.csv",index=False)
    data["manifest"].iloc[idx].to_csv(out/f"{line}_attribution_cells.csv",index=False)
    if checks:print(diagnostics.groupby("pathway").tail(1).to_string(index=False))
    else:print("No positive pathway ablation effects; no gene attribution selected.")
    genes=frame[frame.modality.eq("ATAC")].sort_values("mean_absolute_IG",ascending=False).groupby("pathway").head(10)[["pathway","gene"]]
    links=pd.read_csv(root_out/f"{line}_peak_gene_links.csv")
    candidates=genes.merge(links,on="gene")
    candidates["interpretation"]="candidate linked peak; not peak-level attribution"
    candidates.to_csv(out/f"{line}_candidate_regions.csv",index=False)
    return frame

def summarize_model(model,data,line,out):
    folder=out/f"{line}_{model.mode}"
    gates=model.gates.values().detach().cpu().numpy()
    info=dict(cell_line=line,mode=model.mode,pathways=len(gates),
        exactly_zero=int((gates==0).sum()),nonzero=int((gates>0).sum()),
        above_half=int((gates>.5).sum()),expected_open=float(model.gates.expected_open().sum().detach()),
        parameters=sum(p.numel() for p in model.parameters()))
    save_json(folder/"sparsity_summary.json",info)
    print(info)
    if info["nonzero"]>.8*info["pathways"]:
        print("Most pathways remain open: this run has not achieved a compact pathway selection.")
    return info

def export_matched_review(out):
    comparisons=[];sparsity=[];ablations=[]
    for folder in sorted(out.iterdir()):
        if not folder.is_dir():continue
        summary=folder/"sparsity_summary.json"
        if not summary.exists():continue
        info=json.loads(summary.read_text());line=info["cell_line"];mode=info["mode"]
        diagnostic_path=folder/f"{line}_IG_completeness.csv"
        if diagnostic_path.exists():
            diagnostics=pd.read_csv(diagnostic_path)
            final_checks=diagnostics.groupby("pathway").tail(1)
            info["IG_pathways_attempted"]=len(final_checks)
            info["IG_pathways_passed"]=int(final_checks["passed"].sum())
        sparsity.append(info)
        comparisons.append(pd.read_csv(folder/"comparison.csv"))
        frame=pd.read_csv(folder/f"{line}_validation_ablation.csv")
        frame["cell_line"]=line;frame["mode"]=mode;ablations.append(frame)
    metrics=pd.concat(comparisons,ignore_index=True).drop_duplicates(["cell_line","model","split"])
    metrics.to_csv(out/"model_comparison.csv",index=False)
    sparse=pd.DataFrame(sparsity);sparse.to_csv(out/"sparsity_comparison.csv",index=False)
    ablation=pd.concat(ablations,ignore_index=True);ablation.to_csv(out/"all_validation_ablations.csv",index=False)
    # Validation-only biological dependence comparison; this does not quantify nb08 interpretability.
    for line in sparse.cell_line.unique():
        sub=ablation[ablation.cell_line.eq(line)&ablation.branch.eq("PATHWAY")]
        pivot=sub.pivot(index="pathway",columns="mode",values="AUROC_drop")
        pivot.to_csv(out/f"{line}_matched_pathway_ablation.csv")
        leading=pivot.max(axis=1).nlargest(15).index
        ax=pivot.loc[leading].sort_values("RNA_ATAC").plot.barh(figsize=(10,7),title=f"{line}: matched validation pathway ablations")
        ax.set_xlabel("AUROC drop when pathway is removed")
        ax.figure.tight_layout();ax.figure.savefig(out/f"{line}_matched_ablation.png",dpi=160);plt.show()
    return metrics,sparse
