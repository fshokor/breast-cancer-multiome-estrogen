"""Frozen cross-cell-line treatment prediction. No training or target-data fitting."""
from pathlib import Path
import gc
import hashlib
import json
import math
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import h5py
import torch
from torch import nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix

def make_nb08_model(n_features,config):
    layers=[]
    for width in config["hidden_sizes"]:
        layers.extend([nn.Linear(n_features,width),nn.ReLU(),nn.Dropout(config["dropout"])])
        n_features=width
    return nn.Sequential(*layers,nn.Linear(n_features,1))

class MaskedPathwayLayerV1(nn.Module):
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
        self.norm=nn.LayerNorm(width,elementwise_affine=False)
    def forward(self,x):
        h=nn.functional.linear(x,self.weight*self.mask).reshape(-1,self.p,self.width)
        return self.norm(nn.functional.gelu(h))

class StochasticPathwayGateV1(nn.Module):
    def __init__(self,pathways,sigma):
        super().__init__()
        self.mu=nn.Parameter(torch.zeros(pathways));self.sigma=sigma
    def values(self):
        noise=torch.randn_like(self.mu)*self.sigma if self.training else 0
        return (self.mu+.5+noise).clamp(0,1)
    def expected_open(self):
        return .5*(1+torch.erf((self.mu+.5)/(self.sigma*math.sqrt(2))))

class PathwayMLPV1(nn.Module):
    def __init__(self,structure,config):
        super().__init__()
        p=len(structure["pathways"]);d=config["pathway_width"]
        self.rna=MaskedPathwayLayerV1(structure["rna_mask"],d)
        self.atac=MaskedPathwayLayerV1(structure["atac_mask"],d)
        self.fusion_weight=nn.Parameter(torch.empty(p,2*d,d))
        nn.init.normal_(self.fusion_weight,std=math.sqrt(1/d))
        self.fusion_norm=nn.LayerNorm(d,elementwise_affine=False)
        self.gates=StochasticPathwayGateV1(p,config["gate_sigma"])
        self.head=nn.Sequential(nn.Linear(p*d,config["head_width"]),nn.ReLU(),
            nn.Dropout(config["dropout"]),nn.Linear(config["head_width"],1))
    def fused(self,r,a,ablate=None):
        hr,ha=self.rna(r),self.atac(a)
        if ablate is not None:
            branch,index=ablate
            keep=torch.ones(hr.shape[1],device=hr.device)
            if index is None:keep.zero_()
            else:keep[index]=0
            if branch=="RNA":hr=hr*keep[None,:,None]
            elif branch=="ATAC":ha=ha*keep[None,:,None]
        joined=torch.cat([hr,ha],dim=-1)
        return self.fusion_norm(nn.functional.gelu(torch.einsum("bpi,pid->bpd",joined,self.fusion_weight)))
    def classify(self,fused,ablate=None):
        z=fused*self.gates.values()[None,:,None]
        if ablate is not None and ablate[0]=="PATHWAY":
            keep=torch.ones(z.shape[1],device=z.device);keep[ablate[1]]=0
            z=z*keep[None,:,None]
        return self.head(z.flatten(1)).ravel()
    def forward(self,r,a,ablate=None):
        return self.classify(self.fused(r,a,ablate),ablate)

class MaskedPathwayLayerV2(nn.Module):
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

class StochasticPathwayGateV2(nn.Module):
    def __init__(self,pathways,sigma,initial_mu):
        super().__init__()
        self.mu=nn.Parameter(torch.full((pathways,),float(initial_mu)));self.sigma=sigma
    def values(self):
        noise=torch.randn_like(self.mu)*self.sigma if self.training else 0
        return (self.mu+.5+noise).clamp(0,1)
    def expected_open(self):
        return .5*(1+torch.erf((self.mu+.5)/(self.sigma*math.sqrt(2))))

class PathwayMLPV2(nn.Module):
    def __init__(self,structure,config,mode):
        super().__init__()
        if mode not in ("RNA","ATAC","RNA_ATAC"):raise ValueError("Unknown input mode.")
        self.mode=mode
        p=len(structure["pathways"]);d=config["pathway_width"]
        self.pathways=p;self.width=d
        self.rna=MaskedPathwayLayerV2(structure["rna_mask"],d) if mode!="ATAC" else None
        self.atac=MaskedPathwayLayerV2(structure["atac_mask"],d) if mode!="RNA" else None
        self.fusion_weight=nn.Parameter(torch.empty(p,2*d,d))
        nn.init.normal_(self.fusion_weight,std=math.sqrt(1/d))
        self.fusion_norm=nn.Tanh()
        self.gates=StochasticPathwayGateV2(p,config["gate_sigma"],config["gate_initial_mu"])
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

def transfer_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")

def align_transfer_features(x,available,required):
    """Exact ID alignment; absent features remain zero until source-mean imputation."""
    available=pd.Index(available);required=pd.Index(required)
    if not available.is_unique or not required.is_unique:raise ValueError("Duplicate feature IDs.")
    locations=available.get_indexer(required);valid=np.flatnonzero(locations>=0)
    coo=x[:,locations[valid]].tocoo()
    aligned=sp.csr_matrix((coo.data,(coo.row,valid[coo.col])),shape=(x.shape[0],len(required)),dtype=np.float32)
    return aligned,locations<0

def transfer_metrics(y,score,threshold):
    if not np.isfinite(score).all():raise ValueError("Nonfinite prediction scores.")
    if set(np.asarray(y).tolist())!={0,1}:raise ValueError("Both treatment classes required.")
    pred=score>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(AUROC=float(roc_auc_score(y,score)),average_precision=float(average_precision_score(y,score)),
        accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        F1_E2=float(f1_score(y,pred,zero_division=0)),n_cells=len(y),
        TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))

class TransferContext:
    def __init__(self,base,output,nb08_run_id,max_missing_fraction=.05,batch_size=128):
        self.base=Path(base);self.out=Path(output)/"cross_cell_line";self.out.mkdir(exist_ok=True)
        self.baseline=self.base/"results/tables/nb08"/nb08_run_id
        self.max_missing=max_missing_fraction;self.batch_size=batch_size
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hashes={};self.coverage=[];self.raw_cache={};self.manifests={}
        self.handoff=self.read_json(self.baseline/"integration_handoff.json")
        if self.handoff["positive_class"]!="E2" or self.handoff["run_id"]!=nb08_run_id:
            raise ValueError("Unexpected baseline treatment mapping.")
        self.provenance=self.read_csv(self.baseline/"input_provenance.csv")
        for line in ("MCF7","T47D"):
            path=self.baseline/self.handoff["manifests"][line]["file"]
            if self.record(path)!=self.handoff["manifests"][line]["sha256"]:raise ValueError("Baseline split hash mismatch.")
            m=self.read_csv(path,dtype={"cell_id":str})
            if not m.cell_id.is_unique or set(m.cell_line)!={line} or not np.array_equal(m.label_E2,(m.treatment=="E2").astype(int)):
                raise ValueError("Invalid source/target labels.")
            self.manifests[line]=m
        transfer_json(self.out/"environment.json",{"python":platform.python_version(),"device":str(self.device),
            "packages":{name:importlib.metadata.version(name) for name in ("torch","numpy","pandas","scipy","anndata","h5py","scikit-learn")}})
    def record(self,path):
        path=Path(path)
        if not path.is_file():raise FileNotFoundError(f"Required saved artifact is missing: {path}")
        key=str(path)
        if key not in self.hashes:
            h=hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
            self.hashes[key]=h.hexdigest()
        return self.hashes[key]
    def read_json(self,path):
        self.record(path);return json.loads(Path(path).read_text())
    def read_csv(self,path,**kwargs):
        self.record(path);return pd.read_csv(path,**kwargs)
    def read_npz(self,path):
        self.record(path)
        with np.load(path,allow_pickle=False) as f:return {key:f[key] for key in f.files}
    def test_frame(self,line):
        return self.manifests[line].loc[lambda f:f["split"].eq("test")].reset_index(drop=True)
    def raw(self,line,modality):
        key=(line,modality)
        if key in self.raw_cache:return self.raw_cache[key]
        suffix="RNA_paper_qc" if modality=="RNA" else "ATAC_paired_paper_qc"
        path=self.base/"data/processed"/f"{line}_{suffix}.h5ad"
        expected=self.provenance[(self.provenance.cell_line==line)&(self.provenance.modality==modality)]
        if len(expected)!=1 or self.record(path)!=expected.iloc[0].sha256:
            raise ValueError(f"{line} {modality}: raw input differs from training.")
        with h5py.File(path,"r") as f:
            obs=ad.io.read_elem(f["obs"]);var=ad.io.read_elem(f["var"])
            m=self.manifests[line]
            if list(obs.index.astype(str))!=m.cell_id.tolist() or not np.array_equal(obs.treatment.astype(str),m.treatment):
                raise ValueError("Input cell order/treatment differs from split manifest.")
            x=ad.io.read_elem(f["layers/counts" if modality=="RNA" else "X"])
        if not sp.issparse(x):raise TypeError("Expected sparse counts.")
        ix=np.flatnonzero(m["split"].eq("test"))
        x=x[ix].tocsr().astype(np.float32);x.sum_duplicates();x.eliminate_zeros()
        if not np.isfinite(x.data).all() or (x.data<0).any() or not np.allclose(x.data,np.rint(x.data)):
            raise ValueError("Invalid raw counts.")
        if modality=="RNA":x.data=np.log1p(x.data)
        else:x.data.fill(1)
        result=(x,var.index.astype(str).to_numpy())
        self.raw_cache[key]=result
        return result
    def aligned(self,line,modality,required,tag):
        raw,ids=self.raw(line,modality)
        x,missing=align_transfer_features(raw,ids,required)
        count=int(missing.sum());fraction=count/max(1,len(required))
        self.coverage.append(dict(model=tag,evaluation_line=line,modality=modality,required=len(required),
            missing=count,missing_fraction=fraction,policy="source mean in transformed RNA space" if modality=="RNA" else "strict peak identity"))
        safe=tag.replace("/","_")
        pd.DataFrame({"feature_id":np.asarray(required)[missing]}).to_csv(self.out/f"{safe}_{line}_{modality}_missing_features.csv",index=False)
        if count and (modality=="ATAC" or fraction>self.max_missing):
            raise ValueError(f"{tag}: {count}/{len(required)} {modality} features absent in {line}; coverage policy failed.")
        return x,missing
    def saved_scores(self,path,line,column):
        frame=self.read_csv(path,dtype={"cell_id":str});expected=self.test_frame(line)
        if not frame.cell_id.is_unique or set(frame.cell_id)!=set(expected.cell_id):raise ValueError("Saved prediction cells differ.")
        frame=frame.set_index("cell_id").loc[expected.cell_id]
        if not np.array_equal(frame.label_E2,expected.label_E2):raise ValueError("Saved prediction labels differ.")
        return frame[column].to_numpy(dtype=float)
    def verify_model_splits(self,folder):
        for line in ("MCF7","T47D"):
            if self.record(Path(folder)/f"{line}_splits.csv")!=self.handoff["manifests"][line]["sha256"]:
                # Some nb11 files were reserialized; verify complete manifest values in that case.
                frame=self.read_csv(Path(folder)/f"{line}_splits.csv",dtype={"cell_id":str})
                columns=["cell_id","cell_line","treatment","label_E2","split"]
                if not frame[columns].equals(self.manifests[line][columns]):raise ValueError("Model split mismatch.")
    def save_audit(self):
        pd.DataFrame([{"path":p,"sha256":h} for p,h in self.hashes.items()]).to_csv(self.out/"input_artifact_hashes.csv",index=False)
        pd.DataFrame(self.coverage).to_csv(self.out/"feature_coverage.csv",index=False)

@torch.inference_mode()
def transfer_torch_scores(model,matrices,center_scale,context,mode="single"):
    model.eval();scores=[]
    n=matrices[0][0].shape[0]
    for start in range(0,n,context.batch_size):
        tensors=[]
        for (x,missing),(center,scale) in zip(matrices,center_scale):
            dense=x[start:start+context.batch_size].toarray()
            if np.any(missing):dense[:,missing]=center[missing]
            dense=(dense-center)/scale
            if not np.isfinite(dense).all():raise ValueError("Nonfinite source-scaled inputs.")
            tensors.append(torch.as_tensor(dense,dtype=torch.float32,device=context.device))
        logits=model(tensors[0]).ravel() if mode=="single" else model(*tensors).ravel()
        scores.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(scores)

def load_nb08_transfer(context,source,mode):
    tag=f"{source}_{mode}";folder=context.base/context.handoff["models_directory"]
    spec=context.read_json(folder/f"{tag}_model_spec.json")
    if spec["positive_class"]!="E2" or spec["cell_line"]!=source or spec["modality"]!=mode:
        raise ValueError("nb08 model identity mismatch.")
    if spec["split_sha256"]!=context.handoff["manifests"][source]["sha256"]:raise ValueError("Model split checksum mismatch.")
    prep=context.read_npz(folder/spec["preprocessing"])
    model=make_nb08_model(len(prep["feature_ids"]),spec["config"]).to(context.device)
    context.record(folder/spec["weights"])
    model.load_state_dict(torch.load(folder/spec["weights"],map_location=context.device,weights_only=True),strict=True)
    def scorer(line):
        x,missing=context.aligned(line,mode,prep["feature_ids"],"nb08_"+tag)
        return transfer_torch_scores(model,[(x,missing)],[(prep["center"],prep["scale"])],context)
    def saved(line):return context.baseline/f"{line}_{mode}_test_predictions.csv"
    return scorer,saved,"probability_E2",.5

def transfer_map_folder(context,run,line,modality):
    folder=Path(run);seen=set()
    while True:
        if folder.name in seen:raise ValueError("Cyclic nb09 cache chain.")
        seen.add(folder.name)
        context.verify_model_splits(folder)
        pointer=folder/f"{line}_{modality}_cache.json"
        if not pointer.exists():break
        origin=context.read_json(pointer)["source_run"]
        if not isinstance(origin,str) or "/" in origin or "\\" in origin or origin in ("",".",".."):
            raise ValueError("Invalid cache origin.")
        if origin==folder.name:break
        folder=context.base/"results/tables/nb09"/origin
    return context.base/"data/processed/nb09"/folder.name/f"{line}_{modality}_maps"

def load_nb09_transfer(context,source,mode,run_id):
    run=context.base/"results/tables/nb09"/run_id;context.verify_model_splits(run)
    artifact=context.base/"data/processed/nb09"/run_id;tag=f"{source}_{mode}"
    spec=context.read_json(artifact/f"{tag}_model_spec.json")
    if not spec["selection"]["converged"]:raise ValueError("Cannot evaluate unconverged model.")
    if spec["target_encoding"]!={"E2":1,"Vehicle":-1} or spec["classification_threshold"]!=0:
        raise ValueError("Unexpected kernel score encoding or threshold.")
    model=context.read_npz(artifact/f"{tag}_model.npz")
    modalities=["RNA"] if mode=="RNA" else ["RNA","ATAC"]
    if model["modalities"].tolist()!=modalities or spec["modalities"]!=modalities:raise ValueError("Kernel modality mismatch.")
    resources=[];offset=0;group_order=[]
    for modality in modalities:
        features=context.read_csv(run/f"{source}_{modality}_features.csv")
        kernels=context.read_csv(run/f"{source}_{modality}_kernels.csv")
        maps=transfer_map_folder(context,run,source,modality)
        group_order.extend(kernels["group"].tolist())
        for _,row in kernels.iterrows():
            g=context.read_npz(maps/row["map_file"])
            width=2*g["frequencies"].shape[1]
            weight=model["coef"][offset:offset+width];offset+=width
            if len(weight)!=width:raise ValueError("Kernel coefficient length mismatch.")
            if np.any(weight!=0):resources.append((modality,features.feature_id.astype(str).to_numpy(),g,weight))
    if offset!=len(model["coef"]) or group_order!=spec["group_order"]:raise ValueError("Kernel group order mismatch.")
    def scorer(line):
        score=np.full(len(context.test_frame(line)),float(model["intercept"]),dtype=np.float64)
        aligned={}
        for modality,ids,g,weight in tqdm(resources,desc=f"{source} -> {line}: {mode} kernels",leave=False):
            if modality not in aligned:aligned[modality]=context.aligned(line,modality,ids,"nb09_"+tag)
            x,missing=aligned[modality];cols=g["feature_columns"].astype(int)
            center=g["input_center"];scale=g["input_scale"];missing_group=missing[cols]
            d=g["frequencies"].shape[1]
            for begin in range(0,len(score),context.batch_size):
                dense=x[begin:begin+context.batch_size,cols].toarray()
                if np.any(missing_group):dense[:,missing_group]=center[missing_group]
                angle=((dense-center)/scale)@g["frequencies"]
                block=np.concatenate((np.cos(angle)/np.sqrt(d),np.sin(angle)/np.sqrt(d)),axis=1).astype(np.float32)
                block=(block-g["rff_center"])/float(g["rff_scale"])
                score[begin:begin+len(dense)]+=block@weight
        return score
    def saved(line):return run/f"{line}_{mode}_test_predictions.csv"
    return scorer,saved,"decision_score_E2",0.

def load_nb11_transfer(context,source,mode,run_id,version):
    run=context.base/"results/tables/nb11"/run_id;context.verify_model_splits(run)
    cfg=context.read_json(run/"config.json")
    if int(cfg.get("model_version",1))!=version:raise ValueError("nb11 architecture version mismatch.")
    if version==1 and mode!="RNA_ATAC":raise ValueError("nb11 v1 has only multimodal models.")
    folder=context.base/"data/processed/nb11"/run_id/source
    meta=context.read_json(folder/"prepared.json")
    if context.record(folder/"structure.npz")!=meta["checksums"]["structure.npz"]:
        raise ValueError("Source pathway structure checksum differs.")
    st=context.read_npz(folder/"structure.npz")
    model=(PathwayMLPV2(st,cfg,mode) if version==2 else PathwayMLPV1(st,cfg)).to(context.device)
    weights=folder/"models"/mode/"best.pt" if version==2 else folder/"best.pt"
    context.record(weights)
    model.load_state_dict(torch.load(weights,map_location=context.device,weights_only=True),strict=True)
    tag=f"nb11_v{version}_{source}_{mode}"
    links=None
    if mode!="RNA":
        links=context.read_csv(run/f"{source}_peak_gene_links.csv")
        links=links[links.gene.isin(st["atac_genes"])].copy()
        if links.duplicated(["peak","gene"]).any():raise ValueError("Duplicate source peak/gene links.")
    def scorer(line):
        count=len(context.test_frame(line))
        if mode!="ATAC":
            rx,rm=context.aligned(line,"RNA",st["rna_genes"],tag)
        else:
            rx=sp.csr_matrix((count,len(st["rna_genes"])),dtype=np.float32);rm=np.ones(rx.shape[1],bool)
        if mode!="RNA":
            b,peaks=context.raw(line,"ATAC")
            _,source_peaks=context.raw(source,"ATAC")
            if set(peaks)!=set(source_peaks):
                raise ValueError("ATAC assay peak universes differ; source depth transform cannot be reproduced.")
            peak_index=pd.Index(peaks).get_indexer(links.peak)
            genes=pd.Index(st["atac_genes"]).get_indexer(links.gene)
            if (peak_index<0).any() or (genes<0).any():raise ValueError("Source ATAC links cannot be aligned.")
            assignment=sp.csr_matrix((links.assignment_weight.to_numpy(dtype=np.float32),(peak_index,genes)),
                shape=(len(peaks),len(st["atac_genes"])),dtype=np.float32)
            coverage=np.asarray(assignment.sum(axis=0)).ravel()
            if (coverage<=0).any():raise ValueError("Source accessibility features lost all peak coverage.")
            depth=np.asarray(b.getnnz(axis=1)).ravel().astype(np.float32)
            if (depth<=0).any():raise ValueError("Zero-depth target ATAC cell.")
            reference=float(st["atac_training_median_depth"])
            ax=(b@assignment).multiply(1/np.maximum(coverage,1e-12)).multiply((reference/depth)[:,None]).tocsr()
            ax.eliminate_zeros();ax.data=np.log1p(ax.data);am=np.zeros(ax.shape[1],bool)
            context.coverage.append(dict(model=tag,evaluation_line=line,modality="ATAC_gene_accessibility",
                required=ax.shape[1],missing=0,missing_fraction=0.,policy="source peak links, weights, coverage and depth reference"))
        else:
            ax=sp.csr_matrix((count,len(st["atac_genes"])),dtype=np.float32);am=np.ones(ax.shape[1],bool)
        return transfer_torch_scores(model,[(rx,rm),(ax,am)],
            [(st["rna_mean"],st["rna_sd"]),(st["atac_mean"],st["atac_sd"])],context,mode="paired")
    def saved(line):
        return run/f"{line}_{mode}"/f"{line}_test_predictions.csv" if version==2 else run/f"{line}_test_predictions.csv"
    return scorer,saved,"probability_E2",.5

def run_cross_line_transfer(base,output,nb08_run_id,nb09_run_id,nb11_runs,max_missing_fraction=.05):
    """Source test reproduction is required before every opposite-line evaluation."""
    context=TransferContext(base,output,nb08_run_id,max_missing_fraction)
    jobs=[]
    for source,target in (("MCF7","T47D"),("T47D","MCF7")):
        jobs.extend(dict(family="nb08_MLP",mode=mode,run_id=nb08_run_id,version=0,source=source,target=target) for mode in ("RNA","ATAC"))
        jobs.extend(dict(family="nb09_kernel",mode=mode,run_id=nb09_run_id,version=0,source=source,target=target) for mode in ("RNA","RNA_ATAC"))
        for run in nb11_runs:
            version=run["version"]
            if version not in (1,2):raise ValueError("Only nb11 versions 1 and 2 are supported.")
            jobs.extend(dict(family=f"nb11_v{version}",mode=mode,run_id=run["run_id"],version=version,source=source,target=target)
                        for mode in (("RNA_ATAC",) if version==1 else ("RNA","ATAC","RNA_ATAC")))
    transfer_json(context.out/"transfer_protocol.json",dict(nb08_run_id=nb08_run_id,nb09_run_id=nb09_run_id,
        nb11_runs=nb11_runs,target="E2 versus Vehicle within the opposite cell line",evaluation_subset="nb08 held-out test cells only",
        missing_RNA_policy="source transformed mean; no target statistics",maximum_missing_RNA_fraction=max_missing_fraction,
        missing_ATAC_policy="fail if required peak absent or nb11 assay peak universe differs",
        model_selection="frozen source checkpoints; no target tuning",score_reproduction_atol=1e-4,score_reproduction_rtol=1e-4))
    pd.DataFrame(jobs).to_csv(context.out/"requested_models.csv",index=False)
    records=[]
    for job in tqdm(jobs,desc="Frozen cross-cell-line evaluations"):
        scorer=None
        label=f"{job['family']}_{job['run_id']}_{job['mode']}_{job['source']}_to_{job['target']}"
        row=dict(job,model=job["family"]+" "+job["mode"],status="failed")
        try:
            if job["family"]=="nb08_MLP":scorer,saved,column,threshold=load_nb08_transfer(context,job["source"],job["mode"])
            elif job["family"]=="nb09_kernel":scorer,saved,column,threshold=load_nb09_transfer(context,job["source"],job["mode"],job["run_id"])
            else:scorer,saved,column,threshold=load_nb11_transfer(context,job["source"],job["mode"],job["run_id"],job["version"])
            own=scorer(job["source"])
            original=context.saved_scores(saved(job["source"]),job["source"],column)
            maximum_error=float(np.max(np.abs(own-original)))
            row["source_reproduction_max_absolute_error"]=maximum_error
            if not np.allclose(own,original,atol=1e-4,rtol=1e-4):
                raise ValueError(f"Source test predictions were not reproduced (max error {maximum_error:.6g}); transfer withheld.")
            own_metrics=transfer_metrics(context.test_frame(job["source"]).label_E2,own,threshold)
            row["source_test_AUROC"]=own_metrics["AUROC"]
            # No target transformation is fitted; labels enter metrics only after frozen scoring.
            transferred=scorer(job["target"])
            frame=context.test_frame(job["target"]).copy()
            frame["source_cell_line"]=job["source"];frame["score"]=transferred
            frame["score_type"]=column;frame["decision_threshold"]=threshold
            frame["predicted_label_E2"]=(transferred>=threshold).astype(int)
            frame.to_csv(context.out/f"{label}_predictions.csv",index=False)
            row.update(transfer_metrics(frame.label_E2,transferred,threshold))
            target_native=context.saved_scores(saved(job["target"]),job["target"],column)
            native_metrics=transfer_metrics(frame.label_E2,target_native,threshold)
            row["target_native_AUROC"]=native_metrics["AUROC"]
            row["AUROC_transfer_minus_target_native"]=row["AUROC"]-native_metrics["AUROC"]
            row["constant_predictions"]=bool(np.ptp(transferred)<1e-8)
            row["status"]="complete"
            print(f"{label}: transfer AUROC={row['AUROC']:.4f}; target-native AUROC={row['target_native_AUROC']:.4f}")
        except Exception as error:
            row["error"]=f"{type(error).__name__}: {error}"
            print(f"{label}: NOT EVALUATED — {row['error']}")
        finally:
            records.append(row)
            pd.DataFrame(records).to_csv(context.out/"transfer_results.csv",index=False)
            context.save_audit()
            del scorer;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    result=pd.DataFrame(records)
    done=result[result.status.eq("complete")]
    if len(done):
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        for ax,(source,target) in zip(axes,(("MCF7","T47D"),("T47D","MCF7"))):
            sub=done[(done.source==source)&(done.target==target)].copy()
            if len(sub):
                sub["label"]=sub["model"]+" ["+sub["run_id"].str[:8]+"]"
                sub.set_index("label")[["AUROC","target_native_AUROC"]].plot.barh(ax=ax)
                ax.axvline(.5,color="gray",linestyle="--")
            ax.set(title=f"{source} → {target}",xlabel="Test AUROC",xlim=(0,1.02))
        fig.tight_layout();fig.savefig(context.out/"cross_cell_line_AUROC.png",dpi=160);plt.show();plt.close(fig)
    report=["# Frozen cross-cell-line treatment prediction","",
        f"Completed {len(done)} of {len(jobs)} requested evaluations. Missing/invalid artifacts are listed in transfer_results.csv.",
        "","Source models predict E2 versus Vehicle in the opposite cell line; cell-line identity is not the prediction target.",
        "All evaluations use the opposite line's existing held-out test cells. No training, calibration, threshold selection or target-statistic fitting occurs.",
        "","Target-native scores are the same model family/run trained on the target line and evaluated on those same target test cells.",
        "This comparison measures within-dataset transfer, not independent patient/replicate generalization. Shared libraries and treatment/library confounding remain possible.",
        "Both nb11 versions are included when their saved checkpoints exist. Collapsed ATAC models are retained as negative controls.",
        "Missing RNA features use source means, with the declared coverage limit. Review feature_coverage.csv and missing-feature lists.",
        "A source-prediction reproduction failure prevents transfer for that model. Do not interpret a failed row as a low-performing model.",""]
    for _,row in done.iterrows():
        report.append(f"- {row['model']} ({row['run_id']}), {row['source']} → {row['target']}: AUROC {row['AUROC']:.4f}, target-native {row['target_native_AUROC']:.4f}.")
    (context.out/"transfer_report.md").write_text("\n".join(report)+"\n")
    print(f"Cross-line evaluation: {len(done)}/{len(jobs)} completed. Review failed rows before calling the comparison complete.")
    return result
