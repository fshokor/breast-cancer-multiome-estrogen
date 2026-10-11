"""nb08: separate RNA/ATAC MLPs. Importing defines functions only."""
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import sys
import time
import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score, log_loss,
    brier_score_loss, confusion_matrix, roc_curve, precision_recall_curve)

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

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

def prepare(base, config, run_id):
    out = Path(base)/"results"/"tables"/"nb08"/run_id
    models = Path(base)/"data"/"processed"/"nb08"/run_id
    if out.exists() or models.exists():
        raise FileExistsError("Choose a new RUN_ID; this run already exists.")
    out.mkdir(parents=True)
    models.mkdir(parents=True)
    save_json(out/"config.json", config)
    packages = ["anndata","h5py","numpy","pandas","scipy","scikit-learn","matplotlib","torch"]
    save_json(out/"environment.json", {
        "python":sys.version, "platform":platform.platform(),
        "packages":{p:importlib.metadata.version(p) for p in packages},
        "cuda":torch.version.cuda,
        "device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})
    manifests, provenance = {}, []
    for line in config["cell_lines"]:
        paths = [input_path(base,line,m) for m in ("RNA","ATAC")]
        a, _ = annotations(paths[0])
        b, _ = annotations(paths[1])
        labels = a["treatment"].astype(str).to_numpy()
        if not a.index.equals(b.index) or not np.array_equal(labels,b["treatment"].astype(str)):
            raise ValueError(f"{line}: RNA/ATAC cell order or treatment mismatch.")
        if set(labels) != {"E2","Vehicle"}:
            raise ValueError("Unexpected treatment labels.")
        if any(set(o["cell_line"].astype(str)) != {line} for o in (a,b)):
            raise ValueError("Unexpected cell-line labels.")
        y = (labels == "E2").astype(int)
        trainval, test = train_test_split(np.arange(len(y)), test_size=config["test_fraction"],
                                         stratify=y, random_state=config["seed"])
        train, val = train_test_split(trainval, test_size=config["val_fraction_remaining"],
                                     stratify=y[trainval], random_state=config["seed"]+1)
        assert not set(train)&set(val) and not set(train)&set(test) and not set(val)&set(test)
        assert len(train)+len(val)+len(test) == len(y)
        if any(len(np.unique(y[ix])) != 2 for ix in (train,val,test)):
            raise ValueError("Every split must contain both classes.")
        split = np.full(len(y),"",dtype=object)
        split[train],split[val],split[test] = "train","validation","test"
        manifest = pd.DataFrame({"cell_id":a.index.astype(str),"cell_line":line,
            "treatment":labels,"label_E2":y,"split":split})
        manifest.to_csv(out/f"{line}_splits.csv",index=False)
        manifests[line] = manifest
        for modality,path in zip(("RNA","ATAC"),paths):
            print(f"Fingerprinting {line} {modality}...",flush=True)
            provenance.append({"cell_line":line,"modality":modality,
                "path":str(path.relative_to(base)),"bytes":path.stat().st_size,"sha256":sha256(path)})
    pd.DataFrame(provenance).to_csv(out/"input_provenance.csv",index=False)
    counts = pd.concat([m.groupby(["cell_line","split","treatment"]).size().rename("cells").reset_index()
                        for m in manifests.values()],ignore_index=True)
    counts.to_csv(out/"split_counts.csv",index=False)
    save_json(out/"integration_handoff.json",{
        "positive_class":"E2", "run_id":run_id,
        "manifests":{line:{"file":f"{line}_splits.csv","sha256":sha256(out/f"{line}_splits.csv")}
                     for line in config["cell_lines"]},
        "models_directory":str(models.relative_to(base)),
        "instruction":"nb09 must reuse these exact cell splits; fit on train, select on validation, "
                      "evaluate test only after selection."})
    return out,models,manifests,counts

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

def make_model(n_features,config):
    layers=[]
    for width in config["hidden_sizes"]:
        layers.extend([nn.Linear(n_features,width),nn.ReLU(),nn.Dropout(config["dropout"])])
        n_features=width
    return nn.Sequential(*layers,nn.Linear(n_features,1))

def batches(x,y,indices,center,scale,batch_size,device,rng=None):
    order=np.array(indices,copy=True)
    if rng is not None:
        rng.shuffle(order)
    for start in range(0,len(order),batch_size):
        ix=order[start:start+batch_size]
        dense=x[ix].toarray()  # Densify only one minibatch.
        dense-=center
        dense/=scale
        yield torch.from_numpy(dense).to(device),torch.from_numpy(y[ix].astype(np.float32)).to(device)

def evaluate(model,x,y,ix,center,scale,config,device):
    model.eval()
    total,probabilities=0.,[]
    with torch.no_grad():
        for xb,yb in batches(x,y,ix,center,scale,config["batch_size"],device):
            logits=model(xb).ravel()
            total+=nn.functional.binary_cross_entropy_with_logits(logits,yb,reduction="sum").item()
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return total/len(ix),np.concatenate(probabilities)

def metrics(y,p):
    pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"n_cells":len(y),"E2_prevalence":float(y.mean()),
        "AUROC":float(roc_auc_score(y,p)),"average_precision":float(average_precision_score(y,p)),
        "accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "F1_E2":float(f1_score(y,pred,zero_division=0)),
        "precision_E2":float(precision_score(y,pred,zero_division=0)),
        "recall_E2":float(recall_score(y,pred,zero_division=0)),
        "log_loss":float(log_loss(y,p,labels=[0,1])),"brier_score":float(brier_score_loss(y,p)),
        "TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)}

def plot_diagnostics(history,y,p,result,path,tag):
    fig,ax=plt.subplots(1,4,figsize=(19,4))
    ax[0].plot(history["epoch"],history["train_loss"],label="Train")
    ax[0].plot(history["epoch"],history["validation_loss"],label="Validation")
    ax[0].set(xlabel="Epoch",ylabel="Binary cross-entropy",title="Learning curves")
    ax[0].legend()
    fpr,tpr,_=roc_curve(y,p)
    ax[1].plot(fpr,tpr,label=f'AUROC={result["AUROC"]:.3f}')
    ax[1].plot([0,1],[0,1],"--",color="grey")
    ax[1].set(xlabel="False positive rate",ylabel="True positive rate",title="Test ROC")
    ax[1].legend()
    precision,recall,_=precision_recall_curve(y,p)
    ax[2].plot(recall,precision,label=f'AP={result["average_precision"]:.3f}')
    ax[2].axhline(y.mean(),linestyle="--",color="grey",label="E2 prevalence")
    ax[2].set(xlabel="Recall",ylabel="Precision",title="Test precision–recall")
    ax[2].legend()
    cm=confusion_matrix(y,p>=.5,labels=[0,1])
    ax[3].imshow(cm,cmap="Blues")
    ax[3].set(xticks=[0,1],yticks=[0,1],xticklabels=["Vehicle","E2"],yticklabels=["Vehicle","E2"],
              xlabel="Predicted",ylabel="Actual",title="Test confusion matrix")
    for row in range(2):
        for col in range(2):
            ax[3].text(col,row,str(cm[row,col]),ha="center",va="center",color="black")
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(path,dpi=160,bbox_inches="tight")
    plt.show()
    plt.close(fig)

def train_one(base,out,models,manifest,line,modality,config):
    tag=f"{line}_{modality}"
    if (out/f"{tag}_metrics.json").exists():
        raise FileExistsError("Model completed already. Use a new RUN_ID to retrain.")
    if config != json.loads((out/"config.json").read_text()):
        raise ValueError("Configuration changed after split creation. Start a new run.")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x,center,scale,features=load_features(base,line,modality,manifest,config)
    features.to_csv(out/f"{tag}_features.csv",index=False)
    np.savez_compressed(models/f"{tag}_preprocessing.npz",
        input_columns=features["input_column"].to_numpy(),
        feature_ids=features["feature_id"].to_numpy(dtype=str),center=center,scale=scale)
    y=manifest["label_E2"].to_numpy(dtype=int)
    train,val,test=[np.flatnonzero(manifest["split"].to_numpy()==name) for name in ("train","validation","test")]
    model=make_model(x.shape[1],config).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    rng=np.random.default_rng(config["seed"])
    history,best_state,best_loss,best_epoch,stale=[],None,float("inf"),0,0
    started=time.monotonic()
    print(f"{tag}: {x.shape[1]} features; train/validation/test={len(train)}/{len(val)}/{len(test)}; {device}",flush=True)
    for epoch in range(1,config["max_epochs"]+1):
        model.train()
        total=0.
        for xb,yb in batches(x,y,train,center,scale,config["batch_size"],device,rng):
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(xb).ravel(),yb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss.")
            loss.backward()
            optimizer.step()
            total+=loss.item()*len(yb)
        val_loss,_=evaluate(model,x,y,val,center,scale,config,device)
        if not np.isfinite(val_loss):
            raise FloatingPointError("Nonfinite validation loss.")
        history.append({"epoch":epoch,"train_loss":total/len(train),"validation_loss":val_loss})
        pd.DataFrame(history).to_csv(out/f"{tag}_history.csv",index=False)
        if val_loss<best_loss-config["min_delta"]:
            best_loss,best_epoch,stale=val_loss,epoch,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_state,models/f"{tag}_weights.pt")
        else:
            stale+=1
        print(f"  {epoch:3d}: train={total/len(train):.5f}, validation={val_loss:.5f}",flush=True)
        if stale>=config["patience"]:
            break
    if best_state is None:
        raise RuntimeError("No valid checkpoint.")
    model.load_state_dict(best_state)
    print(f"Restored selected epoch {best_epoch}.",flush=True)
    # No test access during training or checkpoint selection.
    _,pv=evaluate(model,x,y,val,center,scale,config,device)
    _,pt=evaluate(model,x,y,test,center,scale,config,device)
    for name,ix,p in (("validation",val,pv),("test",test,pt)):
        frame=manifest.iloc[ix].copy()
        frame["probability_E2"]=p
        frame["predicted_label_E2"]=(p>=.5).astype(int)
        frame.to_csv(out/f"{tag}_{name}_predictions.csv",index=False)
    result={"cell_line":line,"modality":modality,"seed":config["seed"],
        "n_features":x.shape[1],"n_train":len(train),"n_validation":len(val),
        "best_epoch":best_epoch,"epochs_run":len(history),"best_validation_loss":best_loss,
        "training_seconds":time.monotonic()-started,**metrics(y[test],pt)}
    save_json(out/f"{tag}_validation_metrics.json",metrics(y[val],pv))
    save_json(out/f"{tag}_dummy_metrics.json",metrics(y[test],np.full(len(test),y[train].mean())))
    save_json(models/f"{tag}_model_spec.json",{
        "config":config,"cell_line":line,"modality":modality,"positive_class":"E2",
        "n_features":x.shape[1],"best_epoch":best_epoch,"split_sha256":sha256(out/f"{line}_splits.csv"),
        "transform":"log1p raw counts then train centering/scaling" if modality=="RNA" else "binary counts > 0",
        "weights":f"{tag}_weights.pt","preprocessing":f"{tag}_preprocessing.npz"})
    plot_diagnostics(pd.DataFrame(history),y[test],pt,result,out/f"{tag}_diagnostics.png",tag)
    save_json(out/f"{tag}_metrics.json",result)
    del model,optimizer,x,best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame([result])

def summarize(base,out,config):
    paths=[out/f"{line}_{m}_metrics.json" for line in config["cell_lines"] for m in ("RNA","ATAC")]
    missing=[p.name for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(f"Complete all four models first: {missing}")
    summary=pd.DataFrame([json.loads(p.read_text()) for p in paths])
    for line in config["cell_lines"]:
        a,b=[pd.read_csv(out/f"{line}_{m}_test_predictions.csv",dtype={"cell_id":str}) for m in ("RNA","ATAC")]
        if not a[["cell_id","label_E2"]].equals(b[["cell_id","label_E2"]]):
            raise ValueError("RNA/ATAC test predictions are not paired.")
    summary.to_csv(out/"metrics_summary.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric in zip(axes,("AUROC","average_precision")):
        summary.pivot(index="cell_line",columns="modality",values=metric).plot.bar(ax=ax)
        ax.set(ylim=(0,1.02),ylabel=metric,xlabel="",title=f"Test {metric}")
        ax.tick_params(axis="x",rotation=0)
    fig.tight_layout()
    fig.savefig(out/"unimodal_comparison.png",dpi=160)
    plt.show()
    plt.close(fig)
    exports=Path(base)/"results"/"exports"
    exports.mkdir(parents=True,exist_ok=True)
    archive=shutil.make_archive(str(exports/f"nb08_{out.name}"),"zip",root_dir=out)
    print("Share results:",archive)
    print("Weights/preprocessing remain in data/processed/nb08 for nb09.")
    return summary,Path(archive)
