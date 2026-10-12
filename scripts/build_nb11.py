"""Assemble nb11; never execute its cells."""
from pathlib import Path
import hashlib,json
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/"src/pathway_mlp.py").read_text()
config=json.loads((ROOT/"configs/pathway_mlp.json").read_text())
hallmark=json.loads((ROOT/"configs/hallmark_v7.json").read_text())
annotation=json.loads((ROOT/"configs/atac_reference.json").read_text())
references=json.loads((ROOT/"configs/motif_references.json").read_text())
blacklist=references["resources"]["hg38-blacklist.v2.bed.gz"]
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
cells=[
md("""# nb11 v2 — Matched sparse pathway models: RNA, ATAC and RNA+ATAC

**PyTorch · three input modes per cell line · six fits total · intentionally unexecuted**

This notebook implements the proposed pathway model. It does not use scMKL kernel matrices.
It reuses the frozen Hallmark gene sets, raw paired data and **exact nb08 train/validation/test cells**.

RNA genes → masked RNA pathway nodes ↘  
　　　　　　　　　　　　　　　　　pathway-specific fusion → sparse pathway gates → small MLP → P(E2)  
ATAC gene accessibility → masked ATAC pathway nodes ↗

Each supported Hallmark pathway has four latent nodes per active modality. RNA-only and ATAC-only\nuse the same encoder, fusion output size, sparse gates and classifier as RNA+ATAC. The missing\nbranch is fixed to zero and has no encoder or access to its input values. Total allocated parameter counts\ntherefore differ; we match the architecture recipe rather than force equal parameter counts.\nThe common pathway universe and feature preprocessing are shared for this controlled comparison.\n\nEach supported Hallmark pathway has four latent nodes per modality. A gene connects only to
pathways containing it. Each pathway fuses its own RNA and ATAC representations; the classifier
combines the gated pathway representations. Gates use clipped Gaussian noise during training,
deterministic values during evaluation, and an expected-L0 penalty.

This is an independently implemented research model, not a reproduction of scMKL or P-NET.
Defaults are starting settings, not parameters shown to be optimal in the paper.

## Before running in Colab
1. Select a GPU runtime (CPU works, but may be slower).
2. Keep the project on Google Drive, including the four processed nb02 H5AD files and the complete
   nb08 results folder, with its split manifests, input hashes and test predictions.
3. Check BASE and NB08_RUN_ID below. The default nb08 ID is the earlier run discussed in this project.
4. Run the cells in order. The notebook contains its implementation and reference specifications;
   no repository checkout is required in Colab.
5. **Start a NEW run for v2. Old v1 checkpoints are incompatible.** To resume a v2 run, put its printed nb11 run ID in RESUME_RUN_ID and rerun from setup.
   Keep configuration and implementation unchanged. Completed preprocessing is reused; an interrupted
   preprocessing step restarts for that cell line. Training resumes after the last completed epoch.

Results and checkpoints are saved to Drive under results/tables/nb11/RUN_ID and
data/processed/nb11/RUN_ID. Do not create a new run ID if you want to resume.
"""),
md("""## Inputs, leakage controls and interpretation
- Attempt all 50 frozen Hallmark sets. Retain a pathway only with at least three eligible genes in
  each modality; report coverage and exclusions. No DEG, GSEA NES or nb06 significance filtering.
- RNA: log1p raw counts. ATAC: binary peak accessibility mapped to promoter/gene-body intervals
  from frozen hg38 RefSeq transcripts. Exclude blacklist/nonstandard peaks and coordinate-sensitive
  links. Multi-gene peaks contribute fractional weights.
- ATAC gene scores are weighted accessible fractions, corrected by cell peak depth relative to
  the training median, then log1p transformed. They are custom accessibility proxies, not measured
  gene expression or an exact ArchR gene-score implementation.
- Detection/variance filtering, mean, standard deviation and depth reference use **training cells only**.
  The existing full-cohort QC is inherited. This preprocessing differs from nb08, so any gain is a
  pipeline comparison, not proof that architecture alone caused the gain.
- Validation BCE selects the best epoch from epoch 1. Constant sparsity regularization applies\n  from the first epoch. No early validation optimum is excluded.
  Test cells never select epochs, pathways or attribution examples.
- Ablations and gene attribution use validation cells. They measure dependence of this fitted model,
  not causality. Redundant pathways and off-distribution ablations can affect interpretation.
- Positive/negative GSEA NES describes enrichment direction. A gate is unsigned predictive selection;
  neither a large gate nor ATAC accessibility proves a pathway is active. nb06 provides descriptive
  context, not independent validation.
- Smooth bounded tanh layers replace the previous repeated layer normalization. Gene IG uses\n  adaptive Gauss-Legendre integration (32 to 256 points), completeness checks on every sampled cell\n  and stability between successive resolutions. Failed pathway attributions are withheld.\n- Peak-to-gene links remain provisional. Exported candidate regions are **not peak-level attribution**.
  Gene IG is conditional on one pathway while other pathway representations are held fixed.
- Cell-level splits lack biological-replicate validation; library/treatment confounding can remain.
  The previously inspected test set cannot support an unbiased claim after test-guided tuning.

Architecture references: [P-NET](https://www.nature.com/articles/s41586-021-03922-4) and
[stochastic gates](https://proceedings.mlr.press/v119/yamada20a.html).
"""),
md("## 1. Dependencies and persistent project folder"),
code("""# Installation only; training starts later.
%pip -q install "anndata>=0.10,<0.13" "numpy>=1.26,<3" "pandas>=2,<3" "scipy>=1.11,<2" "h5py>=3.10,<4" "scikit-learn>=1.4,<2" "matplotlib>=3.8,<4" "tqdm>=4.66,<5" "torch>=2.6,<3" ipywidgets"""),
code("""from pathlib import Path
import os
try:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE=Path("/content/drive/MyDrive/breast-cancer-multiome-estrogen")
except ImportError:
    BASE=Path(os.environ.get("MULTIOME_PROJECT_ROOT", "/home/shoko/breast-cancer-multiome-estrogen"))
print("Project:", BASE)
"""),
md("## 2. Settings — one fixed configuration, no hyperparameter sweep"),
code("import json\nfrom datetime import datetime, timezone\n"
     'NB08_RUN_ID = "20261011T103750_408739Z"\nRESUME_RUN_ID = ""  # Paste an existing nb11 run ID to resume.\n'
     'RUN_ID = RESUME_RUN_ID or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")\n'
     "CONFIG = "+repr(config)+"\n"
     "print('nb11 RUN_ID:', RUN_ID)\nprint(json.dumps(CONFIG, indent=2))"),
md("""Defaults: up to 100 epochs, batch 128, model LR 0.0003, gate LR 0.003.
All three modes use head width 16, dropout 0.3, weight decay 0.02 and gate penalty 0.005
from epoch 1. Gates start at 0.25 with noise sigma 0.25. These are revised starting settings,
not established optimal values. Stronger regularization may underfit; inspect validation curves.
Every epoch is eligible for selection. Stop after at least 10 epochs and patience 10.
Progress includes epoch/batch bars, elapsed time and early-stopping state. Each model has its own
atomic checkpoint; completed models load their best weights on resume.
An expected open count is not a count of exact zeros: the notebook reports both.
Do not change settings based on test results.""" ),
md("## 3. Embedded implementation and frozen references"),
code(source),
code("HALLMARK = "+repr(hallmark)+"\nANNOTATION = "+repr(annotation)+"\nBLACKLIST = "+repr(blacklist)+"\nSOURCE_SHA = "+repr(hashlib.sha256(source.encode()).hexdigest())),
md("## 4. Verify paired inputs and reuse the nb08 splits"),
code("""BASELINE, OUTPUT, ARTIFACTS, MANIFESTS = start_run(
    BASE, NB08_RUN_ID, RUN_ID, CONFIG, HALLMARK, ANNOTATION, BLACKLIST, SOURCE_SHA,
    resume=bool(RESUME_RUN_ID))
print("Results:", OUTPUT)
print("Prepared data and checkpoints:", ARTIFACTS)
display(pd.concat([m.groupby(["cell_line","split","treatment"]).size().rename("cells").reset_index()
                   for m in MANIFESTS.values()],ignore_index=True))
COMPARISONS = []
""")]

for line in ("MCF7","T47D"):
    cells += [
        md(f"## {line}: prepare shared pathway features\nAll modes share the same features, masks and train-only scaling. Coverage is restricted to pathways supported in both modalities. This is a matched control, not a separately optimized unimodal pipeline."),
        code(f'DATA = prepare_line(BASE, "{line}", MANIFESTS["{line}"], CONFIG, HALLMARK, ANNOTATION, BLACKLIST, OUTPUT, ARTIFACTS)\n'
             'print("RNA:", DATA["rna"].shape, "ATAC:", DATA["atac"].shape, "pathways:", len(DATA["structure"]["pathways"]))\n'
             f'display(pd.read_csv(OUTPUT/"{line}_pathway_coverage.csv"))')]
    for mode in ("RNA","ATAC","RNA_ATAC"):
        cells += [
            md(f"### {line} — {mode}: train and select by validation BCE\nOne fit, no hyperparameter sweep. Saves its own checkpoint after each completed epoch."),
            code(f'MODEL = train_model(DATA, "{line}", CONFIG, OUTPUT, "{mode}")\n'
                 f'SPARSITY = summarize_model(MODEL, DATA, "{line}", OUTPUT)'),
            md("Evaluate the selected checkpoint. The baseline predictions are reused; nb08 is not retrained."),
            code(f'COMPARISON = evaluate_model(MODEL, DATA, "{line}", CONFIG, OUTPUT, BASELINE)\n'
                 f'COMPARISON.to_csv(OUTPUT/"{line}_{mode}"/"comparison.csv", index=False)\ndisplay(COMPARISON)'),
            md("Validation ablations measure fitted-model dependence. Removing a whole modality is not equivalent to separately training a unimodal model."),
            code(f'ABLATIONS = pathway_ablation(MODEL, DATA, "{line}", CONFIG, OUTPUT)\n'
                 'display(ABLATIONS.sort_values("AUROC_drop",ascending=False).head(15))'),
            md("Checked within-pathway gene attribution. Other pathway latents remain fixed; zero standardized inputs are the training-mean baseline. Rankings are exported only when every sampled cell passes completeness and resolution-stability checks. These checks establish numerical reliability, not biological causality."),
            code(f'GENE_ATTRIBUTION = gene_attribution(MODEL, DATA, "{line}", CONFIG, OUTPUT, ABLATIONS)\n'
                 'display(GENE_ATTRIBUTION.sort_values("mean_absolute_IG",ascending=False).head(20))'),
            code('del MODEL\ngc.collect()\nif torch.cuda.is_available(): torch.cuda.empty_cache()')
        ]
    cells.append(code("del DATA\ngc.collect()"))
cells += [
md("""## Compare prediction, sparsity and biological dependence
Compare each structured unimodal model with its nb08 counterpart, then compare RNA+ATAC against
both matched structured controls. The pathway-ablation tables compare the same named pathways
across modes. Exact-zero/nonzero gates, ablation effects and numerically checked gene attribution
describe interpretability of the fitted structured models. Gates alone do not validate explanations.

nb08 provides predictive baselines, not equivalent attribution/stability measurements. Thus this
notebook can show a more explicit biological organization and whether its pathways affect prediction;
it cannot establish superior explanation fidelity over nb08 without a matched nb08 explanation audit.
A single seed also cannot establish stable pathway selection. Differences from nb08 include feature
preprocessing, regularization and capacity, not only pathway structure."""),
code("""COMPARISON, SPARSITY = export_matched_review(OUTPUT)
display(COMPARISON[COMPARISON["split"].eq("test")])
display(SPARSITY)
RESULTS_ZIP = Path(shutil.make_archive(str(OUTPUT.parent/(RUN_ID+"_results")), "zip", root_dir=OUTPUT))
print("Results ZIP:", RESULTS_ZIP)
print("Checkpoint folder:", ARTIFACTS)
"""),
md("""## Biological follow-up
Compare estrogen-response and other selected pathways with both positive and negative GSEA NES
and nb06 RNA/ATAC evidence, retaining discordant results. Such concordance is descriptive because
the cohort is shared. ATAC-derived gene scores and candidate linked regions are provisional;
region candidates are not peak-level attributions. No patient/replicate generalization is established.
Do not tune on the test scores; earlier results were already inspected.
""")]
nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11"},"project":{"notebook":"nb11","intentionally_unexecuted":True}})
nbf.validate(nb)
target=ROOT/"notebooks/nb11_pathway_multimodal_mlp.ipynb"
nbf.write(nb,target)
print(target)
