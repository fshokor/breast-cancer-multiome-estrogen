# Breast cancer Multiome estrogen response

Setup, structural inspection, and paper-guided QC of the paired RNA + ATAC supplements from
[GSE154873](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE154873).
Organization follows the local multiomics-relationship-modeling reference project:
numbered scientific notebooks, Google Drive data paths, shared Python modules,
separate raw/processed data, and small reproducibility outputs in results/tables.

## Main biological question
**How does estrogen stimulation reshape transcriptional and chromatin-accessibility states in ER-positive breast-cancer cells?**

## Main multimodal question
**Does integrating RNA and chromatin accessibility improve prediction and representation of estrogen-response state beyond either modality alone?**

## Generalization question
**Do estrogen-response patterns learned in MCF-7 transfer to T47D?**

Future work will compare vehicle versus estrogen treatment using RNA differential
expression, pathways, ATAC differential accessibility, motifs and RNA–ATAC concordance.
Prediction will compare RNA only, ATAC only and RNA + ATAC, potentially using scMKL
and MultiVI. Exploratory RNA DEG and GSEA are implemented in nb04; nb05 adds exploratory ATAC analysis; predictive analyses remain future work.

## Local WSL workflow
Run from the repository root:

    conda env create -f environment.yml
    conda activate breast-cancer-multiome-estrogen
    python scripts/download_gse154873.py
    python scripts/inspect_gse154873.py

The environment installed during setup is isolated at .conda; activate it with

    conda activate /home/shoko/breast-cancer-multiome-estrogen/.conda

The local installation uses CPU PyTorch. requirements.txt is the portable package
specification; requirements-lock.txt and environment-lock.linux-64.txt record the
tested local versions. Recreate the local environment using the explicit Conda lock,
then install the pip lock with the CPU PyTorch index available:

    conda create -p .conda --file environment-lock.linux-64.txt
    .conda/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt

For optional TSV extraction:

    python scripts/download_gse154873.py --decompress

This retains original archives. Default inspection reads gzip directly.
Existing archives are validated before skipping; incomplete transfers never receive
the final filename. Move a corrupt archive aside before rerunning.
Raw data and environments are ignored by Git.

## Colab + Google Drive
Upload the available nb00–nb05 notebooks to Colab and run in order.
The notebooks contain generated copies of the canonical scripts, so a remote repository
and manual source-file uploads are unnecessary. They mount Drive and use:

    /content/drive/MyDrive/breast-cancer-multiome-estrogen

Downloads go directly to data/raw/GSE154873/; reports go to results/tables/.
Colab uses its managed Python version; local Python is 3.11.
Both notebooks require only the standard library for this milestone. nb00 includes
an optional installation cell for future scientific packages.

After editing canonical source, regenerate notebooks:

    python scripts/build_notebooks.py

## Contents
- DATA.md — source, study design, inspected file formats, dimensions and limitations
- configs/paths.py — local path conventions
- notebooks/nb00_colab_setup.ipynb — Drive setup and verified GEO download
- notebooks/nb01_data_inspection.ipynb — full structural inspection
- scripts/download_gse154873.py — standard-library downloader
- scripts/inspect_gse154873.py and src/data_inspection.py — streaming inspection
- results/tables/ — small tracked manifests, dimensions, inspection and environment records
- results/figures/ — saved notebook figures
- data/raw/ and data/processed/ — ignored data directories; see data/README.md

## Initial roadmap
| Notebook | Scope |
|---|---|
| nb00 | Colab setup + GEO download |
| nb01 | Data inspection and paired-cell verification |
| nb02 | Scanpy QC, paired preprocessing, RNA PCA/Leiden/UMAP |
| nb03 | Cluster markers, quality, sensitivity and cell-state review |
| nb04 | Exploratory E2 versus Vehicle RNA DEG + GSEA |
| nb05 | ATAC QC, LSI/UMAP, exploratory differential accessibility |
| nb06 | RNA–ATAC pathway comparison in both enrichment directions |
| nb07 | Matched-background motif enrichment, both ATAC directions and sensitivity |
| nb08 (future) | RNA treatment prediction |
| nb09 (future) | ATAC treatment prediction |
| nb10 (future) | scMKL multimodal prediction |
| nb11 (future) | MultiVI latent representation |
| nb12 (future) | Model comparison |
| nb13 (future) | Interpretability |
| nb14 (future) | MCF7 → T47D transfer |
| nb15 (future) | Final biological summary |

nb00 through nb07 are available. nb02 uses Scanpy for the published RNA QC thresholds and preprocessing, preserves
paired ATAC cells, and adds separate RNA PCA, Leiden clustering and UMAP per cell line.
See [QC_PREPROCESSING.md](QC_PREPROCESSING.md) for parameter provenance and results.
scMKL is not installed. nb03 ranks descriptive cluster markers. nb04 adds exploratory treatment DEG and GSEA without biological replicate mapping. nb07 adds exploratory motif analysis; predictive modeling remains future work.
The roadmap can change.

## Validation
Run python scripts/validate_setup.py in the Python 3.11 environment. It checks
package imports, notebook schemas, Python compilation, paths and URLs, exercises
the workflow tests, and executes both notebooks locally against the downloaded
files. The executed notebooks and results/tables/validation.json retain evidence.
The live Colab Drive authorization step is not tested by local execution.

## Scanpy QC and RNA visualization
Use notebooks/nb02_qc_preprocessing.ipynb after inspection. It follows the three
RNA QC thresholds in Kupp et al. (2025), DOI 10.1038/s42003-025-08533-7.
It also produces RNA PCA, Leiden clusters and UMAP with explicit exploratory settings.
See [the QC guide](QC_PREPROCESSING.md) for the distinction between the paper's
standard workflow and scMKL model preprocessing, saved layers, and reproducibility limits.

## Cluster characterization
Use [nb03](notebooks/nb03_cluster_characterization.ipynb) after nb02.
See [the characterization guide](CLUSTER_CHARACTERIZATION.md) for marker tests,
reference signatures, outputs and interpretation limits.

## Exploratory treatment DEG and GSEA
Use [nb04](notebooks/nb04_treatment_deg_gsea.ipynb) for within-line E2 versus Vehicle rankings and Hallmark GSEA.
See [the methods guide](TREATMENT_DEG_GSEA.md) for the replicate audit, statistical scope, settings and outputs.

## ATAC analysis
Use [nb05](notebooks/nb05_atac_analysis.ipynb) for ATAC-specific embeddings and depth-adjusted exploratory accessibility testing.
See [the ATAC guide](ATAC_ANALYSIS.md) for methods, outputs, and provisional coordinate-sensitive annotations.

## RNA and ATAC pathway comparison
Use [nb06](notebooks/nb06_rna_atac_integration.ipynb) to compare both RNA GSEA directions with provisional nearby ATAC evidence.
See [the integration guide](RNA_ATAC_INTEGRATION.md) for coverage, sensitivity analyses and interpretation limits.

## Motif analysis
Use [nb07](notebooks/nb07_motif_analysis.ipynb) for motifs in opening and closing peaks, with matched backgrounds and coordinate sensitivity.
See [the motif guide](MOTIF_ANALYSIS.md) for sequence resources, statistics and limits.
