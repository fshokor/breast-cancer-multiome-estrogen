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
and MultiVI. These downstream analyses are not implemented.

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
Upload only nb00_colab_setup.ipynb and nb01_data_inspection.ipynb to Colab and run in order.
Both notebooks contain generated copies of the canonical scripts, so a remote repository
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
- results/figures/ — reserved for future figures
- data/raw/ and data/processed/ — ignored data directories; see data/README.md

## Initial roadmap
| Notebook | Scope |
|---|---|
| nb00 | Colab setup + GEO download |
| nb01 | Data inspection and paired-cell verification |
| nb02 | Paper-guided QC and paired preprocessing |
| nb03 (future) | RNA differential expression |
| nb04 (future) | RNA GSEA |
| nb05 (future) | ATAC differential accessibility |
| nb06 (future) | ATAC regulatory/motif analysis |
| nb07 (future) | RNA–ATAC pathway concordance |
| nb08 (future) | RNA treatment prediction |
| nb09 (future) | ATAC treatment prediction |
| nb10 (future) | scMKL multimodal prediction |
| nb11 (future) | MultiVI latent representation |
| nb12 (future) | Model comparison |
| nb13 (future) | Interpretability |
| nb14 (future) | MCF7 → T47D transfer |
| nb15 (future) | Final biological summary |

nb00, nb01 and nb02 are available. nb02 applies the published RNA QC thresholds,
preserves paired ATAC cells, and writes sparse AnnData objects.
See [QC_PREPROCESSING.md](QC_PREPROCESSING.md) for parameter provenance and results.
scMKL is not installed. No downstream biological analysis or modeling has been run.
The roadmap can change.

## Validation
Run python scripts/validate_setup.py in the Python 3.11 environment. It checks
package imports, notebook schemas, Python compilation, paths and URLs, exercises
the workflow tests, and executes both notebooks locally against the downloaded
files. The executed notebooks and results/tables/validation.json retain evidence.
The live Colab Drive authorization step is not tested by local execution.

## Paper-guided QC
Use notebooks/nb02_qc_preprocessing.ipynb after inspection. It follows the three
RNA QC thresholds in Kupp et al. (2025), DOI 10.1038/s42003-025-08533-7.
See [the QC guide](QC_PREPROCESSING.md) for the distinction between the paper's
standard workflow and scMKL model preprocessing, saved layers, and reproducibility limits.
