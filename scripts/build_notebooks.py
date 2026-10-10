#!/usr/bin/env python3
"""Generate standalone Colab notebooks from the canonical Python source."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return dict(cell_type="markdown", metadata={}, source=text.strip().splitlines(True))


def code(text):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                source=text.strip().splitlines(True))


def save(name, cells):
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03}"
    nb = dict(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"name": name, "provenance": []}},
        nbformat=4, nbformat_minor=5)
    (ROOT / "notebooks" / name).write_text(json.dumps(nb, indent=2) + "\n")


MOUNT = """try:
    from google.colab import drive
except ImportError:
    IN_COLAB = False
    print("Local execution: using the repository root.")
else:
    IN_COLAB = True
    drive.mount('/content/drive')
"""
PATHS = """from pathlib import Path
import os

if IN_COLAB:
    BASE = Path("/content/drive/MyDrive/breast-cancer-multiome-estrogen")
else:
    BASE = Path(os.environ.get("MULTIOME_PROJECT_ROOT", Path.cwd())).resolve()
    if BASE.name == "notebooks":
        BASE = BASE.parent
    if not (BASE / "environment.yml").is_file():
        raise RuntimeError("Run locally from the repository root or set MULTIOME_PROJECT_ROOT.")

DATA_DIR = BASE / "data" / "raw" / "GSE154873"
RESULTS_DIR = BASE / "results"
for folder in (DATA_DIR, BASE / "data/processed", RESULTS_DIR / "figures", RESULTS_DIR / "tables"):
    folder.mkdir(parents=True, exist_ok=True)
print("BASE:", BASE.resolve())
print("DATA_DIR:", DATA_DIR.resolve())
print("RESULTS_DIR:", RESULTS_DIR.resolve())
"""
downloader = (ROOT / "scripts/download_gse154873.py").read_text().split('\nif __name__ == "__main__":')[0]
inspection = (ROOT / "src/data_inspection.py").read_text()
packages = (ROOT / "requirements.txt").read_text().split()
save("nb00_colab_setup.ipynb", [
    markdown("""# nb00 — Colab setup + GEO download
Mount Drive, create the project folders, and acquire only the six processed paired Multiome files.
This notebook is standalone: no GitHub remote or additional source uploads are required.
Download functions below are generated from scripts/download_gse154873.py.
No biological analysis is performed. In Colab, open the notebook and run the cells in order."""),
    code(MOUNT), code(PATHS),
    markdown("""## Environment
Download and inspection require only Python's standard library.
The local Conda environment uses Python 3.11. Colab controls its own Python version.
Optional installation below prepares the future analysis packages; it does not run any analysis.
For exact local package versions, see requirements-lock.txt and environment-lock.linux-64.txt."""),
    code("import sys, subprocess\nprint(sys.version)\nINSTALL_ANALYSIS_PACKAGES = False\n"
         f"PACKAGES = {packages!r}\n"
         "if INSTALL_ANALYSIS_PACKAGES:\n    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *PACKAGES])"),
    markdown("""## Shared downloader
Downloads use atomic temporary files, bounded retries, expected transfer lengths, gzip CRC validation,
and SHA-256 receipts. Existing valid archives are reused; invalid archives raise a clear error.
Local SHA-256 hashes record provenance; GEO does not supply reference SHA-256 values here.
Keep DECOMPRESS=False to avoid writing approximately 6.35 GB of text. Archives are always retained."""),
    code(downloader),
    code("DECOMPRESS = False\nmanifest = download_all(DATA_DIR, decompress=DECOMPRESS)"),
    code("""import json
receipt = RESULTS_DIR / "tables" / "download_manifest.json"
receipt.write_text(json.dumps(manifest, indent=2) + "\\n")
for key, item in manifest["files"].items():
    path = DATA_DIR / item["filename"]
    assert path.is_file() and path.stat().st_size == item["compressed_bytes"]
    print(key, f'{item["compressed_bytes"]:,} compressed bytes', item["sha256"])
print("All six files verified. Next: nb01_data_inspection.ipynb")
""")
])
save("nb01_data_inspection.ipynb", [
    markdown("""# nb01 — Data inspection and paired-cell verification
## Scientific question
**What exactly is contained in the processed GSE154873 Multiome files, and how are RNA,
ATAC, treatment labels, cell lines and cell barcodes organized?**
Run nb00 first. This notebook scans every matrix entry one row at a time.
It does not filter cells, transpose files, create AnnData, normalize counts, or fit models."""),
    code(MOUNT), code(PATHS),
    markdown("""## Shared inspection functions
Generated from src/data_inspection.py. Metadata are inspected before counts.
Header presence and barcode columns are detected from file contents.
These downloaded metadata are headerless: column_1 and column_2 are assigned positional
names, not GEO-provided headers. Full sample-prefixed barcodes are preserved."""),
    code(inspection),
    code("""metadata_by_line = {}
for line in ("MCF7", "T47D"):
    metadata_by_line[line], barcodes = inspect_metadata(DATA_DIR / f"GSE154873_{line}_multiome_metadata.tsv.gz")
    print(line, json.dumps(metadata_by_line[line], indent=2))
"""),
    markdown("""## Complete matrix scan and pairing audit
All entries are checked for finite, nonnegative integer values and exact zero fraction.
Orientation is established by identifiers, not assumed. Missing identifiers are reported in both
directions for RNA–ATAC, RNA–metadata, and ATAC–metadata. No mismatches are discarded.
Scanning the four dense-text TSVs takes several minutes, especially from Google Drive."""),
    code('report = inspect_dataset(DATA_DIR, RESULTS_DIR / "tables")'),
    code("""for line, dataset in report["cell_lines"].items():
    print("\\n", line)
    for modality in ("rna", "atac"):
        matrix = dataset[modality]
        print(modality.upper(), json.dumps(matrix, indent=2))
    print("Pairing:", json.dumps(dataset["pairing"], indent=2))
    if not (dataset["pairing"]["exact_sets_equal"] and dataset["pairing"]["all_identifiers_unique"]):
        print("STOP: resolve missing or duplicated cell identifiers before preprocessing.")
"""),
    markdown("""## Interpretation and boundary
See DATA.md for the verified study design and exact inspection results.
The observed labels are YW_si_Control_plus_E2 and YW_si_Control_plus_Veh.
The paper identifies a 48-hour estrogen exposure; duration is not a metadata column.
There are no QC, cluster, state, replicate or donor annotations in these two-column metadata.
Integer values are consistent with counts; integer validation alone does not prove upstream processing history.
Do not treat cells as independent biological replicates in later inference.
Next proposed notebook: nb02_qc_preprocessing.ipynb. Stop here for this milestone.""")
])

if __name__ == "__main__":
    print("Generated nb00 and nb01.")
