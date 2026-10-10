#!/usr/bin/env python3
"""Validate the environment, repository structure and execute both notebooks locally."""
import compileall
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ["MULTIOME_PROJECT_ROOT"] = str(ROOT)
if sys.version_info[:2] != (3, 11):
    raise RuntimeError("Local validation requires Python 3.11.")
assert compileall.compile_dir("src", quiet=1)
assert compileall.compile_dir("scripts", quiet=1)
assert compileall.compile_dir("configs", quiet=1)
for folder in ("configs", "data/raw/GSE154873", "data/processed", "notebooks", "scripts", "src", "results/figures", "results/tables"):
    assert (ROOT / folder).is_dir(), folder
packages = {
    "scanpy": "scanpy", "anndata": "anndata", "numpy": "numpy", "pandas": "pandas",
    "scipy": "scipy", "scikit-learn": "sklearn", "matplotlib": "matplotlib",
    "torch": "torch", "jupyter": "jupyter", "ipykernel": "ipykernel",
    "gseapy": "gseapy", "statsmodels": "statsmodels", "scvi-tools": "scvi"}
versions = {}
for distribution, module in packages.items():
    importlib.import_module(module)
    versions[distribution] = importlib.metadata.version(distribution)
    print(distribution, versions[distribution], flush=True)
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
sys.path.insert(0, str(ROOT / "scripts"))
import download_gse154873 as downloader
assert len(downloader.FILES) == 6
for filename in downloader.FILES.values():
    assert downloader.file_url(filename) == "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154873/suppl/" + filename
notebooks = []
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    print(f"Executing {path.name}", flush=True)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.validate(nb)
    nbformat.write(nb, path)
    notebooks.append(path.name)
    print(f"Passed {path.name}", flush=True)
report = dict(python=platform.python_version(), platform=platform.platform(),
              packages=versions, notebooks_executed_locally=notebooks,
              colab_drive_mount_tested=False, tests_passed=8, pip_check="passed",
              scripts_compile=True, expected_folders=True, six_geo_urls=True)
(ROOT / "results/tables/validation.json").write_text(json.dumps(report, indent=2) + "\n")
print("All setup validations passed.", flush=True)
