"""Scanpy QC, paired preprocessing, and exploratory RNA PCA/Leiden/UMAP."""
import gc
import gzip
import hashlib
import json
import warnings
import copy
import importlib.metadata
import scanpy as sc
from array import array
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_sparse_counts(path):
    """Parse one dense-text row at a time, then transpose verified features x cells."""
    values, indices, indptr = array("i"), array("i"), array("q", [0])
    features = []
    with gzip.open(path, "rt") as f:
        cells = f.readline().rstrip("\r\n").split("\t")
        if not cells or any(not cell for cell in cells) or len(set(cells)) != len(cells):
            raise ValueError(f"Missing or duplicate cell IDs in {path}")
        for line_number, line in enumerate(f, start=2):
            feature, sep, text = line.rstrip("\r\n").partition("\t")
            if not sep or text.count("\t") + 1 != len(cells):
                raise ValueError(f"Unexpected row width at {path}:{line_number}")
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                row = np.fromstring(text, sep="\t", dtype=np.float64)
            if (row.size != len(cells) or not np.isfinite(row).all() or
                    (row < 0).any() or (row != np.floor(row)).any() or
                    (row > np.iinfo(np.int32).max).any()):
                raise ValueError(f"Invalid integer counts at {path}:{line_number}")
            nz = np.flatnonzero(row)
            values.frombytes(row[nz].astype(np.int32).tobytes())
            indices.frombytes(nz.astype(np.int32).tobytes())
            indptr.append(len(values))
            features.append(feature)
            if len(features) % 25000 == 0:
                print(f"  {Path(path).name}: {len(features):,} features", flush=True)
    if not features or len(features) != len(set(features)) or any(not x for x in features):
        raise ValueError(f"Missing or duplicated feature IDs: {path}")
    matrix = sparse.csr_matrix((np.frombuffer(values, dtype=np.int32),
                               np.frombuffer(indices, dtype=np.int32),
                               np.frombuffer(indptr, dtype=np.int64)),
                              shape=(len(features), len(cells))).T.tocsr()
    return matrix, pd.Index(cells, name="cell_id"), pd.Index(features, name="feature_id")

def read_metadata(path, expected_cells, cell_line):
    frame = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    if frame.shape[1] != 2:
        raise ValueError("Expected the inspected two-column headerless metadata.")
    frame.columns = ["cell_id", "sample_label"]
    if frame.cell_id.duplicated().any() or not frame.cell_id.tolist() == list(expected_cells):
        raise ValueError("Metadata cells/order do not exactly match the count matrix.")
    frame = frame.set_index("cell_id")
    mapping = {"YW_si_Control_plus_E2": "E2", "YW_si_Control_plus_Veh": "Vehicle"}
    if not set(frame.sample_label).issubset(mapping):
        raise ValueError("Unrecognized treatment labels.")
    frame["treatment"] = frame.sample_label.map(mapping)
    frame["cell_line"] = cell_line
    frame["global_cell_id"] = cell_line + ":" + frame.index
    return frame


def rna_qc(counts, genes, config):
    """Use Scanpy QC and filter APIs, preserving the documented gene-first order."""
    original = ad.AnnData(X=counts, var=pd.DataFrame(index=genes))
    original.var["mt"] = original.var_names.str.startswith(config["mitochondrial_prefix"])
    if not original.var["mt"].any():
        raise ValueError("No mitochondrial genes detected; verify gene naming before QC.")
    sc.pp.calculate_qc_metrics(original, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    gene_keep, gene_detected = sc.pp.filter_genes(
        original, min_cells=config["min_cells_per_gene"], inplace=False)
    selected = original[:, gene_keep].copy()
    sc.pp.calculate_qc_metrics(selected, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    keep_genes, _ = sc.pp.filter_cells(
        selected, min_genes=config["min_genes_per_cell"], inplace=False)
    pct_mt = selected.obs["pct_counts_mt"].to_numpy()
    keep_mt = np.isfinite(pct_mt) & (pct_mt <= config["max_pct_mitochondrial"])
    qc = pd.DataFrame({
        "rna_total_counts_before_gene_filter": original.obs["total_counts"].to_numpy(),
        "rna_n_genes_before_gene_filter": original.obs["n_genes_by_counts"].to_numpy(),
        "rna_total_counts": selected.obs["total_counts"].to_numpy(),
        "rna_n_genes": selected.obs["n_genes_by_counts"].to_numpy(),
        "rna_pct_counts_mt": pct_mt,
        "pass_min_genes": keep_genes, "pass_max_pct_mt": keep_mt,
        "pass_rna_qc": keep_genes & keep_mt})
    var = pd.DataFrame({"n_cells_detected_before_qc": gene_detected,
                        "pass_min_cells": gene_keep,
                        "mt": original.var["mt"].to_numpy()}, index=genes)
    return keep_genes & keep_mt, gene_keep, qc, var


def sparse_log_and_scale(counts, ddof=1):
    """Use Scanpy log1p/scale; store centering offsets to keep all-gene data sparse."""
    if ddof != 1:
        raise ValueError("This workflow uses Scanpy's sample-standard-deviation scaling.")
    if counts.shape[0] <= 1:
        raise ValueError("Too few cells to estimate scaling variance.")
    transformed = ad.AnnData(X=counts.astype(np.float64))
    sc.pp.log1p(transformed)
    log = transformed.X.copy()
    sc.pp.scale(transformed, zero_center=False, max_value=None)
    if not np.isfinite(transformed.X.data).all():
        raise ValueError("Scanpy scaling returned nonfinite values.")
    return log.astype(np.float32), transformed.X.astype(np.float32).tocsr(), transformed.var["mean"].to_numpy(), transformed.var["std"].to_numpy()

def centered_scaled_block(adata, rows, columns):
    """Materialize only an explicitly requested small block of centered z-scores."""
    block = adata.layers["scaled_no_center"][rows, :][:, columns]
    if block.shape[0] * block.shape[1] > 5_000_000:
        raise ValueError("Request a smaller block; full centered matrices are not materialized.")
    offset = adata.var["scale_center_offset"].to_numpy()[columns]
    return block.toarray() - offset

def save_h5ad(adata, path):
    path = Path(path)
    temp = path.with_name(path.stem + ".partial.h5ad")
    adata.write_h5ad(temp, compression="gzip")
    check = ad.read_h5ad(temp, backed="r")
    try:
        if check.shape != adata.shape or not check.obs_names.equals(adata.obs_names):
            raise ValueError(f"Written AnnData validation failed: {path}")
    finally:
        check.file.close()
    temp.replace(path)

def qc_figure(qc, cell_line, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, col, title in zip(axes, ["rna_n_genes", "rna_pct_counts_mt", "atac_n_peaks"],
                             ["Detected RNA genes", "Mitochondrial RNA (%)", "Detected ATAC peaks"]):
        edges = np.histogram_bin_edges(qc[col].dropna(), bins=50)
        ax.hist(qc[col], bins=edges, color="#aebbc9", label="All supplied cells")
        ax.hist(qc.loc[qc.pass_rna_qc, col], bins=edges, histtype="step", linewidth=1.8,
                color="#176b63", label="RNA QC retained")
        ax.set_title(title)
        ax.set_ylabel("Cells")
    axes[0].axvline(config["min_genes_per_cell"], color="#bd4b35", linestyle="--")
    axes[1].axvline(config["max_pct_mitochondrial"], color="#bd4b35", linestyle="--")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{cell_line}: paper-guided QC; ATAC follows the same cell mask")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def visualize_rna(rna, root, cell_line, config):
    """Fit RNA-only descriptive embeddings; treatment labels do not enter fitting."""
    settings = config["visualization"]
    seed = settings["random_state"]
    viz = ad.AnnData(X=rna.layers["counts"].astype(np.float32), obs=rna.obs.copy(),
                    var=pd.DataFrame(index=rna.var_names.copy()))
    sc.pp.normalize_total(viz, target_sum=settings["normalize_total_target_sum"])
    sc.pp.log1p(viz)
    sc.pp.highly_variable_genes(viz, n_top_genes=settings["n_top_genes"],
                               flavor=settings["hvg_flavor"], subset=False)
    mask = viz.var["highly_variable"].to_numpy()
    rna.var["highly_variable"] = mask
    viz.var.to_csv(Path(root)/"results/tables"/f"{cell_line}_hvg.tsv",
                   sep="\t", lineterminator="\n")
    viz = viz[:, mask].copy()
    viz.layers["counts"] = rna.layers["counts"][:, mask].copy()
    viz.layers["log1p_normalized"] = viz.X.copy()
    # Only the ~2,000-HVG visualization subset is centered/densified.
    viz.X = viz.X.astype(np.float64)
    sc.pp.scale(viz, zero_center=True, max_value=settings["scale_max_value"])
    if not np.isfinite(viz.X).all():
        raise ValueError("Scanpy HVG scaling returned nonfinite values.")
    sc.pp.pca(viz, n_comps=settings["pca_components"], svd_solver="arpack",
              random_state=seed, mask_var=None)
    sc.pp.neighbors(viz, n_neighbors=settings["n_neighbors"],
                    n_pcs=settings["neighbors_n_pcs"], use_rep="X_pca",
                    transformer=settings["neighbors_transformer"], random_state=seed)
    sc.tl.leiden(viz, resolution=settings["leiden_resolution"], random_state=seed,
                 flavor=settings["leiden_flavor"], n_iterations=settings["leiden_n_iterations"],
                 directed=False, key_added="leiden")
    sc.tl.umap(viz, min_dist=settings["umap_min_dist"], spread=settings["umap_spread"],
               maxiter=settings["umap_maxiter"], random_state=seed)
    viz.uns["visualization"] = copy.deepcopy(settings)
    viz.uns["visualization"]["interpretation"] = "RNA-derived exploratory clusters, not cell types or predictions"
    rna.obs["leiden"] = viz.obs["leiden"].copy()
    rna.obsm["X_pca"] = viz.obsm["X_pca"].copy()
    rna.obsm["X_umap"] = viz.obsm["X_umap"].copy()
    rna.varm["PCs"] = np.zeros((rna.n_vars, viz.varm["PCs"].shape[1]), dtype=np.float32)
    rna.varm["PCs"][mask] = viz.varm["PCs"]
    for key in ("pca", "neighbors", "umap", "leiden", "visualization"):
        rna.uns[key] = copy.deepcopy(viz.uns[key])
    for key in ("connectivities", "distances"):
        rna.obsp[key] = viz.obsp[key].copy()
    tables, figures = Path(root)/"results/tables", Path(root)/"results/figures"
    summary = pd.crosstab(viz.obs["leiden"], viz.obs["treatment"], dropna=False)
    summary.to_csv(tables/f"{cell_line}_cluster_treatment_counts.tsv",sep="\t",lineterminator="\n")
    embedding = viz.obs[["treatment","leiden"]].copy()
    embedding["UMAP1"], embedding["UMAP2"] = viz.obsm["X_umap"].T
    for i in range(viz.obsm["X_pca"].shape[1]):
        embedding[f"PC{i+1}"] = viz.obsm["X_pca"][:, i]
    embedding.to_csv(tables/f"{cell_line}_rna_embeddings.tsv",sep="\t",lineterminator="\n")
    save_h5ad(viz, Path(root)/"data/processed"/f"{cell_line}_RNA_visualization.h5ad")
    import matplotlib.pyplot as plt
    sc.set_figure_params(dpi=100, dpi_save=160, fontsize=10, frameon=False)
    colors = ["treatment", "leiden", "rna_total_counts", "rna_pct_counts_mt"]
    labels = ["Treatment", "Leiden cluster", "RNA counts", "Mitochondrial RNA (%)"]
    for name, plot in (("pca",sc.pl.pca), ("umap",sc.pl.umap)):
        fig = plot(viz, color=colors, title=[f"{cell_line}: {x}" for x in labels],
                   ncols=2, show=False, return_fig=True, size=8, wspace=0.35)
        fig.savefig(figures/f"{cell_line}_rna_{name}.png",dpi=160,bbox_inches="tight")
        plt.close(fig)
    sc.pl.pca_variance_ratio(viz, n_pcs=settings["pca_components"], log=False, show=False)
    plt.gcf().savefig(figures/f"{cell_line}_pca_variance.png",dpi=160,bbox_inches="tight")
    plt.close()
    result = {"n_hvgs": int(mask.sum()), "n_pcs": int(viz.obsm["X_pca"].shape[1]),
              "n_clusters": int(viz.obs["leiden"].nunique()),
              "cluster_sizes": {str(k):int(v) for k,v in viz.obs["leiden"].value_counts().items()},
              "pca_variance_first_30": float(viz.uns["pca"]["variance_ratio"][:settings["neighbors_n_pcs"]].sum()),
              "parameters": settings}
    del viz
    gc.collect()
    return result

def preprocess_line(root, cell_line, config):
    root = Path(root)
    raw = root / "data/raw/GSE154873"
    output = root / "data/processed"
    tables, figures = root / "results/tables", root / "results/figures"
    for folder in (output, tables, figures):
        folder.mkdir(parents=True, exist_ok=True)
    prefix = f"GSE154873_{cell_line}_multiome_"
    sources = {name: raw / (prefix + suffix + ".tsv.gz") for name, suffix in
               (("rna", "RNA_counts"), ("atac", "ATAC_counts"), ("metadata", "metadata"))}
    manifest = json.loads((raw / "download_manifest.json").read_text())
    source_hashes = {}
    for name, path in sources.items():
        source_hashes[name] = file_hash(path)
        if source_hashes[name] != manifest["files"][f"{cell_line.lower()}_{name}"]["sha256"]:
            raise ValueError(f"Input differs from inspected download: {path}")
    print(f"\n{cell_line}: reading RNA", flush=True)
    counts, cells, genes = read_sparse_counts(sources["rna"])
    expected = config["expected"][cell_line]
    if counts.shape != (expected["cells"], expected["rna_features"]):
        raise ValueError("RNA dimensions differ from inspected paper dataset.")
    obs = read_metadata(sources["metadata"], cells, cell_line)
    cell_keep, gene_keep, qc, var = rna_qc(counts, genes, config)
    qc.index = cells
    qc = obs.join(qc)
    if not cell_keep.any():
        raise ValueError("QC retained no cells.")
    kept_counts = counts[cell_keep, :][:, gene_keep].tocsr()
    del counts
    log, scaled, mean, std = sparse_log_and_scale(kept_counts, config["scaling_ddof"])
    rna = ad.AnnData(X=log, obs=qc.loc[cell_keep].copy(), var=var.loc[gene_keep].copy())
    rna.layers["counts"] = kept_counts
    rna.layers["scaled_no_center"] = scaled
    rna.var["log1p_mean"] = mean
    rna.var["log1p_std"] = std
    rna.var["scale_center_offset"] = mean / std
    rna.uns["preprocessing"] = {
        "reference_doi": config["reference_doi"], "config_json": json.dumps(config),
        "X": "log1p(raw counts), no library-size normalization",
        "counts": "Unmodified integer counts for retained cells and genes",
        "centered_scale": "scaled_no_center - var.scale_center_offset",
        "scale_scope": "descriptive per cell line; refit inside future training folds",
        "filter_order": config["filter_order"], "source_sha256_json": json.dumps(source_hashes)}
    rna.uns["log1p"] = {"base": None}
    print(f"{cell_line}: Scanpy HVGs, PCA, neighbors, Leiden and UMAP", flush=True)
    visualization = visualize_rna(rna, root, cell_line, config)
    qc["leiden"] = pd.Series(rna.obs["leiden"].astype(str), index=rna.obs_names).reindex(qc.index).fillna("excluded")
    rna_path = output / f"{cell_line}_RNA_paper_qc.h5ad"
    save_h5ad(rna, rna_path)
    del rna, kept_counts, log, scaled
    gc.collect()
    print(f"{cell_line}: reading paired ATAC", flush=True)
    atac_counts, atac_cells, peaks = read_sparse_counts(sources["atac"])
    if atac_counts.shape != (expected["cells"], expected["atac_features"]) or not atac_cells.equals(cells):
        raise ValueError("ATAC dimensions or exact cell order differ from RNA.")
    qc["atac_total_peak_counts"] = np.asarray(atac_counts.sum(axis=1)).ravel()
    qc["atac_n_peaks"] = np.diff(atac_counts.indptr)
    qc["atac_zero_counts"] = qc.atac_total_peak_counts == 0
    selected = atac_counts[cell_keep].tocsr()
    del atac_counts
    atac_var = pd.DataFrame(index=peaks)
    coordinates = [peak.rsplit("-", 2) for peak in peaks]
    if any(len(x) != 3 or not x[1].isdigit() or not x[2].isdigit() or int(x[1]) >= int(x[2])
           for x in coordinates):
        raise ValueError("Unrecognized peak coordinates.")
    atac_var["contig"] = [x[0] for x in coordinates]
    atac_var["start_as_supplied"] = [int(x[1]) for x in coordinates]
    atac_var["end_as_supplied"] = [int(x[2]) for x in coordinates]
    atac_var["n_cells_detected_after_rna_qc"] = np.asarray((selected > 0).sum(axis=0)).ravel()
    atac = ad.AnnData(X=selected, obs=qc.loc[cell_keep].copy(), var=atac_var)
    binary = selected.astype(np.uint8)
    binary.data[:] = 1
    atac.layers["binary"] = binary
    atac.uns["preprocessing"] = {
        "reference_doi": config["reference_doi"], "config_json": json.dumps(config),
        "X": "Unmodified integer ATAC peak counts for RNA-QC retained cells; all supplied peaks",
        "binary": "1 for count > 0, otherwise 0",
        "qc_limit": "RNA cell mask only; no fragment-level metrics or ATAC-specific thresholds available",
        "coordinates": "As supplied; genome build and coordinate origin unresolved",
        "source_sha256_json": json.dumps(source_hashes)}
    atac_path = output / f"{cell_line}_ATAC_paired_paper_qc.h5ad"
    save_h5ad(atac, atac_path)
    qc.to_csv(tables / f"{cell_line}_cell_qc.tsv", sep="\t", lineterminator="\n")
    var.to_csv(tables / f"{cell_line}_gene_qc.tsv", sep="\t", lineterminator="\n")
    qc_figure(qc, cell_line, config, figures / f"{cell_line}_paper_qc.png")
    summary = {
        "cells_before": len(cells), "cells_after": int(cell_keep.sum()),
        "genes_before": len(genes), "genes_after": int(gene_keep.sum()),
        "atac_peaks_before": len(peaks), "atac_peaks_after": len(peaks),
        "fail_min_genes": int((~qc.pass_min_genes).sum()),
        "fail_max_pct_mt": int((~qc.pass_max_pct_mt).sum()),
        "fail_both": int(((~qc.pass_min_genes) & (~qc.pass_max_pct_mt)).sum()),
        "atac_zero_count_cells_retained": int(qc.loc[cell_keep, "atac_zero_counts"].sum()),
        "treatment_before": {k:int(v) for k,v in qc.treatment.value_counts().items()},
        "treatment_after": {k:int(v) for k,v in qc.loc[cell_keep].treatment.value_counts().items()},
        "source_sha256": source_hashes, "exact_paired_order": True,
        "visualization": visualization,
        "scanpy_version": importlib.metadata.version("scanpy"),
        "outputs": [str(rna_path.relative_to(root)), str(atac_path.relative_to(root)), f"data/processed/{cell_line}_RNA_visualization.h5ad"]}
    saved_rna, saved_atac = ad.read_h5ad(rna_path, backed="r"), ad.read_h5ad(atac_path, backed="r")
    try:
        if not saved_rna.obs_names.equals(saved_atac.obs_names):
            raise ValueError("Written RNA/ATAC cells no longer match.")
    finally:
        saved_rna.file.close()
        saved_atac.file.close()
    print(json.dumps(summary, indent=2), flush=True)
    del atac, selected, binary
    gc.collect()
    return summary

def run_preprocessing(root, config):
    report = {"reference_doi": config["reference_doi"], "config": config,
              "run_at_utc": datetime.now(timezone.utc).isoformat(), "cell_lines": {}}
    for cell_line in ("MCF7", "T47D"):
        report["cell_lines"][cell_line] = preprocess_line(root, cell_line, config)
    destination = Path(root) / "results/tables/paper_qc_summary.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report
