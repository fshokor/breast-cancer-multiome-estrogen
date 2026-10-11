# GSE154873 data inspection

## Source
- GEO: [GSE154873](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE154873)
- Stable [supplement directory](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154873/suppl/)
- Publication: Ors et al. (2022), *Estrogen regulates divergent transcriptional and epigenetic cell states in breast cancer*, Nucleic Acids Research 50:11492–11508.
  [DOI: 10.1093/nar/gkac908](https://doi.org/10.1093/nar/gkac908);
  [PMID: 36318267](https://pubmed.ncbi.nlm.nih.gov/36318267/).
- Organism: Homo sapiens.
- Assay: 10x single-cell Multiome, paired RNA expression and ATAC chromatin accessibility.
- Selected cell lines: MCF-7 and T47D (T-47D in the paper).
- Local inspection completed on 2026-10-10. Exact entry scans and provenance are
  saved in results/tables/data_inspection.json and download_manifest.json.

## Experimental design
The publication's section “Estrogen stimulation directs transcriptionally and
epigenetically distinct cell states” explicitly describes MCF-7 and T-47D vehicle
controls and 48-hour estrogen treatment for Multiome.
This duration comes from the paper, not a column in the downloaded metadata.

The six supplements contain exactly two sample labels per cell line:
YW_si_Control_plus_E2 and YW_si_Control_plus_Veh.
E2 denotes estrogen stimulation and Veh denotes vehicle; the full submitted labels
are preserved. The si_Control prefix is part of the submitted sample name, not a
separate condition to remove or rename silently. Metadata do not identify replicates,
donors, duration, or dose. Do not infer replication from individual cells.
The general GEO series includes unrelated experiments; its description alone is
not a replacement for inspecting this selected subset.

## Modalities and pairing
RNA and ATAC refer to the same cells within each cell line. Exact identifiers and
their order match the metadata. This is established by complete barcode comparisons
below; no intersection-based filtering was used.

## Files
Only these six processed supplements were downloaded. No organoid, TAMR, older
scRNA-only, RAW.tar, FASTQ or SRA files were acquired.

| File | Compressed bytes | Uncompressed bytes |
|---|---:|---:|
| GSE154873_MCF7_multiome_RNA_counts.tsv.gz | 21,850,925 | 472,334,144 |
| GSE154873_MCF7_multiome_ATAC_counts.tsv.gz | 66,788,657 | 2,659,825,102 |
| GSE154873_MCF7_multiome_metadata.tsv.gz | 34,522 | 411,584 |
| GSE154873_T47D_multiome_RNA_counts.tsv.gz | 23,679,618 | 485,405,233 |
| GSE154873_T47D_multiome_ATAC_counts.tsv.gz | 86,265,641 | 2,732,940,807 |
| GSE154873_T47D_multiome_metadata.tsv.gz | 35,325 | 422,801 |

All downloads completed with gzip CRC validation and local SHA-256 hashes.
Hashes are locally computed provenance, not checksums independently published by GEO.
Archives remain under data/raw/GSE154873/. They were not decompressed to disk.
Optional extraction uses the downloader's --decompress flag and retains archives.
The total compressed and uncompressed sizes are respectively
198,654,688 and 6,351,339,671 bytes.

## Dimensions and count validation
Every numeric entry was read in a streaming scan, holding one TSV row at a time.
Dimensions exclude headers and the leading feature-identifier column.

| Cell line | Modality | Features | Cells | On-disk orientation | Zero entries (%) | Count range |
|---|---|---:|---:|---|---:|---|
| MCF7 | RNA | 36,601 | 6,438 | features × cells | 90.470704 | 0–3041 |
| MCF7 | ATAC | 206,167 | 6,438 | features × cells | 97.071872 | 0–112 |
| T47D | RNA | 36,601 | 6,615 | features × cells | 89.980282 | 0–1661 |
| T47D | ATAC | 206,167 | 6,615 | features × cells | 96.015569 | 0–131 |

All four matrices contain only finite, nonnegative integer values. There are no
duplicate feature identifiers or cell identifiers. These findings are consistent
with count matrices; numeric validation alone cannot establish all upstream
processing history. Files are dense TSV text despite their high fraction of zeros.
No full dense NumPy arrays, transposes, or AnnData objects were created.

### Matrix headers and identifiers
The first row contains cell identifiers only. Data rows have one additional leading
field containing a gene or peak identifier. A naive rectangular TSV import can
misinterpret this omitted index-header field.
RNA feature names are gene symbols/names, for example MIR1302-2HG, FAM138A,
OR4F5, AL627309.1 and AL627309.3; no identifiers start with ENSG.

ATAC peak names use contig-start-end, for example chr1-9785-10705.
They do **not** use chr:start-end. In each matrix, 206,078 peaks have a chr prefix
and 89 are other contigs, for example GL000009.2-124429-125410.
All 89 non-chr identifiers per line are preserved in
results/tables/noncanonical_peak_ids.json; they are not automatically invalid or dropped.
Genome assembly and coordinate origin are not encoded in these TSVs and have not
been independently established in this milestone. Confirm both before motif/annotation work.

## Metadata
Both metadata files are headerless, with exactly two fields per row.
The inspection assigns positional names column_1 and column_2; these are not
original GEO column names.

| Assigned name | Observed content |
|---|---|
| column_1 | Full sample-prefixed cell barcode, e.g. YW_si_Control_plus_E2_AAACAGCCAACGTGCT-1 |
| column_2 | Sample/condition label |

MCF7 metadata shape: 6,438 × 2. T47D metadata shape: 6,615 × 2.
There are no empty fields, repeated cell IDs, QC columns, cluster/state annotations,
replicate columns, or explicit cell-line columns. Cell line is recorded by filename.
Preserve the sample prefix; bare 10x barcode sequences alone are not global cell IDs.
Namespace identifiers by cell line before any future cross-line concatenation.

| Cell line | Exact sample/condition label | Cells |
|---|---|---:|
| MCF7 | YW_si_Control_plus_E2 | 3,443 |
| MCF7 | YW_si_Control_plus_Veh | 2,995 |
| T47D | YW_si_Control_plus_E2 | 3,587 |
| T47D | YW_si_Control_plus_Veh | 3,028 |

Thus vehicle versus E2 is present in both lines, with Veh as the literal vehicle suffix.

## Exact paired-cell audit
| Cell line | RNA IDs | ATAC IDs | Metadata IDs | RNA ∩ ATAC | RNA ∩ metadata | ATAC ∩ metadata |
|---|---:|---:|---:|---:|---:|---:|
| MCF7 | 6,438 | 6,438 | 6,438 | 6,438 | 6,438 | 6,438 |
| T47D | 6,615 | 6,615 | 6,615 | 6,615 | 6,615 | 6,615 |

All three identifier sets and their row/column orders are exactly equal within each
cell line. All six directional missing-ID lists are empty for each line.
No cells were discarded to obtain these matches.

## Issues and next step
- Metadata and matrix header conventions require explicit parsing.
- Only condition labels are supplied; QC and replication annotations are absent.
- ATAC coordinates use hyphens and include 89 non-chr contigs.
- The genome build and coordinate convention require provenance confirmation.
- Drive mounting has not been exercised in a live Colab session; both notebooks
  are also executable locally, where their actual data paths are tested.
- Local environment uses Python 3.11 and CPU PyTorch. No scMKL installation was attempted.

The initial inspection above preceded QC. The subsequent paper-guided QC milestone
is implemented in nb02_qc_preprocessing.ipynb; see QC_PREPROCESSING.md and
results/tables/paper_qc_summary.json for the resulting dimensions.
At that QC milestone, downstream analyses had not run. nb03 later added cluster characterization, and nb04 added exploratory treatment DEG and GSEA; see the follow-up notes.

### Follow-up provenance note
Kupp et al. (2025), DOI 10.1038/s42003-025-08533-7, describes this dataset as a
24-hour estrogen experiment (Methods p.12), conflicting with the 48-hour Multiome
description in Ors et al. above. Metadata cannot resolve this discrepancy.
No duration is assigned to cells; E2/Vehicle labels are preserved.

### Scanpy visualization update
nb02 now uses Scanpy QC/transformations and independently fits RNA PCA, Leiden
clusters and UMAP for each cell line. These exploratory fits are separate from
the paper-QC count/log1p representation and do not constitute cell-type annotation
or ATAC/joint embeddings. See QC_PREPROCESSING.md for exact settings and validation.

### nb04 replicate audit
The GEO family SOFT records identify GSM6418443 and GSM6418444 as the E2 and Vehicle multiomic/GEX libraries, each containing both cell lines. GSM6418445/6 are the paired ATAC libraries. No biological replicate-to-cell mapping is supplied. nb04 therefore performs exploratory cell-level treatment DEG and preranked GSEA; it does not claim replicated treatment inference. See TREATMENT_DEG_GSEA.md and results/tables/nb04/geo_design_audit.json.

### nb05 genome and coordinate follow-up
GEO ATAC sample records GSM6418445/6 explicitly state hg38, supporting that assembly for the processed data. Coordinate origin in the peak names is still unresolved. nb05 preserves coordinates and supplies only provisional nearest-TSS gene candidates agreeing under both zero-based half-open and one-based closed interpretations. No definitive promoter/motif overlap or BED export is made. See ATAC_ANALYSIS.md and configs/atac_reference.json.


## nb07 sequence and motif references (11 October 2026)
nb07 uses checksum-frozen UCSC hg38 2bit sequence, JASPAR2024 CORE vertebrate
non-redundant motifs, and the hg38 blacklist v2. URLs and SHA256 values are in
configs/motif_references.json; the genome also passed UCSC's official MD5 check.
The raw references are under ignored data/raw/reference/nb07.

The original study code audit did not establish the coordinate convention of
the exact GEO TSV export. configs/peak_coordinate_audit.json records the pinned
source and uncertainty. nb07 tests both coordinate hypotheses, preserves upstream
peak IDs/counts, and reports sequence-QC exclusions and sensitivity. It does not
retroactively declare nb05/nb06 annotations definitive.
