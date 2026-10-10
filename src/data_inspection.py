"""Streaming structural inspection; no filtering, normalization or modeling."""
import csv
import gzip
import itertools
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BARCODE = re.compile(r"(?:^|_)[ACGTN]{16}-\d+$")
PEAK_COLON = re.compile(r"^chr[^:]+:\d+-\d+$")
PEAK_HYPHEN = re.compile(r"^chr[^-]+-\d+-\d+$")


def inspect_metadata(path):
    with gzip.open(path, "rt", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"Empty or ragged metadata: {path}")
    headerless = any(BARCODE.search(value) for value in rows[0])
    original_columns = None if headerless else rows.pop(0)
    columns = original_columns or [f"column_{i + 1}" for i in range(len(rows[0]))]
    candidates = [i for i in range(len(columns)) if all(BARCODE.search(row[i]) for row in rows)]
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous barcode column in {path}: {candidates}")
    barcode_index = candidates[0]
    barcodes = [row[barcode_index] for row in rows]
    counts = {columns[i]: dict(sorted(Counter(row[i] for row in rows).items()))
              for i in range(len(columns)) if i != barcode_index}
    duplicates = {key: value for key, value in Counter(barcodes).items() if value > 1}
    report = dict(filename=Path(path).name, shape=[len(rows), len(columns)],
                  header_present=not headerless, original_columns=original_columns,
                  assigned_columns=columns, barcode_column=columns[barcode_index],
                  unique_barcodes=len(set(barcodes)), duplicate_barcodes=duplicates,
                  barcode_examples=barcodes[:3], non_barcode_value_counts=counts,
                  empty_fields=sum(value == "" for row in rows for value in row))
    return report, barcodes


def inspect_matrix(path, metadata_barcodes):
    """Hold one TSV row at a time; validate every numeric entry, including zeros."""
    meta = set(metadata_barcodes)
    row_ids = []
    nonzero = entries = negatives = nonintegers = nonfinite = 0
    minimum, maximum = math.inf, -math.inf
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        first = next(reader)
        if len(first) == len(header) + 1:
            columns, header_style = header, "feature/index column label omitted"
        elif len(first) == len(header):
            columns, header_style = header[1:], "explicit feature/index column label"
        else:
            raise ValueError(f"Header/row width mismatch: {path}")
        if len(set(columns)) != len(columns):
            raise ValueError(f"Duplicate column identifiers: {path}")
        for index, row in enumerate(itertools.chain([first], reader), start=1):
            if len(row) != len(columns) + 1:
                raise ValueError(f"Ragged row {index}: {path}")
            row_ids.append(row[0])
            # Counter limits parsing work for the many repeated zero values.
            frequencies = Counter(row[1:])
            for token, count in frequencies.items():
                try:
                    value = float(token)
                except ValueError as exc:
                    raise ValueError(f"Nonnumeric count at row {index}: {token!r}") from exc
                entries += count
                nonzero += count if value != 0 else 0
                nonfinite += count if not math.isfinite(value) else 0
                negatives += count if value < 0 else 0
                nonintegers += count if not value.is_integer() else 0
                minimum, maximum = min(minimum, value), max(maximum, value)
            if index % 25000 == 0:
                print(f"  {Path(path).name}: {index:,} rows inspected", flush=True)
    column_overlap, row_overlap = len(set(columns) & meta), len(set(row_ids) & meta)
    column_barcode_format = all(BARCODE.search(x) for x in columns)
    row_barcode_format = all(BARCODE.search(x) for x in row_ids)
    if (column_overlap or column_barcode_format) and not (row_overlap or row_barcode_format):
        orientation, barcodes, features = "features x cells", columns, row_ids
    elif (row_overlap or row_barcode_format) and not (column_overlap or column_barcode_format):
        orientation, barcodes, features = "cells x features", row_ids, columns
    else:
        raise ValueError(f"Cannot establish orientation unambiguously: {path}")
    report = dict(filename=Path(path).name, shape_on_disk=[len(row_ids), len(columns)],
                  header_style=header_style, orientation=orientation, n_features=len(features),
                  n_cells=len(barcodes), feature_examples=features[:5], barcode_examples=barcodes[:3],
                  duplicate_features={k: v for k, v in Counter(features).items() if v > 1},
                  duplicate_barcodes={k: v for k, v in Counter(barcodes).items() if v > 1},
                  total_entries=entries, nonzero_entries=nonzero, zero_fraction=1 - nonzero / entries,
                  minimum=minimum, maximum=maximum, negative_entries=negatives,
                  noninteger_entries=nonintegers, nonfinite_entries=nonfinite,
                  all_nonnegative_integer_counts=not (negatives or nonintegers or nonfinite),
                  barcode_overlap_metadata=len(set(barcodes) & meta),
                  feature_format_counts={
                      "chr:start-end": sum(bool(PEAK_COLON.match(x)) for x in features),
                      "chr-start-end": sum(bool(PEAK_HYPHEN.match(x)) for x in features),
                      "ensembl_gene": sum(x.startswith("ENSG") for x in features)})
    return report, barcodes


def compare_barcodes(rna, atac, metadata):
    sets = {"rna": set(rna), "atac": set(atac), "metadata": set(metadata)}
    overlap = {}
    missing = {}
    for left, right in (("rna", "atac"), ("rna", "metadata"), ("atac", "metadata")):
        overlap[f"{left}_intersection_{right}"] = len(sets[left] & sets[right])
        missing[f"{left}_only_vs_{right}"] = sorted(sets[left] - sets[right])
        missing[f"{right}_only_vs_{left}"] = sorted(sets[right] - sets[left])
    return dict(counts={"rna": len(rna), "atac": len(atac), "metadata": len(metadata)},
                intersections=overlap, missing_barcodes=missing,
                exact_sets_equal=sets["rna"] == sets["atac"] == sets["metadata"],
                exact_order_equal=rna == atac == metadata,
                all_identifiers_unique=all(len(ids) == len(set(ids)) for ids in (rna, atac, metadata)))


def inspect_dataset(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(accession="GSE154873", inspected_at_utc=datetime.now(timezone.utc).isoformat(),
                  method="Complete streaming scan of all entries; no cell filtering or transpose",
                  cell_lines={})
    for line in ("MCF7", "T47D"):
        prefix = f"GSE154873_{line}_multiome_"
        metadata, meta_ids = inspect_metadata(data_dir / f"{prefix}metadata.tsv.gz")
        print(f"\n{line} metadata:\n{json.dumps(metadata, indent=2)}", flush=True)
        rna, rna_ids = inspect_matrix(data_dir / f"{prefix}RNA_counts.tsv.gz", meta_ids)
        atac, atac_ids = inspect_matrix(data_dir / f"{prefix}ATAC_counts.tsv.gz", meta_ids)
        pairing = compare_barcodes(rna_ids, atac_ids, meta_ids)
        report["cell_lines"][line] = dict(metadata=metadata, rna=rna, atac=atac, pairing=pairing)
        print(f"{line}: RNA {rna['shape_on_disk']}; ATAC {atac['shape_on_disk']}; pairing {pairing}", flush=True)
    destination = output_dir / "data_inspection.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output_dir / "matrix_dimensions.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["cell_line", "modality", "features", "cells", "orientation", "zero_fraction", "integer_counts"])
        for line, dataset in report["cell_lines"].items():
            for modality in ("rna", "atac"):
                matrix = dataset[modality]
                writer.writerow([line, modality, matrix["n_features"], matrix["n_cells"],
                                 matrix["orientation"], matrix["zero_fraction"], matrix["all_nonnegative_integer_counts"]])
    print(f"Saved inspection: {destination}")
    return report
