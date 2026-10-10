# Data storage
Original GEO archives belong in raw/GSE154873/. The downloader retains them.
Optional decompressed TSVs stay alongside the archives. All biological matrices
are ignored by Git. nb02 writes four sparse AnnData files under processed/:
one RNA paper-QC object and one paired ATAC object per cell line.
See ../DATA.md for initial dimensions and ../QC_PREPROCESSING.md for derived
objects, layers, exact thresholds and provenance.
