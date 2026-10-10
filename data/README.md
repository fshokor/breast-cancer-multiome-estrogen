# Data storage
Original GEO archives belong in raw/GSE154873/. The downloader retains them.
Optional decompressed TSVs stay alongside the archives. All biological matrices
are ignored by Git. nb02 writes six AnnData files under processed/: RNA paper-QC, paired ATAC,
and RNA visualization objects per cell line. Count matrices remain sparse; only
the approximately 2,000-HVG visualization input is centered into a dense matrix.
See ../DATA.md for initial dimensions and ../QC_PREPROCESSING.md for derived
objects, layers, exact thresholds and provenance.
