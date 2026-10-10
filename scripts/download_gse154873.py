#!/usr/bin/env python3
"""Download only six paired Multiome supplements; standard library only."""
import argparse
import gzip
import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ACCESSION = "GSE154873"
BASE_URL = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{ACCESSION[:-3]}nnn/{ACCESSION}/suppl/"
FILES = {
    f"{line.lower()}_{kind.lower()}": f"{ACCESSION}_{line}_multiome_{kind}.tsv.gz"
    for line in ("MCF7", "T47D")
    for kind in ("RNA_counts", "ATAC_counts", "metadata")
}
FILES = {key.replace("_counts", ""): value for key, value in FILES.items()}
CHUNK = 1024 * 1024


def file_url(filename):
    return BASE_URL + urllib.parse.quote(filename, safe="")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_gzip(path):
    """Read to EOF to validate every gzip member/CRC, without retaining payload."""
    total = 0
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            total += len(block)
    if not total:
        raise ValueError(f"Empty gzip payload: {path}")
    return total


def fetch(url, target, attempts=3):
    """Never expose an incomplete transfer as the final filename."""
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "GSE154873-reproducible-setup/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
                length = response.headers.get("Content-Length")
                expected = int(length) if length else None
                received, last = 0, 0.0
                for block in iter(lambda: response.read(CHUNK), b""):
                    output.write(block)
                    received += len(block)
                    now = time.monotonic()
                    if now - last >= 2:
                        fraction = f"/{expected / 1e6:.1f} MB ({100 * received / expected:.1f}%)" if expected else " MB"
                        print(f"  {received / 1e6:.1f}{fraction}", flush=True)
                        last = now
            if not received or (expected is not None and received != expected):
                raise IOError(f"Incomplete transfer: {received} bytes, expected {expected}")
            validate_gzip(partial)
            partial.replace(target)
            print(f"  Complete: {received:,} bytes", flush=True)
            return
        except Exception as exc:
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise RuntimeError(f"Download failed after {attempts} attempts: {url}: {exc}") from exc
            print(f"  Attempt {attempt} failed: {exc}; retrying.", flush=True)
            time.sleep(attempt * 2)


def decompress_archive(path, expected_size, expected_hash=None):
    target = path.with_suffix("")
    if target.exists() and target.stat().st_size == expected_size and expected_hash and sha256(target) == expected_hash:
        print(f"  Decompressed file already verified: {target.stat().st_size:,} bytes")
        return expected_hash
    partial = target.with_name(target.name + ".part")
    try:
        with gzip.open(path, "rb") as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, CHUNK)
        if partial.stat().st_size != expected_size:
            raise IOError(f"Decompression size mismatch: {path}")
        digest = sha256(partial)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    print(f"  Decompressed: {expected_size:,} bytes; compressed original retained")
    return digest


def download_all(destination, decompress=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "download_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = {}
    for key, filename in FILES.items():
        path = destination / filename
        print(f"\n{key}: {filename}", flush=True)
        if path.exists():
            print("  Already exists; validating before skipping download.", flush=True)
            try:
                expanded_size = validate_gzip(path)
                digest = sha256(path)
                old = previous.get("files", {}).get(key, {})
                if old.get("sha256") and old["sha256"] != digest:
                    raise ValueError("SHA-256 differs from the prior manifest")
            except Exception as exc:
                raise RuntimeError(f"Existing file is invalid: {path}. Move it aside and rerun. {exc}") from exc
        else:
            fetch(file_url(filename), path)
            expanded_size = validate_gzip(path)
            digest = sha256(path)
        record = dict(filename=filename, url=file_url(filename), compressed_bytes=path.stat().st_size,
                      uncompressed_bytes=expanded_size, sha256=digest, gzip_crc_valid=True)
        print(f"  Compressed: {record['compressed_bytes']:,} bytes; payload: {expanded_size:,} bytes", flush=True)
        if decompress:
            record["decompressed_sha256"] = decompress_archive(
                path, expanded_size, previous.get("files", {}).get(key, {}).get("decompressed_sha256"))
        records[key] = record
    missing = [name for name in FILES.values() if not (destination / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing required files: {missing}")
    manifest = dict(accession=ACCESSION, verified_at_utc=datetime.now(timezone.utc).isoformat(), files=records)
    temp = manifest_path.with_suffix(".json.part")
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    temp.replace(manifest_path)
    print("\nAll six files verified (gzip CRC and local SHA-256).")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1] / "data/raw/GSE154873")
    parser.add_argument("--decompress", action="store_true", help="Also unpack TSV files; retain .gz originals.")
    parser.add_argument("--list-urls", action="store_true", help="Print the six URLs without downloading.")
    args = parser.parse_args()
    if args.list_urls:
        for filename in FILES.values():
            print(file_url(filename))
        return
    download_all(args.dest, args.decompress)


if __name__ == "__main__":
    main()
