"""Failure-path tests for downloads and structural parsing."""
import gzip
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.data_inspection import inspect_metadata, inspect_matrix, compare_barcodes

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("downloader", ROOT / "scripts/download_gse154873.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class Response(io.BytesIO):
    def __init__(self, payload, size=None):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload) if size is None else size)}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a = "sample_AAAAAAAAAAAAAAAA-1"
        self.b = "sample_CCCCCCCCCCCCCCCC-1"

    def tearDown(self):
        self.temp.cleanup()

    def gz(self, name, content):
        path = self.root / name
        with gzip.open(path, "wt") as handle:
            handle.write(content)
        return path

    def test_headerless_metadata_preserves_first_cell(self):
        path = self.gz("meta.gz", f"{self.a}\tE2\n{self.b}\tVehicle\n")
        report, ids = inspect_metadata(path)
        self.assertEqual(ids, [self.a, self.b])
        self.assertFalse(report["header_present"])
        self.assertEqual(report["shape"], [2, 2])

    def test_missing_header_label_and_exact_counts(self):
        path = self.gz("rna.gz", f"{self.a}\t{self.b}\nG1\t0\t2\nG2\t3\t0\n")
        report, ids = inspect_matrix(path, [self.a, self.b])
        self.assertEqual(report["orientation"], "features x cells")
        self.assertEqual(report["zero_fraction"], 0.5)
        self.assertEqual(report["shape_on_disk"], [2, 2])
        self.assertTrue(report["all_nonnegative_integer_counts"])

    def test_transposed_matrix(self):
        path = self.gz("rna.gz", f"cell\tG1\tG2\n{self.a}\t0\t2\n{self.b}\t1\t0\n")
        report, ids = inspect_matrix(path, [self.a, self.b])
        self.assertEqual(report["orientation"], "cells x features")
        self.assertEqual(ids, [self.a, self.b])

    def test_ragged_matrix_fails(self):
        path = self.gz("rna.gz", f"{self.a}\t{self.b}\nG1\t0\t2\nG2\t3\n")
        with self.assertRaisesRegex(ValueError, "Ragged"):
            inspect_matrix(path, [self.a, self.b])

    def test_pairing_reports_missing_and_order(self):
        result = compare_barcodes([self.a, self.b], [self.a], [self.b, self.a])
        self.assertFalse(result["exact_sets_equal"])
        self.assertEqual(result["missing_barcodes"]["rna_only_vs_atac"], [self.b])
        reordered = compare_barcodes([self.a, self.b], [self.b, self.a], [self.a, self.b])
        self.assertTrue(reordered["exact_sets_equal"])
        self.assertFalse(reordered["exact_order_equal"])

    def test_partial_transfer_never_becomes_final(self):
        payload = gzip.compress(b"data")
        target = self.root / "archive.gz"
        with patch.object(downloader.urllib.request, "urlopen", return_value=Response(payload, len(payload) + 10)):
            with self.assertRaisesRegex(RuntimeError, "Incomplete transfer"):
                downloader.fetch("https://example.org/file.gz", target, attempts=1)
        self.assertFalse(target.exists())
        self.assertFalse(target.with_name(target.name + ".part").exists())

    def test_corrupt_gzip_rejected(self):
        path = self.root / "bad.gz"
        path.write_bytes(gzip.compress(b"data")[:-4])
        with self.assertRaises((EOFError, OSError)):
            downloader.validate_gzip(path)

    def test_rerun_skips_network_and_retains_archive(self):
        filename = "tiny.tsv.gz"
        self.gz(filename, "feature\tcell\nG1\t1\n")
        with patch.object(downloader, "FILES", {"test": filename}), patch.object(downloader, "fetch") as fetch:
            first = downloader.download_all(self.root, decompress=True)
            second = downloader.download_all(self.root, decompress=True)
        fetch.assert_not_called()
        self.assertEqual(first["files"], second["files"])
        self.assertTrue((self.root / filename).exists())
        self.assertTrue((self.root / "tiny.tsv").exists())


if __name__ == "__main__":
    unittest.main()
