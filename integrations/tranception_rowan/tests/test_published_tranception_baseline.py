"""Synthetic fixture tests for precomputed public Tranception benchmark."""
import importlib.util
import io
import csv
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("published_baseline", ROOT / "published_tranception_baseline.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(n=2069, row_name=m.SOURCE["assay"], selection="Stability",
            override=None, duplicate=False):
    override = override or {}
    keys = ["DMS ID", *m.METHODS, "Number of Mutants", "Selection Type", "UniProt ID"]
    values = {"DMS ID": row_name, **{k: 0.1 for k in m.METHODS},
              "Number of Mutants": n, "Selection Type": selection,
              "UniProt ID": "BBC1_YEAST"}
    values.update(override)
    s = io.StringIO()
    writer = csv.DictWriter(s, fieldnames=keys)
    writer.writeheader()
    writer.writerow(values)
    if duplicate:
        writer.writerow(values)
    data = s.getvalue().encode()
    return data, m.git_blob_sha(data)


class BaselineTests(unittest.TestCase):
    def test_exact_evidence_schema_and_scientific_firewalls(self):
        data, digest = fixture()
        r = m.extract(data, expected_blob_sha=digest)
        self.assertEqual(r["number_of_mutants_reported_in_official_benchmark"], 2069)
        self.assertEqual(r["selection_type"], "Stability")
        self.assertEqual(len(r["spearman_rho"]), 9)
        self.assertFalse(r["model_weights_executed_by_janus"])
        self.assertFalse(r["rowan_computation_executed"])
        self.assertFalse(r["directly_comparable_to_1084_single_variant_derivative"])

    def test_git_blob_sha_is_not_plain_sha256(self):
        data, digest = fixture()
        self.assertEqual(len(digest), 40)
        self.assertTrue(m.git_blob_sha(data) == digest)

    def test_tampered_file_rejected(self):
        data, digest = fixture()
        with self.assertRaises(m.EvidenceError):
            m.extract(data+b"\n", expected_blob_sha=digest)

    def test_duplicate_bbc1_rejected(self):
        data, digest = fixture(duplicate=True)
        with self.assertRaises(m.EvidenceError):
            m.extract(data, expected_blob_sha=digest)

    def test_missing_bbc1_rejected(self):
        data, digest = fixture(row_name="OTHER")
        with self.assertRaises(m.EvidenceError):
            m.extract(data, expected_blob_sha=digest)

    def test_unknown_benchmark_count_rejected(self):
        data, digest = fixture(n=1084)
        with self.assertRaises(m.EvidenceError):
            m.extract(data, expected_blob_sha=digest)

    def test_wrong_assay_type_rejected(self):
        data, digest = fixture(selection="Activity")
        with self.assertRaises(m.EvidenceError):
            m.extract(data, expected_blob_sha=digest)

    def test_invalid_rho_rejected(self):
        data, digest = fixture(override={"Tranception L": 1.1})
        with self.assertRaises(m.EvidenceError):
            m.extract(data, expected_blob_sha=digest)

    def test_missing_model_score_rejected(self):
        data, digest = fixture(override={"Tranception L": ""})
        with self.assertRaises(m.EvidenceError):
            m.extract(data, expected_blob_sha=digest)


if __name__ == "__main__":
    unittest.main()
