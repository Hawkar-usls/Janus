"""Small artificial CSV fixtures; no scientific numbers fabricated as results."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit", ROOT / "retrieval_paired_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(extra=None, missing=None):
    cols = ["DMS ID", "UniProt ID", "Selection Type", "Number of Mutants"]
    for a, b in audit.PAIRS.values():
        cols.extend([a, b])
    cols = list(dict.fromkeys(cols))
    basic = {
        "DMS ID": audit.SOURCE["assay"], "UniProt ID": "BBC1_YEAST",
        "Selection Type": "Stability", "Number of Mutants": "2069",
    }
    for a, b in audit.PAIRS.values():
        basic[a] = "0.1"
        basic[b] = "0.2"
    basic.update(missing or {})
    rows = [basic]
    rows.extend(extra or [])
    s = io.StringIO()
    w = csv.DictWriter(s, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
    data = s.getvalue().encode()
    return data, audit.git_blob_sha(data)


class PairedTests(unittest.TestCase):
    def test_frozen_source_meta(self):
        self.assertEqual(len(audit.SOURCE["git_blob_sha1"]), 40)
        self.assertEqual(len(audit.PAIRS), 3)

    def test_unique_bbc1_and_delta(self):
        data, h = fixture()
        r = audit.analyze(data, expected_blob_sha=h)
        self.assertEqual(r["population"]["total_assays"], 1)
        self.assertAlmostEqual(r["bbc1_delta_rho"]["L_PRIMARY"], 0.1)
        self.assertEqual(r["comparisons"]["L_PRIMARY"]["n_proteins"], 1)
        self.assertTrue(r["model_weights_rerun"] is False)

    def test_protein_clustering_does_not_double_count(self):
        row = {
            "DMS ID": "OTHER_2", "UniProt ID": "BBC1_YEAST",
            "Selection Type": "Stability", "Number of Mutants": "40",
        }
        for a, b in audit.PAIRS.values():
            row[a], row[b] = "0.4", "0.5"
        data, h = fixture(extra=[row])
        r = audit.analyze(data, expected_blob_sha=h)
        model = r["comparisons"]["L_PRIMARY"]
        self.assertEqual(model["n_proteins"], 1)
        self.assertEqual(model["n_assay_pairs"], 2)
        self.assertAlmostEqual(model["mean_protein_delta_rho"], .1)

    def test_missing_pair_excluded_not_imputed(self):
        a, b = audit.PAIRS["L_PRIMARY"]
        data, h = fixture(missing={b: ""})
        r = audit.analyze(data, expected_blob_sha=h)
        self.assertEqual(r["comparisons"]["L_PRIMARY"]["status"], "NO_COMPLETE_PROTEIN_PAIRS")
        self.assertEqual(r["comparisons"]["L_PRIMARY"]["excluded_assays_with_missing_pair_or_protein_id"], 1)

    def test_wrong_sha_fails(self):
        data, h = fixture()
        with self.assertRaises(audit.EvidenceError):
            audit.analyze(data+b"\n", expected_blob_sha=h)

    def test_bbc1_missing_fails(self):
        data, h = fixture(missing={"DMS ID": "DMS_TEST"})
        with self.assertRaises(audit.EvidenceError):
            audit.analyze(data, expected_blob_sha=h)

    def test_duplicate_assay_fails(self):
        data, h = fixture()
        header, line = data.split(b"\r\n", 1) if b"\r\n" in data else data.split(b"\n", 1)
        duplicate = data + line
        with self.assertRaises(audit.EvidenceError):
            audit.analyze(duplicate, expected_blob_sha=audit.git_blob_sha(duplicate))

    def test_invalid_coefficient_fails(self):
        key = audit.PAIRS["L_PRIMARY"][1]
        data, h = fixture(missing={key: "1.2"})
        with self.assertRaises(audit.EvidenceError):
            audit.analyze(data, expected_blob_sha=h)

    def test_invalid_count_fails(self):
        data, h = fixture(missing={"Number of Mutants": "1084"})
        with self.assertRaises(audit.EvidenceError):
            audit.analyze(data, expected_blob_sha=h)

    def test_reproducible_bootstrap(self):
        vals = {"A": [0.1, 0.3], "B": [-0.1], "C": [0.15]}
        one = audit.protein_summary(vals)
        two = audit.protein_summary(vals)
        self.assertEqual(one, two)
        self.assertEqual(one["winning_proteins"], 2)
        self.assertEqual(one["losing_proteins"], 1)
        lo, hi = one["bootstrap_mean_protein_delta_95pct_CI"]
        self.assertLessEqual(lo, hi)


if __name__ == "__main__":
    unittest.main()
