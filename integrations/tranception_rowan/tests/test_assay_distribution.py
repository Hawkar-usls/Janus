"""Synthetic controls only; actual public SHA pinned dataset audited in Actions."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("assay_distribution", ROOT / "assay_distribution.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
manifest = json.loads((ROOT / "discovery_manifest.json").read_text(encoding="utf-8"))
wt = manifest["independent_control"]["wildtype"]
mut = manifest["independent_control"]["mutant"]


def bytes_fixture(rows):
    data = "sequences\tlabels\tfold_id\n" + "".join(
        f"{seq}\t{score}\t{fold}\n" for seq, score, fold in rows)
    b = data.encode()
    return b, hashlib.sha256(b).hexdigest()


class AuditTests(unittest.TestCase):
    def test_score_distribution_and_rank(self):
        alt = wt[:7] + "A" + wt[8:]
        self.assertNotEqual(alt, wt)
        raw, sha = bytes_fixture([(mut, -1, 0), (alt, 2, 2)])
        r = mod.analyze(raw, manifest, sha256=sha)
        self.assertEqual(r["row_count"], 2)
        self.assertEqual(r["unique_sequence_count"], 2)
        self.assertEqual(r["K5A"]["rank_ascending"], 1)
        self.assertEqual(r["K5A"]["percent_below"], 0)
        self.assertEqual(r["label_distribution"]["median"], 0.5)
        self.assertEqual(r["fold_counts"], [1, 0, 1, 0, 0])
        self.assertFalse(r["rowan_model_prediction"] == "RUN")

    def test_missing_k5a_rejected(self):
        alt = wt[:7] + "A" + wt[8:]
        raw, sha = bytes_fixture([(alt, 2, 2)])
        with self.assertRaises(mod.SourceIntegrityError):
            mod.analyze(raw, manifest, sha256=sha)

    def test_duplicate_rejected(self):
        raw, sha = bytes_fixture([(mut, -1, 0), (mut, 3, 1)])
        with self.assertRaises(mod.SourceIntegrityError):
            mod.analyze(raw, manifest, sha256=sha)

    def test_tampered_bytes_rejected(self):
        raw, sha = bytes_fixture([(mut, -1, 0)])
        with self.assertRaises(mod.SourceIntegrityError):
            mod.analyze(raw + b"\n", manifest, sha256=sha)

    def test_wildtype_not_single_mutant_flagged(self):
        raw, sha = bytes_fixture([(wt, 0, 0), (mut, 1, 0)])
        r = mod.analyze(raw, manifest, sha256=sha)
        self.assertFalse(r["all_rows_exactly_one_amino_acid_substitution"])

    def test_invalid_aa_rejected(self):
        raw, sha = bytes_fixture([(mut[:-1] + "X", 1, 0)])
        with self.assertRaises(mod.SourceIntegrityError):
            mod.analyze(raw, manifest, sha256=sha)

    def test_nan_rejected(self):
        raw, sha = bytes_fixture([(mut, "nan", 0)])
        with self.assertRaises(mod.SourceIntegrityError):
            mod.analyze(raw, manifest, sha256=sha)

    def test_mutant_position_protected(self):
        original = json.loads(json.dumps(manifest))
        original["independent_control"]["mutant"] = wt
        raw, sha = bytes_fixture([(mut, 0.0, 0)])
        with self.assertRaises(mod.SourceIntegrityError):
            mod.analyze(raw, original, sha256=sha)

    def test_bad_fold_rejected(self):
        raw, sha = bytes_fixture([(mut, 0.0, 10)])
        with self.assertRaises(mod.SourceIntegrityError):
            mod.analyze(raw, manifest, sha256=sha)

    def test_three_way_rank_accounting(self):
        alt1 = wt[:7] + "A" + wt[8:]
        alt2 = wt[:8] + "F" + wt[9:]
        self.assertNotIn(alt2, (alt1, wt, mut))
        raw, sha = bytes_fixture([(alt1, 1, 0), (mut, 2, 1), (alt2, 3, 2)])
        r = mod.analyze(raw, manifest, sha256=sha)
        self.assertEqual(r["K5A"]["number_below"], 1)
        self.assertEqual(r["K5A"]["number_above"], 1)
        self.assertEqual(r["K5A"]["number_tied_including_K5A"], 1)
        self.assertEqual(r["K5A"]["rank_ascending"], 2)


if __name__ == "__main__":
    unittest.main()
