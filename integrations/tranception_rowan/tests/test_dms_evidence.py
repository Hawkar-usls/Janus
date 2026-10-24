"""All fixtures synthetic; these tests never claim to measure a real protein."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("janus_dms_evidence", ROOT / "dms_evidence.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
manifest = json.loads((ROOT / "discovery_manifest.json").read_text(encoding="utf-8"))
wt = manifest["independent_control"]["wildtype"]
mut = manifest["independent_control"]["mutant"]


def fixture(rows):
    v = "sequences\tlabels\tfold_id\n"
    for sequence, label, fold in rows:
        v += f"{sequence}\t{label}\t{fold}\n"
    data = v.encode()
    return data, hashlib.sha256(data).hexdigest()


class DatasetEvidenceTests(unittest.TestCase):
    def test_source_is_revision_and_sha_pinned(self):
        self.assertEqual(len(m.SOURCE["sha256"]), 64)
        self.assertEqual(len(m.SOURCE["dataset_commit"]), 40)
        self.assertTrue(m.SOURCE["url"].startswith("https://huggingface.co/datasets/"))

    def test_verified_synthetic_k5a_lookup_only(self):
        data, sha = fixture([(wt, 0.4, 1), (mut, -1.8, 3)])
        report = m.validate_tsv(data, manifest, expected_sha256=sha)
        self.assertEqual(report["status"], "EXPERIMENTAL_DERIVATIVE_ROW_VERIFIED")
        self.assertEqual(report["processed_row"]["processed_label"], -1.8)
        self.assertEqual(report["processed_row"]["fold_id"], 3)
        self.assertEqual(report["rows_scanned"], 2)
        self.assertFalse(report["biological_replicates_verified"])
        self.assertFalse(report["rowan_job_run"])
        self.assertFalse(report["has_valid_two_arm_correlation"])

    def test_absent_label_not_imputed(self):
        data, sha = fixture([(wt, 0.4, 1)])
        report = m.validate_tsv(data, manifest, expected_sha256=sha)
        self.assertEqual(report["status"], "K5A_ROW_ABSENT")
        self.assertIsNone(report["processed_row"])

    def test_exact_bytes_checksum_matters(self):
        data, sha = fixture([(mut, 0.0, 2)])
        with self.assertRaises(m.SourceIntegrityError):
            m.validate_tsv(data + b"\n", manifest, expected_sha256=sha)

    def test_duplicate_variant_rejected(self):
        data, sha = fixture([(mut, -1.8, 3), (mut, -1.9, 3)])
        with self.assertRaises(m.SourceIntegrityError):
            m.validate_tsv(data, manifest, expected_sha256=sha)

    def test_nan_rejected(self):
        data, sha = fixture([(mut, "nan", 3)])
        with self.assertRaises(m.SourceIntegrityError):
            m.validate_tsv(data, manifest, expected_sha256=sha)

    def test_unrecognized_fold_rejected(self):
        data, sha = fixture([(mut, "0.1", 9)])
        with self.assertRaises(m.SourceIntegrityError):
            m.validate_tsv(data, manifest, expected_sha256=sha)

    def test_missing_fields_rejected(self):
        data = b"mutated_sequence\tDMS_score\nEVPF\t0.1\n"
        with self.assertRaises(m.SourceIntegrityError):
            m.validate_tsv(data, manifest, expected_sha256=hashlib.sha256(data).hexdigest())

    def test_context_change_fails(self):
        data, sha = fixture([(mut, 0.12, 3)])
        other = json.loads(json.dumps(manifest))
        other["independent_control"]["assay_id"] = "OTHER_ASSAY"
        with self.assertRaises(m.SourceIntegrityError):
            m.validate_tsv(data, other, expected_sha256=sha)

    def test_no_second_mutation_allowed(self):
        data, sha = fixture([(mut, 0.12, 3)])
        other = json.loads(json.dumps(manifest))
        other["independent_control"]["mutant"] = wt
        with self.assertRaises(m.SourceIntegrityError):
            m.validate_tsv(data, other, expected_sha256=sha)


if __name__ == "__main__":
    unittest.main()
