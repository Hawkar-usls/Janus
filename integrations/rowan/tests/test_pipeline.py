"""Offline end-to-end checks of hash-linked JANUS/TOPA/Registry envelopes."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "janus_science_pipeline.py"
CASE = ROOT / "examples" / "phenol_case.json"
sys.path.insert(0, str(ROOT))
import janus_rowan_bridge as bridge  # noqa: E402


class PipelineTests(unittest.TestCase):
    def test_plan_end_to_end_no_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "one-run"
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--case", str(CASE),
                 "--output-dir", str(folder)],
                capture_output=True, text=True, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("TOPA_REVIEW=REQUIRED", completed.stdout)
            self.assertTrue(folder.is_dir())
            r = json.loads((folder / "rowan.receipt.json").read_text())
            t = json.loads((folder / "topa.review-request.json").read_text())
            m = json.loads((folder / "meta-registry.candidate.json").read_text())
            self.assertEqual(r["event"], "PLANNED_NO_EXTERNAL_EFFECT")
            self.assertEqual(t["rowan_receipt_sha256"], bridge.sha256(r))
            self.assertEqual(m["topa_request_sha256"], bridge.sha256(t))
            self.assertEqual(m["rowan_receipt_sha256"], bridge.sha256(r))
            self.assertEqual(t["science_claim"], "HYPOTHESIS_ONLY")
            self.assertEqual(m["status"], "REVIEW_REQUIRED_NOT_IMPORTED")
            self.assertFalse(m["automatic_publication"])
            self.assertNotIn("ROWAN_API_KEY", json.dumps([r, t, m]))
            self.assertNotIn("rowan-sk", json.dumps([r, t, m]))

    def test_second_run_never_overwrites_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "one-run"
            cmd = [sys.executable, str(SCRIPT), "--case", str(CASE),
                   "--output-dir", str(folder)]
            subprocess.run(cmd, check=True, capture_output=True)
            original = (folder / "rowan.receipt.json").read_bytes()
            second = subprocess.run(cmd, capture_output=True)
            self.assertEqual(second.returncode, 2)
            self.assertEqual((folder / "rowan.receipt.json").read_bytes(), original)

    def test_no_submit_flag_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "run"
            cmd = [sys.executable, str(SCRIPT), "--case", str(CASE),
                   "--output-dir", str(folder), "--mode", "submit"]
            finished = subprocess.run(cmd, capture_output=True)
            self.assertNotEqual(finished.returncode, 0)
            self.assertFalse(folder.exists())

    def test_no_collect_without_uuid(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "run"
            finished = subprocess.run(
                [sys.executable, str(SCRIPT), "--case", str(CASE),
                 "--output-dir", str(folder), "--mode", "collect"],
                capture_output=True,
            )
            self.assertNotEqual(finished.returncode, 0)
            self.assertFalse(folder.exists())


if __name__ == "__main__":
    unittest.main()
