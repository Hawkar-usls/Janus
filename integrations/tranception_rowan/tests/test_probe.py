import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tranception_probe", ROOT / "probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
manifest = json.loads((ROOT / "discovery_manifest.json").read_text(encoding="utf-8"))


class FakeUser:
    enabled_workflows = ["protein_cofolding", "pka"]


class FakeScores:
    ptm = 0.8
    avg_lddt = 0.7


class FakeResult:
    scores = FakeScores()


class FakeWorkflow:
    uuid = "00000000-0000-4000-8000-000000000005"
    name = "JANUS-TRANCEPTION-ROWAN-BBC1-K5A"
    workflow_type = "protein_cofolding"

    def fetch_latest(self):
        return self

    def done(self):
        return True

    def result(self, wait):
        assert wait is False
        return FakeResult()


class FakeSDK:
    ProteinCofoldingResult = FakeResult

    def __init__(self):
        self.arguments = None

    def whoami(self):
        return FakeUser()

    def submit_protein_cofolding_workflow(self, **kwargs):
        self.arguments = kwargs
        return FakeWorkflow()

    def retrieve_workflow(self, workflow_id, *, result_type):
        assert workflow_id == FakeWorkflow.uuid
        assert result_type is FakeResult
        return FakeWorkflow()


class TranceptionRowanTests(unittest.TestCase):
    def test_full_manifest_and_upstream_lineage(self):
        self.assertEqual(len(manifest["archive"]["files"]), 22)
        self.assertFalse(manifest["archive"]["tree_truncated"])
        self.assertEqual(len({x["path"] for x in manifest["archive"]["files"]}), 22)
        self.assertEqual(manifest["origin"]["class"], "UPSTREAM_DERIVED_WORKING_COPY")

    def test_bbc1_public_control_and_exact_single_substitution(self):
        wt = probe.prepare(manifest, "wildtype")
        mut = probe.prepare(manifest, "K5A")
        self.assertEqual(len(wt["sequence"]), 64)
        self.assertEqual(wt["sequence"][4], "K")
        self.assertEqual(mut["sequence"][4], "A")
        self.assertEqual(sum(a != b for a, b in zip(wt["sequence"], mut["sequence"])), 1)

    def test_plan_never_needs_keys_or_network(self):
        with patch.dict(os.environ, {}, clear=True):
            r = probe.execute(manifest, "wildtype", "plan")
        self.assertEqual(r["event"], "OFFLINE_PLAN_ONLY")
        self.assertIsNone(r["workflow_uuid"])
        self.assertFalse(r["protein_fitness_measured"])
        self.assertFalse(r["registry_published"])
        self.assertFalse(r["tranception_model_run"])

    def test_missing_key_blocks_draft_and_submit(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(PermissionError):
                probe.execute(manifest, "wildtype", "draft", sdk=FakeSDK())
            with self.assertRaises(PermissionError):
                probe.execute(manifest, "wildtype", "submit",
                              authorize_credit_use=True, sdk=FakeSDK())

    def test_authorization_flag_is_mandatory(self):
        with patch.dict(os.environ, {"ROWAN_API_KEY": "fake-unit-test-key"}):
            with self.assertRaises(PermissionError):
                probe.execute(manifest, "K5A", "submit", sdk=FakeSDK())

    def test_limits_fail_closed(self):
        for n in (0, 11, 2.0, True):
            with self.assertRaises(ValueError):
                probe.execute(manifest, "K5A", "plan", max_credits=n)

    def test_draft_has_one_arm_and_does_not_execute(self):
        sdk = FakeSDK()
        with patch.dict(os.environ, {"ROWAN_API_KEY": "fake-unit-test-key"}):
            r = probe.execute(manifest, "K5A", "draft", sdk=sdk)
        self.assertEqual(r["event"], "ROWAN_DRAFT_CREATED_NO_COMPUTATION")
        self.assertTrue(sdk.arguments["is_draft"])
        self.assertEqual(sdk.arguments["model"], "boltz-2")
        self.assertEqual(len(sdk.arguments["initial_protein_sequences"]), 1)
        self.assertEqual(sdk.arguments["max_credits"], 1)

    def test_one_live_submit_requires_budget(self):
        sdk = FakeSDK()
        with patch.dict(os.environ, {"ROWAN_API_KEY": "fake-unit-test-key"}):
            r = probe.execute(manifest, "K5A", "submit",
                              authorize_credit_use=True, sdk=sdk)
        self.assertFalse(sdk.arguments["is_draft"])
        self.assertEqual(r["event"], "ROWAN_SUBMITTED_RESULT_PENDING")
        self.assertIsNone(r["structural_confidence"])

    def test_typed_result_is_not_fitness(self):
        with patch.dict(os.environ, {"ROWAN_API_KEY": "fake-unit-test-key"}):
            r = probe.execute(manifest, "K5A", "collect",
                              workflow_id=FakeWorkflow.uuid, sdk=FakeSDK())
        self.assertEqual(r["structural_confidence"]["ptm"], 0.8)
        self.assertEqual(r["structural_confidence"]["avg_lddt"], 0.7)
        self.assertFalse(r["biological_fitness_inferred"])
        self.assertFalse(r["protein_fitness_measured"])

    def test_wrong_protein_arm_fails(self):
        with self.assertRaises(ValueError):
            probe.prepare(manifest, "other")

    def test_corrupted_archive_blob_sha_fails(self):
        bad = json.loads(json.dumps(manifest))
        bad["archive"]["files"][0]["git_blob_sha"] = "invalid"
        with self.assertRaises(ValueError):
            probe.prepare(bad, "wildtype")

    def test_mutant_sequence_cannot_be_silently_changed(self):
        bad = json.loads(json.dumps(manifest))
        bad["independent_control"]["mutant"] = bad["independent_control"]["wildtype"]
        with self.assertRaises(ValueError):
            probe.prepare(bad, "K5A")


if __name__ == "__main__":
    unittest.main()
