"""Offline tests: no rowan SDK, credentials, or paid compute required."""
import importlib.util
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "janus_rowan_bridge.py"
SPEC = importlib.util.spec_from_file_location("janus_rowan_bridge", PATH)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def example_case():
    return {
        "case_id": "JANUS-ROWAN-DEMO-PHENOL-001",
        "source": {
            "system": "TOPA",
            "reference": "https://github.com/Hawkar-usls/TOPA",
        },
        "hypothesis": "Model may estimate phenol pKa near the experimental range",
        "alternative": "Model/solvent mismatch may dominate the estimate",
        "falsifier": "Compare model output to independent experimental pKa",
        "molecule": {"smiles": "c1ccccc1O"},
        "workflow": {
            "type": "pka", "method": "starling",
            "pka_range": [2, 12], "max_credits": 1,
        },
    }


class FakeResult:
    strongest_acid = 10.0
    strongest_base = None


class FakeWorkflow:
    uuid = "00000000-0000-4000-8000-000000000001"
    workflow_type = "pka"
    status = "COMPLETED_OK"
    credits_charged = 0.1

    def fetch_latest(self):
        return self

    def done(self):
        return True

    def result(self, wait=True):
        assert wait is False
        return FakeResult()


class FakeSDK:
    called = None

    def submit_pka_workflow(self, **kwargs):
        self.called = kwargs
        return FakeWorkflow()

    def retrieve_workflow(self, workflow_id):
        assert workflow_id == FakeWorkflow.uuid
        return FakeWorkflow()


class BridgeTests(TestCase):
    def test_plan_requires_no_credentials_or_import(self):
        with patch.dict(os.environ, {}, clear=True):
            r = bridge.operate(example_case())
        self.assertEqual(r["event"], "PLANNED_NO_EXTERNAL_EFFECT")
        self.assertIsNone(r["workflow_uuid"])
        self.assertEqual(r["research"]["scientific_status"], "UNVERIFIED")
        self.assertEqual(r["research"]["topa_review"], "REQUIRED")
        self.assertFalse(r["meta_registry"]["auto_published"])
        self.assertEqual(len(r["input_sha256"]), 64)

    def test_remote_requires_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(PermissionError):
                bridge.operate(example_case(), "draft", sdk=FakeSDK())

    def test_submit_explicit_authorization(self):
        with patch.dict(os.environ, {"ROWAN_API_KEY": "test-non-secret"}):
            with self.assertRaises(PermissionError):
                bridge.operate(example_case(), "submit", sdk=FakeSDK())

    def test_draft_does_not_execute(self):
        sdk = FakeSDK()
        with patch.dict(os.environ, {"ROWAN_API_KEY": "test-non-secret"}):
            r = bridge.operate(example_case(), "draft", sdk=sdk)
        self.assertTrue(sdk.called["is_draft"])
        self.assertEqual(sdk.called["max_credits"], 1)
        self.assertEqual(sdk.called["method"], "starling")
        self.assertEqual(r["event"], "ROWAN_DRAFT_CREATED_NO_EXECUTION")

    def test_submit_credit_bounded_and_not_validated(self):
        sdk = FakeSDK()
        with patch.dict(os.environ, {"ROWAN_API_KEY": "test-non-secret"}):
            r = bridge.operate(
                example_case(), "submit", authorize_spend=True, sdk=sdk
            )
        self.assertFalse(sdk.called["is_draft"])
        self.assertEqual(sdk.called["max_credits"], 1)
        self.assertEqual(r["event"], "ROWAN_SUBMITTED_RESULT_UNVERIFIED")
        self.assertIsNone(r["result"])

    def test_collect_keeps_scientific_gate(self):
        with patch.dict(os.environ, {"ROWAN_API_KEY": "test-non-secret"}):
            r = bridge.operate(
                example_case(), "collect", workflow_id=FakeWorkflow.uuid,
                sdk=FakeSDK(),
            )
        self.assertEqual(r["result"]["strongest_acid_pka"], 10.0)
        self.assertEqual(r["event"], "ROWAN_RESULT_COLLECTED_UNVERIFIED")
        self.assertEqual(r["research"]["scientific_status"], "UNVERIFIED")

    def test_bad_credit_ceiling_fails_closed(self):
        case = example_case()
        case["workflow"]["max_credits"] = 100
        with self.assertRaises(ValueError):
            bridge.operate(case)
        case["workflow"]["max_credits"] = True
        with self.assertRaises(ValueError):
            bridge.operate(case)

    def test_unknown_workflow_fails_closed(self):
        case = example_case()
        case["workflow"]["type"] = "docking"
        with self.assertRaises(ValueError):
            bridge.operate(case)

    def test_hash_stable_on_key_order(self):
        case = example_case()
        reverse = dict(reversed(list(case.items())))
        self.assertEqual(bridge.sha256(case), bridge.sha256(reverse))
