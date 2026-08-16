import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PORTAL_PATH = ROOT / "portal" / "portal.py"
MANIFEST_PATH = ROOT / "portal" / "manifest.json"
SPEC = importlib.util.spec_from_file_location("janus_portal", PORTAL_PATH)
portal = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(portal)


class PortalTests(unittest.TestCase):
    def setUp(self):
        self.manifest = portal.load_manifest(MANIFEST_PATH)

    def test_default_route_is_demihead_reference_only(self):
        receipt = portal.resolve_route(self.manifest)
        self.assertEqual(receipt["status"], "ROUTE_RESOLVED_REFERENCE_ONLY")
        self.assertEqual(receipt["route"]["destination_id"], "DEMIHEAD")
        self.assertFalse(receipt["effect_authorized"])
        self.assertFalse(receipt["provider_realized"])
        self.assertEqual(receipt["authority_delta"], 0)
        self.assertEqual(receipt["mass_effect_budget_delta"], 0)

    def test_unknown_destination_fails_closed(self):
        receipt = portal.resolve_route(self.manifest, "SOMEWHERE_ELSE")
        self.assertEqual(receipt["status"], "UNKNOWN_DESTINATION_FAIL_CLOSED")
        self.assertIsNone(receipt["route"])
        self.assertFalse(receipt["effect_authorized"])

    def test_language_is_metadata_not_evidence(self):
        ru = portal.resolve_route(self.manifest, "DEMIHEAD", "ru")
        uk = portal.resolve_route(self.manifest, "DEMIHEAD", "uk")
        en = portal.resolve_route(self.manifest, "DEMIHEAD", "en")
        for receipt in (ru, uk, en):
            self.assertEqual(receipt["truth_claim"], "NOT_MADE")
            self.assertFalse(receipt["evidence_state_mutated"])
            self.assertEqual(receipt["route"]["route_ref"], "repo:Hawkar-usls/Demi_Head")

    def test_arbitrary_url_manifest_is_rejected(self):
        manifest = json.loads(json.dumps(self.manifest))
        manifest["destinations"][0]["route_ref"] = "https://example.com/whatever"
        with self.assertRaises(portal.PortalError):
            portal.validate_manifest(manifest)

    def test_destination_cannot_inherit_effect_authority(self):
        manifest = json.loads(json.dumps(self.manifest))
        manifest["destinations"][0]["portal_grants_effect_authority"] = True
        with self.assertRaises(portal.PortalError):
            portal.validate_manifest(manifest)

    def test_decline_has_no_penalty(self):
        receipt = portal.decline_route(self.manifest)
        self.assertEqual(receipt["status"], "ROUTE_DECLINED_VALID")
        self.assertFalse(receipt["penalty"])

    def test_self_test(self):
        self.assertEqual(portal.self_test(self.manifest)["self_test"], "PASS")


if __name__ == "__main__":
    unittest.main()
