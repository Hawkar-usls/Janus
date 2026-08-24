import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp_gateway.organism_tools import ORGAN_REPOS, PRIVATE_ORGAN_ENVS, _resolve_organ


class OrganismToolsTests(unittest.TestCase):
    def test_expected_public_organs_are_allowlisted(self):
        expected = {
            "gateway",
            "memory",
            "proof_spine",
            "causal_witness",
            "guardian_cortex",
            "orchestrator",
            "operator_hands",
            "left_context",
            "right_association",
            "measurement_bench",
            "sensorimotor_mesh",
            "hypothesis_metabolism",
            "symbolic_imagination",
            "anomaly_lab",
            "observatory",
            "simulation_habitat",
            "voice",
            "physical_voice",
            "materials_lab",
            "feline_timing_lab",
            "social_membrane",
        }
        self.assertEqual(expected, set(ORGAN_REPOS))

    def test_resolver_rejects_arbitrary_repository(self):
        with self.assertRaises(ValueError):
            _resolve_organ("owner/anything")

    def test_private_organ_is_fail_closed_without_explicit_binding(self):
        env_name = PRIVATE_ORGAN_ENVS["somatosensory_skin"]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(env_name, None)
            with self.assertRaises(ValueError):
                _resolve_organ("somatosensory_skin")

    def test_private_organ_requires_explicit_locator(self):
        env_name = PRIVATE_ORGAN_ENVS["somatosensory_skin"]
        with patch.dict(os.environ, {env_name: "Hawkar-usls/private-skin-placeholder"}, clear=False):
            key, repo = _resolve_organ("somatosensory_skin")
            self.assertEqual("somatosensory_skin", key)
            self.assertEqual("Hawkar-usls/private-skin-placeholder", repo)

    def test_canonical_manifest_matches_public_allowlist(self):
        manifest = json.loads(Path("organism/JANUS_ORGANISM_v1.json").read_text(encoding="utf-8"))
        public_repos = {
            key: value["repo"]
            for key, value in manifest["members"].items()
            if value.get("mcp_default") is True
        }
        self.assertEqual(ORGAN_REPOS, public_repos)
        self.assertEqual("DENY", manifest["design"]["cross_repo_write_default"])
        self.assertFalse(manifest["design"]["private_repo_names_in_public_manifest"])

    def test_private_manifest_slot_has_no_public_repo_name(self):
        manifest = json.loads(Path("organism/JANUS_ORGANISM_v1.json").read_text(encoding="utf-8"))
        skin = manifest["members"]["somatosensory_skin"]
        self.assertIsNone(skin["repo"])
        self.assertFalse(skin["mcp_default"])
        self.assertEqual(PRIVATE_ORGAN_ENVS["somatosensory_skin"], skin["private_locator_env"])


if __name__ == "__main__":
    unittest.main()
