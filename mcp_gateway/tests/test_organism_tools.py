import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp_gateway.organism_tools import (
    ORGAN_REPOS,
    PRIVATE_ORGAN_ENVS,
    _resolve_organ,
    build_spiral_pass_packet,
    validate_spiral_transition,
)


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

    def test_spiral_constitution_is_explicitly_not_a_cycle(self):
        protocol = json.loads(Path("organism/JANUS_SPIRAL_TRANCEPTION_v1.2.json").read_text(encoding="utf-8"))
        self.assertTrue(protocol["not_a_cycle"])
        self.assertEqual(
            "RETURN_TO_QUESTION_AT_HIGHER_STATE__NEVER_RETURN_TO_IDENTICAL_STATE",
            protocol["spiral_law"],
        )
        self.assertTrue(protocol["tranception"]["available_at_any_node"])
        self.assertEqual("WHOLE_ORGANISM_OPERATOR_NOT_ORGAN", protocol["tranception"]["kind"])

    def test_spiral_packet_is_deterministic_and_5d(self):
        kwargs = {
            "question": "Does the candidate survive the next falsification gate?",
            "turn": 2,
            "parent_state_hash": "abc123",
            "active_constraints": ["B", "A", "A"],
            "evidence_refs": ["e2", "e1"],
        }
        first = build_spiral_pass_packet(**kwargs)
        second = build_spiral_pass_packet(**kwargs)
        self.assertEqual(first["input_state_hash"], second["input_state_hash"])
        self.assertEqual(5, len(first["dimensions"]))
        self.assertTrue(first["checkpoint_anywhere"])
        self.assertEqual(["A", "B"], first["active_constraints"])

    def test_repeated_state_hash_is_plateau_not_new_turn(self):
        result = validate_spiral_transition("same", "same", ["NEW_EVIDENCE"])
        self.assertFalse(result["ascend_allowed"])
        self.assertEqual("PLATEAU_OR_HOLD", result["verdict"])
        self.assertEqual("REPEATED_STATE_HASH_IS_NOT_A_NEW_TURN", result["reason"])

    def test_changed_hash_without_valid_reason_fails_closed(self):
        result = validate_spiral_transition("old", "new", ["PRETTY_IDEA"])
        self.assertFalse(result["ascend_allowed"])
        self.assertEqual("HOLD", result["verdict"])
        self.assertEqual(["PRETTY_IDEA"], result["unknown_state_change_reasons"])

    def test_changed_hash_with_valid_reason_may_ascend(self):
        result = validate_spiral_transition("old", "new", ["NEW_DISCRIMINATING_TEST"])
        self.assertTrue(result["ascend_allowed"])
        self.assertEqual("ASCEND_ALLOWED", result["verdict"])
        self.assertEqual("ASCEND_ALLOWED != CLAIM_CONFIRMED", result["boundary"])


if __name__ == "__main__":
    unittest.main()
