import json
import unittest
from pathlib import Path

from mcp_gateway.organism_tools import ORGAN_REPOS, _resolve_organ


class OrganismToolsTests(unittest.TestCase):
    def test_expected_core_organs_are_allowlisted(self):
        expected = {
            "gateway",
            "memory",
            "proof_spine",
            "guardian_cortex",
            "orchestrator",
            "sensorimotor_mesh",
            "hypothesis_metabolism",
            "symbolic_imagination",
            "anomaly_lab",
            "observatory",
        }
        self.assertEqual(expected, set(ORGAN_REPOS))

    def test_resolver_rejects_arbitrary_repository(self):
        with self.assertRaises(ValueError):
            _resolve_organ("owner/anything")

    def test_canonical_manifest_matches_allowlist(self):
        manifest = json.loads(Path("organism/JANUS_ORGANISM_v1.json").read_text(encoding="utf-8"))
        repos = {key: value["repo"] for key, value in manifest["members"].items()}
        self.assertEqual(ORGAN_REPOS, repos)
        self.assertEqual("DENY", manifest["design"]["cross_repo_write_default"])


if __name__ == "__main__":
    unittest.main()
