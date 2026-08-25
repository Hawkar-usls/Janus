import json
import unittest
from pathlib import Path


class RepositoryEcologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topology = json.loads(Path("organism/JANUS_ORGANISM_v1.json").read_text(encoding="utf-8"))
        cls.ecology_path = Path("organism/JANUS_REPOSITORY_ECOLOGY_v1.3.json")
        cls.ecology_text = cls.ecology_path.read_text(encoding="utf-8")
        cls.ecology = json.loads(cls.ecology_text)

    def test_ecology_extends_current_topology_and_spiral(self):
        self.assertEqual("organism/JANUS_ORGANISM_v1.json", self.ecology["extends_topology"])
        self.assertEqual("organism/JANUS_SPIRAL_TRANCEPTION_v1.2.json", self.ecology["extends_execution"])
        self.assertEqual("1.3.0", self.ecology["version"])

    def test_private_subtissues_fail_closed_and_do_not_publish_repo_names(self):
        private = self.ecology["private_subtissues"]
        self.assertEqual(
            {"private_measurement_substrate", "private_genesis_world"},
            set(private),
        )
        for item in private.values():
            self.assertIsNone(item["repo"])
            self.assertFalse(item["mcp_default"])
            self.assertTrue(item["locator_env"].startswith("JANUS_PRIVATE_"))

        # Public ecology must never disclose the concrete names of private repositories.
        self.assertNotIn("Hawkar-usls/janus-io\"", self.ecology_text)
        self.assertNotIn("Hawkar-usls/JanusMMORPG", self.ecology_text)

    def test_every_subtissue_parent_is_a_canonical_organ(self):
        organs = set(self.topology["members"])
        for collection in ("private_subtissues", "public_subtissues"):
            for item in self.ecology[collection].values():
                self.assertIn(item["parent_organ"], organs)

    def test_upstream_dependencies_do_not_claim_base_authorship(self):
        for item in self.ecology["toolchain_dependencies"].values():
            self.assertFalse(item["base_authorship_claimed"])

    def test_external_research_instruments_are_not_organs(self):
        for item in self.ecology["external_research_instruments"].values():
            self.assertFalse(item["organ"])

    def test_ecology_preserves_zero_authority_transport_law(self):
        self.assertIn("AUTHORITY_DELTA_ON_TRANSPORT = 0", self.ecology["laws"])
        self.assertIn("SUBTISSUE_PARENT != AUTHORITY_INHERITANCE", self.ecology["laws"])


if __name__ == "__main__":
    unittest.main()
