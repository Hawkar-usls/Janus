import unittest

from mcp_gateway.core import checkpoint_document, sanitize_repo_path, safe_slug, topa_packet


class CoreTests(unittest.TestCase):
    def test_safe_repo_path(self):
        self.assertEqual(sanitize_repo_path("data/x.json"), "data/x.json")
        with self.assertRaises(ValueError):
            sanitize_repo_path("../secret")
        with self.assertRaises(ValueError):
            sanitize_repo_path("a/../b")

    def test_safe_slug(self):
        self.assertEqual(safe_slug("SKIN / EXIT CLOCK"), "SKIN-EXIT-CLOCK")

    def test_topa_packet_empirical(self):
        packet = topa_packet("A strange signal exists", "empirical")
        self.assertEqual(packet["required_output"]["status"], "UNRESOLVED")
        self.assertIn("name_falsifiers", packet["host_workflow"])

    def test_topa_packet_math(self):
        packet = topa_packet("P = NP", "mathematical")
        self.assertEqual(packet["required_output"]["status"], "OPEN")
        self.assertIn("search_proof_and_counterexample_routes", packet["host_workflow"])

    def test_checkpoint_has_hash(self):
        doc = checkpoint_document(project="X", kind="progress", summary="s", state="OPEN")
        self.assertEqual(len(doc["content_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
