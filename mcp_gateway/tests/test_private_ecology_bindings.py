import os
import unittest
from unittest.mock import patch

from mcp_gateway.organism_tools import PRIVATE_ORGAN_ENVS, _resolve_organ


class PrivateEcologyBindingTests(unittest.TestCase):
    def test_v13_private_subtissues_are_declared(self):
        self.assertEqual(
            "JANUS_PRIVATE_MEASUREMENT_REPO",
            PRIVATE_ORGAN_ENVS["private_measurement_substrate"],
        )
        self.assertEqual(
            "JANUS_PRIVATE_GENESIS_WORLD_REPO",
            PRIVATE_ORGAN_ENVS["private_genesis_world"],
        )

    def test_private_measurement_substrate_fails_closed(self):
        env_name = PRIVATE_ORGAN_ENVS["private_measurement_substrate"]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(env_name, None)
            with self.assertRaises(ValueError):
                _resolve_organ("private_measurement_substrate")

    def test_private_genesis_world_fails_closed(self):
        env_name = PRIVATE_ORGAN_ENVS["private_genesis_world"]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(env_name, None)
            with self.assertRaises(ValueError):
                _resolve_organ("private_genesis_world")

    def test_private_subtissues_resolve_only_after_explicit_binding(self):
        bindings = {
            "JANUS_PRIVATE_MEASUREMENT_REPO": "owner/private-measurement-placeholder",
            "JANUS_PRIVATE_GENESIS_WORLD_REPO": "owner/private-world-placeholder",
        }
        with patch.dict(os.environ, bindings, clear=False):
            self.assertEqual(
                ("private_measurement_substrate", "owner/private-measurement-placeholder"),
                _resolve_organ("private_measurement_substrate"),
            )
            self.assertEqual(
                ("private_genesis_world", "owner/private-world-placeholder"),
                _resolve_organ("private_genesis_world"),
            )


if __name__ == "__main__":
    unittest.main()
