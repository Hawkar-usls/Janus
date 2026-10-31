"""Static-only checks of archived owner NAS ESP32-P4 research sketch."""
from pathlib import Path
import json
import unittest

ROOT=Path(__file__).resolve().parents[1]/"hardware"/"legacy"/"nas_p4_dual_swarm_v1_1"

class HistoricalP4Proof(unittest.TestCase):
    def test_archived_sources_and_roles(self):
        doc=json.loads((ROOT/"manifest.json").read_text())
        source=(ROOT/"JANUS_P4_DUAL_SWARM_CORE_v1_1.ino").read_text()
        self.assertEqual(doc["target"],"ESP32-P4")
        self.assertFalse(doc["native_wifi"])
        self.assertEqual(set(doc["roles"]),{"P4_A","P4_B","P4_C"})
        self.assertIn("JANUS_DEFAULT_ROLE",source)
        self.assertIn("JP4",source)
    def test_no_build_or_weights(self):
        self.assertTrue((ROOT/"README.md").is_file())
        self.assertEqual(list(ROOT.glob("*.bin")),[])
        self.assertEqual(list(ROOT.glob("*.elf")),[])

if __name__=="__main__":
    unittest.main()
