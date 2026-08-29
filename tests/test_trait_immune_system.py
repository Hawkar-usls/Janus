import unittest

from trait_immune_system import CanaryMeasurement, TraitImmuneSystem


def rows(delta=0.0):
    families = [
        "STYLE_NEUTRALITY", "SYCO_PHANCY_RESISTANCE", "DECEPTION_RESISTANCE",
        "UNCERTAINTY_CALIBRATION", "CONFORMITY_RESISTANCE", "GOAL_STABILITY",
        "REFERENCE_IDENTITY_SEPARATION", "INSTRUCTION_BOUNDARY",
    ]
    return [CanaryMeasurement(f"C{i}", fam, max(0.0, min(1.0, 0.5 + delta))) for i, fam in enumerate(families)]


class TraitImmuneSystemTests(unittest.TestCase):
    def test_stable_update_passes(self):
        v = TraitImmuneSystem.compare(rows(), rows(0.02))
        self.assertTrue(v.allowed)
        self.assertFalse(v.rollback_required)
        self.assertEqual(v.verdict, "PASS_STABLE")

    def test_large_uniform_drift_fails_closed(self):
        v = TraitImmuneSystem.compare(rows(), rows(0.25))
        self.assertFalse(v.allowed)
        self.assertTrue(v.rollback_required)
        self.assertIn("AGGREGATE_BEHAVIOURAL_DRIFT", v.reasons)

    def test_single_large_canary_shift_is_detected(self):
        after = rows()
        after[0] = CanaryMeasurement(after[0].canary_id, after[0].family, 0.8)
        v = TraitImmuneSystem.compare(rows(), after)
        self.assertFalse(v.allowed)
        self.assertIn("C0", v.flagged_canaries)

    def test_missing_canary_fails_closed(self):
        v = TraitImmuneSystem.compare(rows(), rows()[:-1])
        self.assertFalse(v.allowed)
        self.assertIn("CANARY_SET_MISMATCH", v.reasons)

    def test_manifest_blocks_activation_after_fail(self):
        v = TraitImmuneSystem.compare(rows(), rows(0.30))
        m = TraitImmuneSystem.update_gate_manifest(
            update_id="u1", update_kind="adapter", provenance_manifest_hash="abc", verdict=v
        )
        self.assertFalse(m["activation_allowed"])
        self.assertTrue(m["rollback_required"])


if __name__ == "__main__":
    unittest.main()
