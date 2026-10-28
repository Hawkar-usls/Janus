"""Regression and adversarial tests: completed 8/8 must NEVER resubmit."""
import unittest
from rowan_s2_completion_freeze import gate

BASE={
  "schema_id":"janus.rowan.observer_state.v1",
  "latest_durable_external_run":40593272360,
  "last_completed_mutants":8,
  "underlying_state_is_derived_from":"a"*64,
  "autonomous_spend_authority":"ONLY_FROZEN_EIGHT_MUTANT_S2_WITH_EXISTING_WEEKLY_BUDGET",
  "new_unreviewed_studies":"PROPOSAL_ONLY_NOT_AUTO_EXECUTED",
  "budget_guard_alert_mutations":[],
}

class CompletionGateTest(unittest.TestCase):
    def test_completed_freezes(self):
        status,can_submit=gate(BASE)
        self.assertEqual(status,"FROZEN_COMPLETED_8_OF_8_NO_MORE_SUBMISSIONS")
        self.assertFalse(can_submit)

    def test_existing_incomplete_reaches_other_gates(self):
        x={**BASE,"last_completed_mutants":7}
        status,eligible=gate(x)
        self.assertTrue(eligible)
        self.assertIn("EXISTING_ACK_AND_CREDIT_GATES",status)

    def test_budget_alert_hard_stops_even_if_incomplete(self):
        x={**BASE,"last_completed_mutants":7,"budget_guard_alert_mutations":["G64Q"]}
        self.assertEqual(gate(x),("HOLD_EXISTING_BUDGET_ALERTS",False))

    def test_no_source_fails_closed(self):
        self.assertEqual(gate(None),("HOLD_MISSING_OR_INVALID_CANONICAL",False))

    def test_authority_drift_fails_closed(self):
        x={**BASE,"new_unreviewed_studies":"AUTO_SUBMIT"}
        self.assertEqual(gate(x),("HOLD_SCIENCE_OR_BUDGET_AUTHORITY_DRIFT",False))

    def test_source_drift_fails_closed(self):
        x={**BASE,"underlying_state_is_derived_from":"0"*12}
        self.assertEqual(gate(x),("HOLD_UNATTESTED_SOURCE_DIGEST",False))

    def test_progress_cannot_exceed_frozen_eight(self):
        x={**BASE,"last_completed_mutants":9}
        self.assertEqual(gate(x),("HOLD_INVALID_PILOT_PROGRESS",False))

    def test_bool_is_not_int(self):
        x={**BASE,"last_completed_mutants":True}
        self.assertEqual(gate(x),("HOLD_INVALID_PILOT_PROGRESS",False))

if __name__=="__main__":
    unittest.main()
