"""Synthetic, no-network hard checks for cloud-only provenance sequencing."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import unittest

module=Path(__file__).resolve().parents[1]/"rowan_meta_ack_gate.py"
spec=spec_from_file_location("rowan_meta_ack_gate",module)
m=module_from_spec(spec);spec.loader.exec_module(m)

def state(latest=100, alerts=None):
    return {
        "schema_id":"janus.rowan.observer_state.v1",
        "latest_durable_external_run":latest,
        "new_unreviewed_studies":"PROPOSAL_ONLY_NOT_AUTO_EXECUTED",
        "autonomous_spend_authority":"ONLY_FROZEN_EIGHT_MUTANT_S2_WITH_EXISTING_WEEKLY_BUDGET",
        "budget_guard_alert_mutations":alerts or [],
    }

class Tests(unittest.TestCase):
    def test_proper_prior_ack_passes(self):
        self.assertEqual(m.check(100,state(100),101),"META_REGISTRY_ACK_VERIFIED")
    def test_missing_archive_blocks(self):
        with self.assertRaises(m.ArchiveNotConfirmed):
            m.check(100,None,101)
    def test_stale_ack_blocks(self):
        with self.assertRaises(m.ArchiveNotConfirmed):
            m.check(100,state(99),101)
    def test_future_ack_blocks(self):
        with self.assertRaises(m.ArchiveNotConfirmed):
            m.check(100,state(101),101)
    def test_concurrent_same_run_rejected(self):
        with self.assertRaises(m.ArchiveNotConfirmed):
            m.check(100,state(100),100)
    def test_budget_overcharge_stops(self):
        with self.assertRaisesRegex(m.ArchiveNotConfirmed,"BUDGET_ALERT_HARD_STOP"):
            m.check(100,state(100,["G53A"]),101)
    def test_changed_authority_stops(self):
        s=state()
        s["autonomous_spend_authority"]="UNRESTRICTED"
        with self.assertRaises(m.ArchiveNotConfirmed):
            m.check(100,s,101)
    def test_no_previous_first_run_allowed(self):
        self.assertEqual(m.check(None,None,1),"NO_PREVIOUS_SUCCESSFUL_RUN")
    def test_bad_state_schema_stops(self):
        s=state()
        s["schema_id"]="untrusted"
        with self.assertRaises(m.ArchiveNotConfirmed):
            m.check(100,s,101)

if __name__=="__main__":unittest.main()
