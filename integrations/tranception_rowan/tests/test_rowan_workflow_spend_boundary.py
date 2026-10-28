"""No-provider CI guard across JANUS Rowan spending workflow entry points.

Prevent accidental re-enablement of unattended billing after the BBC1
collect-only hardening. These checks only read workflow text.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
WF = ROOT / ".github/workflows"

MANUAL_COSTED = (
    "rowan-1ey0-direct-weekly-pilot.yml",
    "rowan-1ey0-T8A-one-mutant-weekly.yml",
    "rowan-scoby-milk-lalba-single-weekly.yml",
    "rowan-scoby-two-protein-pilots.yml",
    "rowan-research-manual.yml",
)
SCHEDULED_READONLY = (
    "rowan-autonomous-readonly.yml",
    "rowan-credit-gate.yml",
)


class RowanWorkflowSpendBoundary(unittest.TestCase):
    def source(self, name):
        return (WF / name).read_text(encoding="utf-8")

    def test_bbc1_wt_k5a_cron_is_collect_only(self):
        y = self.source("rowan-bbc1-budgeted-autopilot.yml")
        self.assertIn("schedule:", y)
        self.assertIn("rowan_bbc1_collect.py", y)
        self.assertNotIn("run: python integrations/tranception_rowan/rowan_bbc1_autopilot.py", y)
        self.assertNotIn("JANUS_ROWAN_APPROVED_WEEKLY_RUN: 'TRUE'", y)

    def test_s2_submit_depends_on_current_manual_yes_and_freeze(self):
        y = self.source("rowan-bbc1-stratified-pilot.yml")
        self.assertIn("schedule:", y)
        self.assertIn("default: collect_only", y)
        self.assertIn("default: 'NO'", y)
        self.assertEqual(y.count("inputs.authorize_weekly_credit_spend == 'YES'"), 3)
        self.assertNotIn("github.event_name == 'schedule' &&", y)
        self.assertEqual(y.count("steps.completion.outputs.should_submit == 'true'"), 2)
        self.assertIn("rowan_meta_ack_gate.py", y)

    def test_all_other_costed_workflows_manual_only_with_explicit_yes(self):
        for name in MANUAL_COSTED:
            with self.subTest(name=name):
                y = self.source(name)
                self.assertIn("workflow_dispatch:", y)
                self.assertNotIn("\n  schedule:", y)
                self.assertRegex(y, r'default:\\s*[\"\\']?NO[\"\\']?')
                self.assertIn("YES", y)
                self.assertIn("submit", y)
                self.assertIn("ROWAN_API_KEY", y)
                self.assertTrue(
                    "inputs.authorize_weekly_credit_spend == 'YES'" in y or
                    'CREDIT_PERMISSION" != YES' in y,
                    "Missing explicit manual authorization: " + name,
                )

    def test_remaining_scheduled_status_workflows_cannot_submit(self):
        for name in SCHEDULED_READONLY:
            with self.subTest(name=name):
                y = self.source(name)
                self.assertIn("schedule:", y)
                self.assertNotIn("submit_protein_cofolding_workflow", y)
                self.assertNotIn("rowan_bbc1_autopilot.py", y)
                self.assertNotIn("rowan_bbc1_s2_pilot.py", y)
                self.assertNotIn("inputs.authorize_weekly_credit_spend", y)

    def test_ci_rechecks_rowan_workflow_changes(self):
        y = self.source("tranception-rowan-check.yml")
        self.assertEqual(y.count("'.github/workflows/rowan-*.yml'"), 2)
        self.assertIn("python -m unittest discover -s integrations/tranception_rowan/tests -v", y)


if __name__ == "__main__":
    unittest.main()
