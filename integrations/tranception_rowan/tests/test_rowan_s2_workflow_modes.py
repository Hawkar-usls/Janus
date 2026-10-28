"""Static safety contract for the source-bound S2 GitHub Actions collection schedule.

Never contacts Rowan, reads a GitHub secret, or spends credits.
"""
from pathlib import Path
import unittest

WORKFLOW = Path(__file__).resolve().parents[3] / ".github/workflows/rowan-bbc1-stratified-pilot.yml"


class RowanS2WorkflowModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = WORKFLOW.read_text(encoding="utf-8")

    def test_default_manual_run_cannot_submit(self):
        s = self.source
        self.assertIn("default: collect_only", s)
        self.assertIn("          - collect_only", s)
        self.assertIn("          - submit_and_collect", s)
        self.assertIn("inputs.mode == 'submit_and_collect'", s)

    def test_exact_one_preapproved_daily_spend_gate(self):
        s = self.source
        daily = "github.event.schedule == '17 7 * * *'"
        condition = (
            "if: ${{ (github.event_name == 'schedule' && " + daily +
            ") || (github.event_name == 'workflow_dispatch' && " +
            "inputs.mode == 'submit_and_collect') }}"
        )
        # First condition is the new pre-submit completion freeze. Both
        # downstream ACK and provider-submit steps additionally require its
        # should_submit output, so an attested 8/8 can never reach billing.
        guarded = (
            "if: \u0024{{ ((github.event_name == 'schedule' && " + daily +
            ") || (github.event_name == 'workflow_dispatch' && " +
            "inputs.mode == 'submit_and_collect')) && " +
            "steps.completion.outputs.should_submit == 'true' }}"
        )
        self.assertEqual(s.count(condition), 1)
        self.assertEqual(s.count(guarded), 2)
        self.assertIn("id: completion", s)
        self.assertIn("python integrations/tranception_rowan/rowan_s2_completion_freeze.py", s)
        self.assertEqual(s.count("python integrations/tranception_rowan/rowan_meta_ack_gate.py"), 1)
        self.assertEqual(s.count("python integrations/tranception_rowan/rowan_bbc1_s2_pilot.py"), 1)
        ack = s.split("      - name: Require durable Meta Registry ACK for previous completed pilot run\n", 1)[1]
        submit = s.split("      - name: Execute at most one frozen mutant with weekly credits only\n", 1)[1]
        self.assertTrue(ack.startswith("        " + guarded + "\n"))
        self.assertTrue(submit.startswith("        " + guarded + "\n"))
        self.assertIn("JANUS_ROWAN_S2_APPROVED: 'TRUE'", submit)

    def test_six_readonly_collection_schedules_never_select_daily_submit(self):
        s = self.source
        self.assertIn("    - cron: '17 7 * * *'", s)
        self.assertIn("    - cron: '9 0,4,8,12,16,20 * * *'", s)
        self.assertEqual(s.count("    - cron:"), 2)
        coll = s.split("      - name: Read progress and matched experimental labels without spending\n", 1)[1]
        self.assertNotIn("if:", coll.split("        run: python integrations/tranception_rowan/rowan_bbc1_s2_collect.py", 1)[0])
        self.assertEqual(s.count("python integrations/tranception_rowan/rowan_bbc1_s2_collect.py"), 1)
        self.assertIn("bbc1-pilot-eight-public-safe-${{ github.run_id }}", s)
        self.assertIn("persist-credentials: false", s)


if __name__ == "__main__":
    unittest.main()
