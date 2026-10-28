"""Static CI regression: no GitHub cron may authorize new Rowan billing.
This test uses repository text only, never imports Rowan SDK or secrets.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
BUDGETED = ROOT / ".github/workflows/rowan-bbc1-budgeted-autopilot.yml"
S2 = ROOT / ".github/workflows/rowan-bbc1-stratified-pilot.yml"


class NoUnattendedSpend(unittest.TestCase):
    def test_budgeted_autopilot_is_readonly(self):
        source = BUDGETED.read_text(encoding="utf-8")
        self.assertIn("cron: '37 5 * * *'", source)
        self.assertNotIn("run: python integrations/tranception_rowan/rowan_bbc1_autopilot.py", source)
        self.assertIn("run: python integrations/tranception_rowan/rowan_bbc1_collect.py", source)
        self.assertNotIn("JANUS_ROWAN_APPROVED_WEEKLY_RUN: 'TRUE'", source)

    def test_s2_cron_only_collects(self):
        source = S2.read_text(encoding="utf-8")
        self.assertIn("cron: '17 7 * * *'", source)
        self.assertIn("cron: '9 0,4,8,12,16,20 * * *'", source)
        self.assertIn("default: 'NO'", source)
        self.assertEqual(source.count("inputs.authorize_weekly_credit_spend == 'YES'"), 3)
        self.assertNotIn("github.event_name == 'schedule' && github.event.schedule", source)
        self.assertIn("rowan_bbc1_s2_collect.py", source)


if __name__ == "__main__":
    unittest.main()
