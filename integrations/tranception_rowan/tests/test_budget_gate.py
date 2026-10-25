import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

p=Path(__file__).resolve().parents[1]/"budget_gate.py"
spec=importlib.util.spec_from_file_location("budget_gate",p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_free_user_with_signup_grant(self):
        u=SimpleNamespace(weekly_credits=20.,credits=500.,individual_subscription=None,organization=None)
        x=m.evaluate(u)
        self.assertEqual(x["account_class"],"NO_ORGANIZATION")
        self.assertEqual(x["PERSONAL_NONWEEKLY_GE_100"],"YES")
        self.assertNotIn("500.0",repr(x))
        self.assertEqual(x["signup_grant_provenance"],"NOT_DISTINGUISHABLE_FROM_PURCHASED_IN_WHOAMI")
    def test_paid_subscription_not_cleared(self):
        u=SimpleNamespace(weekly_credits=2000.,credits=500.,individual_subscription={"plan":"paid"},organization=None)
        self.assertEqual(m.evaluate(u)["subscription_class"],"SUBSCRIPTION_REQUIRES_REVIEW")
    def test_org_not_cleared(self):
        u=SimpleNamespace(weekly_credits=20.,credits=500.,individual_subscription=None,organization={})
        self.assertEqual(m.evaluate(u)["account_class"],"HAS_ORGANIZATION")
    def test_missing_fields_fail_closed(self):
        self.assertEqual(m.evaluate(SimpleNamespace())["PERSONAL_WEEKLY_GE_10"],"NO_OR_UNKNOWN")
    def test_nonfinite_values_fail_closed(self):
        u=SimpleNamespace(weekly_credits=float("inf"), credits=-1)
        self.assertEqual(m.evaluate(u)["PERSONAL_NONWEEKLY_GE_20"],"NO_OR_UNKNOWN")
    def test_bool_is_not_numeric_balance(self):
        self.assertIsNone(m.finite_nonnegative(True))
    def test_exact_preregistered_30_weekly_credit_threshold_without_raw_values(self):
        for weekly,expected in ((29.99,"NO_OR_UNKNOWN"),(30,"YES"),(30.01,"YES")):
            account=SimpleNamespace(weekly_credits=None,credits=499,
                        individual_subscription=None,
                        organization=SimpleNamespace(weekly_credits=weekly,credits=499))
            bands=m.evaluate(account)
            self.assertEqual(bands["ORG_WEEKLY_GE_30"],expected)
            self.assertEqual(bands["provider_computation"],"NOT_ATTEMPTED")
            self.assertEqual(bands["credit_spending"],"NOT_ATTEMPTED")
            self.assertNotIn(str(weekly),list(bands.values()))
        unknown=m.evaluate(SimpleNamespace(organization=SimpleNamespace()))
        self.assertEqual(unknown["ORG_WEEKLY_GE_30"],"NO_OR_UNKNOWN")

    def test_no_workflow_job_or_paid_call(self):
        u=SimpleNamespace(weekly_credits=20,credits=490,individual_subscription=None,organization=None)
        x=m.evaluate(u)
        self.assertEqual(x["provider_computation"],"NOT_ATTEMPTED")
        self.assertEqual(x["credit_spending"],"NOT_ATTEMPTED")

if __name__=="__main__": unittest.main()
