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
    def test_no_workflow_job_or_paid_call(self):
        u=SimpleNamespace(weekly_credits=20,credits=490,individual_subscription=None,organization=None)
        x=m.evaluate(u)
        self.assertEqual(x["provider_computation"],"NOT_ATTEMPTED")
        self.assertEqual(x["credit_spending"],"NOT_ATTEMPTED")

if __name__=="__main__": unittest.main()
