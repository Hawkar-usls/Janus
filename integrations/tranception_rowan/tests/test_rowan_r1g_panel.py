import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rowan_r1g_panel as r
import rowan_r1g2_credit_guard as g
from types import SimpleNamespace as S
class Tests(unittest.TestCase):
    def test_panel(self):
        self.assertEqual(len(r.CASES),7)
        self.assertEqual(sum(n for _,_,n in r.CASES),27)
        self.assertEqual(r.identity("2LZM","I3T",5)[2][2],"T")
    def test_all_submit_blocked(self):
        # No SDK methods available; verifies rejection before all provider calls.
        for approval,attempt in ((True,"1"),(False,"1"),(True,"2")):
            with self.assertRaises(r.Hold):r.run(S(),"submit",approval,attempt,lambda x:None)
    def test_collect_no_provider_write(self):
        from unittest.mock import patch
        with patch.object(r,"find",return_value=None):
            x=r.run(S(),"collect",False,"1",lambda x:None)
        self.assertEqual(x["status"],"PARTIAL_READ_ONLY");self.assertEqual(x["new_jobs"],0)
    def test_censored(self):
        x=g.report(r.WT["BBC1"],5,"A100_80GB",g.FEATURES)
        self.assertEqual(x["risk"],"OBSERVED_CAP_BREACH")
        self.assertEqual(x["stopped_cost_lower_bound"],"10.26")
        self.assertIsNone(x["estimated_cost"])
    def test_unknown(self):
        for seq,n,hardware,features in [(r.WT["2LZM"],5,"A100_80GB",g.FEATURES),(r.WT["BBC1"],3,"A100_80GB",g.FEATURES),(r.WT["BBC1"],5,"H100",g.FEATURES),(r.WT["BBC1"],5,"A100_80GB",{**g.FEATURES,"use_msa_server":False})]:
            x=g.report(seq,n,hardware,features);self.assertEqual(x["risk"],"UNKNOWN");self.assertFalse(x["new_credit_authorization"])
    def test_existing_single_is_not_authorized(self):
        x=g.report(r.WT["BBC1"],1,"A100_80GB",g.FEATURES)
        self.assertEqual(x["risk"],"OBSERVED_BELOW_CAP_NOT_CALIBRATED")
        self.assertFalse(x["new_credit_authorization"])
    def test_transfer_falsification(self):
        self.assertTrue(g.transfer_falsification(g.load())["transfer_upper_estimate_falsified"])
    def test_full_panel(self):
        x=g.panel();self.assertEqual(len(x["cases"]),7)
        self.assertTrue(all(c["admission"]=="HOLD_PROVIDER_CAP_UNBOUNDED" for c in x["cases"]))
