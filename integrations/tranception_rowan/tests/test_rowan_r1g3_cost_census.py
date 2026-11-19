import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rowan_r1g3_cost_census as c
class T(unittest.TestCase):
    def test_allowlist(self):self.assertEqual(len(c.allowlist()),15)
    def test_grouped_holdout(self):
        rows=[{"protein_group":p,"samples":1,"state":"Status.COMPLETED_OK","elapsed_seconds":t,"length":n} for p,t,n in [("a",100,50),("a",110,50),("b",80,100),("c",150,150)]]
        r=c.qualify({"records":rows});self.assertEqual(len(r["single_sample_leave_one_protein_out"]),3);self.assertFalse(r["paid_submission_authorized"]);self.assertEqual(r["counts"]["five_completed"],0)
    def test_censor_not_training(self):
        rows=[{"protein_group":"a","samples":5,"state":"Status.STOPPED","elapsed_seconds":205.16,"length":64}]
        r=c.qualify({"records":rows});self.assertEqual(r["counts"]["completed"],0);self.assertEqual(r["counts"]["stopped"],1);self.assertEqual(r["single_sample_leave_one_protein_out"],[])
