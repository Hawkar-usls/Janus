import sys,unittest
from pathlib import Path
from types import SimpleNamespace as S
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rowan_r1e_multisample as r
from probe import TARGET_ROWAN_FOLDER_UUID as F
class Fake:
    CofoldingModel=S(BOLTZ_2="BOLTZ_2")
    def __init__(self,weekly=100,job=None):self.weekly=weekly;self.job=job;self.calls=[]
    def retrieve_folder(self,*a):return S(uuid=F)
    def list_workflows(self,**k):return [self.job] if self.job else []
    def whoami(self):return S(organization=S(weekly_credits=self.weekly),individual_subscription=None)
    def submit_protein_cofolding_workflow(self,num_samples=None,**k):self.calls.append(dict(num_samples=num_samples,**k));return S(uuid="unit-test",status="queued")
def prior(n=5,duplicate=False,cost=8.0):
    rows=[S(scores=S(ptm=.90+i*.01,avg_lddt=.95-i*.01),predicted_structure_uuid=str(1 if duplicate else i+1)) for i in range(n)]
    w=S(name=r.NAME,parent_uuid=F,workflow_type="protein_cofolding",status="Status.COMPLETED_OK",credits_charged=cost)
    w.fetch_latest=lambda:w;w.done=lambda:True;w.result=lambda **k:S(predictions=rows,scores=rows[-1].scores)
    return w
class Tests(unittest.TestCase):
    def test_submit(self):
        a=Fake();x=r.operation(a,"submit",True,"1")
        self.assertEqual(x["new_jobs"],1);self.assertEqual(a.calls[0]["num_samples"],5);self.assertEqual(a.calls[0]["max_credits"],10)
    def test_no_unapproved_or_retry(self):
        for approval,attempt in ((False,"1"),(True,"2")):
            a=Fake()
            with self.assertRaises(r.Hold):r.operation(a,"submit",approval,attempt)
            self.assertFalse(a.calls)
    def test_no_budget(self):
        a=Fake(29)
        with self.assertRaises(Exception):r.operation(a,"submit",True,"1")
        self.assertFalse(a.calls)
    def test_collect_no_spend_and_mean_not_primary(self):
        a=Fake(job=prior());x=r.operation(a,"collect",False,"1")
        self.assertEqual(x["status"],"RESULT_COLLECTED_ALL_FIVE");self.assertEqual(x["returned_sample_count"],5);self.assertAlmostEqual(x["summary"]["ptm"]["mean"],.92);self.assertNotEqual(x["summary"]["ptm"]["mean"],x["primary_scores"]["ptm"]);self.assertAlmostEqual(x["credits_per_returned_sample"],1.6);self.assertFalse(a.calls)
    def test_partial_holds(self):
        a=Fake(job=prior(4));self.assertEqual(r.operation(a,"collect",False,"1")["status"],"HOLD_SAMPLE_COUNT_MISMATCH");self.assertFalse(a.calls)
    def test_duplicate_holds(self):
        a=Fake(job=prior(5,True))
        with self.assertRaises(r.Hold):r.operation(a,"collect",False,"1")
        self.assertFalse(a.calls)
    def test_cost_cap(self):
        a=Fake(job=prior(cost=10.1));self.assertEqual(r.operation(a,"collect",False,"1")["status"],"HOLD_CAP_EXCEEDED")
    def test_collect_missing(self):
        a=Fake();self.assertEqual(r.operation(a,"collect",False,"1")["status"],"NOT_FOUND_NO_SPEND");self.assertFalse(a.calls)
    def test_existing_submission_no_spend(self):
        a=Fake(job=prior());self.assertEqual(r.operation(a,"submit",True,"1")["new_jobs"],0);self.assertFalse(a.calls)
