import sys,unittest
from pathlib import Path
from types import SimpleNamespace as S
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rowan_r1g_panel as r
from probe import TARGET_ROWAN_FOLDER_UUID as F
class Fake:
    CofoldingModel=S(BOLTZ_2="BOLTZ_2")
    def __init__(self,weekly=100,pending=False):self.balance=weekly;self.calls=[];self.jobs={};self.pending=pending
    def whoami(self):return S(organization=S(weekly_credits=self.balance),individual_subscription=None)
    def retrieve_folder(self,*a):return S(uuid=F)
    def list_workflows(self,**k):return list(self.jobs.values())
    def submit_protein_cofolding_workflow(self,**k):
        self.calls.append(k);n=k["num_samples"];self.balance-=6
        w=S(uuid=str(len(self.calls)),name=k["name"],parent_uuid=F,workflow_type="protein_cofolding",status="Status.RUNNING" if self.pending else "Status.COMPLETED_OK",credits_charged=6)
        w.fetch_latest=lambda:w;w.done=lambda:not self.pending
        w.result=lambda **k:S(scores=S(ptm=.90,avg_lddt=.94),predictions=[S(predicted_structure_uuid=str(i+1),scores=S(ptm=.90+i*.001,avg_lddt=.94)) for i in range(n)])
        self.jobs[w.name]=w
        return w
class Tests(unittest.TestCase):
    def runfake(self,a,mode="submit",approved=True,attempt="1"):return r.run(a,mode,approved,attempt,lambda x:None,lambda x:None)
    def test_panel(self):
        self.assertEqual(len(r.CASES),7);self.assertEqual(sum(n for _,_,n in r.CASES),27)
        self.assertEqual(r.identity("2LZM","I3T",5)[2][2],"T");self.assertEqual(r.identity("BBC1","E27F",5)[2][26],"F")
    def test_complete(self):
        a=Fake();x=self.runfake(a);self.assertEqual(x["status"],"COMPLETE_ALL_CASES");self.assertEqual(len(a.calls),7);self.assertTrue(all(c["max_credits"]==10 for c in a.calls))
    def test_budget_hold(self):
        a=Fake(42);x=self.runfake(a);self.assertEqual(x["status"],"HOLD_WEEKLY_RESERVE");self.assertEqual(len(a.calls),1)
    def test_collect(self):
        a=Fake();x=self.runfake(a,"collect",False);self.assertEqual(x["status"],"PARTIAL_READ_ONLY");self.assertFalse(a.calls)
    def test_retry_and_approval(self):
        for approved,attempt in ((False,"1"),(True,"2")):
            a=Fake()
            with self.assertRaises(r.Hold):self.runfake(a,approved=approved,attempt=attempt)
            self.assertFalse(a.calls)
    def test_pending(self):
        a=Fake(pending=True);x=self.runfake(a);self.assertEqual(x["status"],"HOLD_PENDING_NO_FURTHER_SPEND");self.assertEqual(len(a.calls),1)
        x=self.runfake(a);self.assertEqual(x["status"],"HOLD_PREEXISTING_PENDING");self.assertEqual(len(a.calls),1)
    def test_existing_no_rebill(self):
        a=Fake();self.runfake(a);x=self.runfake(a);self.assertEqual(x["status"],"COMPLETE_ALL_CASES");self.assertEqual(len(a.calls),7);self.assertEqual(x["new_jobs"],0)
