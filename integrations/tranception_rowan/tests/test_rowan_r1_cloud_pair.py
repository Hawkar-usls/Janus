import sys,unittest
from pathlib import Path
from types import SimpleNamespace as S
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rowan_r1_cloud_pair as r

class Fake:
    CofoldingModel=S(BOLTZ_2="BOLTZ_2")
    def __init__(self,weekly=100,existing=None):
        self.weekly=weekly;self.existing=existing;self.calls=[]
    def retrieve_folder(self,*a):return S(uuid=r.TARGET_ROWAN_FOLDER_UUID)
    def list_workflows(self,**k):return [self.existing] if self.existing else []
    def whoami(self):return S(organization=S(weekly_credits=self.weekly),individual_subscription=None)
    def submit_protein_cofolding_workflow(self,**k):self.calls.append(k);return S(uuid="unit-test",status="queued")

class Tests(unittest.TestCase):
    def test_two_exact_distinct_cases(self):
        a,b=r.identity("WT"),r.identity("T8A")
        self.assertNotEqual(a[0],b[0]); self.assertEqual(a[1][:7]+b[1][7]+a[1][8:],b[1])
    def test_explicit_cap(self):
        for role in r.ROLES:
            a=Fake(); x=r.operation(a,role,"submit",True,"1")
            self.assertEqual(x["new_jobs"],1);self.assertEqual(len(a.calls),1)
            self.assertEqual(a.calls[0]["max_credits"],10)
    def test_no_unapproved_or_retry(self):
        for approved,attempt in [(False,"1"),(True,"2")]:
            a=Fake()
            with self.assertRaises(r.Hold):r.operation(a,"WT","submit",approved,attempt)
            self.assertFalse(a.calls)
    def test_insufficient_credit(self):
        a=Fake(29)
        with self.assertRaises(Exception):r.operation(a,"WT","submit",True,"1")
        self.assertFalse(a.calls)
    def test_readonly(self):
        a=Fake(); self.assertEqual(r.operation(a,"WT","collect",False,"1")["new_jobs"],0);self.assertFalse(a.calls)
    def test_existing_no_spend(self):
        name=r.identity("WT")[0]
        w=S(name=name,parent_uuid=r.TARGET_ROWAN_FOLDER_UUID,workflow_type="protein_cofolding",credits_charged=5,status="done")
        w.fetch_latest=lambda:w;w.done=lambda:True;w.result=lambda **k:S(scores=S(ptm=.9,avg_lddt=.8))
        a=Fake(existing=w)
        self.assertEqual(r.operation(a,"WT","submit",True,"1")["status"],"RESULT_COLLECTED")
        self.assertFalse(a.calls)
