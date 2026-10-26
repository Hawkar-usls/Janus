"""Unit-only tests for credit-bounded 1EY0 Rowan submission."""
import unittest
from types import SimpleNamespace as N
from rowan_1ey0_weekly_api import (WT,WT_SHA256,digest,CAP,NAME,Hold,process,
 TARGET_ROWAN_FOLDER_UUID)
def job(uid="case1",kind="protein_cofolding",folder=TARGET_ROWAN_FOLDER_UUID):
    return N(name=NAME,uuid=uid,workflow_type=kind,parent_uuid=folder,status="running")
class Provider:
    class CofoldingModel:
        BOLTZ_2="BOLTZ_2"
    def __init__(self,weekly=45,rows=None):
        self.user=N(organization=N(weekly_credits=weekly,credits=500),individual_subscription=None)
        self.rows=rows or []
        self.calls=[];self.args=None
    def whoami(self):return self.user
    def retrieve_folder(self,uid):return N(uuid=uid)
    def list_workflows(self,parent_uuid,page,size):return self.rows if page==0 else []
    def submit_protein_cofolding_workflow(self,**kwargs):
        self.calls.append("SUBMIT");self.args=kwargs;return job()
class Tests(unittest.TestCase):
    def test_sequence(self):
        self.assertEqual(len(WT),136);self.assertEqual(digest(WT),WT_SHA256)
    def test_one_allowed_submit_exact_sequence_and_cap(self):
        p=Provider();o=process(p,"submit",True)
        self.assertEqual(o["new_jobs"],1)
        self.assertEqual(p.calls,["SUBMIT"])
        self.assertEqual(p.args["initial_protein_sequences"],[WT])
        self.assertEqual(p.args["max_credits"],CAP)
        self.assertTrue(p.args["use_msa_server"])
        self.assertEqual(p.args["folder_uuid"],TARGET_ROWAN_FOLDER_UUID)
        self.assertEqual(p.args["name"],NAME)
    def test_only_weekly_not_bought(self):
        p=Provider(weekly=29)
        with self.assertRaisesRegex(Hold,"ORG_WEEKLY_POOL"):
            process(p,"submit",True)
        self.assertEqual(p.calls,[])
    def test_unknown_weekly(self):
        p=Provider(weekly=None)
        with self.assertRaisesRegex(Hold,"ORG_WEEKLY_POOL"):
            process(p,"submit",True)
    def test_deduplication(self):
        p=Provider(rows=[job()])
        out=process(p,"submit",True)
        self.assertEqual(out["new_jobs"],0);self.assertEqual(p.calls,[])
    def test_duplicate_names_rejected(self):
        p=Provider(rows=[job("a"),job("b")])
        with self.assertRaisesRegex(Hold,"EXISTING_DUPLICATE"):
            process(p,"submit",True)
    def test_wrong_folder_rejected(self):
        p=Provider(rows=[job(folder="wrong")])
        with self.assertRaisesRegex(Hold,"DIFFERENT_FOLDER"):
            process(p,"submit",True)
    def test_collect_never_submits(self):
        p=Provider()
        self.assertEqual(process(p,"collect",False)["new_jobs"],0)
        self.assertEqual(p.calls,[])
    def test_requires_approval(self):
        p=Provider()
        with self.assertRaisesRegex(Hold,"EXPLICIT_ROWAN_WEEKLY"):
            process(p,"submit",False)
    def test_unknown_subscription_rejected(self):
        p=Provider();p.user.individual_subscription=N()
        with self.assertRaisesRegex(Hold,"SUBSCRIPTION_STATUS"):
            process(p,"submit",True)
    def test_no_fitness_or_mutants_claim(self):
        o=process(Provider(),"submit",True)
        self.assertFalse(o["fitness_validated"]);self.assertEqual(o["fireprot_210_mutations_submitted"],0)
if __name__=="__main__": unittest.main()
