import importlib.util
from pathlib import Path
import json
import sys
from types import SimpleNamespace as Obj
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("rowan_bbc1_autopilot",ROOT/"rowan_bbc1_autopilot.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
manifest=json.loads((ROOT/"discovery_manifest.json").read_text(encoding="utf-8"))
case1=m.prepare(manifest,"wildtype")
case2=m.prepare(manifest,"K5A")

class FakeSDK:
    CofoldingModel=Obj(BOLTZ_2="boltz_2")
    def __init__(self,weekly=110,credits=500,subscription=None,
                 jobs=None,folder_uuid=m.TARGET_ROWAN_FOLDER_UUID,org=True):
        self.user=Obj(organization=Obj(weekly_credits=weekly,credits=credits) if org else None,
                      individual_subscription=subscription)
        self.folder_uuid=folder_uuid
        self.jobs=list(jobs or [])
        self.submissions=[]
    def whoami(self): return self.user
    def retrieve_folder(self,uuid):
        assert uuid==m.TARGET_ROWAN_FOLDER_UUID
        return Obj(uuid=self.folder_uuid)
    def list_workflows(self,parent_uuid,page,size):
        assert parent_uuid==m.TARGET_ROWAN_FOLDER_UUID
        return self.jobs[page*size:(page+1)*size]
    def submit_protein_cofolding_workflow(self,**kwargs):
        self.submissions.append(kwargs)
        return Obj(uuid="fa5e0000-0000-4000-8000-000000000001")

def job(case,folder=m.TARGET_ROWAN_FOLDER_UUID):
    return Obj(name=case["id"],parent_uuid=folder,workflow_type="protein_cofolding")

class AutopilotTests(unittest.TestCase):
    def test_first_job_bounded_to_one_and_weekly(self):
        api=FakeSDK()
        result=m.single_step(api,manifest,authorized=True)
        self.assertEqual(result["state"],"SUBMITTED_ONE_AND_STOPPED")
        self.assertEqual(result["arm"],"wildtype")
        self.assertEqual(len(api.submissions),1)
        args=api.submissions[0]
        self.assertEqual(args["max_credits"],10)
        self.assertEqual(args["folder_uuid"],m.TARGET_ROWAN_FOLDER_UUID)
        self.assertEqual(args["initial_protein_sequences"],[case1["sequence"]])
        self.assertFalse(args["is_draft"])
    def test_second_job_only_if_wildtype_already_exists(self):
        api=FakeSDK(jobs=[job(case1)])
        result=m.single_step(api,manifest,authorized=True)
        self.assertEqual(result["arm"],"K5A")
        self.assertEqual(api.submissions[0]["initial_protein_sequences"],[case2["sequence"]])
    def test_never_duplicate_either_job(self):
        api=FakeSDK(jobs=[job(case1),job(case2)])
        result=m.single_step(api,manifest,authorized=True)
        self.assertEqual(result["state"],"ALL_PREREGISTERED_CASES_ALREADY_EXIST_NO_SUBMISSION")
        self.assertEqual(api.submissions,[])
    def test_approval_switch_fail_closed(self):
        api=FakeSDK()
        with self.assertRaises(m.BudgetHold):
            m.single_step(api,manifest,authorized=False)
        self.assertFalse(api.submissions)
    def test_weekly_budget_floor_30(self):
        api=FakeSDK(weekly=29,credits=500)
        with self.assertRaises(m.BudgetHold):
            m.single_step(api,manifest,authorized=True)
        self.assertFalse(api.submissions)
    def test_five_hundred_nonweekly_not_used_in_place_of_weekly(self):
        api=FakeSDK(weekly=0,credits=500)
        with self.assertRaises(m.BudgetHold):
            m.single_step(api,manifest,authorized=True)
    def test_org_missing_fail_closed(self):
        api=FakeSDK(org=False)
        with self.assertRaises(m.BudgetHold):
            m.single_step(api,manifest,authorized=True)
    def test_subscription_requires_independent_review(self):
        api=FakeSDK(subscription="paid-plan")
        with self.assertRaises(m.BudgetHold):
            m.single_step(api,manifest,authorized=True)
    def test_bad_target_folder_blocks(self):
        api=FakeSDK(folder_uuid="other")
        with self.assertRaises(m.BudgetHold):
            m.single_step(api,manifest,authorized=True)
    def test_foreign_existing_job_blocks(self):
        api=FakeSDK(jobs=[job(case1,folder="other")])
        with self.assertRaises(m.BudgetHold):
            m.single_step(api,manifest,authorized=True)
    def test_same_case_duplicate_blocks(self):
        api=FakeSDK(jobs=[job(case1),job(case1)])
        with self.assertRaises(m.BudgetHold):
            m.single_step(api,manifest,authorized=True)
    def test_unbounded_pagination_fails_closed(self):
        api=FakeSDK(jobs=[Obj(name="other",parent_uuid=m.TARGET_ROWAN_FOLDER_UUID,
                     workflow_type="protein_cofolding") for _ in range(m.PAGE_SIZE*m.MAX_PAGES)])
        with self.assertRaises(m.BudgetHold):
            m.single_step(api,manifest,authorized=True)

if __name__=="__main__":unittest.main()
