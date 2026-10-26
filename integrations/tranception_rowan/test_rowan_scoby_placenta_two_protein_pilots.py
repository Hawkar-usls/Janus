"""No-network synthetic authorization and source tests for both Rowan pilot targets."""
import unittest
from types import SimpleNamespace as N
from rowan_scoby_placenta_two_protein_pilots import (
 TARGETS,source,execute,collect,Hold,TARGET_ROWAN_FOLDER_UUID
)
class API:
 class CofoldingModel: BOLTZ_2="BOLTZ_2"
 def __init__(self,weekly=42,existing=None):
  self.calls=[];self.weekly=weekly;self.prev=existing or []
 def whoami(self):
  self.calls.append("whoami")
  return N(organization=N(weekly_credits=self.weekly,credits=1000),individual_subscription=None)
 def retrieve_folder(self,uuid):return N(uuid=uuid)
 def list_workflows(self,parent_uuid,page,size):return self.prev if page==0 else []
 def submit_protein_cofolding_workflow(self,**kw):
  self.calls.append("submit");self.kw=kw
  return N(uuid="test-uuid",name=kw["name"],status="queued",parent_uuid=TARGET_ROWAN_FOLDER_UUID,workflow_type="protein_cofolding")
def preexisting(case):
 t=TARGETS[case]
 return N(uuid="prior",name=t["name"],parent_uuid=TARGET_ROWAN_FOLDER_UUID,
          workflow_type="protein_cofolding",status="queued")
class Cases(unittest.TestCase):
 def test_pinned_sequence_and_source_each(self):
  for c in TARGETS:
   with self.subTest(c=c):
    t=source(c);self.assertIn(t["length"],(109,156))
 def test_no_third_case(self):
  with self.assertRaisesRegex(Hold,"UNKNOWN_OR_NOT_PREREGISTERED_CASE"):source("third")
 def test_submit_one_acsD_with_cap(self):
  x=API();r=execute(x,"scoby_acsD","submit",True)
  self.assertEqual(x.calls.count("submit"),1);self.assertEqual(r["new_jobs"],1)
  self.assertEqual(x.kw["initial_protein_sequences"],[TARGETS["scoby_acsD"]["sequence"]])
  self.assertEqual(x.kw["max_credits"],10);self.assertTrue(x.kw["use_msa_server"])
 def test_submit_one_fn1_with_cap(self):
  x=API();r=execute(x,"placenta_fn1","submit",True)
  self.assertEqual(x.calls.count("submit"),1)
  self.assertEqual(x.kw["initial_protein_sequences"],[TARGETS["placenta_fn1"]["sequence"]])
  self.assertFalse(r["placental_tissue_or_BC_composite_predicted"])
 def test_weekly_below_30_but_other_credit_large_blocks(self):
  x=API(weekly=29)
  with self.assertRaisesRegex(Hold,"ORG_WEEKLY_BELOW_30_OR_UNKNOWN"):
   execute(x,"scoby_acsD","submit",True)
  self.assertNotIn("submit",x.calls)
 def test_weekly_unknown_blocks(self):
  with self.assertRaisesRegex(Hold,"ORG_WEEKLY_BELOW_30_OR_UNKNOWN"):
   execute(API(weekly=None),"placenta_fn1","submit",True)
 def test_missing_approval_blocks(self):
  with self.assertRaisesRegex(Hold,"USER_EXPLICIT_CREDIT_APPROVAL_MISSING"):
   execute(API(),"scoby_acsD","submit",False)
 def test_duplicate_blocks_repeat_spend(self):
  x=API(existing=[preexisting("scoby_acsD")])
  r=execute(x,"scoby_acsD","submit",True)
  self.assertEqual(r["new_jobs"],0);self.assertNotIn("submit",x.calls)
 def test_two_identical_existing_jobs_blocks(self):
  p=preexisting("placenta_fn1")
  with self.assertRaisesRegex(Hold,"DUPLICATE_PROVIDER_CASES"):
   execute(API(existing=[p,p]),"placenta_fn1","submit",True)
 def test_collect_no_job_never_spends(self):
  x=API();r=execute(x,"placenta_fn1","collect",False)
  self.assertEqual(r["new_jobs"],0);self.assertNotIn("submit",x.calls)
 def test_running_collect_is_readonly(self):
  p=preexisting("scoby_acsD")
  p.fetch_latest=lambda:p
  p.done=lambda:False
  r=execute(API(existing=[p]),"scoby_acsD","collect",False)
  self.assertEqual(r["result_state"],"PENDING_OR_IN_PROGRESS")
  self.assertEqual(r["additional_jobs"],0)
 def test_completed_collect_extracts_score_and_credits(self):
  p=preexisting("scoby_acsD")
  p.fetch_latest=lambda:p
  p.done=lambda:True
  p.credits_charged=4.1
  p.result=lambda wait=False:N(scores=N(ptm=.83,avg_lddt=.9))
  x=API(existing=[p]);r=execute(x,"scoby_acsD","collect",False)
  self.assertEqual(r["ptm"],.83)
  self.assertEqual(r["avg_lddt"],.9)
  self.assertEqual(r["charged_credits"],4.1)
  self.assertNotIn("submit",x.calls)
if __name__=="__main__":unittest.main()
