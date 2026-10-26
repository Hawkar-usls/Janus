"""Provider-mock tests: no network, no actual credits."""
import unittest
from types import SimpleNamespace as S
from rowan_1ey0_T8A_weekly_pair import (
 WT,NAME,MUT,MUT_SHA256,CAP,Hold,sha,mutant,execute,
 TARGET_ROWAN_FOLDER_UUID,collect_one
)
def prior(name=NAME,uuid="old",folder=TARGET_ROWAN_FOLDER_UUID,wft="protein_cofolding"):
 return S(name=name,uuid=uuid,parent_uuid=folder,workflow_type=wft,status="queued")
class Api:
 class CofoldingModel:
  BOLTZ_2="boltz2"
 def __init__(self,weekly=45,rows=None):
  self.weekly=weekly
  self.rows=[] if rows is None else rows
  self.calls=[];self.kwargs=None;self.subscription=None
 def retrieve_folder(self,uuid):return S(uuid=uuid)
 def list_workflows(self,parent_uuid,page,size):return self.rows if page==0 else []
 def whoami(self):
  self.calls.append("whoami")
  return S(organization=S(weekly_credits=self.weekly,credits=900),
           individual_subscription=self.subscription)
 def submit_protein_cofolding_workflow(self,**kw):
  self.calls.append("submit");self.kwargs=kw;return prior()
class Tests(unittest.TestCase):
 def test_mutation_source_sha_and_exact_position(self):
  m=mutant();self.assertEqual(len(m),136)
  self.assertEqual(m[7],"A");self.assertEqual(WT[7],"T")
  self.assertEqual(sha(m.encode()),MUT_SHA256)
 def test_one_job_under_cap_only_weekly(self):
  a=Api();r=execute(a,"submit",True)
  self.assertEqual(r["new_jobs"],1)
  self.assertEqual(a.calls.count("submit"),1)
  self.assertEqual(a.kwargs["initial_protein_sequences"],[mutant()])
  self.assertEqual(a.kwargs["max_credits"],10)
  self.assertEqual(a.kwargs["num_samples"],1)
  self.assertTrue(a.kwargs["use_msa_server"])
  self.assertEqual(a.kwargs["name"],NAME)
 def test_weekly_insufficient_even_when_nonweekly_high(self):
  a=Api(weekly=29.0)
  with self.assertRaisesRegex(Hold,"ORG_WEEKLY_BELOW_30"):
   execute(a,"submit",True)
  self.assertNotIn("submit",a.calls)
 def test_weekly_unknown_fails(self):
  with self.assertRaisesRegex(Hold,"ORG_WEEKLY_BELOW_30"):
   execute(Api(weekly=None),"submit",True)
 def test_existing_case_never_resubmits(self):
  a=Api(rows=[prior()]);r=execute(a,"submit",True)
  self.assertEqual(r["new_jobs"],0)
  self.assertEqual(r["status"],"EXISTING_SINGLE_MUTANT_NO_SPEND")
  self.assertEqual(a.calls,[])
 def test_duplicate_same_name_blocks(self):
  a=Api(rows=[prior(uuid="a"),prior(uuid="b")])
  with self.assertRaisesRegex(Hold,"DUPLICATE_MATCHING_JOBS"):
   execute(a,"submit",True)
 def test_wrong_workflow_type_blocks(self):
  a=Api(rows=[prior(wft="different")])
  with self.assertRaisesRegex(Hold,"WRONG_TYPE_OR_FOLDER"):
   execute(a,"submit",True)
 def test_collect_when_no_case_does_not_submit(self):
  a=Api();r=execute(a,"collect",False)
  self.assertEqual(r["status"],"NO_MUTANT_JOB_TO_COLLECT")
  self.assertEqual(a.calls,[])
 def test_explicit_authorization_required(self):
  a=Api()
  with self.assertRaisesRegex(Hold,"NO_NEW_JOB_EXPLICIT_APPROVAL"):
   execute(a,"submit",False)
 def test_individual_subscription_blocks(self):
  a=Api();a.subscription=S()
  with self.assertRaisesRegex(Hold,"INDIVIDUAL_SUBSCRIPTION_NEEDS_REVIEW"):
   execute(a,"submit",True)
 def test_running_case_readonly(self):
  class Running:
   name=NAME;parent_uuid=TARGET_ROWAN_FOLDER_UUID
   def fetch_latest(self):return self
   def done(self):return False
  self.assertEqual(collect_one(Running())["provider_result_state"],"PENDING_OR_IN_PROGRESS")
 def test_completed_scores_and_credit_readonly(self):
  class Done:
   name=NAME;parent_uuid=TARGET_ROWAN_FOLDER_UUID;credits_charged=4.5
   def fetch_latest(self):return self
   def done(self):return True
   def result(self,wait=False):return S(scores=S(ptm=.92,avg_lddt=.94))
  r=collect_one(Done())
  self.assertEqual(r["provider_result_state"],"PROVIDER_RESULT_RETURNED")
  self.assertEqual(r["credits_charged"],4.5)
  self.assertEqual(r["new_jobs"],0)
 def test_bad_provider_score_holds(self):
  class Done:
   name=NAME;parent_uuid=TARGET_ROWAN_FOLDER_UUID;credits_charged=5
   def fetch_latest(self):return self
   def done(self):return True
   def result(self,wait=False):return S(scores=S(ptm=5,avg_lddt=.94))
  self.assertEqual(collect_one(Done())["provider_result_state"],"PROVIDER_SCORE_UNVERIFIED")
 def test_no_production_claim(self):
  x=execute(Api(),"submit",True)
  self.assertFalse(x["mutation_level_ddg_claim"])
  self.assertFalse(x["shadow_model_weights_modified"])
  self.assertFalse(x["FireProt_210_inference_authorized"])
 def test_unknown_action_blocks(self):
  with self.assertRaisesRegex(Hold,"UNKNOWN_ACTION"):execute(Api(),"other",True)
if __name__=="__main__":unittest.main()
