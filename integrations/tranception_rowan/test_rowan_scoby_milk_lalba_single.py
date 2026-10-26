"""No-network mock-only safeguards; no provider billable calls."""
import unittest
from types import SimpleNamespace as N
import rowan_scoby_milk_lalba_single as m

class FakeRowan:
    class CofoldingModel:
        BOLTZ_2="BOLTZ_2"
    def __init__(self, weekly=45, workflows=None):
        self.weekly=weekly
        self.workflows=workflows or []
        self.submits=[]
    def retrieve_folder(self, uuid):
        return N(uuid=uuid)
    def list_workflows(self,parent_uuid,page,size):
        return self.workflows if page==0 else []
    def whoami(self):
        return N(individual_subscription=None,
                 organization=N(weekly_credits=self.weekly,credits=100000))
    def submit_protein_cofolding_workflow(self,**kwargs):
        self.submits.append(kwargs)
        return N(uuid="new-test-only",name=kwargs["name"],status="queued")
def older(done=False):
    w=N(uuid="existing", name=m.NAME,
        parent_uuid=m.TARGET_ROWAN_FOLDER_UUID,
        workflow_type="protein_cofolding", status="RUNNING")
    w.fetch_latest=lambda:w
    w.done=lambda:done
    if done:
        w.credits_charged=4.5
        w.result=lambda wait=False:N(scores=N(ptm=.9,avg_lddt=.87))
    return w

class TestFrozenLALBA(unittest.TestCase):
    def test_source_and_length(self):
        self.assertTrue(m.validate_source())
        self.assertEqual(len(m.SEQUENCE),123)
        self.assertEqual(m.SHA256,"f7e3b0b48179423a572873eabbb5614d878b0391e868fc8a2be8e0e10e32d9c4")
    def test_unknown_case_holds(self):
        with self.assertRaisesRegex(m.Hold,"UNREGISTERED_CASE"):
            m.execute(FakeRowan(),"milk_unregistered","submit",True)
    def test_no_explicit_permission_holds(self):
        a=FakeRowan()
        with self.assertRaisesRegex(m.Hold,"EXPLICIT_WEEKLY"):
            m.execute(a,m.CASE,"submit",False)
        self.assertEqual(len(a.submits),0)
    def test_unknown_weekly_holds_even_with_large_nonweekly(self):
        for value in (None,29):
            with self.subTest(value=value):
                a=FakeRowan(weekly=value)
                with self.assertRaisesRegex(m.Hold,"ORG_WEEKLY"):
                    m.execute(a,m.CASE,"submit",True)
                self.assertEqual(len(a.submits),0)
    def test_exact_one_submit_and_cap(self):
        a=FakeRowan(weekly=40)
        result=m.execute(a,m.CASE,"submit",True)
        self.assertEqual(result["new_jobs"],1)
        self.assertEqual(len(a.submits),1)
        self.assertEqual(a.submits[0]["initial_protein_sequences"],[m.SEQUENCE])
        self.assertEqual(a.submits[0]["max_credits"],10)
        self.assertEqual(a.submits[0]["num_samples"],1)
        self.assertTrue(a.submits[0]["use_msa_server"])
        self.assertFalse(result["milk_or_placental_tissue_modelled"])
    def test_existing_prevents_double_spend(self):
        a=FakeRowan(workflows=[older()])
        result=m.execute(a,m.CASE,"submit",True)
        self.assertEqual(result["new_jobs"],0)
        self.assertEqual(a.submits,[])
    def test_duplicate_existing_holds(self):
        a=FakeRowan(workflows=[older(),older()])
        with self.assertRaisesRegex(m.Hold,"DUPLICATE_PROVIDER"):
            m.execute(a,m.CASE,"submit",True)
    def test_collect_missing_is_readonly(self):
        a=FakeRowan()
        result=m.execute(a,m.CASE,"collect",False)
        self.assertEqual(result["new_jobs"],0)
        self.assertEqual(a.submits,[])
    def test_collect_existing_readonly_running(self):
        a=FakeRowan(workflows=[older(False)])
        result=m.execute(a,m.CASE,"collect",False)
        self.assertEqual(result["result_state"],"PENDING_OR_IN_PROGRESS")
        self.assertEqual(a.submits,[])
    def test_collect_existing_readonly_completed(self):
        a=FakeRowan(workflows=[older(True)])
        result=m.execute(a,m.CASE,"collect",False)
        self.assertEqual(result["result_state"],"PROVIDER_SCORES_RETURNED")
        self.assertEqual(result["charged_credits"],4.5)
        self.assertEqual(result["new_jobs"],0)
        self.assertEqual(a.submits,[])
    def test_collect_cannot_enable_spending(self):
        with self.assertRaisesRegex(m.Hold,"COLLECTION_MUST_NOT"):
            m.execute(FakeRowan(),m.CASE,"collect",True)
    def test_bad_action_holds(self):
        with self.assertRaisesRegex(m.Hold,"UNKNOWN_ACTION"):
            m.execute(FakeRowan(),m.CASE,"refit",True)

if __name__=="__main__":
    unittest.main()
