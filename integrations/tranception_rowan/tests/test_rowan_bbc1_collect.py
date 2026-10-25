import importlib.util
from pathlib import Path
import sys
import json
from types import SimpleNamespace as Obj
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("rowan_bbc1_collect",ROOT/"rowan_bbc1_collect.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
manifest=json.loads((ROOT/"discovery_manifest.json").read_text())
case1=m.prepare(manifest,"wildtype")
case2=m.prepare(manifest,"K5A")

class Job:
    parent_uuid=m.TARGET_ROWAN_FOLDER_UUID
    workflow_type="protein_cofolding"
    uuid="f00df00d-0000-4000-8000-000000000000"
    def __init__(self,case,done=False,ptm=None,lddt=None,moved=False):
        self.name=case["id"]
        self._done=done
        self._ptm=ptm
        self._lddt=lddt
        if moved:self.parent_uuid="other"
    def fetch_latest(self):return self
    def done(self):return self._done
    def result(self,wait=False):
        assert wait is False
        return Obj(scores=Obj(ptm=self._ptm,avg_lddt=self._lddt))
class API:
    def __init__(self,jobs):
        self.jobs=jobs
    def whoami(self):return Obj()
    def retrieve_folder(self,uuid):return Obj(uuid=uuid)
    def list_workflows(self,parent_uuid,page,size):
        return self.jobs[page*size:(page+1)*size]

class Tests(unittest.TestCase):
    def test_empty_states(self):
        r=m.collect(API([]),manifest)
        self.assertEqual([x["status"] for x in r["entries"]],["NOT_SUBMITTED","NOT_SUBMITTED"])
    def test_pending_safely(self):
        r=m.collect(API([Job(case1)]),manifest)
        self.assertEqual(r["entries"][0]["status"],"PENDING_OR_FAILED")
    def test_numeric_score_only_0_to_1(self):
        r=m.collect(API([Job(case1,done=True,ptm=.82,lddt=.65)]),manifest)
        self.assertEqual(r["entries"][0]["ptm"],.82)
        self.assertFalse(r["protein_fitness_validated"])
    def test_invalid_scores_do_not_fabricate(self):
        r=m.collect(API([Job(case1,done=True,ptm=float("nan"),lddt=1.4)]),manifest)
        self.assertIsNone(r["entries"][0]["ptm"])
        self.assertIsNone(r["entries"][0]["avg_lddt"])
    def test_realistic_two_arm_delta_and_provider_charge(self):
        first=Job(case1,done=True,ptm=.927,lddt=.963)
        second=Job(case2,done=True,ptm=.927,lddt=.973)
        first.credits_charged=6.5
        second.credits_charged=7.0
        result=m.collect(API([first,second]),manifest)
        self.assertEqual(result["paired_confidence_deltas_not_fitness"]["K5A_minus_wildtype_avg_lddt"],.01)
        self.assertEqual(result["paired_confidence_deltas_not_fitness"]["K5A_minus_wildtype_ptm"],0)
        self.assertFalse(result["entries"][0]["charge_over_cap_review"])
        self.assertEqual(result["entries"][1]["provider_credits_charged"],7.0)
        self.assertFalse(result["protein_fitness_validated"])
    def test_unknown_and_over_cap_provider_charge(self):
        first=Job(case1,done=True,ptm=.8,lddt=.9)
        first.credits_charged=12.0
        result=m.collect(API([first]),manifest)
        self.assertTrue(result["entries"][0]["charge_over_cap_review"])
        self.assertEqual(result["entries"][0]["credit_cap"],10)
        self.assertIsNone(result["entries"][1].get("provider_credits_charged"))

    def test_foreign_moved_job_blocks(self):
        with self.assertRaises(m.BudgetHold):
            m.collect(API([Job(case1,moved=True)]),manifest)
    def test_duplicate_blocks(self):
        with self.assertRaises(m.BudgetHold):
            m.collect(API([Job(case1),Job(case1)]),manifest)

if __name__=="__main__":unittest.main()
