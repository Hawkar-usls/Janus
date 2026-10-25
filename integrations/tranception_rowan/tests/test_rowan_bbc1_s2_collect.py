import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as Obj
from unittest.mock import patch
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("s2_collect",ROOT/"rowan_bbc1_s2_collect.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
manifest=json.loads((ROOT/"discovery_manifest.json").read_text())
wt=m.prepare(manifest,"wildtype")
samples=[
 {"case_name":f"JANUS-BBC1-S2-C{i}","quantile_target_percent":i*10,
  "mutation":f"A{i}G","processed_label":float(i)} for i in range(1,9)
]

class Job:
    parent_uuid=m.TARGET_ROWAN_FOLDER_UUID
    workflow_type="protein_cofolding"
    def __init__(self,name,ptm=.8,lddt=.85,done=True,credits=4.5):
        self.name=name;self.ptm=ptm;self.lddt=lddt;self.finished=done
        self.credits_charged=credits
    def fetch_latest(self):return self
    def done(self):return self.finished
    def result(self,wait=False):return Obj(scores=Obj(ptm=self.ptm,avg_lddt=self.lddt))
class API:
    def __init__(self,jobs):self.jobs=jobs;self.submit_count=0
    def whoami(self):return Obj()
    def retrieve_folder(self,uuid):return Obj(uuid=uuid)
    def list_workflows(self,parent_uuid,page,size):
        return self.jobs[page*size:(page+1)*size]
    def submit_protein_cofolding_workflow(self,**kwargs):
        self.submit_count+=1
        raise AssertionError("Collector must never submit")

class Tests(unittest.TestCase):
    def setUp(self):
        p=patch.object(m,"select",return_value=samples)
        p.start();self.addCleanup(p.stop)

    def test_spearman_perfect_order(self):
        self.assertEqual(m.corr_spearman([1,2,3],[2,5,7]),1)
        self.assertEqual(m.corr_spearman([1,2,3],[7,5,2]),-1)

    def test_constant_predictor_undefined(self):
        self.assertIsNone(m.corr_spearman([1,2,3],[9,9,9]))

    def test_ties_bounded(self):
        self.assertTrue(-1<=m.corr_spearman([1,1,3,4],[0,2,2,5])<=1)

    def test_full_pilot_read_only(self):
        jobs=[Job(wt["id"],ptm=.8,lddt=.8)]
        jobs.extend(Job(c["case_name"],ptm=.8+i*.001,lddt=.8+i*.002)
                    for i,c in enumerate(samples))
        api=API(jobs)
        report=m.collect(api,manifest,b"fake-synthetic")
        self.assertTrue(report["all_eight_ready_for_descriptive_association"])
        self.assertEqual(report["spearman_processed_label_vs_delta_structure_confidence"]["ptm"],1)
        self.assertEqual(api.submit_count,0)
        self.assertFalse(report["protein_stability_or_fitness_inferred"])

    def test_pending_prevents_correlation(self):
        jobs=[Job(wt["id"]),Job(samples[0]["case_name"],done=False)]
        report=m.collect(API(jobs),manifest,b"fake")
        self.assertFalse(report["all_eight_ready_for_descriptive_association"])
        self.assertIsNone(report["spearman_processed_label_vs_delta_structure_confidence"]["ptm"])

    def test_missing_numeric_wt_not_ready(self):
        jobs=[Job(wt["id"],ptm=None)]
        jobs.extend(Job(c["case_name"]) for c in samples)
        report=m.collect(API(jobs),manifest,b"fake")
        self.assertFalse(report["all_eight_ready_for_descriptive_association"])

    def test_no_wt_blocks(self):
        with self.assertRaises(m.BudgetHold):
            m.collect(API([]),manifest,b"fake")

    def test_duplicate_strictly_blocks(self):
        jobs=[Job(wt["id"]),Job(wt["id"])]
        with self.assertRaises(m.BudgetHold):
            m.collect(API(jobs),manifest,b"fake")

if __name__=="__main__":unittest.main()
