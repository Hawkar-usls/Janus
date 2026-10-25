import csv
import io
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as Obj
from unittest.mock import patch
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("pilot",ROOT/"rowan_bbc1_s2_pilot.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
manifest=json.loads((ROOT/"discovery_manifest.json").read_text())
wt=manifest["independent_control"]["wildtype"]
AA="ACDEFGHIKLMNPQRSTVWY"

def fixture():
    combos=[]
    for i,original in enumerate(wt):
        for replacement in AA:
            if replacement!=original:
                seq=wt[:i]+replacement+wt[i+1:]
                combos.append(seq)
    assert len(combos)==1216
    rows=[(s,j*.1,0) for j,s in enumerate(combos[:1084])]
    if manifest["independent_control"]["mutant"] not in {r[0] for r in rows}:
        raise AssertionError("K5A missing synthetic fixture")
    out=io.StringIO()
    writer=csv.writer(out,delimiter="\t",lineterminator="\n")
    writer.writerow(["sequences","labels","fold_id"])
    writer.writerows(rows)
    return out.getvalue().encode()

class FakeAPI:
    CofoldingModel=Obj(BOLTZ_2="boltz_2")
    def __init__(self,weekly=100,existing=None,folder=m.TARGET_ROWAN_FOLDER_UUID,org=True,subscription=None):
        self.weekly=weekly
        self.existing=existing or []
        self.folder=folder
        self.org=org
        self.subscription=subscription
        self.sent=[]
    def whoami(self):
        return Obj(organization=Obj(weekly_credits=self.weekly,credits=500) if self.org else None,
                   individual_subscription=self.subscription)
    def retrieve_folder(self,uuid):return Obj(uuid=self.folder)
    def list_workflows(self,parent_uuid,page,size):return self.existing[page*size:(page+1)*size]
    def submit_protein_cofolding_workflow(self,**kwargs):
        self.sent.append(kwargs)
        return Obj(uuid="00000000-0000-4000-8000-000000000123")

def job(name):
    return Obj(name=name,parent_uuid=m.TARGET_ROWAN_FOLDER_UUID,workflow_type="protein_cofolding")

class Tests(unittest.TestCase):
    def setUp(self):
        self.blob=fixture()
        self.patcher=patch.object(m,"validate_tsv",return_value={"status":"EXPERIMENTAL_DERIVATIVE_ROW_VERIFIED"})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def selected(self):
        return m.select(self.blob,manifest)

    def test_eight_stratified_exact_mutations(self):
        xs=self.selected()
        self.assertEqual(len(xs),8)
        self.assertEqual([v["quantile_target_percent"] for v in xs],list(m.QUANTILES))
        self.assertEqual(len({v["position_1_based"] for v in xs}),8)
        self.assertEqual(len({v["case_name"] for v in xs}),8)
        self.assertTrue(all(len(v["sequence"])==64 for v in xs))
        self.assertTrue(all(v["sequence"]!=manifest["independent_control"]["mutant"] for v in xs))

    def test_deterministic_selection(self):
        self.assertEqual(self.selected(),self.selected())

    def test_wrong_row_count_fail_closed(self):
        shorter=b"\n".join(self.blob.split(b"\n")[:-4])
        with self.assertRaises(m.BudgetHold):
            m.select(shorter,manifest)

    def test_one_job_per_run(self):
        api=FakeAPI()
        r=m.step(api,manifest,self.blob,approval=True)
        self.assertEqual(r["status"],"ONE_PILOT_MUTANT_SUBMITTED")
        self.assertEqual(len(api.sent),1)
        self.assertEqual(api.sent[0]["max_credits"],10)
        self.assertEqual(api.sent[0]["folder_uuid"],m.TARGET_ROWAN_FOLDER_UUID)
        self.assertEqual(api.sent[0]["initial_protein_sequences"],[self.selected()[0]["sequence"]])

    def test_existing_one_advances_to_second(self):
        api=FakeAPI(existing=[job(self.selected()[0]["case_name"])])
        r=m.step(api,manifest,self.blob,approval=True)
        self.assertEqual(r["mutation"],self.selected()[1]["mutation"])

    def test_existing_all_never_spends(self):
        api=FakeAPI(existing=[job(v["case_name"]) for v in self.selected()])
        r=m.step(api,manifest,self.blob,approval=True)
        self.assertEqual(r["status"],"ALL_EIGHT_CASES_EXIST_NO_MORE_SUBMISSIONS")
        self.assertEqual(api.sent,[])

    def test_insufficient_weekly_not_replace_with_500(self):
        api=FakeAPI(weekly=20)
        with self.assertRaises(m.BudgetHold):
            m.step(api,manifest,self.blob,approval=True)
        self.assertEqual(api.sent,[])

    def test_missing_org_hold(self):
        api=FakeAPI(org=False)
        with self.assertRaises(m.BudgetHold):
            m.step(api,manifest,self.blob,approval=True)

    def test_subscription_change_hold(self):
        api=FakeAPI(subscription="active_sub")
        with self.assertRaises(m.BudgetHold):
            m.step(api,manifest,self.blob,approval=True)

    def test_no_approval_hold(self):
        api=FakeAPI()
        with self.assertRaises(m.BudgetHold):
            m.step(api,manifest,self.blob,approval=False)

    def test_wrong_folder_hold(self):
        api=FakeAPI(folder="wrong")
        with self.assertRaises(m.BudgetHold):
            m.step(api,manifest,self.blob,approval=True)

    def test_duplicate_existing_hold(self):
        name=self.selected()[0]["case_name"]
        api=FakeAPI(existing=[job(name),job(name)])
        with self.assertRaises(m.BudgetHold):
            m.step(api,manifest,self.blob,approval=True)

if __name__=="__main__":unittest.main()
