import sys,unittest
from pathlib import Path
from types import SimpleNamespace as S
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rowan_r1g_cap_incident as a
from probe import TARGET_ROWAN_FOLDER_UUID as F
class Test(unittest.TestCase):
    def test_overcap_read(self):
        _,name,_=a.identity("BBC1","WT",5)
        j=S(name=name,parent_uuid=F,workflow_type="protein_cofolding",credits_charged=11.2,status="Status.COMPLETED_OK")
        j.fetch_latest=lambda:j;j.done=lambda:True;j.result=lambda **k:S(predictions=[S(predicted_structure_uuid=str(i),scores=S(ptm=.9,avg_lddt=.95)) for i in range(5)])
        api=S(retrieve_folder=lambda *a:S(uuid=F),list_workflows=lambda **k:[j])
        x=a.inspect_existing(api);self.assertTrue(x["cap_exceeded"]);self.assertEqual(x["provider_reported_credits"],11.2);self.assertEqual(x["returned_samples"],5);self.assertEqual(x["new_jobs"],0)
