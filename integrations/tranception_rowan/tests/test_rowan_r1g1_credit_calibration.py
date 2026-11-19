import sys,unittest,contextlib,inspect
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rowan_r1g1_credit_calibration as r
class Test(unittest.TestCase):
    def test_settings(self):
        x=S(data={"initial_protein_sequences":[r.BBC_SEQ],"num_samples":5,"model":"boltz_2","use_msa_server":True})
        self.assertEqual(r.settings(x)["sequence_sha256"],[r.digest(r.BBC_SEQ)])
        self.assertEqual(r.settings(x)["num_samples"],5)
    def test_log_no_raw(self):
        x=r.classify("credit limit exceeded; MSA query; CUDA out of memory")
        self.assertGreater(x["budget_stop"],0);self.assertGreater(x["msa"],0);self.assertGreater(x["gpu_oom"],0)
    def test_missing(self):self.assertEqual(r.settings(S(data=None)),{"available":False})
