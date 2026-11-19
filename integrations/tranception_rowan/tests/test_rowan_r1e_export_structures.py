import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rowan_r1e_export_structures as r
def pdb(seq):
    inv={v:k for k,v in r.AA.items()}
    return "\n".join(f"ATOM  {i:5d}  CA  {inv[aa]:3s} A{i:4d}    {1.0:8.3f}{2.0:8.3f}{3.0:8.3f}  1.00 90.00           C" for i,aa in enumerate(seq,1))
class Tests(unittest.TestCase):
    def test_proper_sequence(self):
        data,n,m=r.verify_atoms("HEADER PRIVATE\n"+pdb(r.WT));self.assertEqual((n,m),(136,136));self.assertNotIn(b"PRIVATE",data)
    def test_missing_residue(self):
        with self.assertRaises(r.Hold):r.verify_atoms(pdb(r.WT[:-1]))
    def test_wrong_sequence(self):
        with self.assertRaises(r.Hold):r.verify_atoms(pdb("A"+r.WT[1:]))
    def test_no_atoms(self):
        with self.assertRaises(r.Hold):r.verify_atoms("END")
