"""Synthetic numerical/sequence and real-source defensive tests; no Rowan network."""
import unittest
import numpy as np
from rowan_scoby_acsd_rcsb_3a8e_readonly import (
    TARGET,EXPECTED_PROVIDER_UUID,MIN_MATCHED_CA,map_chain_to_target,compare_arrays,Hold
)
class GeometryTests(unittest.TestCase):
    def test_fixed_known_source_length(self):
        self.assertEqual(len(TARGET["sequence"]),156)
        self.assertEqual(EXPECTED_PROVIDER_UUID,"d6857380-d46b-4c27-bf79-faf761c9af36")
    def test_exact_alignment_ignores_insert(self):
        seq="ACDE"
        r=[("A",np.array([0.,0,0])),("C",np.array([1.,0,0])),
           ("G",np.array([2.,0,0])),("D",np.array([3.,0,0])),("E",np.array([4.,0,0]))]
        m=map_chain_to_target(seq,r)
        self.assertEqual(len(m),4)
        self.assertIn(3,m)
    def test_rmsd_invariant_to_rigid_translation(self):
        ref={i:np.array([np.cos(i),np.sin(i),i/10.]) for i in range(156)}
        pred={i:v+np.array([4.,-10.,3.]) for i,v in ref.items()}
        o=compare_arrays(ref,pred,156)
        self.assertEqual(o["matched_CA_atoms"],156)
        self.assertLess(o["global_CA_RMSD_A"],1e-4)
    def test_ca_coverage_holds(self):
        ref={i:np.array([i,0.,0.]) for i in range(100)}
        pred=ref.copy()
        with self.assertRaisesRegex(Hold,"INSUFFICIENT_SEQUENCE"):
            compare_arrays(ref,pred,156)
    def test_ca_nonfinite_holds(self):
        ref={i:np.array([i,0.,0.]) for i in range(156)}
        pred={i:np.array([float("nan"),0.,0.]) for i in range(156)}
        with self.assertRaisesRegex(Hold,"INVALID_CA"):
            compare_arrays(ref,pred,156)
if __name__=="__main__":unittest.main()
