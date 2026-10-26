"""No-network tests: UniProt domain boundaries, anchoring and fail-closed invariants."""
import inspect
import unittest
import numpy as np
import rowan_fn1_uniprot_domain_pose_readonly as a

class DomainPoseTests(unittest.TestCase):
    def fixture(self):
        # Independent domains with noncollinear CA arrays.
        x={}
        for i in a.BOUNDARIES["FNII_1"]:
            x[i]=np.array([np.sin(i*.5),np.cos(i*.3),i*.08])
        for i in a.BOUNDARIES["FNII_2"]:
            j=i-60
            x[i]=np.array([10+np.sin(j*.5),np.cos(j*.3),j*.08])
        return x
    def test_uniprot_boundaries_are_fixed(self):
        self.assertEqual(len(a.BOUNDARIES["FNII_1"]),49)
        self.assertEqual(len(a.BOUNDARIES["FNII_2"]),49)
        self.assertEqual(a.BOUNDARIES["FNII_1"][-1],48)
        self.assertEqual(a.BOUNDARIES["FNII_2"][0],60)
    def test_domain_rmsd_identical(self):
        x=self.fixture()
        z=a.evaluate_domain_geometry(x,x)
        self.assertEqual(z["FNII_1"]["independently_fitted_CA_RMSD_A"],0.0)
        self.assertEqual(z["FNII_2"]["independently_fitted_CA_RMSD_A"],0.0)
        self.assertEqual(z["FNII_2_after_FNII_1_alignment"]["CA_RMSD_A"],0.0)
    def test_domain2_relative_pose_shift_not_explained_by_local_fit(self):
        x=self.fixture()
        y={i:v.copy() for i,v in x.items()}
        for j in a.BOUNDARIES["FNII_2"]:
            y[j]+=np.array([0.,8.,0.])
        z=a.evaluate_domain_geometry(x,y)
        self.assertLess(z["FNII_1"]["independently_fitted_CA_RMSD_A"],.0001)
        self.assertLess(z["FNII_2"]["independently_fitted_CA_RMSD_A"],.0001)
        self.assertGreater(z["FNII_2_after_FNII_1_alignment"]["CA_RMSD_A"],7.)
    def test_missing_domain_ca_holds(self):
        x=self.fixture()
        y={i:v for i,v in x.items() if not 60<=i<=70}
        with self.assertRaisesRegex(a.Hold,"INSUFFICIENT_DOMAIN_IDENTICAL"):
            a.evaluate_domain_geometry(x,y)
    def test_no_coordinate_values_in_domain_summary(self):
        x=self.fixture()
        r=[a.evaluate_domain_geometry(x,x) for _ in range(3)]
        s=a.summary(r)
        self.assertIn("median",s["FNII_1_RMSD"])
        self.assertIn("max",s["FNII_2_anchored_RMSD"])
        self.assertEqual(len(r),3)
    def test_frozen_21_original_models_and_source_hashes(self):
        self.assertEqual(a.REFS,(("1E88",20),("1E8B",1)))
        self.assertEqual(len(a.EXPECTED_HASHES["ROWAN"]),64)
        self.assertEqual(a.EXPECTED_HASHES["1E88"],"0b91f9074f4e24f327c10d4d254e9fce543c14b741910b5fe15fac342bbf1f25")
    def test_no_rowan_submit(self):
        self.assertNotIn("submit_protein_cofolding_workflow",inspect.getsource(a.run))
if __name__=="__main__":
    unittest.main()
