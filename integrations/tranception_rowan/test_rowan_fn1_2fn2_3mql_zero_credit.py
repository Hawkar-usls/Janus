"""No-network benchmark input, source integrity and numerics for FN1 external references."""
import inspect
import unittest
import numpy as np
import rowan_fn1_2fn2_3mql_zero_credit as a

class ReferenceTests(unittest.TestCase):
    def test_target_109_is_exact_3MQL_substring(self):
        self.assertEqual(len(a.REF3),220)
        self.assertEqual(a.REF3.count(a.TARGET["sequence"]),1)
        self.assertEqual(a.REF3.index(a.TARGET["sequence"]),50)
    def test_second_FNII_is_exact_2FN2_substring(self):
        self.assertEqual(len(a.FNII2),49)
        self.assertEqual(len(a.REF2),59)
        self.assertEqual(a.REF2.count(a.FNII2),1)
        self.assertEqual(a.REF2.index(a.FNII2),9)
    def test_isolated_model_49aa_identical_alignment(self):
        positions=[(aa,np.array([float(i),0.,0.])) for i,aa in enumerate(a.REF2)]
        mapped=a.subsegment_map(a.FNII2,positions)
        self.assertEqual(len(mapped),49)
        self.assertEqual(mapped[0][0],9.0)
        self.assertEqual(mapped[48][0],57.0)
    def test_subsegment_missing_data_holds(self):
        positions=[(aa,np.array([float(i),0.,0.])) for i,aa in enumerate(a.REF2[:12])]
        with self.assertRaisesRegex(a.Hold,"COVERAGE_INSUFFICIENT"):
            a.subsegment_map(a.FNII2,positions)
    def test_kabsch_identical_domain(self):
        d={i:np.array([np.sin(i*.2),np.cos(i*.5),float(i)/3]) for i in range(49)}
        r=a.domain_fit(d,d)
        self.assertEqual(r["matched_CA"],49)
        self.assertLess(r["CA_RMSD_A"],.00001)
        self.assertEqual(r["GDT_TS_like_1_2_4_8_A"],1.0)
    def test_kabsch_rigid_translation_invariant(self):
        d={i:np.array([np.sin(i),np.cos(i),float(i)]) for i in range(49)}
        p={i:coord+np.array([8.,-3.,-1.]) for i,coord in d.items()}
        r=a.domain_fit(d,p)
        self.assertLess(r["CA_RMSD_A"],.00001)
    def test_missing_45_ca_holds(self):
        d={i:np.array([i,0.,0.]) for i in range(49)}
        with self.assertRaisesRegex(a.Hold,"TOO_SHORT"):
            a.domain_fit(d,{i:d[i] for i in range(44)})
    def test_20_model_min_median_max_not_best_only(self):
        rows=[{"CA_RMSD_A":k,"GDT_TS_like_1_2_4_8_A":k/10} for k in [1,2,3]]
        x=a.summary(rows,["CA_RMSD_A"])
        self.assertEqual(x["CA_RMSD_A"],{"median":2.0,"min":1.0,"max":3.0})
    def test_provider_is_readonly(self):
        self.assertNotIn("submit_protein_cofolding_workflow",inspect.getsource(a.run))
        self.assertIn("get_predicted_structure",inspect.getsource(a.run))
if __name__=="__main__": unittest.main()
