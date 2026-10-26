"""Scientific fail-closed no-network tests: FN1 21 frozen NMR conformers."""
import unittest
import numpy as np
import rowan_fn1_1e88_1e8b_nmr_readonly as a

class FN1NMRNumericalTests(unittest.TestCase):
    def mock_coords(self):
        return {i:np.asarray([np.sin(i/3),np.cos(i/3),i*.25],dtype=float) for i in range(109)}
    def test_frozen_uniprot_identity_and_reference_offset(self):
        self.assertEqual(len(a.TARGET["sequence"]),109)
        self.assertEqual(len(a.FULL_REF_SEQUENCE),160)
        self.assertEqual(a.FULL_REF_SEQUENCE.find(a.TARGET["sequence"]),50)
        self.assertEqual(a.FULL_REF_SEQUENCE.count(a.TARGET["sequence"]),1)
        self.assertEqual(a.REFS,(("1E88",20),("1E8B",1)))
    def test_structure_compare_identical(self):
        x=self.mock_coords()
        r=a.compare_models(x,x,109)
        self.assertEqual(r["matched_exact_CA"],109)
        self.assertLess(r["global_CA_RMSD_A"],.00001)
        self.assertLess(r["pairwise_dRMSD_A"],.00001)
        self.assertEqual(r["contact_F1"],1.0)
    def test_global_translation_invariant(self):
        x=self.mock_coords()
        y={i:v+np.asarray([8.,-2.,7.]) for i,v in x.items()}
        r=a.compare_models(x,y,109)
        self.assertEqual(r["contact_F1"],1.0)
        self.assertLess(r["global_CA_RMSD_A"],.00001)
    def test_common_CA_threshold(self):
        x=self.mock_coords()
        y={i:v for i,v in x.items() if i<99}
        with self.assertRaisesRegex(a.Hold,"INSUFFICIENT_COMMON_MATCHED"):
            a.compare_models(x,y,109)
    def test_all_models_summary_not_best_only(self):
        x=self.mock_coords()
        y={i:v+np.asarray([0.,0.,.8*np.sin(i)]) for i,v in x.items()}
        z={i:v+np.asarray([0.,0.,1.6*np.sin(i)]) for i,v in x.items()}
        R=[a.compare_models(x,ref,109) for ref in (x,y,z)]
        S=a.aggregate(R)
        self.assertEqual(S["global_CA_RMSD_A"]["min"],0.0)
        self.assertGreater(S["global_CA_RMSD_A"]["max"],S["global_CA_RMSD_A"]["median"])
    def test_exact_sequence_mapping_embedded_with_flanks(self):
        obs=[(aa,np.array([float(i),0.,0.])) for i,aa in enumerate(a.FULL_REF_SEQUENCE)]
        mapped=a.map_CA_to_target(a.TARGET["sequence"],obs)
        self.assertEqual(len(mapped),109)
        self.assertEqual(float(mapped[0][0]),50.0)
        self.assertEqual(float(mapped[108][0]),158.0)
    def test_missing_target_segment_holds(self):
        z=[(aa,np.array([i,0.,0.])) for i,aa in enumerate(a.FULL_REF_SEQUENCE[:50])]
        with self.assertRaisesRegex(a.Hold,"INSUFFICIENT_EXACT"):
            a.map_CA_to_target(a.TARGET["sequence"],z)
    def test_sanitized_zero_spend_source_contains_no_submit(self):
        import inspect
        source=inspect.getsource(a.run)
        self.assertNotIn("submit_protein_cofolding_workflow",source)
        self.assertIn("get_predicted_structure",source)

if __name__=="__main__":
    unittest.main()
