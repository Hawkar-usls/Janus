"""Entirely synthetic geometry tests; never invoke Rowan provider."""
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace as Obj
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("rowan_bbc1_geometry",ROOT/"rowan_bbc1_geometry.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
manifest=json.loads((ROOT/"discovery_manifest.json").read_text(encoding="utf-8"))
cases=[m.prepare(manifest,arm) for arm in m.CASES]
reverse={v:k for k,v in m.AA3.items()}

def fake_coords():
    return np.array([[i*1.4, math.sin(i/3)*3.,math.cos(i/5)*2.] for i in range(64)])

def pdb_bytes(sequence,coords,chain="A"):
    lines=[]
    for i,(aa,xyz) in enumerate(zip(sequence,coords),start=1):
        lines.append(f"ATOM  {i:5d}  CA  {reverse[aa]:>3s} {chain}{i:4d}    "
                     f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 40.00           C")
    return ("\n".join(lines)+"\nEND\n").encode()

class FakeProtein:
    def __init__(self,data):self.data=data
    def download_pdb_file(self,path,name):
        (Path(path)/(name+".pdb")).write_bytes(self.data)
class FakeWorkflow:
    workflow_type="protein_cofolding"
    parent_uuid=m.TARGET_ROWAN_FOLDER_UUID
    def __init__(self,case,data):
        self.name=case["id"]
        self.data=data
    def fetch_latest(self):return self
    def done(self):return True
    def result(self,wait=False):return Obj(get_predicted_structure=lambda:FakeProtein(self.data))
class FakeProvider:
    def __init__(self,workflows):self.jobs=workflows
    def whoami(self):return Obj()
    def retrieve_folder(self,uuid):return Obj(uuid=uuid)
    def list_workflows(self,parent_uuid,page,size):
        assert parent_uuid==m.TARGET_ROWAN_FOLDER_UUID
        return self.jobs[page*size:(page+1)*size]

class Tests(unittest.TestCase):
    def test_pdb_exact_sequence(self):
        x=fake_coords()
        out=m.parse_pdb_ca(pdb_bytes(cases[0]["sequence"],x),cases[0]["sequence"])
        self.assertEqual(out.shape,(64,3))
        self.assertTrue(np.allclose(out,x,atol=.001))

    def test_wildtype_and_k5a_differ_at_exact_one_site(self):
        self.assertEqual(sum(a!=b for a,b in zip(cases[0]["sequence"],cases[1]["sequence"])),1)
        self.assertEqual(cases[0]["sequence"][4],"K")
        self.assertEqual(cases[1]["sequence"][4],"A")

    def test_rotation_and_translation_invariant_rmsd(self):
        x=fake_coords()
        theta=math.radians(37)
        rot=np.array([[math.cos(theta),-math.sin(theta),0],
                      [math.sin(theta),math.cos(theta),0],
                      [0,0,1.]])
        y=x@rot+np.array([10,5,-2])
        r=m.kabsch_summary(x,y)
        self.assertLess(r["global_Calpha_rmsd_angstrom"],.00001)

    def test_local_structural_shift_detected(self):
        x=fake_coords()
        y=x.copy()
        y[4]=y[4]+[0,0,2]
        r=m.kabsch_summary(x,y)
        self.assertGreater(r["global_Calpha_rmsd_angstrom"],.05)
        self.assertGreater(r["mutation_position_5_Calpha_displacement_angstrom"],1.)

    def test_sequence_mismatch_rejected(self):
        x=fake_coords()
        with self.assertRaises(m.GeometryHold):
            m.parse_pdb_ca(pdb_bytes(cases[0]["sequence"],x),cases[1]["sequence"])

    def test_incomplete_ca_chain_rejected(self):
        x=fake_coords()
        raw=pdb_bytes(cases[0]["sequence"][:-1],x[:-1])
        with self.assertRaises(m.GeometryHold):
            m.parse_pdb_ca(raw,cases[0]["sequence"])

    def test_duplicate_ca_rejected(self):
        x=fake_coords()
        raw=pdb_bytes(cases[0]["sequence"],x)
        first=raw.split(b"\n",1)[0]
        with self.assertRaises(m.GeometryHold):
            m.parse_pdb_ca(first+b"\n"+raw,cases[0]["sequence"])

    def test_ambiguous_chain_rejected(self):
        x=fake_coords()
        raw=pdb_bytes(cases[0]["sequence"],x,chain="A")+pdb_bytes(cases[0]["sequence"],x,chain="B")
        with self.assertRaises(m.GeometryHold):
            m.parse_pdb_ca(raw,cases[0]["sequence"])

    def test_bad_geometry_dimensions_rejected(self):
        with self.assertRaises(m.GeometryHold):
            m.kabsch_summary(np.zeros((63,3)),np.zeros((63,3)))

    def test_complete_readonly_provider_rmsd(self):
        coords=fake_coords()
        a=FakeWorkflow(cases[0],pdb_bytes(cases[0]["sequence"],coords))
        b=FakeWorkflow(cases[1],pdb_bytes(cases[1]["sequence"],coords))
        report=m.analyze(FakeProvider([a,b]),manifest)
        self.assertLess(report["geometry"]["global_Calpha_rmsd_angstrom"],1e-6)
        self.assertTrue(report["input_sequence_verified_by_pdb_residue_identity"])
        self.assertFalse(report["raw_structures_published"])
        self.assertFalse(report["fitness_change_inferred"])

    def test_unsubmitted_case_blocks(self):
        coords=fake_coords()
        a=FakeWorkflow(cases[0],pdb_bytes(cases[0]["sequence"],coords))
        with self.assertRaises(m.GeometryHold):
            m.analyze(FakeProvider([a]),manifest)

if __name__=="__main__":unittest.main()
