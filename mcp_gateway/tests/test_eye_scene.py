import json
import unittest
from mcp_gateway.organism_tools import (
    read_eye_scene, EYE_SPATIAL_CAPABILITY_PATH, EYE_SPATIAL_BINDING_PATH,
    EYE_SPATIAL_SCENE_PATH, EYE_SPATIAL_REGISTRY, EYE_SPATIAL_CAPABILITY_REF,
    EYE_SPATIAL_BINDING_REF, EYE_SPATIAL_SCENE_REF,
)

class FakeAPI:
    def __init__(self):
        self.calls=[]
        self.cap={"schema":"janus.eye.organ_capability.v1","organ_id":"JANUS_EYE","policy":{"read_only_data":True,"production_promotion":False}}
        self.bind={"schema":"janus.eye.spatial_binding.v1","authority_delta":0}
        self.record={"run_id":45833030382,"production_promotion":False,"receipt":{"schema":"janus.eye.scene_receipt.v1","scene_id":"KUSTO_ZERO_NODE","authority":"MODELLED","frame":"test metric","units":{"height":"m"},"nodes":[],"coverage_fraction":1,"claim_ceiling":"DESCRIPTIVE","limitations":["modelled"]}}
    def get_file(self,repo,path,ref):
        self.calls.append((repo,path,ref))
        assert repo==EYE_SPATIAL_REGISTRY
        data,expected={EYE_SPATIAL_CAPABILITY_PATH:(self.cap,EYE_SPATIAL_CAPABILITY_REF),EYE_SPATIAL_BINDING_PATH:(self.bind,EYE_SPATIAL_BINDING_REF),EYE_SPATIAL_SCENE_PATH:(self.record,EYE_SPATIAL_SCENE_REF)}[path]
        assert ref==expected
        return {"content":json.dumps(data),"sha":"test"}

class EyeRouteTests(unittest.TestCase):
    def test_pinned_read_only(self):
        api=FakeAPI();r=read_eye_scene(api)
        self.assertEqual(len(api.calls),3)
        self.assertEqual(r["authority"],"MODELLED")
        self.assertEqual(r["authority_delta"],0)
        self.assertFalse(r["write_back"])
    def test_unknown_no_network(self):
        api=FakeAPI()
        with self.assertRaises(ValueError):read_eye_scene(api,"https://example.com")
        self.assertFalse(api.calls)
    def test_authority_upgrade_rejected(self):
        api=FakeAPI();api.record["receipt"]["authority"]="MEASURED"
        with self.assertRaises(ValueError):read_eye_scene(api)
    def test_binding_privilege_rejected(self):
        api=FakeAPI();api.bind["authority_delta"]=1
        with self.assertRaises(ValueError):read_eye_scene(api)
