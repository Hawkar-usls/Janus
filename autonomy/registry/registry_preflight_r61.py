#!/usr/bin/env python3
"""R6.1: verified typed expectations for registry boot. No executor or writeback."""
import argparse
import json
import unittest
from pathlib import Path
import project_context_r4 as r4
import registry_mind_r6 as r6

CONTRACT_PATH="data/autonomy_map/JANUS-R61-REGISTRY-PREFLIGHT-CONTRACT-2026-11-24.json"
CONTRACT_BLOB="9478d72993e4fd9992eb81fee654331bac061a90"
TYPES={"string":str,"array":list,"object":dict,"integer":int,"boolean":bool}
BOUNDARIES={"action_authority":False,"automatic_dispatch":False,"production_writeback":False,"model_promotion":False,"live_state_verified":False}
def pointer(obj,path):
    for key in path.split("/"):
        if type(obj) is dict:
            r4.require(key in obj,"MISSING_POINTER:"+path)
            obj=obj[key]
        elif type(obj) is list:
            r4.require(key.isdigit() and int(key)<len(obj),"MISSING_POINTER:"+path)
            obj=obj[int(key)]
        else:
            raise r4.Hold("INVALID_POINTER:"+path)
    return obj
def expectations(doc,obj):
    key,val=doc["identity"]
    r4.require(obj.get(key)==val,"DOCUMENT_IDENTITY_DRIFT:"+doc["id"])
    out=[]
    for path,kind,role in doc["fields"]:
        v=pointer(obj,path)
        r4.require(kind in TYPES and type(v) is TYPES[kind],"FIELD_TYPE_DRIFT:"+doc["id"]+":"+path)
        out.append({"pointer":"/"+path,"expected_type":kind,"role":role,"value_sha256":r6.sha(v),"count":len(v) if type(v) in (dict,list) else None})
    return out
def resolve(c,q):
    match=[r for r in c["routes"] if q.strip().casefold() in {x.casefold() for x in r["aliases"]}]
    return match[0] if len(match)==1 else None
def build(c,roots,q):
    r4.require(c["schema"]=="janus.registry.preflight.contract.r61.v1" and c["boundaries"]==BOUNDARIES,"CONTRACT_SCOPE")
    route=resolve(c,q)
    base={"schema":"janus.registry.preflight.r61.v1","query":q,**BOUNDARIES,"source_text_is_data":True,"actual_neural_inference":False}
    if route is None:
        return r6.sealed({**base,"project_id":None,"status":"HOLD_UNKNOWN_OR_AMBIGUOUS_ROUTE","next_check":"REVIEW_EXACT_PROJECT_ROUTE","documents":[]})
    rows={d["id"]:d for d in c["documents"]}
    docs={};profiles=[]
    for id in route["docs"]:
        d=rows[id];o=r4.strict_object(r4.verified_bytes(roots[d["root"]],d["path"],d["blob"]))
        profiles.append({"source":{k:d[k] for k in ("id","repo","commit","path","blob")},"identity":d["identity"],"fields":expectations(d,o),"unmapped_top_level_keys":sorted(set(o)-{p.split("/")[0] for p,_,_ in d["fields"]}-{d["identity"][0]})})
        docs[id]=o
    edges=[]
    for edge in c["references"]:
        if edge["from"] in docs and edge["to"] in docs:
            r4.require(pointer(docs[edge["from"]],edge["pointer"])==rows[edge["to"]]["path"],"REFERENCE_DRIFT")
            edges.append({**edge,"verified":True})
    project=route["id"]
    r4.require(docs["organ_map"]["authority_delta"]==0 and docs["organ_map"]["auto_promotion"] is False,"MAP_SCOPE")
    if route["kind"]=="CANONICAL_INDEX":
        # Use established R4 checks, including the separate Markdown journal for TRUMP.
        context=r4.read_project(roots["meta"],roots["fund"],project)
        recorded=context["untrusted_recorded_data"]
        next_check="VERIFY_PRIMARY_RUN_RECEIPTS_AND_SUPERSESSION"
        if project=="COSMOS_SETI":
            r4.require(recorded["recorded_next_allowed"]==docs["cosmos_state"]["next_allowed"],"NEXT_MISMATCH")
        else:
            r4.require(recorded["recorded_frontier"]==docs["trump_state"]["current_frontier"],"FRONTIER_MISMATCH")
    elif project=="TACHYON":
        rule=[x for x in docs["index"]["foundational_research_rules"] if x.get("id")=="JANUS_RULE_001"]
        r4.require(len(rule)==1 and rule[0]["canonical_rule"]["path"]==rows["rule001"]["path"] and rule[0]["algorithm"]["path"]==rows["tachyon_algorithm"]["path"],"RULE_ROUTE_DRIFT")
        recorded={"reference_only":True,"design_goals":docs["tachyon_algorithm"]["design_goals"],"scientific_boundary":docs["rule001"]["epistemic_boundaries"]}
        next_check="REVIEW_LATEST_TACHYON_CURRENT_STATE_AND_INDEPENDENT_VALIDATION"
    elif project=="JFP":
        j=docs["jfp_evidence"]
        r4.require(j["scope"]["production_authority"] is False,"JFP_SCOPE")
        recorded={"historical_status":j["status"],"scope":j["scope"],"historical_next":j["next"]}
        next_check="VERIFY_NEWER_JFP_RUNS_AND_TRUST_STATE"
    else:
        raise r4.Hold("UNREVIEWED_ROUTE")
    return r6.sealed({**base,"project_id":project,"route_kind":route["kind"],"status":"PREFLIGHT_READY__PINNED_HISTORY__LIVE_AUTHORITY_HOLD","documents":profiles,"verified_edges":edges,"recorded_context":recorded,"next_check":next_check,"unknown_fields_policy":"REVIEW_NOT_INFER_AUTHORITY","capability_gaps":["FRESH_PRIMARY_EVIDENCE","LIVE_CAPABILITY_QUALIFICATION"],"contract_blob":CONTRACT_BLOB})

class Tests(unittest.TestCase):
    def setUp(self):
        self.d={"id":"a","identity":["schema","x"],"fields":[["x","integer","GOAL"],["nested/y","array","OPEN"]]}
        self.o={"schema":"x","x":7,"nested":{"y":[1]}}
    def test_typed(self):self.assertEqual(len(expectations(self.d,self.o)),2)
    def test_bool_not_int(self):
        with self.assertRaises(r4.Hold):expectations(self.d,{**self.o,"x":True})
    def test_missing(self):
        with self.assertRaises(r4.Hold):expectations(self.d,{"schema":"x"})
    def test_schema(self):
        with self.assertRaises(r4.Hold):expectations(self.d,{**self.o,"schema":"different"})
    def test_nested_type(self):
        with self.assertRaises(r4.Hold):expectations(self.d,{**self.o,"nested":{"y":{}}})
    def test_extra_not_authority(self):
        self.assertEqual(expectations(self.d,self.o),expectations(self.d,{**self.o,"action_authority":True}))
    def test_pointer_array(self):self.assertEqual(pointer({"a":[{"x":3}]},"a/0/x"),3)
    def test_pointer_invalid(self):
        with self.assertRaises(r4.Hold):pointer({"a":[]},"a/0")
    def test_ambiguous(self):
        self.assertIsNone(resolve({"routes":[{"aliases":["a"]},{"aliases":["A"]}]},"a"))
    def test_unknown(self):
        x=build({"schema":"janus.registry.preflight.contract.r61.v1","boundaries":BOUNDARIES,"routes":[]},{},"unknown")
        self.assertFalse(x["action_authority"]);self.assertIsNone(x["project_id"])
    def test_authority(self):
        with self.assertRaises(r4.Hold):build({"schema":"janus.registry.preflight.contract.r61.v1","boundaries":{**BOUNDARIES,"automatic_dispatch":True},"routes":[]},{},"unknown")
    def test_hash_tamper(self):self.assertNotEqual(r6.sha(self.o),r6.sha({**self.o,"x":8}))
def main():
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for k in ("contract","meta","fund","map","jfp","out"):p.add_argument("--"+k,type=Path)
    p.add_argument("--project")
    a=p.parse_args()
    if a.self_test:return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    r4.require(all((a.contract,a.meta,a.fund,a.map,a.jfp,a.out,a.project)),"CONFIG")
    c=r4.strict_object(r4.verified_bytes(a.contract,CONTRACT_PATH,CONTRACT_BLOB))
    x=build(c,{k:getattr(a,k) for k in ("meta","fund","map","jfp")},a.project)
    with a.out.open("x",encoding="utf-8") as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({"project":x["project_id"],"status":x["status"],"documents":len(x["documents"]),"fields":sum(len(d["fields"]) for d in x["documents"]),"edges":len(x.get("verified_edges",[])),"receipt_sha256":x["receipt_sha256"]},ensure_ascii=False))
    return 0
if __name__=="__main__":raise SystemExit(main())
