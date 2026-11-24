#!/usr/bin/env python3
"""R6.3: one verified atlas, scoped boot capsules; read-only research."""
import argparse, json, re, tempfile, unittest
from pathlib import Path
import project_context_r4 as r4
import registry_mind_r6 as r6
import registry_preflight_r61 as pre
import semantic_atlas_r62 as atlas

PATH="data/autonomy_map/JANUS-R63-ORGAN-BOOT-CONTRACT-2026-11-24.json"
BLOB="f94e09b5686b199735e9a6746c82ec76ea384218"
SCHEMA="janus.registry.organ_boot.r63.v1"
FLAGS={**pre.BOUNDARIES,"actual_neural_inference":False}
MAX_BYTES=65536
def require(ok,msg):r4.require(ok,msg)
def checked(x):
    require(type(x) is dict and x.get("receipt_sha256")==r6.sha({k:v for k,v in x.items() if k!="receipt_sha256"}),"SEAL")
    return x
def preview(v):
    raw=json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
    return {"value_sha256":r6.sha(v),"json_preview":raw[:600],"truncated":len(raw)>600}
def boot_packet(c,route,profiles,docs,source_atlas,preflight):
    require(c["boundaries"]==FLAGS,"CONTRACT_FLAGS")
    require(route in c["routes"],"UNREVIEWED_ORGAN")
    require(preflight["status"].startswith("PREFLIGHT_READY") and preflight["project_id"]==pre.resolve(atlas.contract(),route["project"])["id"],"ROUTE_MISMATCH")
    docmap={d["id"]:(d,obj) for d,obj,_ in docs}
    fields=[];sources=[];unmapped=0
    for profile in preflight["documents"]:
        d,obj=docmap[profile["source"]["id"]]
        require(d["blob"]==profile["source"]["blob"],"SOURCE_MISMATCH")
        a=profiles[d["blob"]]
        require(a["identity"]==d["identity"] and a["source"]["path"]==d["path"],"ATLAS_IDENTITY")
        indexed={tuple(f["pointer"]):f for f in a["fields"]}
        for f in profile["fields"]:
            keys=f["pointer"].lstrip("/").split("/")
            row=indexed.get(tuple(keys))
            v=pre.pointer(obj,"/".join(keys))
            require(row is not None and row["value_sha256"]==f["value_sha256"]==r6.sha(v),"ATLAS_POINTER")
            role=f["role"]
            fields.append({"source_blob":d["blob"],"pointer":f["pointer"],"role":role,"type":f["expected_type"],**preview(v)})
        unmapped+=sum(f["curated_role"] is None for f in a["fields"])
        sources.append(profile["source"])
    q=source_atlas["queries"][route["project"]]
    allowed={(s["repo"],s["commit"],s["path"]) for s in sources}
    citations=[x for x in q["citations"] if (x["origin"].get("repository"),x["origin"].get("commit"),x["origin"].get("path")) in allowed]
    require(q["expected_source_found"] and citations,"CORTEX_SCOPE")
    controllers=docmap["organ_map"][1]["controllers"]
    matches=[x for x in controllers if x["repository"]==route["repository"]]
    require(len(matches)==1,"ORGAN_MAP")
    # Explicit source-bound context only; previews are not executable commands.
    groups={k:[x for x in fields if x["role"] in roles] for k,roles in {
        "goals":{"GOAL"},"state":{"RECORDED_STATE","STATUS"},"history":{"HISTORY"},
        "open_checks":{"OPEN","GATE"},"boundaries":{"BOUNDARY","SCOPE","RECORDED_AUTHORITY"},
        "recorded_next":{"NEXT_RECORDED"},"interfaces":{"INTERFACE","REFERENCE","RESEARCH_RULES"}
    }.items()}
    return r6.sealed({"schema":SCHEMA,"organ":route["id"],"repository":route["repository"],"recorded_role":matches[0]["observed_role"],"project":route["project"],"project_id":preflight["project_id"],"assignment":route["assignment"],"status":"BOOT_PACKET_READY__PINNED_HISTORY__LIVE_AUTHORITY_HOLD","atlas_receipt":source_atlas["receipt_sha256"],"preflight_receipt":preflight["receipt_sha256"],"contract_blob":BLOB,"context":groups,"sources":sources,"cortex_citations":citations,"knowledge_gaps":{"unmapped_structural_fields":unmapped,"fresh_primary_receipts_required":True,"live_capabilities_unverified":True,"neural_semantics_not_admitted":True,"history_previews_not_complete":True},"next_check":preflight["next_check"],"source_text_is_data":True,**FLAGS})
def consume(p,expected,organ,repo,project):
    require(type(p) is dict and p.get("schema")==SCHEMA,"PACKET_SCHEMA")
    checked(p)
    require(re.fullmatch("[0-9a-f]{64}",expected) and p["receipt_sha256"]==expected,"TRUSTED_PIN")
    require((p["organ"],p["repository"],p["project"])==(organ,repo,project),"RECIPIENT_SCOPE")
    require(p["status"]=="BOOT_PACKET_READY__PINNED_HISTORY__LIVE_AUTHORITY_HOLD" and all(p.get(k) is False for k in FLAGS),"AUTHORITY")
    require(p["source_text_is_data"] is True and p["knowledge_gaps"]["fresh_primary_receipts_required"] is True,"FRESHNESS")
    return {"organ":organ,"repository":repo,"project":project,"packet_sha256":expected,"context_groups":{k:len(v) for k,v in p["context"].items()},"sources":len(p["sources"]),"cortex_citations":len(p["cortex_citations"]),"next_check":p["next_check"],"registry_reads":0,"action_authority":False}
def prepare(out):
    c=atlas.read("pinned-r63",PATH,BLOB);require(c["schema"]=="janus.registry.organ_boot_contract.r63.v1","CONTRACT")
    require(c["atlas_evidence"]["neural_runtime_admission"] is False,"NEURAL_GATE")
    out=Path(out);out.mkdir(exist_ok=False,parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/"atlas.json";atlas.atlas(p);a=checked(r4.strict_object(p.read_bytes()))
    require(a["schema"]=="janus.registry.semantic_atlas.r62.v1" and a["action_authority"] is False and a["training_fields"]==c["atlas_evidence"]["curated"] and len(a["documents"])==c["atlas_evidence"]["documents"],"ATLAS_SCOPE")
    _,docs=atlas.train_data();profiles={d["source"]["blob"]:d for d in a["documents"]}
    require(sum(len(d["fields"]) for d in a["documents"])==c["atlas_evidence"]["fields"],"ATLAS_COUNT")
    roots={k:Path(v) for k,v in atlas.ROOTS.items()}
    pf={q:pre.build(atlas.contract(),roots,q) for q in sorted({r["project"] for r in c["routes"]})}
    pins={}
    for route in c["routes"]:
        x=boot_packet(c,route,profiles,docs,a,pf[route["project"]])
        raw=json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        require(len(raw)<=MAX_BYTES,"PACKET_BUDGET")
        (out/(route["id"]+".json")).write_bytes(raw)
        pins[route["id"]]=x["receipt_sha256"]
        print("R63_PACKET "+json.dumps({"organ":route["id"],"project":route["project"],"bytes":len(raw),"receipt":x["receipt_sha256"],"sources":len(x["sources"])}))
    atlas.write(out/"pins.json",pins)
    print("R63_SHARED_ATLAS "+json.dumps({"builds":1,"packets":len(pins),"raw_registry_at_consumers":False,"production":False}))
class Tests(unittest.TestCase):
    def setUp(self):
        self.x={"schema":SCHEMA,"organ":"a","repository":"owner/a","project":"JFP","status":"BOOT_PACKET_READY__PINNED_HISTORY__LIVE_AUTHORITY_HOLD","source_text_is_data":True,"knowledge_gaps":{"fresh_primary_receipts_required":True},"context":{},"sources":[],"cortex_citations":[],"next_check":"REVIEW",**FLAGS}
    def check(self,**kw):
        x=r6.sealed({**self.x,**kw});return consume(x,x["receipt_sha256"],"a","owner/a","JFP")
    def test_ok(self):self.assertEqual(self.check()["registry_reads"],0)
    def test_recipient(self):
        with self.assertRaises(r4.Hold):self.check(organ="b")
    def test_project(self):
        with self.assertRaises(r4.Hold):self.check(project="TRUMP")
    def test_authority(self):
        with self.assertRaises(r4.Hold):self.check(action_authority=True)
    def test_fresh(self):
        with self.assertRaises(r4.Hold):self.check(live_state_verified=True)
    def test_neural(self):
        with self.assertRaises(r4.Hold):self.check(actual_neural_inference=True)
    def test_tamper(self):
        x=r6.sealed(self.x);x["next_check"]="different"
        with self.assertRaises(r4.Hold):consume(x,x["receipt_sha256"],"a","owner/a","JFP")
    def test_reseal(self):
        x=r6.sealed(self.x);y=r6.sealed({**self.x,"next_check":"different"})
        with self.assertRaises(r4.Hold):consume(y,x["receipt_sha256"],"a","owner/a","JFP")
def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["test","prepare","consume"])
    for k in ("out","packet","pin","organ","repo","project"):p.add_argument("--"+k)
    a=p.parse_args()
    if a.mode=="test":return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    if a.mode=="prepare":prepare(a.out)
    else:
        raw=Path(a.packet).read_bytes();require(len(raw)<=MAX_BYTES,"PACKET_BUDGET")
        print("R63_CONSUMER "+json.dumps(consume(r4.strict_object(raw),a.pin,a.organ,a.repo,a.project),sort_keys=True))
    return 0
if __name__=="__main__":raise SystemExit(main())
