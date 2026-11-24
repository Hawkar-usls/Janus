"""R6 Registry Mind: deterministic read-only boot packet.
R4 pinned context + Cortex retrieval + source-bound advisory proposals.
Never dispatches tasks, grants permissions, executes JSON text or promotes models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
import project_context_r4 as r4

MAP_COMMIT="9306037335bb4532049633b7f5f082e75891ff51"
MAP_PATH="data/autonomy_map/OPERATING_CONTRACT.json"
MAP_BLOB="e00aebd4a8a1a0d9aef57ecc882be234bee1be8c"
EVIDENCE_COMMIT="48fb78caa34c3fc5d9aa0486fcf608ddfa53fc32"
EVIDENCE_PATH="data/autonomy_map/JANUS-R5-R51-REGISTRY-INTELLIGENCE-EVIDENCE-2026-11-24.json"
EVIDENCE_BLOB="302f1b07d1b810451a05b79fa8c546eb1aeaaff9"
CORTEX_COMMIT="00910a3e1ea2d50650c20308ab66e3c2f7be46ab"
CORTEX_BLOB="626b74379e195d2e72d7c5b727198f6a3e297f04"
ROLE_SET={"GOAL","OPEN","NEXT","HISTORY","EVIDENCE","BOUNDARY","STATUS","AUTHORITY","UNKNOWN"}

def sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,
             separators=(",",":"),allow_nan=False).encode()).hexdigest()

def sealed(obj):
    return {**obj,"receipt_sha256":sha(obj)}

def read_doc(root,path,blob):
    return r4.strict_object(r4.verified_bytes(root,path,blob))

def resolve(index, query):
    if type(query) is not str or not 1<=len(query)<=80:
        return None
    matches=set()
    for p in index["projects"]:
        if query.strip().casefold() in {p["id"].casefold(),*(a.strip().casefold() for a in p["aliases"])}:
            matches.add(p["id"])
    if len(matches)!=1:
        return None
    project=matches.pop()
    return project if project in r4.ROUTES else None

def advisory(proposal, state, blob, allowed_paths):
    # No freeform text, confidence, instructions, authority fields or model code.
    # Unknown paths must be explicitly offered by caller. Never alters known fields.
    if type(proposal) is not dict or set(proposal)!={"pointer","source_blob","value_sha256","role"}:
        return {"status":"REJECTED_FORMAT","action_authority":False}
    path=proposal["pointer"]
    if type(path) is not list or not 1<=len(path)<=8 or not all(type(k) is str for k in path):
        return {"status":"REJECTED_POINTER","action_authority":False}
    if tuple(path) not in allowed_paths or proposal["source_blob"]!=blob or not isinstance(proposal["role"],str) or proposal["role"] not in ROLE_SET:
        return {"status":"REJECTED_SCOPE","action_authority":False}
    node=state
    for k in path:
        if type(node) is not dict or k not in node:
            return {"status":"REJECTED_POINTER","action_authority":False}
        node=node[k]
    if sha(node)!=proposal["value_sha256"]:
        return {"status":"REJECTED_HASH","action_authority":False}
    return {"status":"UNVERIFIED_ROLE_PROPOSAL_ONLY","pointer":path,"role":proposal["role"],
            "source_blob":blob,"value_sha256":sha(node),"action_authority":False}

def history(project,state,context):
    if project=="TRUMP":
        return {"historical_failure":context["recorded_historical_failure"],
                "must_not_repeat_as_new":context["must_not_repeat_as_new"],
                "coverage":"PINNED_STATE_FIELDS_ONLY"}
    rows=[]
    for stage,row in state["preceding"].items():
        if type(row) is dict:
            status=row.get("status",row.get("gate"))
            if isinstance(status,str) and len(status)<=200 and ("HOLD" in status or "FAIL" in status):
                rows.append({"stage":stage,"recorded_status":status})
    return {"recorded_negative_or_open_predecessors":rows,
            "coverage":"PINNED_PRECEDING_MAP_ONLY_NOT_FULL_JOURNAL_SEMANTICS"}

def cortex(meta,fund,demi):
    r4.verified_bytes(demi,"janus_model/memory_cortex_growth.py",CORTEX_BLOB)
    from janus_model import memory_cortex_growth as g
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        s=g.new_state()
        for p,route in r4.ROUTES.items():
            src=meta if route["repo"]==r4.META else fund
            for kind in ("state","journal"):
                if kind=="journal" and p=="TRUMP":
                    continue
                raw=r4.verified_bytes(src,route[kind],route[kind+"_blob"])
                g.add_text(root,s,raw,{"repository":route["repo"],"commit":route["commit"],"path":route[kind]})
        g.publish(root,s,"R6_DISPOSABLE_BOOT_REPLAY")
        g.write_json(root/"STATE.json",g.sealed(s))
        results={}
        for project,query in (("COSMOS_SETI","open_gates next_allowed"),("TRUMP","current_frontier preregistration_status")):
            x=g.query(root,query,limit=3)
            r4.require(x["status"]=="VERIFIED_SOURCE_POINTERS_ONLY" and x["citations"],"CORTEX_NO_VERIFIED_POINTER")
            results[project]={"status":x["status"],"citations":x["citations"],"indexed_documents":x["indexed_documents"]}
        return results

def boot(meta,fund,map_root,evidence_root,demi,query,proposal=None,memory=None):
    index=read_doc(meta,r4.INDEX,r4.INDEX_BLOB)
    project=resolve(index,query)
    base={"schema":"janus.registry.boot_context.r6.v1","requested_project":query,
          "project_id":project,"action_authority":False,"production_writeback":False,
          "live_state_verified":False,"automatic_dispatch":False,
          "source_text_is_data":True,"model_promotion":False}
    if project is None:
        return sealed({**base,"status":"HOLD_UNKNOWN_OR_AMBIGUOUS_PROJECT",
                       "goal":None,"current_state":None,"history":None,
                       "capabilities":[],"open_tasks":["REVIEW_EXACT_PROJECT_ROUTE"],
                       "constraints":["NO_GUESSED_AUTHORITY"],"neural_advisory":{"status":"NOT_RUN"},
                       "knowledge":{"status":"NOT_QUERIED"},"sources":[]})
    ctx=r4.read_project(meta,fund,project)
    route=ctx["source"]; d=ctx["untrusted_recorded_data"]
    state=read_doc(meta if route["repo"]==r4.META else fund,route["state"],route["state_blob"])
    contract=read_doc(map_root,MAP_PATH,MAP_BLOB)
    r4.require(contract["schema"]=="janus.autonomy.github_operating_contract.v1" and contract["authority_delta"]==0 and contract["auto_promotion"] is False,"MAP_AUTHORITY_DRIFT")
    evidence=read_doc(evidence_root,EVIDENCE_PATH,EVIDENCE_BLOB)
    r4.require(evidence["r51"]["correct"]=={"key_only":9,"contextual":9,"rules":22} and evidence["r51"]["production_promotion"] is False,"EVIDENCE_DRIFT")
    capabilities=[{"repository":x["repository"],"recorded_role":x["observed_role"],
                   "live_availability_verified":False} for x in contract["controllers"]]
    if project=="COSMOS_SETI":
        goal=d["recorded_goal"]
        current={"stage_candidates":d["stage_candidates"],"candidates_differ":d["stage_candidates_differ"],
                 "release_gate":d["recorded_release_gate"],"selected_live_stage":None}
        tasks=d["recorded_open_gates"]
        constraints=[d["claim_ceiling"],"NO_AUTOMATIC_STAGE_SELECTION"]
        next_step=d["recorded_next_allowed"]
        # Explicit unknown candidate only; do not allow replacement of known roles.
        allowed={("as_of",)}
    else:
        goal=d["recorded_frontier"]["correct_question"]
        current={"recorded_frontier":d["recorded_frontier"],"recorded_scoped_authority":d["recorded_scoped_authority"],"selected_live_stage":None}
        tasks=[d["recorded_frontier"]["gate"],d["recorded_frontier"]["preregistration_status"]]
        constraints=list(d["recorded_claim_limits"].values())
        next_step="VERIFY_LATEST_SCIENTIFIC_SEAL_AND_PREREGISTRATION"
        allowed={("status",)}
    if memory is None:
        memory=cortex(meta,fund,demi)
    note=advisory(proposal,state,route["state_blob"],allowed) if proposal is not None else {"status":"MODEL_NOT_CONNECTED","action_authority":False}
    return sealed({**base,"status":"BOOT_CONTEXT_READY__PINNED_HISTORY__LIVE_STATE_HOLD",
        "goal":goal,"current_state":current,"history":history(project,state,d),
        "capabilities":capabilities,"capability_scope":"HISTORICAL_DRAFT_MAP_NOT_LIVE",
        "open_tasks":tasks,"recorded_next_step":next_step,"constraints":constraints,
        "knowledge":{**memory[project],"scope":"DISPOSABLE_SOURCE_INDEX_NOT_DURABLE_PRODUCTION"},
        "neural_advisory":note,"actual_neural_inference_executed":False,
        "learning_history":{"R5":evidence["r5"]["correct"],"R5_1":evidence["r51"]["correct"],
                            "disclosed_exams_not_new_holdout":True,"learned_gain_established":False},
        "required_review":ctx["required_review"],
        "sources":[{"repo":r4.META,"commit":r4.META_COMMIT,"path":r4.INDEX,"blob":r4.INDEX_BLOB},
                   {"repo":route["repo"],"commit":route["commit"],"path":route["state"],"blob":route["state_blob"]},
                   {"repo":route["repo"],"commit":route["commit"],"path":route["journal"],"blob":route["journal_blob"]},
                   {"repo":r4.META,"commit":MAP_COMMIT,"path":MAP_PATH,"blob":MAP_BLOB,"draft":True},
                   {"repo":r4.META,"commit":EVIDENCE_COMMIT,"path":EVIDENCE_PATH,"blob":EVIDENCE_BLOB,"draft":True},
                   {"repo":"Hawkar-usls/Janus-Demiurge","commit":CORTEX_COMMIT,"path":"janus_model/memory_cortex_growth.py","blob":CORTEX_BLOB}]})

class Controls(unittest.TestCase):
    def setUp(self):
        self.state={"candidate":"data","goal":"known"}
        self.p={"pointer":["candidate"],"source_blob":"x","value_sha256":sha("data"),"role":"GOAL"}
    def check(self,p):
        return advisory(p,self.state,"x",{("candidate",)})
    def test_valid_advisory_is_unverified(self):
        self.assertEqual(self.check(self.p)["status"],"UNVERIFIED_ROLE_PROPOSAL_ONLY")
        self.assertFalse(self.check(self.p)["action_authority"])
    def test_forged_permission(self):
        self.assertEqual(self.check({**self.p,"action_authority":True})["status"],"REJECTED_FORMAT")
    def test_known_override(self):
        self.assertEqual(self.check({**self.p,"pointer":["goal"]})["status"],"REJECTED_SCOPE")
    def test_wrong_blob(self):
        self.assertEqual(self.check({**self.p,"source_blob":"bad"})["status"],"REJECTED_SCOPE")
    def test_wrong_value(self):
        self.assertEqual(self.check({**self.p,"value_sha256":"bad"})["status"],"REJECTED_HASH")
    def test_unknown_role(self):
        self.assertEqual(self.check({**self.p,"role":"EXECUTE"})["status"],"REJECTED_SCOPE")
    def test_ambiguous_alias(self):
        idx={"projects":[{"id":"TRUMP","aliases":["same"]},{"id":"COSMOS_SETI","aliases":["same"]}]}
        self.assertIsNone(resolve(idx,"same"))
    def test_digest_tamper(self):
        x=sealed({"action_authority":False})
        self.assertNotEqual(x["receipt_sha256"],sha({"action_authority":True}))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("meta","fund","map-root","evidence-root","demi","out"):
        p.add_argument("--"+k,type=Path)
    p.add_argument("--project")
    a=p.parse_args()
    if a.self_test:
        return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls)).wasSuccessful() else 1
    if not all((a.meta,a.fund,a.map_root,a.evidence_root,a.demi,a.out,a.project)):
        p.error("all roots, project and output required")
    x=boot(a.meta,a.fund,a.map_root,a.evidence_root,a.demi,a.project)
    with a.out.open("x",encoding="utf-8") as f:
        json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write("\n")
    print(json.dumps({"project":x["project_id"],"status":x["status"],"receipt_sha256":x["receipt_sha256"]}))
    return 0
if __name__=="__main__":
    raise SystemExit(main())
