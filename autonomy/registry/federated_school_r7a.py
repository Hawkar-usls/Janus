"""R7A Federated School: bounded first-party teacher transport and admission.
No third-party runner, dispatch, model training, or production write.
JSON results are untrusted data; exact recorded source values are not scientific truth.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import unittest
import project_context_r4 as r4

WORKERS=("typed_r4","direct_json")
ROUTE=r4.ROUTES["COSMOS_SETI"]
FIELDS=(("goal",),("release_gate","status"),("next_allowed",))
QUESTIONS=("What goal is recorded in this snapshot?",
           "What release gate is recorded in this snapshot?",
           "What next step is recorded in this snapshot?")
SOURCE={"repository":r4.META,"commit":r4.META_COMMIT,"path":ROUTE["state"],"blob":ROUTE["state_blob"]}
SCHEMA="janus.federated_school.training_request.v1"

def require(test,reason):
    if not test:
        raise ValueError(reason)

def canonical(x):
    return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()
def seal(x):
    return {**x,"sha256":digest(x)}
def unseal(x):
    require(type(x) is dict and "sha256" in x,"MISSING_SEAL")
    y={k:v for k,v in x.items() if k!="sha256"}
    require(x["sha256"]==digest(y),"SEAL_MISMATCH")
    return y

def read(path):
    with Path(path).open("rb") as f:
        raw=f.read(65537)
    require(len(raw)<=65536,"PACKET_TOO_LARGE")
    return r4.strict_object(raw)

def write(path,x):
    with Path(path).open("x",encoding="utf-8") as f:
        f.write(canonical(x)+"\n")

def expected_tasks():
    return [{"id":f"R7A-{i+1:02d}","pointer":list(p),"question":q}
            for i,(p,q) in enumerate(zip(FIELDS,QUESTIONS))]

def request(boot):
    b={k:v for k,v in boot.items() if k!="receipt_sha256"}
    require(boot.get("receipt_sha256")==digest(b),"BOOT_SEAL_MISMATCH")
    require(b["schema"]=="janus.registry.boot_context.r6.v1" and b["project_id"]=="COSMOS_SETI","BOOT_SCOPE")
    require(b["status"]=="BOOT_CONTEXT_READY__PINNED_HISTORY__LIVE_STATE_HOLD","BOOT_STATUS")
    require(all(b.get(k) is False for k in ("action_authority","live_state_verified","production_writeback","automatic_dispatch")),"BOOT_AUTHORITY")
    require(any(s.get("repo")==SOURCE["repository"] and all(s.get(k)==SOURCE[k] for k in ("commit","path","blob")) for s in b["sources"]),"BOOT_SOURCE")
    return seal({"schema":SCHEMA,"project":"COSMOS_SETI","boot_sha256":boot["receipt_sha256"],
                 "source":SOURCE,"partition":"TRAIN_ONLY_DISCLOSED_SOURCE",
                 "tasks":expected_tasks(),"workers":list(WORKERS),"action_authority":False,
                 "max_tasks":3,"max_workers":2})

def validate_request(req):
    x=unseal(req)
    require(set(x)=={"schema","project","boot_sha256","source","partition","tasks","workers","action_authority","max_tasks","max_workers"},"REQUEST_FIELDS")
    require(x["schema"]==SCHEMA and x["project"]=="COSMOS_SETI" and x["source"]==SOURCE,"REQUEST_SOURCE")
    require(x["tasks"]==expected_tasks() and x["workers"]==list(WORKERS) and x["partition"]=="TRAIN_ONLY_DISCLOSED_SOURCE","REQUEST_TASKS")
    require(x["max_tasks"]==3 and x["max_workers"]==2 and x["action_authority"] is False,"REQUEST_BUDGET_OR_AUTHORITY")
    require(type(x["boot_sha256"]) is str and len(x["boot_sha256"])==64 and all(c in "0123456789abcdef" for c in x["boot_sha256"]),"BOOT_HASH")
    return x

def direct_values(meta):
    raw=r4.verified_bytes(meta,SOURCE["path"],SOURCE["blob"])
    doc=r4.strict_object(raw)
    return [doc["goal"],doc["release_gate"]["status"],doc["next_allowed"]]

def typed_values(meta,fund):
    d=r4.read_project(meta,fund,"COSMOS_SETI")["untrusted_recorded_data"]
    return [d["recorded_goal"],d["recorded_release_gate"],d["recorded_next_allowed"]]

def response(req,worker,values):
    validate_request(req)
    require(worker in WORKERS and len(values)==3 and all(type(v) is str and 0<len(v)<=2000 for v in values),"WORKER_SCOPE")
    return seal({"schema":"janus.federated_school.training_result.v1","request_sha256":req["sha256"],
                 "worker":worker,"rows":[{"id":t["id"],"answer":v} for t,v in zip(req["tasks"],values)],
                 "action_authority":False})

def admit(req,results,expected):
    validate_request(req)
    require(len(results)==2,"WORKER_COUNT")
    seen=set()
    for result in results:
        x=unseal(result)
        require(set(x)=={"schema","request_sha256","worker","rows","action_authority"},"RESULT_FIELDS")
        require(x["schema"]=="janus.federated_school.training_result.v1" and x["request_sha256"]==req["sha256"] and x["action_authority"] is False,"RESULT_BINDING")
        require(x["worker"] in WORKERS and x["worker"] not in seen,"DUPLICATE_OR_UNKNOWN_WORKER")
        seen.add(x["worker"])
        require(type(x["rows"]) is list and len(x["rows"])==3,"ROW_COUNT")
        for row,t,v in zip(x["rows"],req["tasks"],expected):
            require(type(row) is dict and set(row)=={"id","answer"} and row["id"]==t["id"] and type(row["answer"]) is str and row["answer"]==v,"ORACLE_MISMATCH")
    require(seen==set(WORKERS),"MISSING_WORKER")
    # One source question = one example, not one example per worker.
    examples=[{"id":t["id"],"question":t["question"],"answer":v,
               "source":SOURCE,"pointer":t["pointer"],"partition":"TRAIN_ONLY_DISCLOSED_SOURCE"}
              for t,v in zip(req["tasks"],expected)]
    return seal({"schema":"janus.federated_school.admission.v1","status":"SOURCE_EXACT_REPLAY_TRAIN_CANDIDATES_ONLY",
                 "request_sha256":req["sha256"],"workers_verified":len(seen),"worker_rows_verified":6,
                 "unique_examples":3,"examples":examples,"external_workers_executed":0,
                 "neural_inference":False,"student_training":False,"production_promotion":False,
                 "scientific_truth_verified":False,"external_blind":False,"action_authority":False})

def oracle(meta):
    # Re-read exact bytes independently of either worker; reference contract is fixed.
    doc=r4.strict_object(r4.verified_bytes(meta,SOURCE["path"],SOURCE["blob"]))
    out=[]
    for path in FIELDS:
        cur=doc
        for key in path:
            require(type(cur) is dict and key in cur,"ORACLE_POINTER")
            cur=cur[key]
        require(type(cur) is str and 0<len(cur)<=2000,"ORACLE_VALUE")
        out.append(cur)
    return out

class Tests(unittest.TestCase):
    def setUp(self):
        b={"schema":"janus.registry.boot_context.r6.v1","project_id":"COSMOS_SETI",
           "status":"BOOT_CONTEXT_READY__PINNED_HISTORY__LIVE_STATE_HOLD",
           "action_authority":False,"live_state_verified":False,"production_writeback":False,
           "automatic_dispatch":False,"sources":[{"repo":SOURCE["repository"],**SOURCE}]}
        self.boot={**b,"receipt_sha256":digest(b)}
        self.req=request(self.boot)
        self.values=["goal","HOLD","review"]
        self.a=response(self.req,WORKERS[0],self.values)
        self.b=response(self.req,WORKERS[1],self.values)
    def test_accept_and_dedupe(self):
        self.assertEqual(admit(self.req,[self.a,self.b],self.values)["unique_examples"],3)
    def test_duplicate(self):
        with self.assertRaises(ValueError): admit(self.req,[self.a,self.a],self.values)
    def test_wrong_answer_even_agreement(self):
        a=response(self.req,WORKERS[0],["goal","PASS","review"])
        b=response(self.req,WORKERS[1],["goal","PASS","review"])
        with self.assertRaises(ValueError): admit(self.req,[a,b],self.values)
    def test_wrong_request(self):
        b=seal({**unseal(self.b),"request_sha256":"x"})
        with self.assertRaises(ValueError): admit(self.req,[self.a,b],self.values)
    def test_permission(self):
        b=seal({**unseal(self.b),"action_authority":True})
        with self.assertRaises(ValueError): admit(self.req,[self.a,b],self.values)
    def test_tamper(self):
        with self.assertRaises(ValueError): admit(self.req,[self.a,{**self.b,"worker":"bad"}],self.values)
    def test_budget(self):
        with self.assertRaises(ValueError): validate_request(seal({**unseal(self.req),"max_workers":50}))
    def test_unknown_boot(self):
        b={**self.boot,"project_id":"UNKNOWN"}
        b["receipt_sha256"]=digest({k:v for k,v in b.items() if k!="receipt_sha256"})
        with self.assertRaises(ValueError): request(b)
    def test_boot_tamper(self):
        with self.assertRaises(ValueError): request({**self.boot,"automatic_dispatch":True})
    def test_exam_partition(self):
        with self.assertRaises(ValueError): validate_request(seal({**unseal(self.req),"partition":"EXAM"}))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["test","request","worker","admit"])
    p.add_argument("--boot",type=Path)
    p.add_argument("--request",type=Path)
    p.add_argument("--meta",type=Path)
    p.add_argument("--fund",type=Path)
    p.add_argument("--worker",choices=WORKERS)
    p.add_argument("--result",type=Path,action="append")
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.mode=="test":
        return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    if a.mode=="request":
        x=request(read(a.boot))
    elif a.mode=="worker":
        req=read(a.request)
        validate_request(req)
        vals=typed_values(a.meta,a.fund) if a.worker=="typed_r4" else direct_values(a.meta)
        x=response(req,a.worker,vals)
    else:
        req=read(a.request)
        x=admit(req,[read(p) for p in a.result],oracle(a.meta))
    write(a.out,x)
    print(canonical({"mode":a.mode,"sha256":x["sha256"],"status":x.get("status"),"unique_examples":x.get("unique_examples"),"worker_rows_verified":x.get("worker_rows_verified")}))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
