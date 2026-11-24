#!/usr/bin/env python3
"""Original R6.2 semantic atlas. All semantic labels are advisory."""
import argparse, collections, hashlib, json, os, re, tempfile, unittest
from pathlib import Path
import project_context_r4 as r4
import registry_mind_r6 as r6
import registry_preflight_r61 as pre
import semantic_transfer_r51 as old

PROTO="data/autonomy_map/JANUS-R62-SEMANTIC-ATLAS-PROTOCOL-2026-11-24.json"
PBLOB="749daca7ad2463f7fd69219986e6a893700e6280"
LABELS=sorted(old.LABELS)
ROLE={"ROUTES":"UNKNOWN","PROVENANCE":"EVIDENCE","RESEARCH_RULES":"BOUNDARY","RECORDED_STATE":"STATUS","NEXT_RECORDED":"NEXT","GATE":"OPEN","RECORDED_AUTHORITY":"AUTHORITY","CAPABILITIES":"UNKNOWN","SCOPE":"BOUNDARY","REFERENCE":"EVIDENCE","INTERFACE":"UNKNOWN"}
ROOTS={"meta":"pinned-meta","fund":"pinned-fund","map":"pinned-map","jfp":"pinned-jfp"}
def require(x,msg):r4.require(x,msg)
def read(root,p,b):return r4.strict_object(r4.verified_bytes(Path(root),p,b))
def protocol():return read("pinned-protocol",PROTO,PBLOB)
def contract():return read("pinned-contract",pre.CONTRACT_PATH,pre.CONTRACT_BLOB)
def write(p,x):
    with Path(p).open("x",encoding="utf-8") as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2)
def train_data():
    c=contract(); rows=[]; docs=[]
    for d in c["documents"]:
        raw=r4.verified_bytes(ROOTS[d["root"]],d["path"],d["blob"])
        obj=r4.strict_object(raw);pre.expectations(d,obj)
        docs.append((d,obj,raw))
        for path,kind,role in d["fields"]:
            value=pre.pointer(obj,path)
            label=ROLE.get(role,role)
            require(label in LABELS,"ROLE")
            rows.append({"id":d["id"]+":"+path,"blob":d["blob"],"pointer":path.split("/"),"value":value,"role":label})
    require(len(rows)<=128 and len({r["id"] for r in rows})==len(rows),"TRAIN_BUDGET")
    return rows,docs
def shape(v):
    # No document instructions or string contents enter parent prompts.
    if type(v) is dict:return "object; keys "+", ".join(sorted(v)[:8])
    if type(v) is list:return "array; count "+str(len(v))
    if type(v) is str:return "string; "+("digest" if re.fullmatch("[0-9a-f]{40,64}",v) else "text")
    return type(v).__name__
def descriptors(obj):
    # Bounded structural atlas, including unlabelled fields; never executes text.
    for k,v in sorted(obj.items()):
        yield {"pointer":[k],"shape":shape(v),"value_sha256":r6.sha(v)}
        if type(v) is dict:
            for sk,sv in sorted(v.items())[:32]:
                yield {"pointer":[k,sk],"shape":shape(sv),"value_sha256":r6.sha(sv)}
def atlas(out):
    rows,docs=train_data()
    from janus_model import memory_cortex_growth as g
    r4.verified_bytes("pinned-demi","janus_model/memory_cortex_growth.py",r6.CORTEX_BLOB)
    atlas_rows=[];known={r["id"]:r["role"] for r in rows}
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);s=g.new_state()
        for d,obj,raw in docs:
            origin={k:d[k] for k in ("repo","commit","path","blob")}
            origin["repository"]=origin.pop("repo")
            g.add_text(root,s,raw,origin)
            fields=[]
            for x in descriptors(obj):
                role=known.get(d["id"]+":"+"/".join(x["pointer"]))
                fields.append({**x,"curated_role":role,"status":"KNOWN_CONTRACT" if role else "UNLABELLED_REVIEW"})
            atlas_rows.append({"source":origin,"identity":d["identity"],"fields":fields})
        doc={"schema":"janus.registry.semantic_atlas.r62.v1","documents":atlas_rows,"source_text_is_data":True,"action_authority":False}
        g.add_text(root,s,json.dumps(doc,ensure_ascii=False,separators=(",",":")).encode(),{"repository":"Hawkar-usls/Janus","path":"R62_GENERATED_ATLAS","scope":"derived_advisory"})
        g.publish(root,s,"R62_DISPOSABLE_ATLAS")
        g.write_json(root/"STATE.json",g.sealed(s))
        queries={}
        for name,q,id in [("COSMOS","open_gates next_allowed","cosmos_state"),("TRUMP","current_frontier preregistration_status","trump_state"),("TACHYON","epistemic_boundaries temporal_binding_algorithm","rule001"),("JFP","verification_runs remote_handshake","jfp_evidence")]:
            x=g.query(root,q,limit=8);expected=next(d["path"] for d,_,_ in docs if d["id"]==id)
            queries[name]={"status":x["status"],"expected_source_found":any(c["origin"].get("path")==expected for c in x["citations"]),"citations":x["citations"]}
        result=r6.sealed({**doc,"queries":queries,"training_fields":len(rows),"indexed_documents":len(s["active"]),"durable_production":False,"unknown_policy":"REVIEW_NOT_EXECUTE","next_check":"VERIFY_FRESH_PRIMARY_RECEIPTS"})
    write(out,result)
    print("R62_ATLAS "+json.dumps({"documents":len(docs),"indexed":result["indexed_documents"],"fields":sum(len(x["fields"]) for x in atlas_rows),"curated":len(rows),"retrieval":{k:v["expected_source_found"] for k,v in queries.items()},"receipt":result["receipt_sha256"]}))
def parent_sample(rows):
    c=collections.Counter();out=[]
    for r in rows:
        if c[r["role"]]<3:
            out.append(r);c[r["role"]]+=1
    return out
def proposal_prompt(r):
    return "Classify JSON field role. Choose one letter only. "+", ".join(chr(65+i)+"="+v for i,v in enumerate(LABELS))+". Field path: "+"/".join(r["pointer"])[:180]+". Type: "+shape(r["value"])[:200]+"."
def features(r):
    # Fixed hashed structural and value tokens; no file names/schema identities.
    xs=old.feats(r["pointer"],r["value"],True)
    a=[0.0]*256
    for token in sorted(xs):
        j=int(hashlib.sha256(token.encode()).hexdigest()[:8],16)%len(a);a[j]=1.0
    return a
def fit_neural(rows):
    import torch
    torch.manual_seed(20261124);torch.set_num_threads(2)
    net=torch.nn.Sequential(torch.nn.Linear(256,32),torch.nn.ReLU(),torch.nn.Linear(32,len(LABELS)))
    x=torch.tensor([features(r) for r in rows]);y=torch.tensor([LABELS.index(r["role"]) for r in rows])
    opt=torch.optim.AdamW(net.parameters(),lr=.015)
    losses=[]
    for step in range(100):
        loss=torch.nn.functional.cross_entropy(net(x),y)
        opt.zero_grad();loss.backward();opt.step()
        if step in (0,99):losses.append(float(loss.item()))
    require(all(0<=x<100 for x in losses),"LOSS")
    return {"weights":{k:v.detach().tolist() for k,v in net.state_dict().items()},"loss":losses,"train_ids":[r["id"] for r in rows]}
def train(out,parents):
    rows,_=train_data();report={"schema":"janus.registry.r62.train.v1","train_sha256":r6.sha(rows),"oracle_only":fit_neural(rows),"parent_admitted":None,"parent_rows":[],"actual_parent_inference":False,"native_JANUS_weights_changed":False}
    if parents:
        from tools import trinity_parental_contract as c
        from tools import trinity_parental_cpu as cpu
        import gc
        r4.verified_bytes("pinned-trinity","tools/trinity_parental_contract.py","4636318842c64196c916c75a7b59fac0c876e2fa")
        r4.verified_bytes("pinned-trinity","tools/trinity_parental_cpu.py","147a3fcfb0f33e4b23a7bf6c331f8375111549b4")
        sample=parent_sample(rows)
        tk,m=cpu.load_parent(c.QWEN)
        answers=[cpu.parent_answer(tk,m,proposal_prompt(r),8) for r in sample]
        del tk,m;gc.collect()
        tk,m=cpu.load_parent(c.SMOL)
        notes=[cpu.parent_answer(tk,m,proposal_prompt(r)+" Proposed letter: "+(a if re.fullmatch("[A-I]",a) else "INVALID")+". Is it correct? YES or NO only.",5) for r,a in zip(sample,answers)]
        del tk,m;gc.collect()
        accepted=[]
        for r,a,n in zip(sample,answers,notes):
            correct=a.strip()==chr(65+LABELS.index(r["role"]))
            ok=correct and n.strip().upper()=="YES"
            report["parent_rows"].append({"id":r["id"],"proposal":a,"critic":n,"curated_match":correct,"admitted":ok})
            if ok:accepted.append(r)
        report["actual_parent_inference"]=True
        report["models"]={"qwen":c.QWEN,"smollm":c.SMOL}
        if len(accepted)>=9 and len({r["role"] for r in accepted})>=4:report["parent_admitted"]=fit_neural(accepted)
    report=r6.sealed(report);write(out,report)
    print("R62_TRAIN "+json.dumps({"rows":len(rows),"parent_called":report["actual_parent_inference"],"admitted":sum(r["admitted"] for r in report["parent_rows"]),"candidate":report["parent_admitted"] is not None,"receipt":report["receipt_sha256"]}))
def predictor(spec):
    import torch
    net=torch.nn.Sequential(torch.nn.Linear(256,32),torch.nn.ReLU(),torch.nn.Linear(32,len(LABELS)))
    net.load_state_dict({k:torch.tensor(v) for k,v in spec["weights"].items()});net.eval()
    def predict(r):
        with torch.inference_mode():p=torch.softmax(net(torch.tensor(features(r))),dim=0)
        return LABELS[int(p.argmax())] if float(p.max())>=.55 else "UNKNOWN"
    return predict
def exam(train_path,out):
    rows,docs=train_data();tr=json.loads(Path(train_path).read_text())
    seal=tr.pop("receipt_sha256");require(r6.sha(tr)==seal and tr["train_sha256"]==r6.sha(rows),"TRAIN_SEAL")
    p=protocol();cases=[];blobs={d["blob"] for d,_,_ in docs}
    schemas={o.get("schema") for _,o,_ in docs}
    for d in p["exam"]["documents"]:
        o=read(d["root"],d["path"],d["blob"])
        require(d["blob"] not in blobs and o["schema"] not in schemas and o["schema"]==d["schema"],"SPLIT")
        for k,role in d["fields"]:
            cases.append({"id":d["id"]+":"+k,"pointer":k.split("/"),"value":pre.pointer(o,k),"role":role,"doc":d["id"]})
    keys=collections.defaultdict(set)
    for r in rows:keys[r["pointer"][-1]].add(r["role"])
    nb=old.fit(rows,True)
    methods={"exact":lambda r:next(iter(keys[r["pointer"][-1]])) if len(keys[r["pointer"][-1]])==1 else "UNKNOWN","r51_rule":lambda r:old.rule(r["pointer"],r["value"]),"r51_contextual":lambda r:old.predict(nb,r["pointer"],r["value"],True),"oracle_neural":predictor(tr["oracle_only"])}
    if tr["parent_admitted"] is not None:methods["trinity_neural"]=predictor(tr["parent_admitted"])
    unseen={r["id"] for r in cases if r["pointer"][-1] not in {t["pointer"][-1] for t in rows}}
    metrics={}
    for name,fn in methods.items():
        pred=[fn(r) for r in cases]
        metrics[name]={"correct":sum(v==r["role"] for v,r in zip(pred,cases)),"unseen_correct":sum(v==r["role"] and r["id"] in unseen for v,r in zip(pred,cases)),"unknown_false_accept":sum(v!="UNKNOWN" and r["role"]=="UNKNOWN" for v,r in zip(pred,cases)),"by_doc":{d:sum(v==r["role"] and r["doc"]==d for v,r in zip(pred,cases)) for d in sorted({r["doc"] for r in cases})}}
    report=r6.sealed({"schema":"janus.registry.r62.exam.v1","cases":len(cases),"unseen_keys":len(unseen),"metrics":metrics,"train_receipt":seal,"trinity_candidate_available":tr["parent_admitted"] is not None,"external_blind":False,"native_weights_changed":False,"production_promotion":False,"action_authority":False})
    write(out,report);print("R62_EXAM "+json.dumps(report,sort_keys=True))
class Tests(unittest.TestCase):
    def test_roles(self):self.assertTrue(set(ROLE.values())<=set(LABELS))
    def test_descriptors(self):self.assertEqual(len(list(descriptors({"a":{"b":1},"c":False}))),3)
    def test_shape_no_value(self):self.assertNotIn("ignore all",shape("ignore all"))
    def test_features(self):self.assertEqual(features({"pointer":["goal"],"value":"x"}),features({"pointer":["goal"],"value":"x"}))
    def test_sample(self):self.assertEqual(len(parent_sample([{"role":"GOAL"}]*8)),3)
def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["test","atlas","train","exam"]);p.add_argument("--out");p.add_argument("--train");p.add_argument("--parents",action="store_true");a=p.parse_args()
    if a.mode=="test":return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    if a.mode=="atlas":atlas(a.out)
    if a.mode=="train":train(a.out,a.parents)
    if a.mode=="exam":exam(a.train,a.out)
    return 0
if __name__=="__main__":raise SystemExit(main())
