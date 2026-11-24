"""R70C bounded offline classifier distillation, separate train/exam jobs."""
import hashlib, json, sys, zipfile, statistics, time
from collections import Counter
from pathlib import Path
import torch
import assimilation_r70b as b
def require(x,msg):
    if not x: raise ValueError(msg)
def load_json(path):
    return json.loads(Path(path).read_bytes())
def read_sources(spec):
    docs={}
    for s in spec["sources"]:
        raw=(Path(s["id"])/s["path"]).read_bytes()
        require(len(raw)<=1000000,"source too large")
        require(hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()==s["blob"],"source blob")
        d=json.loads(raw);require(d["schema"]==s["schema"],"schema")
        docs[s["id"]]=d
    return docs
def net():
    return torch.nn.Sequential(torch.nn.Linear(256,32),torch.nn.ReLU(),torch.nn.Linear(32,len(b.LABELS)))
def parent(spec,archive):
    b.load_model(spec,archive) # verifies complete archive, TRAIN bytes and sealed receipt
    with zipfile.ZipFile(archive) as z: t=json.loads(z.read("TRAIN.json"))
    m=net()
    m.load_state_dict({k:torch.tensor(v) for k,v in t["oracle_only"]["weights"].items()})
    return m
def leaves(d,path=()):
    if type(d) is dict and len(path)<8:
        for k,v in sorted(d.items()): yield from leaves(v,path+(k,))
    elif type(d) in (str,int,bool) and path and path[-1] not in {"schema","id","artifact_id"}:
        yield list(path),d
def weights(m): return {k:v.detach().tolist() for k,v in m.state_dict().items()}
def fit(m,x,y,steps,lr):
    c=Counter(y.tolist()); w=torch.tensor([1/max(1,c[i]) for i in range(len(b.LABELS))],dtype=torch.float32)
    opt=torch.optim.AdamW(m.parameters(),lr=lr)
    m.train();losses=[]
    for i in range(steps):
        loss=torch.nn.functional.cross_entropy(m(x),y,weight=w)
        opt.zero_grad();loss.backward();opt.step()
        if i in (0,steps-1): losses.append(float(loss.item()))
    m.eval();return losses
def train(spec,archive,out):
    torch.manual_seed(spec["seed"]);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    docs=read_sources(spec)
    rows=[]; seen=set()
    for id,d in docs.items():
        for path,v in leaves(d):
            key=b.digest(b.canonical([path,v]))
            if key in seen: continue
            seen.add(key)
            rows.append((id,path,v,b.strengthened_rule(path,v)))
    require(20<=len(rows)<=spec["max_rows"],"train size")
    require(len({r[3] for r in rows})>=4,"class support")
    x=torch.tensor([b.features(p,v) for _,p,v,_ in rows])
    y=torch.tensor([b.LABELS.index(label) for *_,label in rows])
    p=parent(spec["model"],archive); p0=b.digest(b.canonical(weights(p)))
    a=fit(p,x,y,spec["steps"],spec["learning_rate"])
    torch.manual_seed(spec["seed"]);s=net()
    z=fit(s,x,y,spec["steps"],spec["learning_rate"])
    body={"schema":"janus.r70c.candidate.v1","parent_receipt":spec["model"]["receipt"],"parent_weights_sha256":p0,"train_spec_sha256":b.digest(b.canonical(spec)),"train_sources":spec["sources"],"train_rows":len(rows),"teacher_label_counts":dict(Counter(r[3] for r in rows)),"train_rows_sha256":b.digest(b.canonical(rows)),"inherited":weights(p),"scratch":weights(s),"inherited_loss":a,"scratch_loss":z,"production_changed":False,"action_authority":False}
    require(b.digest(b.canonical(body["inherited"]))!=p0,"weights did not change")
    seal=b.digest(b.canonical(body));body["receipt_sha256"]=seal
    Path(out).write_bytes(b.canonical(body))
    print("R70C_TRAIN "+json.dumps({k:v for k,v in body.items() if k not in {"inherited","scratch"}},sort_keys=True))
def predictor(m,threshold):
    m.eval()
    def f(path,v):
        with torch.inference_mode(): p=torch.softmax(m(torch.tensor(b.features(path,v))),dim=0)
        return b.LABELS[int(p.argmax())] if float(p.max())>=threshold else "UNKNOWN"
    return f
def exam(tr,ex,archive,candidate,expected_seal):
    torch.set_num_threads(1)
    c=load_json(candidate);seal=c.pop("receipt_sha256")
    require(seal==expected_seal==b.digest(b.canonical(c)),"candidate seal")
    require(c["train_spec_sha256"]==b.digest(b.canonical(tr)) and c["train_sources"]==tr["sources"],"train spec")
    require(c["parent_receipt"]==tr["model"]["receipt"],"parent receipt")
    for field in ("path","blob","schema"):
        require(not {s[field] for s in tr["sources"]}&{s[field] for s in ex["sources"]},"split overlap")
    p=parent(tr["model"],archive)
    require(b.digest(b.canonical(weights(p)))==c["parent_weights_sha256"],"parent weights")
    models={"parent":p}
    for k in ("inherited","scratch"):
        m=net();m.load_state_dict({k2:torch.tensor(v) for k2,v in c[k].items()});models[k]=m
    methods={k:predictor(m,tr["threshold"]) for k,m in models.items()}
    methods.update({"teacher":b.strengthened_rule,"r51_rule":b.rule})
    docs=read_sources(ex);rows=[];outputs={k:[] for k in methods};outputs["exact_ceiling"]=[]
    times={k:[] for k in methods}
    for doc,role,paths,target in ex["cases"]:
        require(len(set(paths))==len(paths) and (target is None or target in paths),"case")
        vals={p:b.get(docs[doc],p) for p in paths};r={"doc":doc,"role":role,"target":target,"pred":{}}
        for k,fn in methods.items():
            start=time.perf_counter_ns()
            matched=[p for p in paths if role!="UNKNOWN" and fn(p.split("/"),vals[p])==role]
            times[k].append(time.perf_counter_ns()-start)
            pred=matched[0] if len(matched)==1 else None
            r["pred"][k]=pred;outputs[k].append(vals[pred] if pred is not None else b.HOLD)
        r["pred"]["exact_ceiling"]=target;outputs["exact_ceiling"].append(vals[target] if target is not None else b.HOLD);rows.append(r)
    metrics={}
    for k,values in outputs.items():
        comp=[b.typed(values[a],values[j],op) for a,j,op,_ in ex["composition"]]
        exp=[v for *_,v in ex["composition"]]
        metrics[k]={"route_correct":sum(r["pred"][k]==r["target"] for r in rows),"composition_correct":sum(type(a) is type(v) and a==v for a,v in zip(comp,exp)),"unsupported_route":sum(r["target"] is None and r["pred"][k] is not None for r in rows),"wrong_nonabstaining":sum(r["pred"][k] is not None and r["pred"][k]!=r["target"] for r in rows),"by_doc":{doc:sum(r["doc"]==doc and r["pred"][k]==r["target"] for r in rows) for doc in docs},"composition_predictions":comp}
        if k in times: metrics[k]["median_route_ns"]=int(statistics.median(times[k]))
    require(metrics["exact_ceiling"]["route_correct"]==len(rows) and metrics["exact_ceiling"]["composition_correct"]==len(ex["composition"]),"gold composition mismatch")
    paired={}
    for k in ("parent","scratch","teacher","r51_rule"):
        paired[k]={"inherited_only":sum(r["pred"]["inherited"]==r["target"] and r["pred"][k]!=r["target"] for r in rows),"other_only":sum(r["pred"]["inherited"]!=r["target"] and r["pred"][k]==r["target"] for r in rows)}
    report={"schema":"janus.r70c.internal_exam.v1","candidate_receipt":seal,"train_rows":c["train_rows"],"route_cases":len(rows),"composition_cases":len(ex["composition"]),"metrics":metrics,"paired":paired,"rows":rows,"external_blind":False,"production_changed":False,"action_authority":False,"native_model_tested":False,"status":"INTERNAL_DISTILLATION_FIRSTLOOK__EXTERNAL_VALIDATION_HOLD"}
    print("R70C_EXAM "+json.dumps(report,sort_keys=True))
if __name__=="__main__":
    tr=load_json("research/assimilation/R70C_TRAIN.json")
    if sys.argv[1]=="train": train(tr,Path(sys.argv[2]),sys.argv[3])
    elif sys.argv[1]=="exam": exam(tr,load_json("research/assimilation/R70C_EXAM.json"),Path(sys.argv[2]),sys.argv[3],sys.argv[4])
    else: raise ValueError("unknown mode")
