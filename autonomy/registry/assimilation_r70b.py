"""Frozen R62 role-routing contribution to source-grounded scalar composition."""
import hashlib, json, math, re, time, zipfile, sys, statistics
from pathlib import Path
HOLD="HOLD"
LABELS=sorted({"GOAL","OPEN","NEXT","HISTORY","EVIDENCE","BOUNDARY","STATUS","AUTHORITY","UNKNOWN"})
def digest(b): return hashlib.sha256(b).hexdigest()
def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def words(x): return re.findall(r"[a-z][a-z0-9]*",x.lower())[:160]
def features(path,value):
    fs={"key:"+x for x in words(path[-1])}
    fs|={"parent:"+x for x in words(" ".join(path[:-1]))}
    fs.add("type:"+type(value).__name__)
    if isinstance(value,str):
        fs|={"value:"+x for x in words(value[:900])}
        if re.fullmatch("[0-9a-f]{40,64}",value): fs.add("shape:digest")
        if value.startswith("https://"): fs.add("shape:url")
    a=[0.0]*256
    for token in sorted(fs): a[int(digest(token.encode())[:8],16)%256]=1.
    return a
# R51 rule exactly for scalar input (source blob 55c05b4ca05984cf2c5a3cd91134801d9c44a69b).
def rule(path,value):
    t=set(words(path[-1]))
    if t & {"execute","permission","override"}: return "UNKNOWN"
    if t & {"boundary","boundaries","limits","ceiling","policy","policies","control","rule","firewalls"}: return "BOUNDARY"
    if t & {"next","recommendation"}: return "NEXT"
    if t & {"gate","gates","unresolved","pending","custodian","obligations"}: return "OPEN"
    if t & {"history","historical","parents","preceding","repeat"}: return "HISTORY"
    if t & {"authority"}: return "AUTHORITY"
    if t & {"goal","purpose","aim","objective","question","summary"}: return "GOAL"
    if t & {"status","maturity","verdict"}: return "STATUS"
    if t & {"source","sources","metrics","sha","commit","receipt","audit","ci","evidence","seal","digest"}: return "EVIDENCE"
    return "UNKNOWN"
def strengthened_rule(path,value):
    # Deterministic structural routing uses ancestry and explicit unknown fallback.
    last=path[-1]; tokens=set(words(" ".join(path)))
    if "boundaries" in path or last in {"limitations","unsupported"}: return "BOUNDARY"
    if "evidence" in tokens and last in {"blob","commit","sha","receipt"}: return "EVIDENCE"
    return rule(path,value)
def load_model(spec,archive):
    assert digest(archive.read_bytes())==spec["zip_sha256"]
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())=={"TRAIN.json","EXAM.json"}
        raw=z.read("TRAIN.json")
    assert digest(raw)==spec["train_sha256"]
    t=json.loads(raw); seal=t.pop("receipt_sha256")
    assert seal==spec["receipt"]==digest(canonical(t))
    assert len(t["oracle_only"]["train_ids"])==40 and t["parent_admitted"] is None
    import torch
    torch.set_num_threads(1)
    net=torch.nn.Sequential(torch.nn.Linear(256,32),torch.nn.ReLU(),torch.nn.Linear(32,len(LABELS)))
    net.load_state_dict({k:torch.tensor(v) for k,v in t["oracle_only"]["weights"].items()});net.eval()
    def predict(path,value):
        with torch.inference_mode(): p=torch.softmax(net(torch.tensor(features(path,value))),dim=0)
        return LABELS[int(p.argmax())] if float(p.max())>=.55 else "UNKNOWN"
    return predict
def get(d,path):
    for key in path.split("/"): d=d[key]
    assert type(d) in (int,str,bool)
    return d
def typed(a,b,op):
    if a==HOLD or b==HOLD: return HOLD
    if op=="equal" and type(a) is type(b): return a==b
    if op=="both_false" and type(a) is type(b) is bool: return not a and not b
    return HOLD
def main():
    manifest=json.loads(Path("research/assimilation/R70B_EXAM.json").read_text())
    neural=load_model(manifest["model"],Path(sys.argv[1]))
    docs={}
    for s in manifest["sources"]:
        raw=(Path("exam-"+s["id"])/s["path"]).read_bytes()
        assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()==s["blob"]
        d=json.loads(raw);assert d["schema"]==s["schema"];docs[s["id"]]=d
    cases=manifest["cases"]; rows=[]
    methods={"frozen_neural":neural,"r51_rule":rule,"structural_rule":strengthened_rule}
    outputs={k:[] for k in methods}
    outputs["without_memory"]=[];outputs["exact_pointer_ceiling"]=[]
    times={k:[] for k in methods}
    for doc,role,paths,target in cases:
        values={p:get(docs[doc],p) for p in paths}; row={"doc":doc,"role":role,"target":target,"pred":{}}
        for name,fn in methods.items():
            start=time.perf_counter_ns()
            matched=[p for p in paths if fn(p.split("/"),values[p])==role and role!="UNKNOWN"]
            times[name].append(time.perf_counter_ns()-start)
            pred=matched[0] if len(matched)==1 else None
            row["pred"][name]=pred;outputs[name].append(values[pred] if pred is not None else HOLD)
        row["pred"]["without_memory"]=None;row["pred"]["exact_pointer_ceiling"]=target
        outputs["without_memory"].append(HOLD);outputs["exact_pointer_ceiling"].append(values[target] if target is not None else HOLD)
        rows.append(row)
    metrics={}
    for name,vals in outputs.items():
        comp=[typed(vals[a],vals[b],op) for a,b,op,_ in manifest["composition"]]
        expected=[v for _,_,_,v in manifest["composition"]]
        metrics[name]={"route_correct":sum(r["pred"][name]==r["target"] for r in rows),"composition_correct":sum(type(a) is type(b) and a==b for a,b in zip(comp,expected)),"unsupported_route":sum(r["target"] is None and r["pred"][name] is not None for r in rows),"wrong_nonabstaining_route":sum(r["pred"][name] is not None and r["pred"][name]!=r["target"] for r in rows),"composition_predictions":comp}
        if name in times: metrics[name]["median_route_ns"]=int(statistics.median(times[name]))
    paired={}
    for baseline in ("r51_rule","structural_rule"):
        a=[r["pred"]["frozen_neural"]==r["target"] for r in rows];b=[r["pred"][baseline]==r["target"] for r in rows]
        paired[baseline]={"neural_only":sum(x and not y for x,y in zip(a,b)),"rule_only":sum(y and not x for x,y in zip(a,b))}
    assert metrics["exact_pointer_ceiling"]["route_correct"]==len(cases)
    assert metrics["exact_pointer_ceiling"]["composition_correct"]==len(manifest["composition"])
    report={"schema":"janus.r70b.internal_firstlook.v1","sources_verified":len(docs),"frozen_model_verified":True,"route_cases":len(cases),"composition_cases":len(manifest["composition"]),"metrics":metrics,"paired":paired,"rows":rows,"external_blind":False,"training_executed":False,"native_model_tested":False,"production_changed":False,"action_authority":False,"status":"INTERNAL_FIRSTLOOK_ONLY__INDEPENDENT_EXAM_HOLD"}
    print("R70B_EXAM "+json.dumps(report,sort_keys=True))
if __name__=="__main__": main()
