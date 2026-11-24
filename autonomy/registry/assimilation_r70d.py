"""R70D: bounded first-party offline balanced distillation, no production changes."""
import hashlib, json, sys, zipfile, random, time, statistics
from collections import Counter, defaultdict
from pathlib import Path
import torch
import assimilation_r70b as b
import assimilation_r70c as c
def req(x,msg):
    if not x: raise ValueError(msg)
def checked_doc(root,s):
    raw=(Path(root)/s["path"]).read_bytes()
    req(len(raw)<=1000000 and hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()==s["blob"],"source pin")
    d=json.loads(raw)
    req(type(d) is dict,"object source")
    if "schema" in s: req(d.get("schema")==s["schema"],"source schema")
    return d
def leaves(d,p=()):
    if len(p)>8:return
    if type(d) is dict:
        for k,v in sorted(d.items()):yield from leaves(v,p+(k,))
    elif type(d) is list:
        for i,v in enumerate(d):
            if type(v) in (dict,list):yield from leaves(v,p+(str(i),))
    elif type(d) in (str,int,bool) and p and p[-1] not in {"schema","id","artifact_id","conversation_id","review_message_id"}:
        yield list(p),d
def read_parent(spec,archive):
    req(b.digest(archive.read_bytes())==spec["zip_sha256"],"parent zip")
    with zipfile.ZipFile(archive) as z:
        req(z.namelist()==["CANDIDATE.json"],"parent archive members")
        x=json.loads(z.read("CANDIDATE.json"))
    seal=x.pop("receipt_sha256")
    req(seal==spec["receipt"]==b.digest(b.canonical(x)),"parent seal")
    m=c.net();m.load_state_dict({k:torch.tensor(v) for k,v in x["inherited"].items()});return m
def train(spec,archive,out):
    torch.manual_seed(spec["seed"]);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    groups=defaultdict(list);seen=set()
    for s in spec["sources"]:
        d=checked_doc("pool",s)
        for p,v in leaves(d):
            norm=[("*" if k.isdecimal() else k) for k in p]
            key=b.digest(b.canonical([norm,v]))
            if key in seen:continue
            seen.add(key)
            label=b.strengthened_rule(p,v)
            groups[label].append((s["blob"],p,v,label))
    available={k:len(v) for k,v in groups.items()}
    rng=random.Random(spec["seed"])
    for rows in groups.values():rng.shuffle(rows)
    selected=[];counts=Counter()
    while len(selected)<spec["budget"]:
        progress=False
        for label in sorted(groups):
            limit=spec["max_unknown"] if label=="UNKNOWN" else spec["max_class"]
            if groups[label] and counts[label]<limit and len(selected)<spec["budget"]:
                selected.append(groups[label].pop());counts[label]+=1;progress=True
        if not progress:break
    print("R70D_POOL "+json.dumps({"documents":len(spec["sources"]),"available":available,"selected":dict(counts),"rows":len(selected)},sort_keys=True))
    req(spec["min_rows"]<=len(selected)<=spec["max_rows"],"INSUFFICIENT_DISTINCT_BALANCED_SOURCE_ROWS")
    req(len(counts)>=5 and counts["UNKNOWN"]<=spec["max_unknown"],"class balance")
    x=torch.tensor([b.features(p,v) for _,p,v,_ in selected]);y=torch.tensor([b.LABELS.index(l) for *_,l in selected])
    m=read_parent(spec["parent"],archive);p0=b.digest(b.canonical(c.weights(m)))
    losses=c.fit(m,x,y,spec["steps"],spec["learning_rate"])
    torch.manual_seed(spec["seed"]);scratch=c.net();sl=c.fit(scratch,x,y,spec["steps"],spec["learning_rate"])
    candidate={"schema":"janus.r70d.candidate.v1","parent_receipt":spec["parent"]["receipt"],"parent_weights_sha256":p0,"train_spec_sha256":b.digest(b.canonical(spec)),"rows":len(selected),"labels":dict(counts),"available_labels":available,"train_rows_sha256":b.digest(b.canonical(selected)),"distinct_feature_vectors":len({tuple(b.features(p,v)) for _,p,v,_ in selected}),"train_source_blobs":sorted({blob for blob,*_ in selected}),"inherited":c.weights(m),"scratch":c.weights(scratch),"loss":losses,"scratch_loss":sl,"production_changed":False,"action_authority":False}
    req(b.digest(b.canonical(candidate["inherited"]))!=p0,"weights unchanged")
    candidate["receipt_sha256"]=b.digest(b.canonical(candidate))
    Path(out).write_bytes(b.canonical(candidate))
    print("R70D_TRAIN "+json.dumps({k:v for k,v in candidate.items() if k not in {"inherited","scratch"}},sort_keys=True))
def evaluate(spec,docs,methods):
    outputs={k:[] for k in methods};outputs["exact_ceiling"]=[];rows=[];times=defaultdict(list)
    for doc,role,paths,target in spec["cases"]:
        vals={p:b.get(docs[doc],p) for p in paths};r={"doc":doc,"role":role,"target":target,"pred":{}}
        for k,fn in methods.items():
            start=time.perf_counter_ns()
            matches=[p for p in paths if role!="UNKNOWN" and fn(p.split("/"),vals[p])==role]
            times[k].append(time.perf_counter_ns()-start)
            pred=matches[0] if len(matches)==1 else None
            r["pred"][k]=pred;outputs[k].append(vals[pred] if pred is not None else b.HOLD)
        r["pred"]["exact_ceiling"]=target;outputs["exact_ceiling"].append(vals[target] if target is not None else b.HOLD);rows.append(r)
    metrics={}
    for k,values in outputs.items():
        comp=[b.typed(values[i],values[j],op) for i,j,op,_ in spec["composition"]]
        exp=[v for *_,v in spec["composition"]]
        metrics[k]={"route_correct":sum(r["pred"][k]==r["target"] for r in rows),"composition_correct":sum(type(a) is type(v) and a==v for a,v in zip(comp,exp)),"unsupported":sum(r["target"] is None and r["pred"][k] is not None for r in rows),"wrong_nonabstaining":sum(r["pred"][k] is not None and r["pred"][k]!=r["target"] for r in rows),"by_doc":{d:sum(r["doc"]==d and r["pred"][k]==r["target"] for r in rows) for d in docs},"composition_predictions":comp}
        if k in times:metrics[k]["median_route_ns"]=int(statistics.median(times[k]))
    req(metrics["exact_ceiling"]["route_correct"]==len(rows) and metrics["exact_ceiling"]["composition_correct"]==len(spec["composition"]),"gold mismatch")
    paired={k:{"new_only":sum(r["pred"]["inherited"]==r["target"] and r["pred"][k]!=r["target"] for r in rows),"other_only":sum(r["pred"]["inherited"]!=r["target"] and r["pred"][k]==r["target"] for r in rows)} for k in methods if k!="inherited"}
    return {"route_cases":len(rows),"composition_cases":len(spec["composition"]),"metrics":metrics,"paired":paired,"rows":rows}
def exam(tr,ex,archive,candidate,expected):
    torch.set_num_threads(1)
    d=json.loads(Path(candidate).read_bytes());seal=d.pop("receipt_sha256")
    req(seal==expected==b.digest(b.canonical(d)),"candidate seal")
    req(d["train_spec_sha256"]==b.digest(b.canonical(tr)) and d["parent_receipt"]==tr["parent"]["receipt"],"lineage")
    old=json.loads(Path("research/assimilation/R70C_EXAM.json").read_bytes())
    for field in ("path","blob"):
        a={s[field] for s in tr["sources"]};f={s[field] for s in ex["sources"]};o={s[field] for s in old["sources"]}
        req(not(a&f or a&o or f&o),"source split")
    p=read_parent(tr["parent"],archive)
    req(b.digest(b.canonical(c.weights(p)))==d["parent_weights_sha256"],"parent weights")
    models={"parent":p}
    for k in ("inherited","scratch"):
        m=c.net();m.load_state_dict({k2:torch.tensor(v) for k2,v in d[k].items()});models[k]=m
    methods={k:c.predictor(m,tr["threshold"]) for k,m in models.items()}
    methods.update({"teacher":b.strengthened_rule,"r51_rule":b.rule})
    fresh=evaluate(ex,{s["id"]:checked_doc("fresh",s) for s in ex["sources"]},methods)
    regress=evaluate(old,{s["id"]:checked_doc(s["id"],s) for s in old["sources"]},methods)
    pm=fresh["metrics"]["parent"];im=fresh["metrics"]["inherited"];tm=fresh["metrics"]["teacher"]
    rloss=regress["paired"]["parent"]["other_only"]
    report={"schema":"janus.r70d.internal_exam.v1","candidate_receipt":seal,"training_rows":d["rows"],"fresh":fresh,"disclosed_regression_only":regress,"regression_route_losses":rloss,"fresh_parent_gain":im["route_correct"]-pm["route_correct"],"teacher_gap":im["route_correct"]-tm["route_correct"],"research_gate": "PARENT_GAIN_NO_ROUTE_LOSS" if im["route_correct"]>pm["route_correct"] and rloss==0 and im["wrong_nonabstaining"]<=pm["wrong_nonabstaining"] and regress["metrics"]["inherited"]["composition_correct"]>=regress["metrics"]["parent"]["composition_correct"] else "HOLD_REGRESSION_OR_NO_GAIN","external_blind":False,"production_changed":False,"action_authority":False,"native_model_tested":False}
    print("R70D_EXAM "+json.dumps(report,sort_keys=True))
if __name__=="__main__":
    tr=json.loads(Path("research/assimilation/R70D_TRAIN.json").read_bytes())
    if sys.argv[1]=="train":train(tr,Path(sys.argv[2]),sys.argv[3])
    elif sys.argv[1]=="exam":exam(tr,json.loads(Path("research/assimilation/R70D_EXAM.json").read_bytes()),Path(sys.argv[2]),sys.argv[3],sys.argv[4])
    else:raise ValueError("mode")
