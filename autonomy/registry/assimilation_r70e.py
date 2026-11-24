"""R70E finite offline anti-forgetting ablation; no persistent policy updates."""
import json,sys,random,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import torch
import assimilation_r70b as b
import assimilation_r70c as c
import assimilation_r70d as d
def req(x,msg):
    if not x:raise ValueError(msg)
def spec(name):return json.loads(Path("research/assimilation/"+name+".json").read_bytes())
def replay_c(tr):
    rows=[];seen=set()
    for s in tr["sources"]:
        doc=d.checked_doc(s["id"],s)
        for p,v in c.leaves(doc):
            key=b.digest(b.canonical([p,v]))
            if key in seen:continue
            seen.add(key);rows.append((s["id"],p,v,b.strengthened_rule(p,v)))
    return rows
def replay_d(tr):
    groups=defaultdict(list);seen=set()
    for s in tr["sources"]:
        doc=d.checked_doc("pool",s)
        for p,v in d.leaves(doc):
            norm=["*" if k.isdecimal() else k for k in p]
            key=b.digest(b.canonical([norm,v]))
            if key in seen:continue
            seen.add(key);groups[b.strengthened_rule(p,v)].append((s["blob"],p,v,b.strengthened_rule(p,v)))
    rng=random.Random(tr["seed"])
    for rows in groups.values():rng.shuffle(rows)
    selected=[];counts=Counter()
    while len(selected)<tr["budget"]:
        progress=False
        for label in sorted(groups):
            limit=tr["max_unknown"] if label=="UNKNOWN" else tr["max_class"]
            if groups[label] and counts[label]<limit and len(selected)<tr["budget"]:
                selected.append(groups[label].pop());counts[label]+=1;progress=True
        if not progress:break
    return selected
def new_rows(tr):
    groups=defaultdict(list);seen=set()
    for s in tr["sources"]:
        for p,v in d.leaves(d.checked_doc("newpool",s)):
            key=b.digest(b.canonical([["*" if k.isdecimal() else k for k in p],v]))
            if key in seen:continue
            seen.add(key);l=b.strengthened_rule(p,v);groups[l].append((s["blob"],p,v,l))
    rng=random.Random(tr["seed"])
    for rows in groups.values():rng.shuffle(rows)
    selected=[];counts=Counter()
    while len(selected)<tr["new_budget"]:
        progress=False
        for l in sorted(groups):
            cap=24 if l=="UNKNOWN" else 40
            if groups[l] and counts[l]<cap and len(selected)<tr["new_budget"]:
                selected.append(groups[l].pop());counts[l]+=1;progress=True
        if not progress:break
    req(len(selected)>=tr["minimum_new"],"insufficient new rows")
    return selected
def tensors(rows):
    return torch.tensor([b.features(p,v) for _,p,v,_ in rows]),torch.tensor([b.LABELS.index(l) for *_,l in rows])
def train(tr,archive,out):
    torch.manual_seed(tr["seed"]);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rc=replay_c(spec("R70C_TRAIN"));rd=replay_d(spec("R70D_TRAIN"))
    req(len(rc)==63 and b.digest(b.canonical(rc))==tr["previous_train_receipts"]["r70c"],"R70C exact replay")
    req(len(rd)==400 and b.digest(b.canonical(rd))==tr["previous_train_receipts"]["r70d"],"R70D exact replay")
    new=new_rows(tr);seen=set();replay=[]
    for row in rc+rd:
        k=b.digest(b.canonical(row[1:]))
        if k not in seen:seen.add(k);replay.append(row)
    new=[r for r in new if b.digest(b.canonical(r[1:])) not in seen]
    req(len(new)>=tr["minimum_new"],"overlap reduced fresh pool")
    nx,ny=tensors(new);rx,ry=tensors(replay)
    parent=d.read_parent(tr["parent"],archive);base=c.weights(parent);basehash=b.digest(b.canonical(base))
    variants={}
    for name,use_replay,use_l2 in (("plain",False,False),("replay",True,False),("l2",False,True),("combined",True,True)):
        m=c.net();m.load_state_dict(parent.state_dict());m.train()
        anchor=[v.detach().clone() for v in m.parameters()]
        opt=torch.optim.AdamW(m.parameters(),lr=tr["learning_rate"])
        rng=random.Random(tr["seed"]);losses=[];new_draws=old_draws=0
        for step in range(tr["steps"]):
            n=tr["batch"]//2 if use_replay else tr["batch"]
            ni=[rng.randrange(len(new)) for _ in range(n)];x=nx[ni];y=ny[ni];new_draws+=n
            if use_replay:
                oi=[rng.randrange(len(replay)) for _ in range(tr["batch"]-n)]
                x=torch.cat((x,rx[oi]));y=torch.cat((y,ry[oi]));old_draws+=len(oi)
            loss=torch.nn.functional.cross_entropy(m(x),y)
            if use_l2:
                loss=loss+tr["l2_lambda"]*sum((v-a).square().sum() for v,a in zip(m.parameters(),anchor))/sum(v.numel() for v in m.parameters())
            opt.zero_grad();loss.backward();opt.step()
            if step in (0,tr["steps"]-1):losses.append(float(loss.item()))
        variants[name]={"weights":c.weights(m),"loss":losses,"new_draws":new_draws,"replay_draws":old_draws,"weight_l2":float(sum((v-a).square().sum() for v,a in zip(m.parameters(),anchor)).sqrt().item())}
    body={"schema":"janus.r70e.candidate.v1","parent_receipt":tr["parent"]["receipt"],"parent_weights_sha256":basehash,"train_spec_sha256":b.digest(b.canonical(tr)),"replay_c":len(rc),"replay_d":len(rd),"unique_replay":len(replay),"new_rows":len(new),"new_labels":dict(Counter(r[3] for r in new)),"new_rows_sha256":b.digest(b.canonical(new)),"variants":variants,"production_changed":False,"action_authority":False}
    body["receipt_sha256"]=b.digest(b.canonical(body));Path(out).write_bytes(b.canonical(body))
    print("R70E_TRAIN "+json.dumps({**{k:v for k,v in body.items() if k!="variants"},"variants":{k:{x:y for x,y in v.items() if x!="weights"} for k,v in variants.items()}},sort_keys=True))
def evaluate(ex,docs,methods):
    rows=[];out={k:[] for k in methods}
    for doc,role,paths,target in ex["cases"]:
        vals={p:b.get(docs[doc],p) for p in paths};r={"target":target,"pred":{},"doc":doc}
        for k,fn in methods.items():
            matches=[p for p in paths if role!="UNKNOWN" and fn(p.split("/"),vals[p])==role]
            pred=matches[0] if len(matches)==1 else None
            r["pred"][k]=pred;out[k].append(vals[pred] if pred is not None else b.HOLD)
        rows.append(r)
    expected=[v for *_,v in ex["composition"]]
    metrics={}
    for k,vals in out.items():
        comp=[b.typed(vals[i],vals[j],op) for i,j,op,_ in ex["composition"]]
        metrics[k]={"route_correct":sum(r["pred"][k]==r["target"] for r in rows),"composition_correct":sum(type(a) is type(v) and a==v for a,v in zip(comp,expected)),"wrong_nonabstaining":sum(r["pred"][k] is not None and r["pred"][k]!=r["target"] for r in rows),"lost_vs_D":sum(r["pred"]["parent_D"]==r["target"] and r["pred"][k]!=r["target"] for r in rows),"lost_vs_C":sum(r["pred"]["parent_C"]==r["target"] and r["pred"][k]!=r["target"] for r in rows),"new_vs_D":sum(r["pred"][k]==r["target"] and r["pred"]["parent_D"]!=r["target"] for r in rows)}
    return {"route_cases":len(rows),"composition_cases":len(expected),"metrics":metrics}
def exam(tr,archive,oldarchive,candidate,seal):
    torch.set_num_threads(1);r=json.loads(Path(candidate).read_bytes());pin=r.pop("receipt_sha256")
    req(pin==seal==b.digest(b.canonical(r)) and r["train_spec_sha256"]==b.digest(b.canonical(tr)),"candidate seal")
    p=d.read_parent(tr["parent"],archive);req(b.digest(b.canonical(c.weights(p)))==r["parent_weights_sha256"],"parent")
    pc=d.read_parent(spec("R70D_TRAIN")["parent"],oldarchive)
    methods={"parent_D":c.predictor(p,tr["threshold"]),"parent_C":c.predictor(pc,tr["threshold"]),"teacher":b.strengthened_rule}
    for k,v in r["variants"].items():
        m=c.net();m.load_state_dict({x:torch.tensor(y) for x,y in v["weights"].items()});methods[k]=c.predictor(m,tr["threshold"])
    oldc=spec("R70C_EXAM");oldd=spec("R70D_EXAM");fresh=spec("R70E_EXAM")
    sets=[{s["blob"] for s in t["sources"]} for t in (spec("R70C_TRAIN"),spec("R70D_TRAIN"),tr,oldc,oldd,fresh)]
    req(all(not (sets[i]&sets[j]) for i in range(len(sets)) for j in range(i)),"source overlap")
    results={"fresh":evaluate(fresh,{s["id"]:d.checked_doc("freshe",s) for s in fresh["sources"]},methods),"regression_C":evaluate(oldc,{s["id"]:d.checked_doc(s["id"],s) for s in oldc["sources"]},methods),"regression_D":evaluate(oldd,{s["id"]:d.checked_doc("freshd",s) for s in oldd["sources"]},methods)}
    gates={}
    for k in r["variants"]:
        fm=results["fresh"]["metrics"];cm=results["regression_C"]["metrics"];dm=results["regression_D"]["metrics"]
        gates[k]={"fresh_gain":fm[k]["route_correct"]-fm["parent_D"]["route_correct"],"old_C_lost_vs_C":cm[k]["lost_vs_C"],"old_C_lost_vs_D":cm[k]["lost_vs_D"],"old_D_lost_vs_D":dm[k]["lost_vs_D"],"no_loss_gate":cm[k]["lost_vs_C"]==cm[k]["lost_vs_D"]==dm[k]["lost_vs_D"]==0 and cm[k]["composition_correct"]>=max(cm["parent_C"]["composition_correct"],cm["parent_D"]["composition_correct"]) and dm[k]["composition_correct"]>=dm["parent_D"]["composition_correct"] and fm[k]["wrong_nonabstaining"]<=fm["parent_D"]["wrong_nonabstaining"] and fm[k]["route_correct"]>fm["parent_D"]["route_correct"]}
    print("R70E_EXAM "+json.dumps({"schema":"janus.r70e.internal_exam.v1","candidate_receipt":pin,"results":results,"gates":gates,"external_blind":False,"production_changed":False,"action_authority":False,"status":"RESEARCH_ONLY__NO_AUTOMATIC_ADMISSION"},sort_keys=True))
if __name__=="__main__":
    tr=spec("R70E_TRAIN")
    if sys.argv[1]=="train":train(tr,Path(sys.argv[2]),sys.argv[3])
    elif sys.argv[1]=="exam":exam(tr,Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],sys.argv[5])
    else:raise ValueError("mode")
