"""R70F: bounded offline memory retention experiment; no autonomous updates."""
import json,sys,zipfile,random
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
import assimilation_r70b as b
import assimilation_r70c as c
import assimilation_r70d as d
import assimilation_r70e as e
def req(x,msg):
    if not x:raise ValueError(msg)
def spec(n):return e.spec(n)
def parent_models(tr,root):
    ms={}
    for k in ("C","D"):
        ms[k]=d.read_parent(tr["parents"][k],root/("r70"+k.lower()+".zip"))
    p=tr["parents"]["E"];raw=(root/"r70e.zip").read_bytes()
    req(b.digest(raw)==p["zip_sha256"],"E zip")
    with zipfile.ZipFile(root/"r70e.zip") as z:
        req(z.namelist()==["CANDIDATE.json"],"E archive")
        x=json.loads(z.read("CANDIDATE.json"))
    pin=x.pop("receipt_sha256")
    req(pin==p["receipt"]==b.digest(b.canonical(x)),"E receipt")
    m=c.net();m.load_state_dict({k:torch.tensor(v) for k,v in x["variants"][p["variant"]]["weights"].items()});ms["E"]=m
    for m in ms.values():m.eval()
    return ms
def replay(tr):
    ec=spec("R70E_TRAIN");rc=e.replay_c(spec("R70C_TRAIN"));rd=e.replay_d(spec("R70D_TRAIN"))
    req(b.digest(b.canonical(rc))==ec["previous_train_receipts"]["r70c"] and b.digest(b.canonical(rd))==ec["previous_train_receipts"]["r70d"],"old training receipts")
    seen=set();old=[]
    for row in rc+rd:
        key=b.digest(b.canonical(row[1:]))
        if key not in seen:seen.add(key);old.append(row)
    re=[r for r in e.new_rows(ec) if b.digest(b.canonical(r[1:])) not in seen]
    req(b.digest(b.canonical(re))==tr["expected_replay_E"],"E training receipt")
    for row in re:
        key=b.digest(b.canonical(row[1:]))
        if key not in seen:seen.add(key);old.append(row)
    Path("newpool").rename("newpool-e-retained")
    Path("newpool-f").rename("newpool")
    try:
        new=[r for r in e.new_rows(tr) if b.digest(b.canonical(r[1:])) not in seen]
    finally:
        Path("newpool").rename("newpool-f")
        Path("newpool-e-retained").rename("newpool")
    req(len(new)>=tr["minimum_new"],"new pool")
    return old,new
def train(tr,root,out):
    torch.manual_seed(tr["seed"]);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    old,new=replay(tr);rx,ry=e.tensors(old);nx,ny=e.tensors(new)
    ms=parent_models(tr,root);parent=ms["E"];temp=tr["temperature"]
    with torch.no_grad():
        targets={k:torch.softmax(m(rx)/temp,dim=-1) for k,m in ms.items()}
    # Each replay example has a verified teacher pseudo-label. For multi-parent
    # soft targets include only ancestors predicting this label at the frozen threshold.
    with torch.no_grad():
        masks={k:((torch.softmax(m(rx),dim=-1).argmax(-1)==ry)&(torch.softmax(m(rx),dim=-1).max(-1).values>=tr["threshold"])).float() for k,m in ms.items()}
        # Fall back to E soft distribution if none qualify; same frozen target policy for all steps.
        count=sum(masks.values());multi=sum(targets[k]*masks[k][:,None] for k in ms)/count.clamp_min(1)[:,None]
        multi=torch.where((count>0)[:,None],multi,targets["E"])
    variants={}
    for name,mode in (("anchor_only","none"),("lock_E","E"),("lock_CDE","multi")):
        m=c.net();m.load_state_dict(parent.state_dict());m.train();anchors=[v.detach().clone() for v in m.parameters()]
        opt=torch.optim.AdamW(m.parameters(),lr=tr["lr"]);rng=random.Random(tr["seed"]);losses=[]
        for step in range(tr["steps"]):
            ni=[rng.randrange(len(new)) for _ in range(32)];ri=[rng.randrange(len(old)) for _ in range(32)]
            logits=m(torch.cat((nx[ni],rx[ri])));y=torch.cat((ny[ni],ry[ri]))
            loss=F.cross_entropy(logits,y)+tr["l2_lambda"]*sum((v-a).square().sum() for v,a in zip(m.parameters(),anchors))/sum(v.numel() for v in m.parameters())
            if mode!="none":
                target=(targets["E"] if mode=="E" else multi)[ri]
                loss=loss+tr["distill_weight"]*temp*temp*F.kl_div(F.log_softmax(logits[32:]/temp,dim=-1),target,reduction="batchmean")
            opt.zero_grad();loss.backward();opt.step()
            if step in (0,tr["steps"]-1):losses.append(float(loss.item()))
        variants[name]={"weights":c.weights(m),"loss":losses,"weight_l2":float(sum((v-a).square().sum() for v,a in zip(m.parameters(),anchors)).sqrt().item())}
    body={"schema":"janus.r70f.candidate.v1","train_spec_sha256":b.digest(b.canonical(tr)),"parents":{k:v["receipt"] for k,v in tr["parents"].items()},"parent_weights":{k:b.digest(b.canonical(c.weights(m))) for k,m in ms.items()},"replay_rows":len(old),"new_rows":len(new),"new_rows_sha256":b.digest(b.canonical(new)),"multi_target_eligible":{k:int(v.sum()) for k,v in masks.items()},"variants":variants,"production_changed":False}
    body["receipt_sha256"]=b.digest(b.canonical(body));Path(out).write_bytes(b.canonical(body))
    print("R70F_TRAIN "+json.dumps({**{k:v for k,v in body.items() if k!="variants"},"variants":{k:{x:y for x,y in v.items() if x!="weights"} for k,v in variants.items()}},sort_keys=True))
def evaluate(ex,docs,methods):
    gold=[r[3] for r in ex["cases"]];exp=[r[3] for r in ex["composition"]];results={}
    for k,fn in methods.items():
        pred=[];values=[]
        for doc,role,paths,target in ex["cases"]:
            vals={p:b.get(docs[doc],p) for p in paths}
            hits=[p for p in paths if role!="UNKNOWN" and fn(p.split("/"),vals[p])==role]
            p=hits[0] if len(hits)==1 else None;pred.append(p);values.append(vals[p] if p is not None else b.HOLD)
        comp=[b.typed(values[i],values[j],op) for i,j,op,_ in ex["composition"]]
        results[k]={"route":[p==g for p,g in zip(pred,gold)],"comp":[type(v) is type(g) and v==g for v,g in zip(comp,exp)],"wrong":sum(p is not None and p!=g for p,g in zip(pred,gold))}
    metrics={}
    for k,r in results.items():
        metrics[k]={"route_correct":sum(r["route"]),"composition_correct":sum(r["comp"]),"wrong_nonabstaining":r["wrong"],"route_lost_union_CDE":sum(any(results[p]["route"][i] for p in ("C","D","E")) and not v for i,v in enumerate(r["route"])),"comp_lost_union_CDE":sum(any(results[p]["comp"][i] for p in ("C","D","E")) and not v for i,v in enumerate(r["comp"]))}
    return {"route_cases":len(gold),"composition_cases":len(exp),"metrics":metrics}
def exam(tr,root,candidate,pin):
    torch.set_num_threads(1);r=json.loads(Path(candidate).read_bytes());seal=r.pop("receipt_sha256")
    req(seal==pin==b.digest(b.canonical(r)) and r["train_spec_sha256"]==b.digest(b.canonical(tr)),"candidate pin")
    ms=parent_models(tr,root);req(r["parent_weights"]=={k:b.digest(b.canonical(c.weights(m))) for k,m in ms.items()},"parent lineage")
    for k,v in r["variants"].items():
        m=c.net();m.load_state_dict({x:torch.tensor(y) for x,y in v["weights"].items()});ms[k]=m
    methods={k:c.predictor(m,tr["threshold"]) for k,m in ms.items()};methods["teacher"]=b.strengthened_rule
    sets=[{s["blob"] for s in t["sources"]} for t in [spec("R70C_TRAIN"),spec("R70D_TRAIN"),spec("R70E_TRAIN"),tr,spec("R70C_EXAM"),spec("R70D_EXAM"),spec("R70E_EXAM"),spec("R70F_EXAM")]]
    req(all(not sets[i]&sets[j] for i in range(len(sets)) for j in range(i)),"source overlap")
    results={}
    for stage,location in (("C",None),("D","freshd"),("E","freshe"),("F","freshf")):
        ex=spec("R70"+stage+"_EXAM")
        results[stage]=evaluate(ex,{s["id"]:d.checked_doc(location or s["id"],s) for s in ex["sources"]},methods)
    gates={}
    for k in r["variants"]:
        fm=results["F"]["metrics"];lost=sum(results[s]["metrics"][k]["route_lost_union_CDE"]+results[s]["metrics"][k]["comp_lost_union_CDE"] for s in "CDE")
        gates[k]={"old_union_losses":lost,"fresh_gain_vs_E":fm[k]["route_correct"]-fm["E"]["route_correct"],"fresh_gain_vs_teacher":fm[k]["route_correct"]-fm["teacher"]["route_correct"],"pass":lost==0 and fm[k]["route_correct"]>max(fm["E"]["route_correct"],fm["teacher"]["route_correct"]) and fm[k]["composition_correct"]>=max(fm["E"]["composition_correct"],fm["teacher"]["composition_correct"]) and all(results[s]["metrics"][k]["wrong_nonabstaining"]<=results[s]["metrics"]["E"]["wrong_nonabstaining"] for s in "CDEF")}
    print("R70F_EXAM "+json.dumps({"schema":"janus.r70f.internal_exam.v1","candidate_receipt":seal,"results":results,"gates":gates,"external_blind":False,"production_changed":False,"action_authority":False,"status":"RESEARCH_ONLY__NO_AUTOMATIC_ADMISSION"},sort_keys=True))
if __name__=="__main__":
    tr=spec("R70F_TRAIN")
    if sys.argv[1]=="train":train(tr,Path(sys.argv[2]),sys.argv[3])
    elif sys.argv[1]=="exam":exam(tr,Path(sys.argv[2]),sys.argv[3],sys.argv[4])
    else:raise ValueError("mode")
