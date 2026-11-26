"""H38J bounded autonomous synthetic curriculum and gated shadow learning.
No model-generated truth; local three-check admission follows TRINITY R1.4's
separation of proposal and verification, not the original arithmetic oracle.
"""
import sys,copy,json,hashlib,time
from pathlib import Path
from collections import Counter
import torch
from torch import nn
from safetensors.torch import save_file,load_file
import r84h38i_fractal_reconstruction as i
import r84h38h_semantic_kaleidoscope as h
import r84h38g_semantic_mosaic as m
import r84h38f_semantic_crystal as f
import r84h38e_semantic_invariance as e
import r84h38d_world_understanding as d
import r84h38c_semantic_world_graph as g
ck=f.ck;read=f.read;write=f.write;sha=f.sha
# Authored finite generator; labels come from construction, never prediction.
TEMPLATES=[
["A occurred before B","A happened before B","A took place before B","A precedes B","after A, B occurred","only after A did B happen"],
["A occurred after B","A happened after B","A took place after B","A follows B","before A, B happened"],
["A did not occur before B","A did not happen before B"],
["A did not occur after B","A did not happen after B"],
["A caused B","A causes B"],
["A was caused by B"],
["A did not cause B"],
[],
["A requires B","A required B"],
["A is required by B"],
["A does not require B"],
[],
["A prevents B","A prevented B"],
["A was prevented by B"],
["A did not prevent B"],
[]]
def synth():
    for y,forms in enumerate(TEMPLATES):
        for j,t in enumerate(forms):
            names=["gena"+str(y)+"_"+str(j),"genb"+str(y)+"_"+str(j)]
            text=t.replace("A",names[0]).replace("B",names[1])
            n=e.norm(text,names);ck(n is not None,"GEN_NORMALIZE")
            yield {"text":n[0],"surface":text,"names":names,"y":y,"gold":e.decode(y,n[1])}
def admission(z):
    # Three separately checked obligations: round-trip, polarity contrast,
    # exact graph proof. All operate in the declared synthetic domain.
    n=e.norm(z["surface"],z["names"]);a=d.parse(z["surface"])
    if n is None or n[0]!=z["text"] or a is None or a!=z["gold"] or e.decode(z["y"],n[1])!=a:return None
    q={"schema":"janus.event.graph.v1","facts":[{"id":"src",**a}],"claim":a}
    yes=g.solve(q);no=g.solve({**q,"claim":{**a,"positive":not a["positive"]}})
    if yes["status"]!="ENTAILED" or no["status"]!="CONTRADICTED" or yes["proof"]!=["src"] or no["proof"]!=["src"]:return None
    return {"origin":"FINITE_SYNTHETIC_CONSTRUCTION","label":z["y"],"roundtrip":a,"positive_proof":yes,"contrast_proof":no,"external_truth_verified":False,"training_allowed":True}
def load_i(p):
    r=read(p/"receipt.json");ms=[]
    for z in r["runs"]["repair"]:
        q=p/f"repair-{z['seed']}.safetensors";ck(sha(q)==z["sha"],"I_WEIGHT");n=m.Mosaic();n.load_state_dict(load_file(str(q)));ms.append(n.eval())
    return r,ms
def fit(parent,rows):
    ms=[];cpu=0
    x=m.batch([z["text"] for z in rows]);y=torch.tensor([f.parts(z["y"]) for z in rows]);known=y[:,3]==1
    for old in parent:
        net=copy.deepcopy(old);start=time.process_time()
        # Relation tile is frozen; repair only polarity/direction/certainty.
        for k in range(1,4):
            tile=net.tiles[k];opt=torch.optim.AdamW(tile.parameters(),lr=.002);mask=known if k<3 else torch.ones_like(known)
            for _ in range(120):
                l=nn.functional.cross_entropy(tile(x[k][mask]),y[mask,k]);opt.zero_grad();l.backward();opt.step()
        ck(all(torch.equal(a,b) for a,b in zip(old.tiles[0].parameters(),net.tiles[0].parameters())),"FROZEN_TILE")
        cpu+=time.process_time()-start;ms.append(net.eval())
    return ms,cpu
def save_models(ms,out,name):
    ans=[]
    for j,n in enumerate(ms):
        p=out/f"{name}-{j}.safetensors";save_file(n.state_dict(),str(p));ans.append({"file":p.name,"sha":sha(p)})
    return ans
def load_models(out,rows):
    ms=[]
    for z in rows:
        p=out/z["file"];ck(sha(p)==z["sha"],"CHECKPOINT");n=m.Mosaic();n.load_state_dict(load_file(str(p)));ms.append(n.eval())
    return ms
def train(hdir,gdir,idir,demi,out):
    g.witness();torch.set_num_threads(2);out.mkdir();hr,bank,ns=i.load_h(hdir);gm=i.load_g(gdir,hr["bank_sha"]);ir,parent=load_i(idir);ck(ir["bank_sha"]==hr["bank_sha"],"I_BANK")
    base=bank["train"]+read(idir/"repair_rows.json");dev=bank["dev"];rule=ir["rules"]["repair"]["rule"]
    known={z["text"] for z in base+dev};root=out/"cortex";root.mkdir();cortex=e.cortex(demi)
    proposals=list(synth());ck(len({z["text"] for z in proposals})==len(proposals),"GEN_DUP")
    # Deliberately corrupt labels, polarity and text: verifier must reject.
    bad=copy.deepcopy(proposals[0]);bad["y"]=2
    bad2=copy.deepcopy(proposals[0]);bad2["gold"]["positive"]=False
    bad3=copy.deepcopy(proposals[0]);bad3["surface"]=bad3["names"][0]+" may cause "+bad3["names"][1]
    ck(all(admission(z) is None for z in [bad,bad2,bad3]),"ADMISSION_NEGATIVE_CONTROLS")
    def raw(t,ms):return i.trace(t,hr,ns,bank,ms,rule)
    def guarded(t,ms):
        a=i.trace(t,hr,ns,bank,gm)["value"]
        return a if a is not None else raw(t,ms)["value"]
    def assess(ms):
        ps=[guarded(z["text"],ms) for z in dev]
        return {"correct":[j for j,(p,z) in enumerate(zip(ps,dev)) if p is not None and p==z["y"]],"wrong":[j for j,(p,z) in enumerate(zip(ps,dev)) if p is not None and p!=z["y"]]}
    current=parent;rows=[];cycles=[];all_traces=[];last_candidate=parent;committed=save_models(current,out,"initial")
    for cycle in range(2):
        # Reload serialized state and model at every cycle boundary.
        current=load_models(out,committed);state=cortex.load_state(root);before=assess(current)
        eligible=[];skips=Counter()
        for z in proposals:
            t=z["text"];digest=hashlib.sha256(t.encode()).hexdigest()
            if t in known:skips["BASE_OR_DEV_DUP"]+=1;continue
            if digest in state["records"]:skips["CORTEX_DUP"]+=1;continue
            p=raw(t,current);ck(i.verify(p),"RETURN")
            if p["children"][0]["value"] is None or p["value"] is not None:skips["NOT_COMPONENT_REFERRAL"]+=1;continue
            # Selection uses only rejection diagnostics, no gold labels.
            priority=sum(r.startswith(("polarity:","direction:","certainty:")) for r in p["reasons"])
            eligible.append((priority,t,z,p))
        selected=[]
        for _,t,z,p in sorted(eligible,key=lambda q:(-q[0],q[1])):
            a=admission(z)
            if a is None:skips["VERIFIER_VETO"]+=1;continue
            origin={**a,"return_sha":p["sha256"],"cycle":cycle}
            ck(cortex.add_text(root,state,t.encode(),origin),"CORTEX_ADD")
            ck(not cortex.add_text(root,state,t.encode(),origin),"CORTEX_DEDUP")
            selected.append({"text":t,"y":z["y"],"origin":origin});all_traces.append({"cycle":cycle,"text":t,"return":p,"admission":a})
            if len(selected)==6:break
        if not selected:
            cycles.append({"cycle":cycle,"selected":0,"skips":dict(skips),"decision":"NO_NEW_VERIFIED_REFERRALS"});break
        rows.extend(selected)
        manifest=cortex.publish(root,state,"H38J_VERIFIED_SYNTHETIC_CURRICULUM");cortex.write_json(root/"STATE.json",cortex.sealed(state))
        # Real memory readback from persisted state, not just an in-memory set.
        persisted=cortex.load_state(root);ck(len(persisted["active"])==len(rows),"CORTEX_RELOAD")
        for z in rows:
            dg=hashlib.sha256(z["text"].encode()).hexdigest();ck(cortex.read_object(root,dg,persisted["records"][dg]["size"]).decode()==z["text"],"MEMORY_BYTES")
        candidate,cpu=fit(current,base+rows);after=assess(candidate);last_candidate=candidate
        lost=sorted(set(before["correct"])-set(after["correct"]))
        # Strict quality promotion; rejected weights never replace parent.
        accept=not after["wrong"] and not lost and len(after["correct"])>len(before["correct"])
        proposed=save_models(candidate,out,f"candidate-{cycle}")
        if accept:committed=proposed
        cycles.append({"cycle":cycle,"eligible":len(eligible),"selected":len(selected),"skips":dict(skips),"memory_generation":manifest["generation"],"memory_documents":manifest["indexed_documents"],"manifest_sha":manifest["receipt_sha256"],"before":before,"after":after,"lost":lost,"decision":"SHADOW_ACCEPT" if accept else "ROLLBACK_KEEP_PARENT","candidate":proposed,"active":committed,"cpu":cpu})
    final=load_models(out,committed);last=save_models(last_candidate,out,"last-proposal")
    result={"schema":"h38j.train.v1","bank_sha":hr["bank_sha"],"i_receipt_sha":sha(idir/"receipt.json"),"proposals":len(proposals),"cycles":cycles,"verified_new":len(rows),"admission_negative_controls":3,"active":committed,"last":last,"rule":rule,"exam_read":False,"production":False,"scope":"Bounded automatic curriculum; authored grammar and exact local verifier; native disposable Cortex; separate quality gate. Not general autonomous learning or existing TRINITY arithmetic execution."}
    write(out/"receipt.json",result);write(out/"new_rows.json",rows);write(out/"returns.json",all_traces);print("R84H38J_TRAIN",json.dumps(result,sort_keys=True),flush=True)
def evaluate(hdir,gdir,idir,edir,demi,trained,exam,oldc,oldd,olde,oldf,oldg,oldh,oldi,out):
    g.witness();torch.set_num_threads(2);out.mkdir();hr,bank,ns=i.load_h(hdir);gm=i.load_g(gdir,hr["bank_sha"]);ir,im=load_i(idir);r=read(trained/"receipt.json");ck(r["bank_sha"]==hr["bank_sha"] and r["i_receipt_sha"]==sha(idir/"receipt.json"),"PARENT_PINS")
    active=load_models(trained,r["active"]);last=load_models(trained,r["last"]);rule=r["rule"]
    er=read(edir/"receipt.json");eb=read(edir/"bank.json");ck(sha(edir/"bank.json")==er["bank_sha"],"E_BANK");em=[]
    for z in er["runs"]:
        p=edir/f"model-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"E_WEIGHT");n=e.Net();n.load_state_dict(load_file(str(p)));em.append(n.eval())
    di=e.Dictionary(e.cortex(demi),edir/"cortex",eb["train"]);cortex=e.cortex(demi);newrows=read(trained/"new_rows.json")
    state=cortex.load_state(trained/"cortex");ck(len(state["active"])==len(newrows)==r["verified_new"],"PERSISTED_MEMORY")
    for z in newrows:
        dg=hashlib.sha256(z["text"].encode()).hexdigest();ck(cortex.read_object(trained/"cortex",dg,state["records"][dg]["size"]).decode()==z["text"],"MEMORY_READ")
    refs={z["text"] for z in bank["train"]+bank["dev"]+eb["train"]+eb["dev"]+read(idir/"repair_rows.json")+newrows}
    prior=read(oldd)["cases"]
    def namesD(z):return sorted({a[k] for a in z["atoms"]+[z["claim_atom"]] if a for k in ("left","right")})
    for z in prior:
        for t in z["sentences"]+[z["claim"]]:
            n=e.norm(t,namesD(z))
            if n:refs.add(n[0])
    for p in (olde,oldf,oldg,oldh,oldi):
        for z in read(p)["rows"]:
            n=e.norm(z["text"],z["names"])
            if n:refs.add(n[0])
    cache={}
    def eval_t(t,names,k):
        if k in ("legacy","parent","stack"):
            a=d.parse(t)
            if a is not None:return a
        n=e.norm(t,names)
        if n is None:return None
        text,events=n
        if text not in cache:
            j=e.pick(e.probs(em,[text])[0],di.search(text),er["rules"]["agreement"],"agreement")
            qs={"frozen":i.trace(text,hr,ns,bank,gm),"parent_tile":i.trace(text,hr,ns,bank,im,rule),"active":i.trace(text,hr,ns,bank,active,rule),"last":i.trace(text,hr,ns,bank,last,rule)}
            ck(all(i.verify(q) for q in qs.values()),"RETURN");cache[text]=(j,qs)
        j,qs=cache[text]
        if k in ("legacy","parent","stack") and j is not None:return e.decode(j,events)
        if k=="legacy":return None
        if k in ("parent","stack"):
            for route in (("frozen","parent_tile") if k=="parent" else ("frozen","parent_tile","active")):
                if qs[route]["value"] is not None:return e.decode(qs[route]["value"],events)
            return None
        a=qs[k]["value"];return e.decode(a,events) if a is not None else None
    modes=("legacy","parent","parent_tile","active","last","stack");ex=read(exam);details=[]
    for z in ex["rows"]:
        n=e.norm(z["text"],z["names"]);ck(n is None or n[0] not in refs,"FRESH_OVERLAP")
        if z["y"]<16:ck(n is not None and e.decode(z["y"],n[1])==z["gold"],"GOLD")
        details.append({**z,"pred":{k:eval_t(z["text"],z["names"],k) for k in modes}})
    def metrics(rows,k,groups=()):
        return {"correct":sum(z["gold"] is not None and z["pred"][k]==z["gold"] for z in rows),"wrong":sum(z["gold"] is not None and z["pred"][k] is not None and z["pred"][k]!=z["gold"] for z in rows),"unknown_false":sum(z["gold"] is None and z["pred"][k] is not None for z in rows),"parent_lost":sum(z["gold"] is not None and z["pred"]["parent"]==z["gold"] and z["pred"][k]!=z["gold"] for z in rows),"groups":sum(all(rows[j]["gold"] is not None and rows[j]["pred"][k]==rows[j]["gold"] for j in group) for group in groups)}
    stats={k:metrics(details,k,ex["groups"]) for k in modes}
    def solve(ss,q,names,k):
        fs=[eval_t(t,names,k) for t in ss];a=eval_t(q,names,k)
        if a is None or any(v is None for v in fs):return "HOLD_UNPARSED"
        return g.solve({"schema":"janus.event.graph.v1","facts":[{"id":f"s{j}",**v} for j,v in enumerate(fs)],"claim":a})["status"]
    def graphs(cases,keys):
        out={k:{"correct":0,"false_certainty":0} for k in keys}
        for z in cases:
            for k in keys:
                a=solve(z["sentences"],z["claim"],z["names"],k);out[k]["correct"]+=a==z["expected"];out[k]["false_certainty"]+=a in ("ENTAILED","CONTRADICTED") and a!=z["expected"]
        return out
    for z in ex["graphs"]:
        for t in z["sentences"]+[z["claim"]]:
            n=e.norm(t,z["names"]);ck(n is None or n[0] not in refs,"GRAPH_OVERLAP")
    graph=graphs(ex["graphs"],modes)
    oldI=read(oldi);oldrows=[{**z,"pred":{k:eval_t(z["text"],z["names"],k) for k in modes}} for z in oldI["rows"]];replayI={k:metrics(oldrows,k,oldI["groups"]) for k in modes}
    oldgraph=graphs(oldI["graphs"],("parent","stack"));ck(replayI["parent"]["correct"]==24 and oldgraph["parent"]["correct"]==9,"I_PARITY")
    replayD=graphs([{**z,"names":namesD(z)} for z in prior],("legacy","parent","stack"));ck(replayD["legacy"]["correct"]==46,"D_PARITY")
    old=read(oldc)["cases"];ck(len(old)==76 and all(g.solve(z["request"])["status"]==z["expected"] for z in old),"C_PARITY")
    result={"schema":"h38j.result.v1","exam_sha":sha(exam),"known_n":sum(z["gold"] is not None for z in details),"unknown_n":sum(z["gold"] is None for z in details),"groups_n":len(ex["groups"]),"graph_n":len(ex["graphs"]),"routes":stats,"graphs":graph,"h38i_replay":replayI,"h38i_graph_replay":oldgraph,"h38d_replay":replayD,"h38c_replay":76,"cycles":r["cycles"],"verified_new":r["verified_new"],"memory_readback":len(newrows),"quality_gate":"HOLD","production":False,"scope":"Bounded synthetic self-curriculum, rule construction and exact three-obligation local verifier. No external truth, no original TRINITY execution, no persistent scheduling, no production. Internal fresh exam, supplied events."}
    write(out/"result.json",result);write(out/"cases.json",details);print("R84H38J_RESULT",json.dumps(result,sort_keys=True),flush=True)
if __name__=="__main__":{"train":train,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
