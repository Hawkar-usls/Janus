"""H38I: frozen failure decomposition, four-branch return receipts, supervised repair.
H38H disclosed labels may train this successor, NEVER its fresh exam.
"""
import sys,json,hashlib,time
from pathlib import Path
from collections import Counter
import torch
from torch import nn
from safetensors.torch import save_file,load_file
import r84h38h_semantic_kaleidoscope as h
import r84h38g_semantic_mosaic as m
import r84h38f_semantic_crystal as f
import r84h38e_semantic_invariance as e
import r84h38d_world_understanding as d
import r84h38c_semantic_world_graph as g
ck=f.ck;read=f.read;write=f.write;sha=f.sha
SLOTS=("relation","polarity","direction","certainty")
def load_h(p):
    r=read(p/"receipt.json");ck(sha(p/"bank.json")==r["bank_sha"],"H_BANK");ns={}
    for k,rows in r["runs"].items():
        ns[k]=[]
        for z in rows:
            q=p/f"{k}-{z['seed']}.safetensors";ck(sha(q)==z["sha"],"H_WEIGHT");net=h.Net();net.load_state_dict(load_file(str(q)));ns[k].append(net.eval())
    return r,read(p/"bank.json"),ns
def load_g(p,banksha):
    r=read(p/"receipt.json");ck(sha(p/"bank.json")==r["bank_sha"]==banksha,"G_BANK");ms=[]
    for z in r["runs"]:
        q=p/f"mosaic-{z['seed']}.safetensors";ck(sha(q)==z["sha"],"G_WEIGHT");net=m.Mosaic();net.load_state_dict(load_file(str(q)));ms.append(net.eval())
    return ms
def packet(slot,value,reasons,confidence=None,margin=None,children=None):
    p={"slot":slot,"value":value,"reasons":sorted(set(reasons)),"confidence":confidence,"margin":margin,"children":children or []}
    p["sha256"]=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest();return p
def verify(p):
    q={k:v for k,v in p.items() if k!="sha256"}
    return p["sha256"]==hashlib.sha256(json.dumps(q,sort_keys=True,separators=(",",":")).encode()).hexdigest() and all(verify(x) for x in p["children"])
def combine(nodes):
    ck([p["slot"] for p in nodes]==list(SLOTS) and all(verify(p) for p in nodes),"RETURN_INTEGRITY")
    reasons=[f"{p['slot']}:{r}" for p in nodes for r in p["reasons"]]
    v=[p["value"] for p in nodes];label=4*v[0]+2*v[1]+v[2] if not reasons else None
    return packet("crystal",label,reasons,children=nodes)
def relation(t,hr,ns,bank):
    c=h.candidates(t,ns,bank["train"]);j=h.decide(c,"mosaic",hr["rules"]["mosaic"]["rule"]);a,p,margin=c["mosaic"]
    return packet("relation",j,["REJECTED"] if j is None else [],p,margin)
def components(t,ms,rule):
    p=f.perrow(m.probs(ms,[t]),0);js,conf,mar=m.raw(p);return leaves(t,js,conf,mar,rule)
def leaves(t,js,conf,mar,rule):
    pol,di=m.cues(t);out=[]
    for i,name,cue in [(1,"polarity",pol),(2,"direction",di),(3,"certainty",1)]:
        checks=[]
        if cue is None:checks.append("CUE_UNDEFINED")
        elif js[i]!=cue:checks.append("MODEL_CUE_DISAGREE" if i<3 else "UNKNOWN_CLASS")
        if conf[i]<rule[0]:checks.append("LOW_CONFIDENCE")
        if mar[i]<rule[1]:checks.append("LOW_MARGIN")
        out.append(packet(name,js[i],checks,conf[i],mar[i],children=[packet("model",js[i],[],conf[i],mar[i]),packet("cue",cue,[] if cue is not None else ["UNDEFINED"])]))
    return out
def trace(t,hr,ns,bank,ms,rule=(.9,.25)):
    return combine([relation(t,hr,ns,bank),*components(t,ms,rule)])
def normalize(z):
    a=e.norm(z["text"],z["names"]);return a
def new_tiles(seed,rows):
    torch.manual_seed(seed);model=m.Mosaic();x=m.batch([z["text"] for z in rows]);y=torch.tensor([f.parts(z["y"]) for z in rows]);known=y[:,3]==1;start=time.process_time()
    # Freeze relation tile (unused); train only the three failed components.
    for i in range(1,4):
        tile=model.tiles[i];opt=torch.optim.AdamW(tile.parameters(),lr=.008);mask=known if i<3 else torch.ones_like(known)
        for _ in range(240):
            loss=nn.functional.cross_entropy(tile(x[i][mask]),y[mask,i]);opt.zero_grad();loss.backward();opt.step()
    return model.eval(),time.process_time()-start
def train(hdir,gdir,oldexam,out):
    g.witness();torch.set_num_threads(2);out.mkdir();hr,bank,ns=load_h(hdir);gm=load_g(gdir,hr["bank_sha"]);old=read(oldexam);ck(sha(oldexam)=="e45ed35da91c34517a75d982f902d1bb2a7f0af982a51a8b6f9ca8177832ecca","H_EXAM")
    diag=[];fail=[];counts=Counter();raw_errors=Counter();relation_n=correct=0
    for i,z in enumerate(old["rows"]):
        n=normalize(z)
        if n is None:continue
        t,events=n;p=trace(t,hr,ns,bank,gm);y=z["y"];pred=e.decode(p["value"],events) if p["value"] is not None else None
        if y<16 and p["children"][0]["value"]==h.rel(y):
            relation_n+=1
            if pred==z["gold"]:correct+=1
            else:
                ck(p["value"] is None,"OLD_FALSE_CERTAINTY")
                for reason in p["reasons"]:counts[reason]+=1
                for k,leaf in enumerate(p["children"]):
                    if k and leaf["value"]!=f.parts(y)[k]:raw_errors[SLOTS[k]]+=1
                fail.append({"text":t,"y":y,"origin":"DISCLOSED_H38H_GOLD","row":i,"return_sha":p["sha256"]})
        diag.append({"row":i,"text":z["text"],"normalized":t,"gold":z["gold"],"prediction":pred,"return":p})
    ck((relation_n,correct,len(fail))==(29,11,18),"H38H_PARITY")
    ck(all(verify(z["return"]) for z in diag),"TRACE")
    # Tamper one child while leaving the original parent hash.
    tamper=json.loads(json.dumps(diag[0]["return"]));tamper["children"][1]["value"]=99;ck(not verify(tamper),"TAMPER")
    base=bank["train"];dev=bank["dev"];known={z["text"] for z in base+dev};repair=[z for z in fail if z["text"] not in known];ck(len(repair)==len(fail) and len({z["text"] for z in repair})==18,"REPAIR_OVERLAP")
    runs={};rules={}
    for mode,rows in [("control",base),("repair",base+repair)]:
        runs[mode]=[];ms=[]
        for seed in (393,394):
            net,cpu=new_tiles(seed,rows);p=out/f"{mode}-{seed}.safetensors";save_file(net.state_dict(),str(p));ms.append(net);runs[mode].append({"seed":seed,"sha":sha(p),"cpu":cpu,"trained_parameters":sum(v.numel() for tile in net.tiles[1:] for v in tile.parameters())})
        # Both arms use same calibration, never the new exam.
        pool=[]
        for conf in (.5,.65,.8,.9,.95,1.01):
            for margin in (0,.1,.25,.4):
                rule=(conf,margin);pred=[trace(z["text"],hr,ns,bank,ms,rule)["value"] for z in dev]
                good=sum(a is not None and a==z["y"] for a,z in zip(pred,dev));bad=sum(a is not None and a!=z["y"] for a,z in zip(pred,dev))
                if not bad:pool.append((good,rule))
        good,rule=max(pool);rules[mode]={"rule":rule,"dev_correct":good}
    result={"schema":"h38i.train.v1","h38h_exam_sha":sha(oldexam),"bank_sha":hr["bank_sha"],"relation_correct":relation_n,"crystal_correct":correct,"diagnosed":len(fail),"reasons_nonexclusive":dict(counts),"raw_component_errors_nonexclusive":dict(raw_errors),"train_control_n":len(base),"train_repair_n":len(base+repair),"dev_n":len(dev),"runs":runs,"rules":rules,"return_tamper_rejected":True,"new_exam_read":False,"production":False}
    write(out/"receipt.json",result);write(out/"diagnostic.json",diag);write(out/"repair_rows.json",repair);print("R84H38I_TRAIN",json.dumps(result,sort_keys=True),flush=True)
def evaluate(hdir,gdir,edir,demi,trained,exam,oldc,oldd,olde,oldf,oldg,oldh,out):
    g.witness();torch.set_num_threads(2);out.mkdir();hr,bank,ns=load_h(hdir);gm=load_g(gdir,hr["bank_sha"]);r=read(trained/"receipt.json");ck(r["bank_sha"]==hr["bank_sha"] and r["h38h_exam_sha"]==sha(oldh),"TRAIN_PINS")
    nets={"frozen":gm};rules={"frozen":(.9,.25)}
    for mode,rows in r["runs"].items():
        nets[mode]=[];rules[mode]=r["rules"][mode]["rule"]
        for z in rows:
            p=trained/f"{mode}-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"REPAIR_WEIGHT");net=m.Mosaic();net.load_state_dict(load_file(str(p)));nets[mode].append(net.eval())
    er=read(edir/"receipt.json");eb=read(edir/"bank.json");ck(sha(edir/"bank.json")==er["bank_sha"],"E_BANK");em=[]
    for z in er["runs"]:
        p=edir/f"model-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"E_WEIGHT");net=e.Net();net.load_state_dict(load_file(str(p)));em.append(net.eval())
    di=e.Dictionary(e.cortex(demi),edir/"cortex",eb["train"]);cache={}
    def eval_t(t,names,mode):
        n=e.norm(t,names)
        if mode in ("legacy","stack"):
            a=d.parse(t)
            if a is not None:return a,None
        if n is None:return None,None
        text,events=n
        if mode in ("legacy","stack"):
            j=e.pick(e.probs(em,[text])[0],di.search(text),er["rules"]["agreement"],"agreement")
            if j is not None:return e.decode(j,events),None
            if mode=="legacy":return None,None
        def p(k):
            if (text,k) not in cache:cache[text,k]=trace(text,hr,ns,bank,nets[k],rules[k])
            return cache[text,k]
        if mode=="stack":
            q=p("frozen")
            if q["value"] is None:q=p("repair")
        else:q=p(mode)
        ck(verify(q),"RETURN")
        return (e.decode(q["value"],events) if q["value"] is not None else None),q
    refs={z["text"] for z in bank["train"]+bank["dev"]+eb["train"]+eb["dev"]+read(trained/"repair_rows.json")}
    prior=read(oldd)["cases"]
    def namesD(z):return sorted({a[k] for a in z["atoms"]+[z["claim_atom"]] if a for k in ("left","right")})
    for z in prior:
        for t in z["sentences"]+[z["claim"]]:
            n=e.norm(t,namesD(z))
            if n:refs.add(n[0])
    for p in (olde,oldf,oldg,oldh):
        for z in read(p)["rows"]:
            n=normalize(z)
            if n:refs.add(n[0])
    ex=read(exam);modes=("legacy","frozen","control","repair","stack");details=[]
    for z in ex["rows"]:
        n=normalize(z);ck(n is None or n[0] not in refs,"FRESH_OVERLAP")
        if z["y"]<16:ck(n is not None and e.decode(z["y"],n[1])==z["gold"],"GOLD")
        pairs={k:eval_t(z["text"],z["names"],k) for k in modes};details.append({**z,"pred":{k:v[0] for k,v in pairs.items()},"return":{k:v[1] for k,v in pairs.items()}})
    def metrics(rows,k,groups):
        s={"correct":0,"wrong":0,"unknown_false":0,"groups":0,"frozen_lost":0}
        for z in rows:
            a=z["pred"][k];gold=z["gold"];s["correct"]+=gold is not None and a==gold;s["wrong"]+=gold is not None and a is not None and a!=gold;s["unknown_false"]+=gold is None and a is not None;s["frozen_lost"]+=gold is not None and z["pred"]["frozen"]==gold and a!=gold
        s["groups"]=sum(all(rows[i]["gold"] is not None and rows[i]["pred"][k]==rows[i]["gold"] for i in group) for group in groups);return s
    stats={k:metrics(details,k,ex["groups"]) for k in modes}
    def solve(ss,q,names,k):
        fs=[eval_t(t,names,k)[0] for t in ss];a=eval_t(q,names,k)[0]
        if a is None or any(v is None for v in fs):return "HOLD_UNPARSED"
        return g.solve({"schema":"janus.event.graph.v1","facts":[{"id":f"s{i}",**v} for i,v in enumerate(fs)],"claim":a})["status"]
    graph={k:{"correct":0,"false_certainty":0} for k in modes}
    for z in ex["graphs"]:
        for t in z["sentences"]+[z["claim"]]:
            n=e.norm(t,z["names"]);ck(n is None or n[0] not in refs,"GRAPH_OVERLAP")
        for k in modes:
            a=solve(z["sentences"],z["claim"],z["names"],k);graph[k]["correct"]+=a==z["expected"];graph[k]["false_certainty"]+=a in ("ENTAILED","CONTRADICTED") and a!=z["expected"]
    replay={k:{"correct":0,"false_certainty":0,"legacy_lost":0} for k in ("legacy","stack")}
    for z in prior:
        names=namesD(z);b=solve(z["sentences"],z["claim"],names,"legacy")
        for k in replay:
            a=solve(z["sentences"],z["claim"],names,k);replay[k]["correct"]+=a==z["expected"];replay[k]["false_certainty"]+=a in ("ENTAILED","CONTRADICTED") and a!=z["expected"];replay[k]["legacy_lost"]+=b==z["expected"] and a!=z["expected"]
    ck(replay["legacy"]["correct"]==46,"LEGACY_REPLAY");old=read(oldc)["cases"];ck(len(old)==76 and all(g.solve(z["request"])["status"]==z["expected"] for z in old),"H38C_REPLAY")
    oldrows=read(oldh)["rows"];oldstats={}
    for k in modes:
        oldstats[k]={"correct":sum(z["gold"] is not None and eval_t(z["text"],z["names"],k)[0]==z["gold"] for z in oldrows),"wrong":sum((a:=eval_t(z["text"],z["names"],k)[0]) is not None and a!=z["gold"] for z in oldrows)}
    ck(oldstats["frozen"]["correct"]==11 and oldstats["frozen"]["wrong"]==0,"H38H_REPLAY")
    report={"schema":"h38i.result.v1","exam_sha":sha(exam),"known_n":sum(z["gold"] is not None for z in details),"unknown_n":sum(z["gold"] is None for z in details),"groups_n":len(ex["groups"]),"graph_n":len(ex["graphs"]),"routes":stats,"graphs":graph,"h38h_disclosed_replay":oldstats,"h38d_replay":replay,"h38c_replay":76,"diagnostic":{k:r[k] for k in ("diagnosed","reasons_nonexclusive","raw_component_errors_nonexclusive","return_tamper_rejected")},"cortex_queries":di.calls,"quality_gate":"HOLD","production":False,"scope":"Four-slot hierarchical Genesis-Return with nested model/cue receipts; 18 disclosed H38H gold labels added to repair arm, not self-generated truth. Control has fewer training examples, equal seeds and updates. Fresh same-author exam with supplied event inventory. No independent external validation or automatic self-training."}
    write(out/"result.json",report);write(out/"cases.json",details);print("R84H38I_RESULT",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":{"train":train,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
