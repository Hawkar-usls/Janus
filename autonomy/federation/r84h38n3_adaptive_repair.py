"""H38N3: diagnose representation conflicts, avoid failed strategies, bounded adaptive learning."""
import sys,json,hashlib,copy,time,math
from pathlib import Path
from collections import Counter,defaultdict
import torch
from torch import nn
from safetensors.torch import save_file,load_file
import r84h38n2_neural_symbiosis as n2
import r84h38n1_semantic_recovery as n1
import r84h38n_semantic_correction as n0
import r84h38j_fractal_self_learning as j
import r84h38i_fractal_reconstruction as i
import r84h38h_semantic_kaleidoscope as h
import r84h38e_semantic_invariance as e
import r84h38d_world_understanding as d
import r84h38c_semantic_world_graph as g
import r84h38f_semantic_crystal as f
ck=f.ck;read=f.read;write=f.write;sha=f.sha
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def feat(t,mode,mirror=False):
    # Position-sensitive cues and explicit view tag; no gold information in features.
    x=h.features(t).clone();w=t.split()
    add=["pos:"+str(k)+":"+v for k,v in enumerate(w)]
    add+=["last:"+str(k)+":"+v for k,v in enumerate(reversed(w))]
    add+=["role:"+str(k)+":"+v for k,v in enumerate(w) if v in ("evone","evtwo")]
    if mode=="TAGGED_MIRROR":
        flag="reverse:" if mirror else "forward:"
        add+=[flag+v for v in w]+[flag+" ".join(w[max(0,k-1):k+2]) for k in range(len(w))]
        add+=[flag+"view"]*4
    for v in add:x[int.from_bytes(hashlib.blake2s(v.encode(),digest_size=4).digest(),"little")%len(x)]+=.25
    return x/x.norm().clamp_min(1e-8)
def tile():return nn.Sequential(nn.Linear(e.B,16),nn.GELU(),nn.Linear(16,2))
def collision_audit(rows,mode):
    mapping=defaultdict(list)
    for z in rows:
        if z["y"]>=16:continue
        t=z["text"];y=z["y"]%2
        views=[(t,y,False),(n1.reverse(t),1-y,True)] if mode!="DIRECT_POSITION" else [(t,y,False)]
        for x,v,b in views:
            key=(h.features(x) if mode=="LEGACY" else feat(x,mode,b)).numpy().tobytes()
            mapping[hashlib.sha256(key).hexdigest()].append({"text":t,"view":x,"mirror":b,"target":v})
    bad=[{"feature_sha":k,"rows":v} for k,v in mapping.items() if len({z["target"] for z in v})>1]
    return bad
def predict(child,t,mode):
    if mode=="DIRECT_POSITION":
        x=feat(t,mode).unsqueeze(0)
        with torch.inference_mode():
            ps=torch.stack([n(x).softmax(-1)[0] for n in child]).min(0).values
        # Canonicalized reverse gives the same prediction mapped back; no independent vote.
        return ps,ps.flip(0)
    x=torch.stack([feat(t,mode),feat(n1.reverse(t),mode,True)])
    with torch.inference_mode():
        ps=torch.stack([n(x).softmax(-1) for n in child]).min(0).values
    return ps[0],ps[1]
def components(parent,child,t,mode):
    ps=n1.probs(parent,[t,n1.reverse(t)]);a,b=predict(child,t,mode)
    ps[2]=torch.stack([a,b]);return n1.raw([p[0] for p in ps]),n1.raw([p[1] for p in ps])
def raw_label(a):return 4*a[0][0]+2*a[1][0]+a[2][0] if a[3][0]==1 else None
def mirror_admit(a,b,rule):
    x=n1.admit(a,rule);y=n1.admit(b,rule)
    return x if x is not None and x==n1.mirror_label(y) else None
def adaptive(parent,child,t,r):
    a,b=components(parent,child,t,r["selected"]);v=mirror_admit(a,b,r["rule"])
    leaves=[i.packet(s,a[k][0],[] if a[k][0]==(1-b[k][0] if k==2 else b[k][0]) else ["VIEW_DISAGREEMENT"],a[k][1],a[k][2],children=[i.packet("mirror",b[k][0],[])]) for k,s in enumerate(n1.SLOTS)]
    p=i.packet("GENESIS_RETURN",{"candidate":v,"strategy":r["selected"],"truth_verified":False},[] if v is not None else ["HOLD"],children=leaves);ck(i.verify(p),"RETURN")
    return {"adaptive_raw":raw_label(a),"adaptive_mirror":v},p
def load_child(root,r):
    out=[]
    for z in r["runs"]:
        p=root/z["file"];ck(sha(p)==z["sha"],"CHILD_HASH");n=tile();n.load_state_dict(load_file(str(p)));out.append(n.eval())
    return out
def train(hdir,idir,ldir,ndir,fac,pre,prior,registry,spiral,out):
    g.witness();torch.set_num_threads(2);out.mkdir();p=read(pre);ck(not p["overlap"] and p["n_receipt_sha"]==sha(ndir/"receipt.json") and p["training_refs_sha"]==sha(ndir/"training_refs.json"),"PREFLIGHT")
    ck(e.blob(spiral)==n2.SPIRAL,"SPIRAL_PIN")
    old=read(prior/"receipt.json");record=read(registry)
    ck(record["run_id"]==45845651569 and old["decision"]=="ROLLBACK_KEEP_FROZEN" and old["component_errors_before"]==old["component_errors_after"]=={"direction":5},"PRIOR_FAILURE")
    ck(any(z.get("tag")=="R84H38N2_TRAIN" and z["data"]==old for z in record["receipts"]),"REGISTRY_RECEIPT")
    rows,dev=n1.training_rows(hdir,idir,ldir,ndir);fr=read(fac/"receipt.json");ck(old["parent_sha"]==sha(fac/"receipt.json"),"PARENT_PIN")
    fs=n1.load_factor(fac,fr);di,counts=n2.diagnose(fs,rows);ck(di==read(prior/"diagnostics.json") and counts=={"direction":5},"DIAG_REPLAY")
    known=[z for z in rows if z["y"]<16];legacy=collision_audit(rows,"LEGACY");memory={"schema":"h38n3.error_memory.v1","parent_receipt":sha(prior/"receipt.json"),"registry_receipt":sha(registry),"errors":di,"legacy_feature_collisions":legacy,"strategies":[{"id":"LEGACY_WARM_REWEIGHT","status":"REJECTED_NO_GAIN","errors_before":5,"errors_after":5}],"production":False}
    baseps=n1.probs(fs,[z["text"] for z in dev]);baser=n1.probs(fs,[n1.reverse(z["text"]) for z in dev]);br=fr["rules"]["mirror"]
    basepred=[mirror_admit(n1.raw([p[k] for p in baseps]),n1.raw([p[k] for p in baser]),br) for k in range(len(dev))]
    basegood={k for k,(a,z) in enumerate(zip(basepred,dev)) if a is not None and a==z["y"]}
    base_direction={k for k,z in enumerate(dev) if z["y"]<16 and int(baseps[2][k].argmax())==z["y"]%2}
    attempts=[];models={}
    for mode in ("DIRECT_POSITION","TAGGED_MIRROR"):
        conflicts=collision_audit(rows,mode);ts=[];ys=[]
        for z in known:
            t=z["text"];y=z["y"]%2
            ts.append(feat(t,mode));ys.append(y)
            if mode=="TAGGED_MIRROR":ts.append(feat(n1.reverse(t),mode,True));ys.append(1-y)
        x=torch.stack(ts);y=torch.tensor(ys);children=[];runs=[]
        for seed in (395,396):
            torch.manual_seed(seed);net=tile();opt=torch.optim.AdamW(net.parameters(),lr=.006);start=time.process_time()
            for _ in range(300):
                loss=nn.functional.cross_entropy(net(x),y);opt.zero_grad();loss.backward();opt.step()
            children.append(net.eval());runs.append({"seed":seed,"cpu":time.process_time()-start})
        devpairs=[components(fs,children,z["text"],mode) for z in dev]
        direction={k for k,(z,(a,b)) in enumerate(zip(dev,devpairs)) if z["y"]<16 and a[2][0]==z["y"]%2 and b[2][0]==1-z["y"]%2}
        bp=[]
        for c in (.35,.5,.65,.8,.9,.95,1.01):
            for margin in (0,.1,.25,.4):
                rule=(c,margin);pred=[mirror_admit(a,b,rule) for a,b in devpairs]
                good={k for k,(v,z) in enumerate(zip(pred,dev)) if v is not None and v==z["y"]};wrong=sum(v is not None and v!=z["y"] for v,z in zip(pred,dev))
                if not wrong:bp.append((len(good&basegood),len(good),rule,sorted(good)))
        _,_,rule,good=max(bp);lost=sorted(basegood-set(good));errors=[]
        for z in known:
            a,b=predict(children,z["text"],mode)
            if int(a.argmax())!=z["y"]%2 or int(b.argmax())!=1-z["y"]%2:errors.append(z["text"])
        attempt={"id":mode,"feature_conflicts":len(conflicts),"training_errors":len(errors),"training_error_texts":errors,"dev_direction_correct":len(direction),"dev_direction_lost":sorted(base_direction-direction),"dev_admitted_correct":good,"dev_lost":lost,"dev_wrong":0,"rule":rule,"runs":runs}
        attempts.append(attempt);models[mode]=children;memory["strategies"].append(attempt)
    # Rank only by calibration, before exam. Require no lost admissions and strict gain.
    winner=max(attempts,key=lambda z:(not z["dev_lost"],len(z["dev_admitted_correct"]),z["dev_direction_correct"],-z["feature_conflicts"],-z["training_errors"]))
    accept=not winner["dev_lost"] and len(winner["dev_admitted_correct"])>len(basegood)
    mode=winner["id"];runs=[]
    for n,z in zip(models[mode],winner["runs"]):
        name=f"direction-{z['seed']}.safetensors";q=out/name;save_file(n.state_dict(),str(q));runs.append({**z,"file":name,"sha":sha(q)})
    memory["selected"]=mode;memory["decision"]="SHADOW_CANDIDATE" if accept else "ROLLBACK_KEEP_FROZEN";memory["sha256"]=digest(memory)
    write(out/"error_memory.json",memory)
    # Persist resumable ledger, separate source discovery request; no generated labels from sources.
    q={"schema":"h38n3.gap.v1","failed_slots":["direction"],"terms":["semantic role labeling","relation extraction","contrastive learning","word order"],"purpose":"SEARCH_METADATA_ONLY_NOT_TRUTH","memory_sha":memory["sha256"]}
    write(out/"gap_request.json",q)
    rec={"schema":"h38n3.train.v1","preflight_sha":sha(pre),"parent_sha":sha(fac/"receipt.json"),"joint_sha":sha(ndir/"receipt.json"),"prior_failed_sha":sha(prior/"receipt.json"),"memory_sha":memory["sha256"],"legacy_feature_conflicts":len(legacy),"legacy_collision_examples":legacy[:3],"base_dev_correct":sorted(basegood),"base_dev_direction_correct":len(base_direction),"attempts":attempts,"selected":mode,"rule":winner["rule"],"decision":memory["decision"],"runs":runs,"exam_read":False,"production":False}
    write(out/"receipt.json",rec);print("R84H38N3_TRAIN",json.dumps(rec,sort_keys=True),flush=True)

def evaluate(hdir,gdir,idir,edir,ldir,demi,trained,factor,preflight_receipt,repair,exam,oldc,oldd,olde,oldf,oldg,oldh,oldi,oldj,oldl,oldn1,oldn2,out):
    g.witness();torch.set_num_threads(2);out.mkdir();hr,bank,ns=i.load_h(hdir);gm=i.load_g(gdir,hr["bank_sha"]);ir,im=j.load_i(idir);lr=read(ldir/"receipt.json");r=read(trained/"receipt.json");ck(r["bank_sha"]==hr["bank_sha"] and r["l_receipt_sha"]==sha(ldir/"receipt.json"),"PINS")
    active=j.load_models(ldir,lr["active"]);rule=lr["rule"];er=read(edir/"receipt.json");eb=read(edir/"bank.json");ck(sha(edir/"bank.json")==er["bank_sha"],"E_BANK");em=[]
    for z in er["runs"]:
        p=edir/f"model-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"E_WEIGHT");n=e.Net();n.load_state_dict(load_file(str(p)));em.append(n.eval())
    di=e.Dictionary(e.cortex(demi),edir/"cortex",eb["train"]);nets={}
    for k,rows in r["runs"].items():
        nets[k]=[]
        for z in rows:
            p=trained/f"{k}-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"N_WEIGHT");n=n0.Net();n.load_state_dict(load_file(str(p)));nets[k].append(n.eval())
    fr=read(factor/"receipt.json");pr=read(preflight_receipt);ck(not pr["overlap"] and pr["exam_sha"]==sha(exam) and fr["n_receipt_sha"]==sha(trained/"receipt.json"),"PREFLIGHT_PIN")
    fs=n1.load_factor(factor,fr);rr=read(repair/"receipt.json");ck(rr["preflight_sha"]==sha(preflight_receipt) and rr["parent_sha"]==sha(factor/"receipt.json") and rr["joint_sha"]==sha(trained/"receipt.json"),"REPAIR_PIN")
    child=load_child(repair,rr);cache={};returns={}
    def allpred(t,names):
        direct=d.parse(t);n=e.norm(t,names)
        if n is None:return {k:direct if k in ("parent","guard","stack","factor_stack","hybrid","learned","revision") else None for k in modes}
        text,events=n
        if text not in cache:
            ej=e.pick(e.probs(em,[text])[0],di.search(text),er["rules"]["agreement"],"agreement")
            qs=[i.trace(text,hr,ns,bank,ms,rule if ms is not gm else (.9,.25)) for ms in (gm,im,active)]
            old=ej if ej is not None else next((q["value"] for q in qs if q["value"] is not None),None)
            ps={k:n0.probabilities(v,[text])[0] for k,v in nets.items()}
            ys={k:n0.select(p,r["rules"][k]["rule"]) for k,p in ps.items()}
            cache[text]=(old,ys,{k:h.top(v)[0] for k,v in ps.items()},n0.cue(text),n1.factor_score(fs,text,fr),adaptive(fs,child,text,rr))
        old,ys,raw,role,fac,ad=cache[text];parent=direct if direct is not None else e.decode(old,events) if old is not None else None
        def agrees(a):
            return role is not None and a is not None and (e.REL.index(a["rel"]),events.index(a["left"]))==role
        v={k:e.decode(y,events) if y is not None else None for k,y in ys.items()}
        guard=parent if role is None or agrees(parent) else None
        # A known exact grammar answer remains authoritative; otherwise veto
        # contradictory old inference and require learned agreement to fill.
        if direct is not None:guard=direct
        stack=guard if guard is not None else v["diverse"] if agrees(v["diverse"]) else None
        fa={k:e.decode(y,events) if y is not None else None for k,y in fac.items()}
        corrected=guard if guard is not None else fa["factor_mirror"] if agrees(fa["factor_mirror"]) else None
        av,packet=ad;returns[text]=packet
        av={k:e.decode(y,events) if y is not None else None for k,y in av.items()}
        sv=av["adaptive_mirror"];cv=av["adaptive_raw"]
        hybrid=corrected if corrected is not None else sv if agrees(sv) else None
        learned=corrected if corrected is not None else cv if agrees(cv) else None
        revision=corrected if direct is not None or sv is None or not agrees(sv) else sv
        return {"parent":parent,"control":v["control"],"diverse":v["diverse"],"guard":guard,"stack":stack,"factor_stack":corrected,"fusion":sv,"hybrid":hybrid,"learned":learned,"revision":revision,**av,**fa,"control_raw":e.decode(raw["control"],events) if raw["control"]<16 else None,"diverse_raw":e.decode(raw["diverse"],events) if raw["diverse"]<16 else None}
    modes=("parent","control","diverse","guard","stack","control_raw","diverse_raw","factor_raw","factor_direct","factor_mirror","factor_stack","fusion","hybrid","learned","revision","adaptive_raw","adaptive_mirror");ex=read(exam);refs=set(read(trained/"training_refs.json"))|{z["text"] for z in eb["train"]+eb["dev"]};prior=read(oldd)["cases"]
    def namesD(z):return sorted({a[k] for a in z["atoms"]+[z["claim_atom"]] if a for k in ("left","right")})
    for z in prior:
        for t in z["sentences"]+[z["claim"]]:
            n=e.norm(t,namesD(z))
            if n:refs.add(n[0])
    for p in (olde,oldf,oldg,oldh,oldi,oldj,oldl):
        x=read(p)
        for z in x["rows"]:
            n=e.norm(z["text"],z["names"])
            if n:refs.add(n[0])
        for z in x.get("graphs",[]):
            for t in z["sentences"]+[z["claim"]]:
                n=e.norm(t,z["names"])
                if n:refs.add(n[0])
    def fresh(t,names):
        n=e.norm(t,names);ck(n is None or n[0] not in refs,"FRESH_OVERLAP:"+str(n))
    def rows(x,check=False):
        result=[]
        for z in x["rows"]:
            if check:
                fresh(z["text"],z["names"]);n=e.norm(z["text"],z["names"])
                if z["y"]<16:ck(n and e.decode(z["y"],n[1])==z["gold"],"GOLD")
            result.append({**z,"pred":allpred(z["text"],z["names"])})
        return result
    def metrics(rs,k,groups):
        ans={"correct":0,"wrong":0,"unknown_false":0,"lost":0,"relation_correct":0,"direction_correct":0,"groups":0}
        for z in rs:
            a=z["pred"][k];b=z["gold"]
            ans["correct"]+=b is not None and a==b;ans["wrong"]+=b is not None and a is not None and a!=b;ans["unknown_false"]+=b is None and a is not None;ans["lost"]+=b is not None and z["pred"]["parent"]==b and a!=b
            ans["relation_correct"]+=b is not None and a is not None and a["rel"]==b["rel"]
            ans["direction_correct"]+=b is not None and a is not None and (a["left"],a["right"])==(b["left"],b["right"])
        ans["groups"]=sum(all(rs[x]["gold"] is not None and rs[x]["pred"][k]==rs[x]["gold"] for x in group) for group in groups);return ans
    def graph(cases,keys,check=False):
        ans={k:{"correct":0,"false_certainty":0,"lost":0} for k in keys}
        for z in cases:
            if check:
                for t in z["sentences"]+[z["claim"]]:fresh(t,z["names"])
            ps=[allpred(t,z["names"]) for t in z["sentences"]+[z["claim"]]];verdict={}
            for k in keys:
                vals=[p[k] for p in ps];verdict[k]="HOLD_UNPARSED" if any(v is None for v in vals) else n2.checked_graph(vals)
                a=verdict[k];ans[k]["correct"]+=a==z["expected"];ans[k]["false_certainty"]+=a in ("ENTAILED","CONTRADICTED") and a!=z["expected"]
            for k in keys:ans[k]["lost"]+=verdict["parent"]==z["expected"] and verdict[k]!=z["expected"]
        return ans
    rs=rows(ex,True);stats={k:metrics(rs,k,ex["groups"]) for k in modes};gs=graph(ex["graphs"],modes,True)
    il=read(oldi);ll=read(oldl);replayI=rows(il);replayL=rows(ll);hist={k:{"I":metrics(replayI,k,il["groups"]),"L":metrics(replayL,k,ll["groups"])} for k in ("parent","guard","stack","factor_stack","hybrid","learned","revision")}
    histgraph={"I":graph(il["graphs"],("parent","guard","stack","factor_stack","hybrid","learned","revision")),"L":graph(ll["graphs"],("parent","guard","stack","factor_stack","hybrid","learned","revision")),"D":graph([{**z,"names":namesD(z)} for z in prior],("parent","guard","stack","factor_stack","hybrid","learned","revision"))}
    ck(hist["parent"]["I"]["correct"]==24 and hist["parent"]["L"]["correct"]==11 and histgraph["I"]["parent"]["correct"]==9 and histgraph["D"]["parent"]["correct"]==46,"PARENT_PARITY")
    old=read(oldc)["cases"];ck(len(old)==76 and all(g.solve(z["request"])["status"]==z["expected"] for z in old),"C_REPLAY")
    err=[z for z in replayL if z["pred"]["parent"] is not None and z["pred"]["parent"]!=z["gold"]];ck(len(err)==1,"ERROR_PARITY")
    prev=read(oldn1);rn=rows(prev);keys=("factor_stack","hybrid","learned","revision")
    replayN1={k:metrics(rn,k,prev["groups"]) for k in keys}
    ck(replayN1["factor_stack"]["correct"]==12 and replayN1["factor_stack"]["wrong"]==1 and hist["factor_stack"]["L"]["correct"]==13,"N1_PARITY")
    def relative(rs,k):return sum(z["gold"] is not None and z["pred"]["factor_stack"]==z["gold"] and z["pred"][k]!=z["gold"] for z in rs)
    losses={k:{"fresh":relative(rs,k),"I":relative(replayI,k),"L":relative(replayL,k),"N1":relative(rn,k)} for k in keys}
    histgraph["N1"]=graph(prev["graphs"],("parent",)+keys)
    pn2=read(oldn2);rn2=rows(pn2);replayN2={k:metrics(rn2,k,pn2["groups"]) for k in keys}
    ck(replayN2["factor_stack"]["correct"]==9 and replayN2["factor_stack"]["wrong"]==0,"N2_PARITY")
    histgraph["N2"]=graph(pn2["graphs"],("parent",)+keys)
    for k in keys:losses[k]["N2"]=relative(rn2,k)
    write(out/"return_receipts.json",returns)
    audit=[{"text":z["text"],"gold":z["gold"],"prediction":z["pred"]["learned"],"training_allowed":False} for z in rs if z["gold"] is not None and z["pred"]["learned"]==z["gold"]]
    write(out/"verified_audit.json",audit)
    report={"schema":"h38n3.result.v1","n2_replay":replayN2,"n1_replay":replayN1,"relative_losses":losses,"return_n":len(returns),"audit_n":len(audit),"repair":rr,"exam_sha":sha(exam),"known_n":32,"unknown_n":12,"routes":stats,"graphs":gs,"historical":hist,"historical_graphs":histgraph,"h38c_replay":76,"disclosed_error":err[0],"quality_gate":"HOLD","preflight_sha":sha(preflight_receipt),"factor_receipt_sha":sha(factor/"receipt.json"),"production":False,"scope":"Same-author fresh exam, adaptive direction repair vs frozen baseline; learned mirror views correlated; no external semantic oracle or promotion."}
    write(out/"result.json",report);write(out/"cases.json",rs);print("R84H38N3_RESULT",json.dumps(report,sort_keys=True),flush=True)



if __name__=="__main__":{"preflight":n1.preflight,"train":train,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
