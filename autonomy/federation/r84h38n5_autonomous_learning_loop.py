"""H38N4: preserve exact error memory; combine positional and tagged direction models."""
import sys,json,hashlib,copy,time
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
import r84h38n3_adaptive_repair as n3
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
MODES=("DIRECT_POSITION","TAGGED_MIRROR")
from r84h38n4_experience_crystallization import load_child, adaptive, candidate, choose

def evaluate(hdir,gdir,idir,edir,ldir,demi,trained,factor,preflight_receipt,repair,exam,oldc,oldd,olde,oldf,oldg,oldh,oldi,oldj,oldl,oldn1,oldn2,oldn3,out):
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
            cache[text]=(old,ys,{k:h.top(v)[0] for k,v in ps.items()},n0.cue(text),n1.factor_score(fs,text,fr),adaptive(fs,child,nets['diverse'],text,rr))
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
        learned=corrected if corrected is not None else sv if agrees(sv) else None
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
    pn3=read(oldn3);rn3=rows(pn3);replayN3={k:metrics(rn3,k,pn3["groups"]) for k in keys}
    ck(replayN3["factor_stack"]["correct"]==5 and replayN3["factor_stack"]["wrong"]==0,"N3_PARITY")
    histgraph["N3"]=graph(pn3["graphs"],("parent",)+keys)
    for k in keys:losses[k]["N3"]=relative(rn3,k)
    comparisons=[(stats,gs),(replayN1,histgraph["N1"]),(replayN2,histgraph["N2"]),(replayN3,histgraph["N3"])]
    comparisons += [({k:hist[k][v] for k in keys},histgraph[v]) for v in ("I","L")]
    gate=all(m["hybrid"]["wrong"]<=m["factor_stack"]["wrong"] and m["hybrid"]["unknown_false"]<=m["factor_stack"]["unknown_false"] and q["hybrid"]["false_certainty"]<=q["factor_stack"]["false_certainty"] and q["hybrid"]["correct"]>=q["factor_stack"]["correct"] for m,q in comparisons) and all(v==0 for v in losses["hybrid"].values()) and stats["hybrid"]["correct"]>stats["factor_stack"]["correct"]
    # Reopen historical cases, not training labels. Exact old metrics are replayed.
    def audit_case(z,scope):
        n=e.norm(z["text"],z["names"]);packet=returns.get(n[0]) if n else None
        p=z["pred"];gold=z["gold"];bad=p["hybrid"]
        fields=[] if gold is None or bad is None else [k for k in ("rel","positive","left","right") if gold[k]!=bad[k]]
        return {"scope":scope,"text":z["text"],"gold":gold,"pred":p,"differing_fields":fields,"origin_route":"INHERITED" if p["factor_stack"]==bad else "NEW_CONSENSUS","normalized":n[0] if n else None,"role_cue":n0.cue(n[0]) if n else None,"packet":packet,"training_allowed":False}
    introduced=[audit_case(z,"H38N1_DISCLOSED") for z in rn if z["pred"]["hybrid"] is not None and z["pred"]["hybrid"]!=z["gold"] and z["pred"]["factor_stack"]!=z["pred"]["hybrid"]]
    fresh_wrong=[audit_case(z,"H38N4A_DISCLOSED") for z in rs if z["pred"]["hybrid"] is not None and z["pred"]["hybrid"]!=z["gold"]]
    ck(len(introduced)==1 and len(fresh_wrong)==1,"ERROR_PARITY_N4")
    gains=[audit_case(z,scope) for scope,items in (("I",replayI),("L",replayL),("N1",rn),("N2",rn2),("N3",rn3),("N4A",rs)) for z in items if z["gold"] is not None and z["pred"]["hybrid"]==z["gold"] and z["pred"]["factor_stack"]!=z["gold"]]
    ck(len(gains)==7,"GAIN_PARITY")
    diagnostic={"schema":"h38n5.diagnosis.v1","introduced":introduced,"fresh_wrong":fresh_wrong,"historical_gains":gains,"scope":"Disclosed evaluation evidence, not gradient data or proof of internal causality.","training_allowed":False,"production":False}
    write(out/"diagnosis.json",diagnostic);print("R84H38N5_DIAGNOSIS",json.dumps(diagnostic,sort_keys=True),flush=True)
    write(out/"return_receipts.json",returns)
    audit=[{"text":z["text"],"gold":z["gold"],"prediction":z["pred"]["learned"],"training_allowed":False} for z in rs if z["gold"] is not None and z["pred"]["learned"]==z["gold"]]
    write(out/"verified_audit.json",audit)
    report={"schema":"h38n4.result.v1","n3_replay":replayN3,"admission":"SHADOW_PASS" if gate else "ROLLBACK_KEEP_FROZEN","n2_replay":replayN2,"n1_replay":replayN1,"relative_losses":losses,"return_n":len(returns),"audit_n":len(audit),"repair":rr,"exam_sha":sha(exam),"known_n":32,"unknown_n":12,"routes":stats,"graphs":gs,"historical":hist,"historical_graphs":histgraph,"h38c_replay":76,"disclosed_error":err[0],"quality_gate":"HOLD","preflight_sha":sha(preflight_receipt),"factor_receipt_sha":sha(factor/"receipt.json"),"production":False,"scope":"Same-author fresh exam; consensus direction and joint class veto; no external semantic oracle or production promotion."}
    write(out/"result.json",report);write(out/"cases.json",rs);print("R84H38N5_REPLAY",json.dumps(report,sort_keys=True),flush=True)





def cortex_witness(demi,state_root,registry_mem,out):
    c=e.cortex(demi);raw=registry_mem.read_bytes();mem=json.loads(raw);body={k:v for k,v in mem.items() if k!="sha256"}
    ck(n3.digest(body)==mem["sha256"],"MEMORY_SEAL")
    state=c.load_state(state_root);key=hashlib.sha256(raw).hexdigest()
    status="ACTIVE" if key in state["active"] else "PENDING" if key in state["pending"] else "NOT_INGESTED"
    if status!="NOT_INGESTED":ck(c.read_object(state_root,key,len(raw))==raw,"OBJECT_BYTES")
    rec={"schema":"h38n5.cortex.witness.v1","memory_sha256":mem["sha256"],"source_sha256":key,"generation":state["generation"],"status":status,"production":False,"training_allowed":False}
    out.mkdir();write(out/"cortex.json",rec);print("R84H38N5_CORTEX",json.dumps(rec,sort_keys=True),flush=True)
if __name__=="__main__":{"evaluate":evaluate,"cortex":cortex_witness}[sys.argv[1]](*map(Path,sys.argv[2:]))
