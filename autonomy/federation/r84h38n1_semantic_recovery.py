"""H38N1: preflight-first recovery, factorized learning and reverse-view checks."""
import sys,json,re,hashlib,time
from pathlib import Path
from collections import Counter
import torch
from torch import nn
from safetensors.torch import save_file,load_file
import r84h38n_semantic_correction as n0
import r84h38j_fractal_self_learning as j
import r84h38i_fractal_reconstruction as i
import r84h38h_semantic_kaleidoscope as h
import r84h38e_semantic_invariance as e
import r84h38d_world_understanding as d
import r84h38c_semantic_world_graph as g
import r84h38f_semantic_crystal as f
ck=f.ck;read=f.read;write=f.write;sha=f.sha
SLOTS=("relation","polarity","direction","certainty")
def review(demi,scout,verified,out):
    out.mkdir();c=e.cortex(demi);old=read(verified/"sources.json");rows=[]
    for p in sorted((scout/"candidates.parts").glob("part-*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():rows.append(json.loads(line))
    ck(len(rows)==10 and len(old["records"])==10,"SOURCES")
    prev={z["candidate_id"]:z for z in old["records"]};notes=[]
    for z in rows:
        a=prev[z["candidate_id"]];lead=c.canonical_lead(z["url"]);rec={"primary_id":a["primary_id"],"url":z["url"],"title":a["title"],"scope":"ABSTRACT_OR_REPOSITORY_DESCRIPTION_ONLY","full_text_reviewed":False,"training_allowed":False}
        try:
            ck(lead and lead[0]==a["primary_id"],"ID");raw=c.fetch_primary(lead[1]);txt=c.primary_text(lead[0],raw)
            ck(hashlib.sha256(txt).hexdigest()==a["text_sha256"],"SOURCE_TEXT_CHANGED")
            body=txt.decode().split("\n\n",1)[1];sentences=re.split(r"(?<=[.!?])\s+",body.strip())
            selected=[s[:600] for s in sentences if re.search(r"\b(method|approach|framework|pipeline|propos\w*|train\w*|evaluat\w*|benchmark|dataset|model|graph|constraint|audit|verif\w*|extract\w*|protocol)\b",s,re.I)][:4]
            rec.update(status="MATCHED_PRIMARY_TEXT",text_sha256=a["text_sha256"],method_evidence=selected,method_evidence_n=len(selected))
        except Exception as ex:rec.update(status="HOLD",error=type(ex).__name__+":"+str(ex)[:140])
        notes.append(rec)
    report={"schema":"h38n1.source_review.v1","records":notes,"matched":sum(z["status"]=="MATCHED_PRIMARY_TEXT" for z in notes),"full_text_reviewed":0,"scientific_claims_admitted":0,"production":False}
    write(out/"method_review.json",report);print("R84H38N1_REVIEW",json.dumps(report,sort_keys=True),flush=True)
def training_rows(hdir,idir,ldir,ndir):
    bank=read(hdir/"bank.json");nr=read(ndir/"receipt.json");ck(sha(hdir/"bank.json")==nr["bank_sha"] and sha(ldir/"receipt.json")==nr["l_receipt_sha"],"BANK_PIN")
    base=bank["train"]+read(idir/"repair_rows.json")+read(ldir/"new_rows.json");aug=read(ndir/"curriculum.json");dev=bank["dev"];rows=base+aug
    ck(len(rows)==nr["train_n"]["diverse"] and len(base)==nr["train_n"]["control"],"BANK_COUNT")
    ck(sorted({z["text"] for z in rows+dev})==read(ndir/"training_refs.json"),"BANK_REFS")
    return rows,dev
def preflight(hdir,idir,ldir,ndir,edir,exam,out,*prior):
    g.witness();out.mkdir();rows,dev=training_rows(hdir,idir,ldir,ndir);eb=read(edir/"bank.json");ck(sha(edir/"bank.json")==read(edir/"receipt.json")["bank_sha"],"E_BANK")
    refs={z["text"] for z in rows+dev+eb["train"]+eb["dev"]}
    def add(t,names):
        a=e.norm(t,names)
        if a:refs.add(a[0])
    for p in prior:
        x=read(p)
        for z in x.get("rows",[]):add(z["text"],z["names"])
        for z in x.get("graphs",[]):
            for t in z["sentences"]+[z["claim"]]:add(t,z["names"])
        for z in x.get("cases",[]):
            if "atoms" not in z:continue
            names=sorted({a[k] for a in z["atoms"]+[z["claim_atom"]] if a for k in ("left","right")})
            for t in z["sentences"]+[z["claim"]]:add(t,names)
    ex=read(exam);hits=[];known=[]
    def scan(t,names):
        a=e.norm(t,names)
        if a and a[0] in refs:hits.append(a[0])
        return a
    for z in ex["rows"]:
        a=scan(z["text"],z["names"])
        if z["y"]<16:
            ck(a and e.decode(z["y"],a[1])==z["gold"],"GOLD");known.append(a[0])
    for z in ex["graphs"]:
        for t in z["sentences"]+[z["claim"]]:scan(t,z["names"])
    ck(len(known)==len(set(known))==32,"EXAM_INTERNAL_DUP")
    report={"schema":"h38n1.preflight.v1","exam_sha":sha(exam),"n_receipt_sha":sha(ndir/"receipt.json"),"training_refs_sha":sha(ndir/"training_refs.json"),"prior_count":len(prior),"reference_count":len(refs),"overlap":sorted(set(hits)),"known":len(known),"production":False}
    write(out/"preflight.json",report);print("R84H38N1_PREFLIGHT",json.dumps(report,sort_keys=True),flush=True);ck(not hits,"PREFLIGHT_OVERLAP")
def reverse(t):
    # Reverse token order then relabel positions, so direction label flips.
    return " ".join("evtwo" if w=="evone" else "evone" if w=="evtwo" else w for w in reversed(t.split()))
def view(t,k):
    if k==0:return h.view(t)
    return t
class Factor(nn.Module):
    def __init__(self):
        super().__init__()
        self.tiles=nn.ModuleList([nn.Sequential(nn.Linear(e.B,16),nn.GELU(),nn.Linear(16,s)) for s in (4,2,2,2)])
def probs(ms,ts):
    with torch.inference_mode():
        return [torch.stack([n.tiles[k](h.batch([view(t,k) for t in ts])).softmax(-1) for n in ms]).min(0).values for k in range(4)]
def raw(ps):return [h.top(p) for p in ps]
def admit(a,rule):
    if a[3][0]!=1 or any(p<rule[0] or m<rule[1] for _,p,m in a):return None
    return 4*a[0][0]+2*a[1][0]+a[2][0]
def mirror_label(y):return y^1 if y is not None and y<16 else y
def decompose(ms,t,rules):
    ps=probs(ms,[t,reverse(t)]);a=raw([p[0] for p in ps]);b=raw([p[1] for p in ps])
    leaves=[]
    for k,slot in enumerate(SLOTS):
        aa,bb=a[k],b[k];expected=1-bb[0] if k==2 else bb[0]
        reasons=[] if aa[0]==expected else ["REVERSE_ROLE_DISAGREE"]
        leaves.append(i.packet(slot,aa[0],reasons,aa[1],aa[2],children=[i.packet("forward",aa[0],[],aa[1],aa[2]),i.packet("reverse",bb[0],[],bb[1],bb[2])]))
    ret=i.packet("crystal",None,[x for p in leaves for x in p["reasons"]],children=leaves);ck(i.verify(ret),"RETURN")
    aa=admit(a,rules["direct"]);bb=admit(b,rules["mirror"]);am=admit(a,rules["mirror"])
    return {"factor_raw":4*a[0][0]+2*a[1][0]+a[2][0] if a[3][0]==1 else None,"factor_direct":aa,"factor_mirror":am if am is not None and am==mirror_label(bb) and not ret["reasons"] else None},ret
def factor_score(ms,t,r):return decompose(ms,t,r["rules"])[0]
def load_factor(root,r):
    ms=[]
    for z in r["runs"]:
        p=root/z["file"];ck(sha(p)==z["sha"],"FACTOR_WEIGHT");n=Factor();n.load_state_dict(load_file(str(p)));ms.append(n.eval())
    return ms
def train(hdir,idir,ldir,ndir,pre,out):
    g.witness();torch.set_num_threads(2);out.mkdir();p=read(pre);ck(not p["overlap"] and p["n_receipt_sha"]==sha(ndir/"receipt.json") and p["training_refs_sha"]==sha(ndir/"training_refs.json"),"PREFLIGHT")
    rows,dev=training_rows(hdir,idir,ldir,ndir);ts=[];ys=[]
    for z in rows:
        t=z["text"];y=z["y"];ts.extend([t,reverse(t)]);ys.extend([f.parts(y),f.parts(mirror_label(y))])
    y=torch.tensor(ys);known=y[:,3]==1;xs=[h.batch([view(t,k) for t in ts]) for k in range(4)];runs=[];ms=[]
    for seed in (395,396):
        torch.manual_seed(seed);n=Factor();start=time.process_time()
        for k in range(4):
            tile=n.tiles[k];opt=torch.optim.AdamW(tile.parameters(),lr=.006);mask=known if k<3 else torch.ones_like(known)
            for _ in range(300):
                loss=nn.functional.cross_entropy(tile(xs[k][mask]),y[mask,k]);opt.zero_grad();loss.backward();opt.step()
        path=out/f"factor-{seed}.safetensors";save_file(n.state_dict(),str(path));ms.append(n.eval());runs.append({"seed":seed,"file":path.name,"sha":sha(path),"cpu":time.process_time()-start,"params":sum(v.numel() for v in n.parameters())})
    a=probs(ms,[z["text"] for z in dev]);b=probs(ms,[reverse(z["text"]) for z in dev]);rules={};cal={}
    for mode in ("direct","mirror"):
        pool=[]
        for c in (.35,.5,.65,.8,.9,.95,1.01):
            for margin in (0,.1,.25,.4):
                rule=(c,margin);pred=[]
                for ix,z in enumerate(dev):
                    u=admit(raw([p[ix] for p in a]),rule);v=admit(raw([p[ix] for p in b]),rule)
                    pred.append(u if mode=="direct" or u==mirror_label(v) else None)
                good=sum(k is not None and k==z["y"] for k,z in zip(pred,dev));bad=sum(k is not None and k!=z["y"] for k,z in zip(pred,dev))
                if not bad:pool.append((good,rule))
        good,rule=max(pool);rules[mode]=rule;cal[mode]=good
    rec={"schema":"h38n1.train.v1","n_receipt_sha":sha(ndir/"receipt.json"),"preflight_sha":sha(pre),"train_n":len(rows),"augmented_n":len(ts),"dev_n":len(dev),"runs":runs,"rules":rules,"calibration_correct":cal,"exam_read":False,"production":False}
    write(out/"receipt.json",rec);print("R84H38N1_TRAIN",json.dumps(rec,sort_keys=True),flush=True)

def evaluate(hdir,gdir,idir,edir,ldir,demi,trained,factor,preflight_receipt,exam,oldc,oldd,olde,oldf,oldg,oldh,oldi,oldj,oldl,out):
    g.witness();torch.set_num_threads(2);out.mkdir();hr,bank,ns=i.load_h(hdir);gm=i.load_g(gdir,hr["bank_sha"]);ir,im=j.load_i(idir);lr=read(ldir/"receipt.json");r=read(trained/"receipt.json");ck(r["bank_sha"]==hr["bank_sha"] and r["l_receipt_sha"]==sha(ldir/"receipt.json"),"PINS")
    active=j.load_models(ldir,lr["active"]);rule=lr["rule"];er=read(edir/"receipt.json");eb=read(edir/"bank.json");ck(sha(edir/"bank.json")==er["bank_sha"],"E_BANK");em=[]
    for z in er["runs"]:
        p=edir/f"model-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"E_WEIGHT");n=e.Net();n.load_state_dict(load_file(str(p)));em.append(n.eval())
    di=e.Dictionary(e.cortex(demi),edir/"cortex",eb["train"]);nets={}
    for k,rows in r["runs"].items():
        nets[k]=[]
        for z in rows:
            p=trained/f"{k}-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"N_WEIGHT");n=n0.Net();n.load_state_dict(load_file(str(p)));nets[k].append(n.eval())
    fr=read(factor/"receipt.json");pr=read(preflight_receipt);ck(fr["preflight_sha"]==sha(preflight_receipt) and pr["exam_sha"]==sha(exam) and fr["n_receipt_sha"]==sha(trained/"receipt.json"),"PREFLIGHT_PIN")
    fs=load_factor(factor,fr);cache={}
    def allpred(t,names):
        direct=d.parse(t);n=e.norm(t,names)
        if n is None:return {k:direct if k in ("parent","guard","stack","factor_stack") else None for k in modes}
        text,events=n
        if text not in cache:
            ej=e.pick(e.probs(em,[text])[0],di.search(text),er["rules"]["agreement"],"agreement")
            qs=[i.trace(text,hr,ns,bank,ms,rule if ms is not gm else (.9,.25)) for ms in (gm,im,active)]
            old=ej if ej is not None else next((q["value"] for q in qs if q["value"] is not None),None)
            ps={k:n0.probabilities(v,[text])[0] for k,v in nets.items()}
            ys={k:n0.select(p,r["rules"][k]["rule"]) for k,p in ps.items()}
            cache[text]=(old,ys,{k:h.top(v)[0] for k,v in ps.items()},n0.cue(text),factor_score(fs,text,fr))
        old,ys,raw,role,fac=cache[text];parent=direct if direct is not None else e.decode(old,events) if old is not None else None
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
        return {"parent":parent,"control":v["control"],"diverse":v["diverse"],"guard":guard,"stack":stack,"factor_stack":corrected,**fa,"control_raw":e.decode(raw["control"],events) if raw["control"]<16 else None,"diverse_raw":e.decode(raw["diverse"],events) if raw["diverse"]<16 else None}
    modes=("parent","control","diverse","guard","stack","control_raw","diverse_raw","factor_raw","factor_direct","factor_mirror","factor_stack");ex=read(exam);refs=set(read(trained/"training_refs.json"))|{z["text"] for z in eb["train"]+eb["dev"]};prior=read(oldd)["cases"]
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
                vals=[p[k] for p in ps];verdict[k]="HOLD_UNPARSED" if any(v is None for v in vals) else g.solve({"schema":"janus.event.graph.v1","facts":[{"id":f"s{x}",**a} for x,a in enumerate(vals[:-1])],"claim":vals[-1]})["status"]
                a=verdict[k];ans[k]["correct"]+=a==z["expected"];ans[k]["false_certainty"]+=a in ("ENTAILED","CONTRADICTED") and a!=z["expected"]
            for k in keys:ans[k]["lost"]+=verdict["parent"]==z["expected"] and verdict[k]!=z["expected"]
        return ans
    rs=rows(ex,True);stats={k:metrics(rs,k,ex["groups"]) for k in modes};gs=graph(ex["graphs"],modes,True)
    il=read(oldi);ll=read(oldl);replayI=rows(il);replayL=rows(ll);hist={k:{"I":metrics(replayI,k,il["groups"]),"L":metrics(replayL,k,ll["groups"])} for k in ("parent","guard","stack","factor_stack")}
    histgraph={"I":graph(il["graphs"],("parent","guard","stack","factor_stack")),"L":graph(ll["graphs"],("parent","guard","stack","factor_stack")),"D":graph([{**z,"names":namesD(z)} for z in prior],("parent","guard","stack","factor_stack"))}
    ck(hist["parent"]["I"]["correct"]==24 and hist["parent"]["L"]["correct"]==11 and histgraph["I"]["parent"]["correct"]==9 and histgraph["D"]["parent"]["correct"]==46,"PARENT_PARITY")
    old=read(oldc)["cases"];ck(len(old)==76 and all(g.solve(z["request"])["status"]==z["expected"] for z in old),"C_REPLAY")
    err=[z for z in replayL if z["pred"]["parent"] is not None and z["pred"]["parent"]!=z["gold"]];ck(len(err)==1,"ERROR_PARITY")
    report={"schema":"h38n1.result.v1","exam_sha":sha(exam),"known_n":32,"unknown_n":12,"routes":stats,"graphs":gs,"historical":hist,"historical_graphs":histgraph,"h38c_replay":76,"disclosed_error":err[0],"quality_gate":"HOLD","preflight_sha":sha(preflight_receipt),"factor_receipt_sha":sha(factor/"receipt.json"),"production":False,"scope":"Same-author replacement exam; frozen H38N joint models versus factorized models with same training bank; mirrored token view is a correlated consistency control, not an independent semantic oracle."}
    write(out/"result.json",report);write(out/"cases.json",rs);print("R84H38N1_RESULT",json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":{"review":review,"preflight":preflight,"train":train,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
