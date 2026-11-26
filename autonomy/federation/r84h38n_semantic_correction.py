"""H38N: source verification, graph-grounded synthetic language and role-aware correction.
All scientific claims stay quarantined. No automatic model promotion.
"""
import sys,json,re,hashlib,time
from pathlib import Path
from collections import Counter
import torch
from torch import nn
from safetensors.torch import save_file,load_file
import r84h38m_knowledge_diversification as m
import r84h38j_fractal_self_learning as j
import r84h38i_fractal_reconstruction as i
import r84h38h_semantic_kaleidoscope as h
import r84h38e_semantic_invariance as e
import r84h38d_world_understanding as d
import r84h38c_semantic_world_graph as g
import r84h38f_semantic_crystal as f
ck=f.ck;read=f.read;write=f.write;sha=f.sha
# Training-only surface forms with role assignments. No arbitrary text from sources.
FORMS=[
["A precedes the occurrence of B","A takes place in advance of B"],
["A follows the occurrence of B","A takes place subsequent to B"],
["A does not precede the occurrence of B","A never takes place in advance of B"],
["A does not follow the occurrence of B","A never takes place subsequent to B"],
["A gives rise to B","A brings about B","A triggers the occurrence of B"],
["A originates as a consequence of B","A arises from B","A is brought about by B"],
["A does not give rise to B","A does not bring about B","A does not trigger the occurrence of B"],
["A does not originate as a consequence of B","A does not arise from B","A is not brought about by B"],
["A depends on B","A needs B to occur"],
["A is needed for B","A is a requirement for B"],
["A does not depend on B","A does not need B to occur"],
["A is not needed for B","A is not a requirement for B"],
["A blocks the occurrence of B","A precludes B from occurring"],
["A is blocked by B","A is precluded from occurring by B"],
["A does not block the occurrence of B","A does not preclude B from occurring"],
["A is not blocked by B","A is not precluded from occurring by B"]]
# An independent narrowly scoped semantic-role prior. It vetoes incompatible
# learned outputs, never supplies polarity or a positive prediction alone.
ROLE=[
(0,0,r"\b(before|earlier|prior to|in advance of|precedes)\b"),
(0,1,r"\b(after|later|subsequent to|following|follows)\b"),
(1,0,r"\b(causes?|cause of|gives? rise to|bring[s]? about|trigger[s]?|produce[s]?|brought about the occurrence)\b"),
(1,1,r"\b(arise[s]? (as a consequence of|from)|arose as a consequence of|originate[s]? as a consequence of|result[s]? from|consequence of|caused by|brought about by)\b"),
(2,0,r"\b(depends? on|depends? upon|requires?|needs?|demands?)\b"),
(2,1,r"\b(needed for|required by|requirement for|prerequisite of|prerequisite for|indispensable for)\b"),
(3,0,r"\b(prevents?|precludes?|blocks?|excludes?|rules? out|renders? .* impossible)\b"),
(3,1,r"\b(prevented by|precluded .* by|blocked by|excluded by|ruled out by|rendered impossible by)\b")]
def cue(t):
    matches={(r,s) for r,s,p in ROLE if re.search(p,t)}
    return next(iter(matches)) if len(matches)==1 else None
def verify_sources(demi,scout,out):
    out.mkdir();c=e.cortex(demi);receipt=read(scout/"H38M_SCOUT.json");rows=[]
    for p in sorted((scout/"candidates.parts").glob("part-*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():rows.append(json.loads(line))
    ck(len(rows)==receipt["summary"]["new_candidates"]==10,"SCOUT_COUNT")
    ck(len({z["candidate_id"] for z in rows})==10,"SCOUT_DUP")
    results=[];seen=set()
    for z in rows:
        lead=c.canonical_lead(z["url"]);rec={"candidate_id":z["candidate_id"],"url":z["url"],"status":"HOLD_UNSUPPORTED","scientific_truth":False,"training_allowed":False}
        if lead:
            ident,url=lead;rec["primary_id"]=ident
            if ident in seen:rec["status"]="DUPLICATE_PRIMARY_ID"
            else:
                seen.add(ident)
                try:
                    raw=c.fetch_primary(url);txt=c.primary_text(ident,raw)
                    rec.update(status="PRIMARY_METADATA_VERIFIED",provider_sha256=hashlib.sha256(raw).hexdigest(),text_sha256=hashlib.sha256(txt).hexdigest(),bytes=len(txt),title=txt.decode().split("\n",1)[0][:240])
                except Exception as ex:rec.update(status="HOLD_PROVIDER_FAILURE",error=type(ex).__name__+":"+str(ex)[:120])
        results.append(rec)
    report={"schema":"h38n.sources.v1","candidates":len(rows),"verified":sum(z["status"]=="PRIMARY_METADATA_VERIFIED" for z in results),"statuses":dict(Counter(z["status"] for z in results)),"records":results,"scientific_claims_admitted":0,"production":False}
    write(out/"sources.json",report);print("R84H38N_SOURCES",json.dumps(report,sort_keys=True),flush=True)
def curriculum(tasks):
    rows=read(tasks/"diversified/tasks.json");fam={}
    for z in rows:
        ck(i.verify(z["return"]),"RETURN");m.verify_task(z["request"],z["expected"])
        fam.setdefault(z["family"],z)
    ck(len(fam)==28,"FAMILIES")
    out={};coverage={}
    for key,z in sorted(fam.items()):
        atoms=[{k:v for k,v in a.items() if k!="id"} for a in z["request"]["facts"]]+[z["request"]["claim"]]
        unique=set()
        for a in atoms:
            r=e.REL.index(a["rel"]);v=0 if a["positive"] else 1
            for s in (0,1):
                y=4*r+2*v+s
                for form in FORMS[y]:
                    lft,rgt=(a["right"],a["left"]) if s else (a["left"],a["right"])
                    t=form.replace("A",lft).replace("B",rgt);n=e.norm(t,[a["left"],a["right"]]);ck(n and e.decode(y,n[1])==a,"RENDER")
                    # Syntax label independent from graph verdict. A false claim is
                    # still a valid syntactic relation, not an admitted world fact.
                    if n[0] in out:ck(out[n[0]]["y"]==y,"CONFLICT")
                    else:out[n[0]]={"text":n[0],"y":y,"origin":"SYNTHETIC_GRAPH_SYNTAX","families":[]}
                    if key not in out[n[0]]["families"]:out[n[0]]["families"].append(key)
                    unique.add(n[0])
        coverage[key]=len(unique)
    return list(out.values()),coverage,len(rows)
def probabilities(ms,ts):
    x=h.batch(ts)
    with torch.inference_mode():
        return torch.stack([n(x).softmax(-1) for n in ms]).min(0).values
class Net(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(e.B,64),nn.GELU(),nn.Linear(64,17))
    def forward(self,x):return self.net(x)
def select(p,rule):
    y,c,margin=h.top(p)
    return y if y<16 and c>=rule[0] and margin>=rule[1] else None
def train(hdir,idir,ldir,tasks,out):
    g.witness();torch.set_num_threads(2);out.mkdir();hr,bank,ns=i.load_h(hdir);lr=read(ldir/"receipt.json")
    ck(lr["bank_sha"]==hr["bank_sha"],"BANK")
    base=bank["train"]+read(idir/"repair_rows.json")+read(ldir/"new_rows.json");dev=bank["dev"];seen={z["text"]:z["y"] for z in base+dev}
    aug,coverage,graph_n=curriculum(tasks);fresh=[]
    for z in aug:
        if z["text"] in seen:ck(seen[z["text"]]==z["y"],"BASE_LABEL_CONFLICT")
        else:fresh.append(z)
    # The exact disclosed L failure is a diagnostic only, never training text.
    diag=read(tasks/"diagnostic/diagnostic.json");ck(len(diag)==1,"DIAG")
    forbidden=e.norm(diag[0]["text"],list({diag[0]["gold"]["left"],diag[0]["gold"]["right"]}))[0]
    ck(forbidden not in {z["text"] for z in base+dev+fresh},"DISCLOSED_ERROR_TRAIN_OVERLAP")
    runs={};rules={};allrows={}
    for mode,rows in [("control",base),("diverse",base+fresh)]:
        x=h.batch([z["text"] for z in rows]);y=torch.tensor([z["y"] for z in rows]);ms=[];runs[mode]=[]
        for seed in (395,396):
            torch.manual_seed(seed);n=Net();opt=torch.optim.AdamW(n.parameters(),lr=.006);start=time.process_time()
            for _ in range(300):
                loss=nn.functional.cross_entropy(n(x),y);opt.zero_grad();loss.backward();opt.step()
            p=out/f"{mode}-{seed}.safetensors";save_file(n.state_dict(),str(p));ms.append(n.eval());runs[mode].append({"seed":seed,"sha":sha(p),"parameters":sum(v.numel() for v in n.parameters()),"cpu":time.process_time()-start})
        ps=probabilities(ms,[z["text"] for z in dev]);pool=[]
        for c in (.35,.5,.65,.8,.9,.95,1.01):
            for margin in (0,.1,.25,.4):
                r=(c,margin);pred=[select(p,r) for p in ps];good=sum(a is not None and a==z["y"] for a,z in zip(pred,dev));bad=sum(a is not None and a!=z["y"] for a,z in zip(pred,dev))
                if not bad:pool.append((good,r))
        good,r=max(pool);rules[mode]={"rule":r,"dev_correct":good};allrows[mode]=len(rows)
    write(out/"curriculum.json",fresh);write(out/"training_refs.json",sorted({z["text"] for z in base+dev+fresh}))
    r={"schema":"h38n.train.v1","bank_sha":hr["bank_sha"],"l_receipt_sha":sha(ldir/"receipt.json"),"train_n":allrows,"dev_n":len(dev),"new_unique_templates":len(fresh),"families":coverage,"graph_scenarios_rechecked":graph_n,"runs":runs,"rules":rules,"exam_read":False,"production":False}
    write(out/"receipt.json",r);print("R84H38N_TRAIN",json.dumps(r,sort_keys=True),flush=True)
def evaluate(hdir,gdir,idir,edir,ldir,demi,trained,exam,oldc,oldd,olde,oldf,oldg,oldh,oldi,oldj,oldl,out):
    g.witness();torch.set_num_threads(2);out.mkdir();hr,bank,ns=i.load_h(hdir);gm=i.load_g(gdir,hr["bank_sha"]);ir,im=j.load_i(idir);lr=read(ldir/"receipt.json");r=read(trained/"receipt.json");ck(r["bank_sha"]==hr["bank_sha"] and r["l_receipt_sha"]==sha(ldir/"receipt.json"),"PINS")
    active=j.load_models(ldir,lr["active"]);rule=lr["rule"];er=read(edir/"receipt.json");eb=read(edir/"bank.json");ck(sha(edir/"bank.json")==er["bank_sha"],"E_BANK");em=[]
    for z in er["runs"]:
        p=edir/f"model-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"E_WEIGHT");n=e.Net();n.load_state_dict(load_file(str(p)));em.append(n.eval())
    di=e.Dictionary(e.cortex(demi),edir/"cortex",eb["train"]);nets={}
    for k,rows in r["runs"].items():
        nets[k]=[]
        for z in rows:
            p=trained/f"{k}-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"N_WEIGHT");n=Net();n.load_state_dict(load_file(str(p)));nets[k].append(n.eval())
    cache={}
    def allpred(t,names):
        direct=d.parse(t);n=e.norm(t,names)
        if n is None:return {k:direct if k in ("parent","guard","stack") else None for k in modes}
        text,events=n
        if text not in cache:
            ej=e.pick(e.probs(em,[text])[0],di.search(text),er["rules"]["agreement"],"agreement")
            qs=[i.trace(text,hr,ns,bank,ms,rule if ms is not gm else (.9,.25)) for ms in (gm,im,active)]
            old=ej if ej is not None else next((q["value"] for q in qs if q["value"] is not None),None)
            ps={k:probabilities(v,[text])[0] for k,v in nets.items()}
            ys={k:select(p,r["rules"][k]["rule"]) for k,p in ps.items()}
            cache[text]=(old,ys,{k:h.top(v)[0] for k,v in ps.items()},cue(text))
        old,ys,raw,role=cache[text];parent=direct if direct is not None else e.decode(old,events) if old is not None else None
        def agrees(a):
            return role is not None and a is not None and (e.REL.index(a["rel"]),events.index(a["left"]))==role
        v={k:e.decode(y,events) if y is not None else None for k,y in ys.items()}
        guard=parent if role is None or agrees(parent) else None
        # A known exact grammar answer remains authoritative; otherwise veto
        # contradictory old inference and require learned agreement to fill.
        if direct is not None:guard=direct
        stack=guard if guard is not None else v["diverse"] if agrees(v["diverse"]) else None
        return {"parent":parent,"control":v["control"],"diverse":v["diverse"],"guard":guard,"stack":stack,"control_raw":e.decode(raw["control"],events) if raw["control"]<16 else None,"diverse_raw":e.decode(raw["diverse"],events) if raw["diverse"]<16 else None}
    modes=("parent","control","diverse","guard","stack","control_raw","diverse_raw");ex=read(exam);refs=set(read(trained/"training_refs.json"))|{z["text"] for z in eb["train"]+eb["dev"]};prior=read(oldd)["cases"]
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
    il=read(oldi);ll=read(oldl);replayI=rows(il);replayL=rows(ll);hist={k:{"I":metrics(replayI,k,il["groups"]),"L":metrics(replayL,k,ll["groups"])} for k in ("parent","guard","stack")}
    histgraph={"I":graph(il["graphs"],("parent","guard","stack")),"L":graph(ll["graphs"],("parent","guard","stack")),"D":graph([{**z,"names":namesD(z)} for z in prior],("parent","guard","stack"))}
    ck(hist["parent"]["I"]["correct"]==24 and hist["parent"]["L"]["correct"]==11 and histgraph["I"]["parent"]["correct"]==9 and histgraph["D"]["parent"]["correct"]==46,"PARENT_PARITY")
    old=read(oldc)["cases"];ck(len(old)==76 and all(g.solve(z["request"])["status"]==z["expected"] for z in old),"C_REPLAY")
    err=[z for z in replayL if z["pred"]["parent"] is not None and z["pred"]["parent"]!=z["gold"]];ck(len(err)==1,"ERROR_PARITY")
    report={"schema":"h38n.result.v1","exam_sha":sha(exam),"known_n":32,"unknown_n":12,"routes":stats,"graphs":gs,"historical":hist,"historical_graphs":histgraph,"h38c_replay":76,"disclosed_error":err[0],"quality_gate":"HOLD","production":False,"scope":"Same-author normalized-disjoint exam; operational/hypothetical graph syntax training only; guard vs learned ablation; no independent external validation."}
    write(out/"result.json",report);write(out/"cases.json",rs);print("R84H38N_RESULT",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":{"sources":verify_sources,"train":train,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
