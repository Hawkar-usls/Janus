"""H27 source-referenced symbolic critic; never equate agreement with evidence."""
import sys,json,re,time,importlib.util
from pathlib import Path
import r84h25_metacognition as h25
import r84h26_self_critic as h26
import r84h19_exploration as h19
import r84h18_fog as h18
import r8_learning as r
HERE=Path(__file__).parent
EXAM="01831460b93f8b5124e19b951ced9b02d5a92b83"
EVIDENCE="a18a855a292ecd8683d3fa35d5e5d458582bc3fb"
TRINITY="7792bae07705e040504052bcf673342b661858f7"
MOTTO="Мы должны узнать"
def ck(v,m):h25.ck(v,m)
def read(p):return h25.read(p)
def write(p,x):h25.write(p,x)
def digest(x):return h25.digest(x)
def log(t,x):h25.log(t,x)
class Critic:
    def __init__(self,labels):
        raw=(HERE/"R84H27_EVIDENCE.json").read_bytes();ck(r.blob(raw)==EVIDENCE,"EVIDENCE_PIN");self.rules=json.loads(raw)["rules"]
        ck(len(self.rules)==20 and len({x["label"] for x in self.rules})==20 and all(x["label"] in labels for x in self.rules),"RULES")
        self.labels=labels
    def inspect(self,text):
        t=h18.clean(text)
        if not 0<len(t)<=500 or "?" in t or re.search(r"\b(?:compare|hypothetical\w*|do not assume|not asserted|suppose|imagine)\b",t):return {"verdict":"HOLD_SCOPE","matches":[]}
        matches=[]
        for row in self.rules:
            hits=[re.search(g,t,re.ASCII) for g in row["groups"]]
            if all(hits):
                matches.append({"label":row["label"],"answer":int(self.labels[row["label"]]),"source":row["source"],"claim":row["claim"],"matched":[m.group() for m in hits]})
        return {"verdict":"SUPPORTED_CANDIDATE" if len(matches)==1 else "HOLD_CONFLICT" if matches else "HOLD_NO_EVIDENCE","matches":matches}
def replay(ctx,folder,h25feedback):
    z=read(folder/"result.json");rows=read(folder/"feedback.json");h25.restore(folder/"feedback.sqlite",rows)
    ck(z["safety_gate"]=="PASS_SHADOW" and z["improvement_gate"]=="HOLD_NO_PROVEN_GAIN" and z["base"]["agreement_wrong"]==9 and len(rows)==100 and not z["production"],"H26_RECEIPT")
    ex=h19.load_exam("R84H26_FROZEN_EXAM.json",h26.EXAM);prev=read(h25feedback/"feedback.json");h25.restore(h25feedback/"feedback.sqlite",prev)
    refs={h18.clean(t) for t,_ in ctx.fit+ctx.dev+ctx.oldknown}|{h18.clean(t) for t in ctx.oldunknown}|{h18.clean(x["text"]) for x in prev}
    known=[x for x in ex["known"] if h18.clean(x[0]) not in refs];unknown=[x for x in ex["unknown"] if h18.clean(x) not in refs]
    actual,cost=ctx.observations(known,unknown,"H26_FRESH_EVAL")
    ck(len(actual)==len(rows) and all(all(x[k]==y[k] for k in x) for x,y in zip(actual,rows)),"H26_REPLAY")
    return rows,prev,cost
def verdict(row,critic,enabled=None):
    v=critic.inspect(row["text"])
    if v["verdict"]=="SUPPORTED_CANDIDATE":
        hit=v["matches"][0]
        if enabled is None or hit["label"] in enabled:return hit["answer"],v
    return row.get("neural"),v
def stats(rows,critic,enabled=None):
    m={"known_n":0,"base_correct":0,"correct":0,"supported":0,"supported_wrong":0,"rescued":0,"lost":0,"joint_wrong":0,"joint_wrong_rescued":0,"unknown_n":0,"unknown_supported":0}
    detail=[]
    for z in rows:
        a,v=verdict(z,critic,enabled);supported=v["verdict"]=="SUPPORTED_CANDIDATE" and (enabled is None or v["matches"][0]["label"] in enabled)
        row={"id":z["id"],"kind":z["kind"],"text":z["text"],"evidence":v,"enabled_support":supported}
        if z["kind"]=="known":
            good=a==z["truth"];base=z["neural_correct"];joint=z["neural"]==z["lexical"] and not base
            m["known_n"]+=1;m["base_correct"]+=base;m["correct"]+=good;m["supported"]+=supported;m["supported_wrong"]+=supported and not good;m["rescued"]+=good and not base;m["lost"]+=base and not good;m["joint_wrong"]+=joint;m["joint_wrong_rescued"]+=joint and good
            row.update({"truth":z["truth"],"neural":z["neural"],"lexical":z["lexical"],"answer":a,"correct":good,"joint_wrong":joint})
        else:m["unknown_n"]+=1;m["unknown_supported"]+=supported
        detail.append(row)
    return m,detail
def trinity(demi):
    p=demi/"tools/trinity_r14_verified_oracle.py";ck(r.blob(p.read_bytes())==TRINITY,"TRINITY_PIN")
    spec=importlib.util.spec_from_file_location("trinity",p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    controls=[("Calculate 8 + 9. Answer with one integer.","17","APPROVE"),("Calculate 8 + 9. Answer with one integer.","18","VETO_WRONG"),("Calculate 8 + 9. Answer with one integer.","seventeen","VETO_FORMAT"),("Explain a cache hit.","cache_hit","HOLD_UNSUPPORTED")]
    ck(all(mod.verify(q,a)["verdict"]==v for q,a,v in controls),"TRINITY_CONTROLS");return len(controls)
def gate(*args):
    *paths,h25feedback,h26feedback,demi,out=args;ctx=h25.Context(*paths);out.mkdir(parents=True,exist_ok=False)
    rows,prev,cost=replay(ctx,h26feedback,h25feedback);c=Critic(ctx.labels)
    raw,details=stats(rows,c);ck(raw["joint_wrong"]==9,"JOINT_FAILURE_COUNT")
    # Per-label validation on disclosed H25/H26 only. No H27 exam access.
    domains={}
    for rule in c.rules:
        label=rule["label"];a,_=stats(prev,c,{label});b,_=stats(rows,c,{label})
        # Abstain if any unsupported-topic false match or any incorrectly
        # supported classification, including wrong->wrong substitutions.
        accepted=a["supported_wrong"]+b["supported_wrong"]==0 and a["unknown_supported"]+b["unknown_supported"]==0 and a["lost"]+b["lost"]==0 and a["supported"]+b["supported"]>=3 and a["rescued"]+b["rescued"]>=2
        domains[label]={"h25":a,"h26":b,"accepted":accepted}
    enabled={k for k,v in domains.items() if v["accepted"]};guarded,guarddetails=stats(rows,c,enabled)
    ck(guarded["lost"]==0,"VALIDATION_REGRESSION")
    controls=trinity(demi)
    diagnostics=[z for z in guarddetails if z.get("joint_wrong")]
    ck(len(diagnostics)==9,"DIAGNOSTICS")
    rec={"schema":"r84h27.gate.v1","evidence_blob":EVIDENCE,"h26_feedback_sha":digest(rows),"h25_feedback_sha":digest(prev),"domains":domains,"enabled":sorted(enabled),"raw_diagnostic":raw,"guarded_diagnostic":guarded,"trinity_controls":controls,"exam_read":False,"production":False,"motto":MOTTO}
    write(out/"gate.json",rec);write(out/"joint_failures.json",diagnostics);log("R84H27_GATE",rec);log("R84H27_DIAGNOSTIC",diagnostics)
def evaluate(*args):
    *paths,h25feedback,h26feedback,demi,gateout,out=args;ctx=h25.Context(*paths);out.mkdir(parents=True,exist_ok=False)
    policy=read(gateout/"gate.json");prior=read(h26feedback/"feedback.json");prev=read(h25feedback/"feedback.json");ck(policy["h26_feedback_sha"]==digest(prior) and policy["h25_feedback_sha"]==digest(prev) and policy["evidence_blob"]==EVIDENCE and not policy["exam_read"] and not policy["production"],"GATE_PIN")
    raw=(HERE/"R84H27_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN");ex=json.loads(raw)
    refs={h18.clean(t) for t,_ in ctx.fit+ctx.dev+ctx.oldknown}|{h18.clean(t) for t in ctx.oldunknown}|{h18.clean(z["text"]) for z in prior+prev}
    known=[z for z in ex["known"] if h18.clean(z[0]) not in refs];unknown=[t for t in ex["unknown"] if h18.clean(t) not in refs]
    ck(len(known)>=35 and len(unknown)>=18 and len({h18.clean(t) for t,_ in known}|{h18.clean(t) for t in unknown})==len(known)+len(unknown),"EXAM_OVERLAP")
    rows,cost=ctx.observations(known,unknown,"H27_FRESH_EVAL");c=Critic(ctx.labels);enabled=set(policy["enabled"])
    ts=time.process_time();unguarded,_=stats(rows,c);guarded,details=stats(rows,c,enabled);critic_cpu=time.process_time()-ts
    conflicts=[{"text":t,"evidence":c.inspect(t)} for t in ex["conflict"]];held=sum(z["evidence"]["verdict"]!="SUPPORTED_CANDIDATE" for z in conflicts)
    pred,routes,histcpu=ctx.neural(ctx.hist);parity=sum(ctx.prev[i]["answer"]==int(pred[i]) and ctx.prev[i]["route"]==int(routes[i]) for i in range(671));ck(parity==671,"HISTORY")
    safe=guarded["lost"]==0 and guarded["unknown_supported"]==0 and held==len(conflicts)
    improved=guarded["correct"]>guarded["base_correct"]
    report={"schema":"r84h27.result.v1","motto":MOTTO,"prior_joint_wrong":9,"prior_raw_rescued":policy["raw_diagnostic"]["joint_wrong_rescued"],"prior_guarded_rescued":policy["guarded_diagnostic"]["joint_wrong_rescued"],"enabled_labels":policy["enabled"],"fresh_raw":unguarded,"fresh_guarded":guarded,"conflict_held":held,"conflict_n":len(conflicts),"overlap_excluded":60-len(known)-len(unknown),"historical_parity":parity,"trinity_controls":trinity(demi),"cost":cost,"critic_cpu_seconds":critic_cpu,"historical_cpu_seconds":histcpu,"safety":"PASS_SHADOW" if safe else "HOLD","improvement":"PASS_SHADOW" if safe and improved else "HOLD_NO_PROVEN_GAIN","production":False,"scope":"hand-authored technical reference predicates, not a truth oracle for arbitrary prose; source URLs and evidence matches recorded, no live source re-fetch or license admission; question-form unknowns easy scope rejection; H25/H26 disclosed validation, fresh same-author rule-oriented probes; native TRINITY arithmetic remains unsupported for prose; historical compatibility replay separate, no neural weights or production updates"}
    write(out/"result.json",report);write(out/"cases.json",details);write(out/"conflicts.json",conflicts);log("R84H27_RESULT",report)
if __name__=="__main__":{"gate":gate,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
