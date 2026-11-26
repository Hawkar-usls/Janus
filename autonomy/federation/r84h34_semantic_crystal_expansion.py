"""H34: bounded additional evidence paths; frozen H33 weights, no live writes."""
import sys,re,time,math
from pathlib import Path
import torch
from safetensors.torch import load_file
import r84h33_json_crystallization as h33
import r84h32_json_semantic_compiler as h32
import r84h31_semantic_evolution as h31
import r84h29_semantic_resonance as h29
import r84h27_trinity_critic as h27
import r84h30_semantic_inheritance as h30
import r84h19_exploration as h19
import verified_capabilities as exact
EXAM="582ea0e9aa40c545af77964042121f7a95eb0b0a"
STATES=h32.STATES;HOLD=h32.HOLD;ATTR=h32.ATTR
ck=h32.ck;read=h32.read;write=h32.write;log=h32.log;digest=h32.digest
# Author-specified contrast cues, not an independent truth oracle.
# Require an explicit domain anchor and a single polarity. Ambiguity holds.
RULES={
"cached_available":(r"cache|cached|local (?:tier|store|lookup|copy)|fast(?:-access)? (?:tier|store|layer)|resident copy",r"miss(?:es|ing)?|absent|evict|lacks?|empty|no (?:local|resident|cached)|unsuccessful|cannot (?:find|provide|be served)|lookup (?:fails|failed)",r"hit|already (?:has|present|contains|available)|resident (?:copy|entry)|existing local|without (?:contacting|consulting|accessing|a remote)|no upstream|answers locally|serves the lookup"),
"persist_first":(r"persist|durab|stable (?:media|storage)|backing disk|permanent storage|volatile|flush",r"acknowledg\w* before|confirm\w* before|success (?:before|while)|before (?:the bytes|durab|persist)|after (?:the )?(?:success|acknowledg|confirmation)|later flush|only later|while .*volatile",r"durab\w* (?:storage )?before|persist\w* before|before (?:acknowledg|confirmation|success)|only (?:after|once)|after .* (?:persist|durab)|waits? for .*durab"),
"multi_holder":(r"permit|token|admission|holder|exclusive|concurrent|simultaneous",r"single (?:holder|owner|worker)|sole exclusive|only one|unique permit|exclusive (?:token|owner)|second .*excluded|simultaneous .*forbidden",r"multiple (?:callers|holders|workers)|several (?:workers|holders|callers)|distinct (?:tickets|permits)|separate permits|more than one|(?:three|four|five|six) simultaneous"),
"full_replica":(r"server|node|machine|shard|partition|replica|dataset",r"disjoint|nonoverlapping|non-overlapping|shard|only its own portion|separate key ranges|distinct .*partition",r"full copy|complete cop(?:y|ies)|entirety|entire dataset|all records|complete logical|identical complete"),
"newer_recovery":(r"recover|deploy|release|upgrade|revision|build|patch",r"rolls? back|earlier (?:working|stable|release)|preceding stable|previous (?:working|stable|release)|restor\w* .*predecessor|reinstal\w* .*earlier",r"rolls? forward|newer (?:corrected|fixed|release)|later (?:fixed|repaired|corrected)|advanc\w* .*patch|newly repaired|newer revision"),
"all_or_none":(r"commit|transaction|linked (?:edits|updates|changes)|coupled (?:edits|modifications)",r"partial commit (?:is )?permitted|permits? a partial|allows? a partial|commit independently|one change to survive|one .*commit alone|partial (?:result|completion) is possible",r"commit together|neither may commit|forbids? a partial|indivisible commit|all .* or none|both .* or (?:neither|are discarded)|cannot leave a partial")}
def cue(t):
    if not h33.scope(t) or re.search(r"\b(?:uncertain|not chosen|not specify|no conclusion|no information|may (?:or|either)|compare|hypothetical)\b",t):return None
    matches=[]
    for a,(anchor,neg,pos) in RULES.items():
        if re.search(anchor,t):
            n=bool(re.search(neg,t));p=bool(re.search(pos,t))
            if n and p:return None
            if n or p:matches.append(STATES.index((a,int(p))))
    return matches[0] if len(matches)==1 else None
class Context:
    def __init__(self,demi,source,trained):
        self.rec=read(trained/"receipt.json");self.tr=read(trained/"train.json");self.dev=read(trained/"dev.json");records=read(trained/"correspondences.json")
        ck(self.rec["train_sha"]==digest(self.tr) and self.rec["dev_sha"]==digest(self.dev) and self.rec["dictionary_sha"]==digest(records) and self.rec["train_n"]==236 and not self.rec["exam_read"] and not self.rec["production"],"H33_PIN")
        self.d=h33.Dictionary(demi,trained/"cortex",records);self.sem=h29.Semantic(demi,source);ck(self.sem.hashes==self.rec["teacher_hashes"],"TEACHER_PIN")
        self.z=next(z for z in self.rec["runs"] if z["seed"]==self.rec["selected"]);self.policy=self.z["policy"];ck(self.policy is not None,"H33_POLICY")
        p=trained/("split-%d.safetensors"%self.z["seed"]);ck(h29.sha(p)==self.z["sha"],"WEIGHTS")
        self.model=h33.Split();self.model.load_state_dict(load_file(str(p)));self.model.eval()
    def observations(self,ts):
        n,cpu=h33.probabilities(self.model,ts);l=[self.d.search(t) for t in ts];old=h33.evidence(self.sem,ts,n,l)
        rows=[];ps=[];index=[]
        for i,t in enumerate(ts):
            j,proof=h33.select(t,n[i],l[i],old.get(i),self.policy);k,s,_=h32.candidate(l[i][0]);q=cue(t)
            row={"text":t,"baseline":j,"candidate":q,"proof":proof,"alt":None,"alternative":None};rows.append(row)
            if j is not None or q is None or q!=k or not h33.scope(t) or l[i][1][q] is None:continue
            a,v=STATES[q];aidx=ATTR.index(a);mass=[sum(n[i][z*2:z*2+2]) for z in range(len(ATTR))]+[n[i][HOLD]]
            other=max(x for z,x in enumerate(mass) if z!=aidx)
            # Original score and margin floors remain unchanged; attribute
            # marginal is a different quantity, explicitly not joint confidence.
            # Alternative requires lexical exact class, explicit polarity cue,
            # neural attribute support and two NLI contrast hypotheses.
            if mass[aidx]<self.policy["confidence"] or mass[aidx]-other<self.policy["margin"] or s<self.policy["lexical"]:continue
            if n[i][q]<=n[i][q^1] or l[i][0][q]<=l[i][0][q^1]:continue
            row["attribute_mass"]=mass[aidx];row["attribute_margin"]=mass[aidx]-other
            index.append(i)
            for h in range(2):
                ps.extend([(t,h31.HYP[a][v][h].format(o="entity")),(t,h31.HYP[a][1-v][h].format(o="entity"))])
        vals=self.sem.pairs(ps) if ps else []
        for k,i in enumerate(index):
            checks=[]
            for j in (0,1):
                support=vals[4*k+2*j];op=vals[4*k+2*j+1]
                checks.append({"support":support[1],"support_contradiction":support[0],"opposite_contradiction":op[0],"opposite_entailment":op[1]})
            rows[i]["alt"]=checks
            # No lower thresholds. Both template contrasts must meet H33
            # original semantic/opposite thresholds; same teacher, correlated.
            if all(x["support"]>=self.policy["semantic"] and x["opposite_contradiction"]>=self.policy["opposite"] and x["opposite_entailment"]<=.20 for x in checks):
                rows[i]["alternative"]=rows[i]["candidate"]
        return rows,cpu
def choose(z,enabled):return z["baseline"] if z["baseline"] is not None else z["alternative"] if enabled else None
def gate(demi,source,trained,out):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);c=Context(demi,source,trained)
    rows,cpu=c.observations([z["text"] for z in c.dev])
    def m(on):
        r={"correct":0,"wrong":0,"unknown_accepted":0}
        for z,truth in zip(rows,c.dev):
            p=choose(z,on);y=truth["label"]
            if p is not None:
                if y==HOLD:r["unknown_accepted"]+=1
                elif p==y:r["correct"]+=1
                else:r["wrong"]+=1
        return r
    old,new=m(False),m(True);ck(old["correct"]==4 and old["wrong"]==0 and old["unknown_accepted"]==0,"H33_DEV_REPLAY")
    enabled=new["wrong"]==0 and new["unknown_accepted"]==0 and new["correct"]>old["correct"]
    report={"schema":"r84h34.gate.v1","train_sha":c.rec["train_sha"],"dictionary_sha":c.rec["dictionary_sha"],"weight_sha":c.z["sha"],"policy":c.policy,"enabled":enabled,"baseline_dev":old,"candidate_dev":new,"dev_n":len(rows),"teacher_hashes":c.sem.hashes,"exam_read":False,"production":False,"cpu_seconds":cpu,"trinity_arithmetic_controls":h27.trinity(demi),"scope":"Frozen H33 weights and dictionary; alternate evidence path only, no threshold reduction or exam reuse for calibration."}
    write(out/"gate.json",report);write(out/"dev_evidence.json",rows);log("R84H34_GATE",report)
def corpus(ex):
    ts=list(dict.fromkeys([z["text"] for z in ex["statements"]]+ex["unknown"]+[t for z in ex["pairs"] for t in (z["premise"],z["hypothesis"])]))
    norms={t:h32.normalize(t,ex["subjects"]) for t in ts}
    return norms
def evaluate(demi,source,trained,gateout,out):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);c=Context(demi,source,trained);g=read(gateout/"gate.json")
    ck(g["train_sha"]==c.rec["train_sha"] and g["dictionary_sha"]==c.rec["dictionary_sha"] and g["weight_sha"]==c.z["sha"] and g["policy"]==c.policy and g["teacher_hashes"]==c.sem.hashes and not g["exam_read"],"GATE_PIN")
    old=h19.load_exam("R84H33_FROZEN_EXAM.json",h33.EXAM);older=h19.load_exam("R84H32_FROZEN_EXAM.json",h32.EXAM);ex=h19.load_exam("R84H34_FROZEN_EXAM.json",EXAM)
    ref={h32.key(z["text"]) for z in c.tr+c.dev}
    for prior in (old,older):ref|={h32.key(n[1]) for n in corpus(prior).values() if n is not None}
    def assess(data,exclude):
        norm=corpus(data);blocked={t for t,n in norm.items() if n and h32.key(n[1]) in exclude}
        known=[z for z in data["statements"] if z["text"] not in blocked];unknown=[t for t in data["unknown"] if t not in blocked];pairs=[z for z in data["pairs"] if z["premise"] not in blocked and z["hypothesis"] not in blocked]
        ck(len(known)>=30 and len(unknown)>=18 and len(pairs)>=24,"EXAM_OVERLAP_OR_SIZE")
        valid=[t for t,n in norm.items() if n and t not in blocked];rows,cpu=c.observations([norm[t][1] for t in valid]);lookup=dict(zip(valid,rows))
        ck(all(exact.relation({"schema":"janus.relation.v1","facts":q["facts"],"claim":q["claim"]})==q["label"] for q in pairs),"ORACLE")
        def compile(t,mode):
            if t not in lookup:return None
            z=lookup[t];j=choose(z,mode=="candidate" or (mode=="guarded" and g["enabled"]))
            if j is None:return None
            a,v=STATES[j];return {"subject":norm[t][0],"attribute":a,"value":v}
        res={};details={}
        for mode in ("baseline","candidate","guarded"):
            a=[compile(z["text"],mode) for z in known];u=[compile(t,mode) for t in unknown];answers=[]
            for q in pairs:
                p,s=compile(q["premise"],mode),compile(q["hypothesis"],mode)
                answers.append(exact.relation({"schema":"janus.relation.v1","facts":[p],"claim":s}) if p and s else None)
            res[mode]={"facts_n":len(known),"facts_exact":sum(f==z["fact"] for f,z in zip(a,known)),"fact_errors":sum(f is not None and f!=z["fact"] for f,z in zip(a,known)),"unknown_n":len(unknown),"unknown_accepted":sum(f is not None for f in u),"pairs_n":len(pairs),"pairs_correct":sum(f==q["label"] for f,q in zip(answers,pairs)),"pairs_wrong":sum(f is not None and f!=q["label"] for f,q in zip(answers,pairs)),"pairs_answered":sum(f is not None for f in answers)}
            details[mode]={"facts":a,"unknown":u,"pairs":answers}
        confirmed=[{"text":z["text"],"fact":f,"evidence":lookup[z["text"]],"status":"CONFIRMED_ON_SYNTHETIC_EXAM","training_allowed":False,"production":False} for z,f in zip(known,details["guarded"]["facts"]) if f==z["fact"]]
        return res,{"details":details,"observations":lookup,"excluded":sorted(blocked)},confirmed,cpu
    prior,_,_,_=assess(old,set());ck(prior["baseline"]["facts_exact"]==3 and prior["baseline"]["fact_errors"]==0 and prior["baseline"]["unknown_accepted"]==0,"H33_REPLAY")
    fresh,details,confirmed,cpu=assess(ex,ref);z=fresh["guarded"];b=fresh["baseline"];p=prior["guarded"]
    safe=z["fact_errors"]==0 and z["unknown_accepted"]==0 and z["pairs_wrong"]==0 and p["fact_errors"]==0 and p["unknown_accepted"]==0 and p["pairs_wrong"]==0
    gain=z["facts_exact"]>b["facts_exact"];report={"schema":"r84h34.result.v1","gate_enabled":g["enabled"],"dev":g,"fresh":fresh,"h33_regression":prior,"excluded_exact_overlap":len(details["excluded"]),"new_confirmed":len(confirmed),"dictionary_documents":len(c.tr),"parameters":c.z["parameters"],"cortex_queries":c.d.calls,"neural_cpu_seconds":cpu,"safety":"PASS_SHADOW" if safe and z["pairs_answered"]>=12 and gain else "HOLD","production":False,"scope":"Same-author finite domain, frozen H33 model, added authored cues plus two same-MiniLM contrast checks, alternate attribute confidence is not joint confidence; no training on H33 or H34 exams, no external truth guarantee, no production changes."}
    write(out/"result.json",report);write(out/"cases.json",details);write(out/"confirmed_correspondences.json",confirmed);log("R84H34_RESULT",report)
if __name__=="__main__":{"gate":gate,"evaluate":evaluate,"history":h30.history}[sys.argv[1]](*map(Path,sys.argv[2:]))
