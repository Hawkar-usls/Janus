"""H37 comparison-only job; parent never used by standalone jobs."""
import sys
from pathlib import Path
import torch
import r84h37_polarity_independence as h37
import r84h36_semantic_independence as h36
import r84h34_semantic_crystal_expansion as h34
import r84h33_json_crystallization as h33
import r84h32_json_semantic_compiler as h32
import r84h19_exploration as h19
import verified_capabilities as exact
ck=h37.ck;read=h37.read;write=h37.write;log=h37.log
def compare(demi,source,trained,old,offline,out):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);c=h34.Context(demi,source,trained);rec=read(offline/"offline.json");detail=read(offline/"accepts.json");oldrec=read(old/"receipt.json")
    ck(rec["schema"]=="r84h37.offline.v1" and not rec["teacher_modules_loaded"] and not rec["teacher_weights_loaded"] and rec["network_namespace_no_routes"] and not rec["production"],"OFFLINE_RECEIPT")
    models=[]
    from safetensors.torch import load_file
    for z in oldrec["runs"]:
        p=old/("verify-%d.safetensors"%z["seed"]);ck(h37.sha(p)==z["sha"],"H36_PIN");m=h36.Verifier();m.load_state_dict(load_file(str(p)));models.append(m.eval())
    exams={"fresh":h19.load_exam("R84H37_FROZEN_EXAM.json",h37.EXAM),"33":h19.load_exam("R84H33_FROZEN_EXAM.json",h33.EXAM),"34":h19.load_exam("R84H34_FROZEN_EXAM.json",h34.EXAM),"36":h19.load_exam("R84H36_FROZEN_EXAM.json",h36.EXAM)}
    results={}
    for name,ex in exams.items():
        d=detail[name];norm=h37.corpus(ex);blocked=set(d["excluded"]);known=[z for z in ex["statements"] if z["text"] not in blocked];unknown=[t for t in ex["unknown"] if t not in blocked];pairs=[z for z in ex["pairs"] if z["premise"] not in blocked and z["hypothesis"] not in blocked]
        valid=[t for t,n in norm.items() if n and t not in blocked];ck(set(valid)==set(d["accepts"]),"OFFLINE_INPUTS");ts=[norm[t][1] for t in valid];obs,_=h36.observe(c,models,ts);lookup=dict(zip(valid,obs))
        answers={}
        for mode in ("parent","h36_same_candidate","h37_same_candidate","h37_standalone"):
            def compile(t):
                if t not in lookup:return None
                z=lookup[t];a=d["accepts"][t]["h37"]
                if mode=="parent":j=z["baseline"]
                elif mode=="h36_same_candidate":j=h36.select(z,oldrec["rule"],"own")
                elif mode=="h37_same_candidate":j=z["candidate"] if z["candidate"] in a else None
                else:j=a[0] if len(a)==1 else None
                if j is None:return None
                attr,v=h37.STATES[j];return {"subject":norm[t][0],"attribute":attr,"value":v}
            facts=[compile(z["text"]) for z in known];u=[compile(t) for t in unknown];pa=[]
            for q in pairs:
                f,g=compile(q["premise"]),compile(q["hypothesis"]);pa.append(exact.relation({"schema":"janus.relation.v1","facts":[f],"claim":g}) if f and g else None)
            answers[mode]={"facts":facts,"unknown":u,"pairs":pa}
        m={}
        for mode,z in answers.items():
            a,u,p=z["facts"],z["unknown"],z["pairs"]
            m[mode]={"facts_n":len(known),"facts_correct":sum(f==q["fact"] for f,q in zip(a,known)),"fact_errors":sum(f is not None and f!=q["fact"] for f,q in zip(a,known)),"unknown_n":len(u),"unknown_accepted":sum(x is not None for x in u),"pairs_n":len(p),"pairs_correct":sum(x==q["label"] for x,q in zip(p,pairs)),"pairs_wrong":sum(x is not None and x!=q["label"] for x,q in zip(p,pairs)),"parent_correct_lost":sum(x==q["fact"] and y!=q["fact"] for x,y,q in zip(answers["parent"]["facts"],a,known))}
        if name=="fresh":
            targets=[(z["text"],h37.STATES.index((z["fact"]["attribute"],z["fact"]["value"]))) for z in known]+[(t,h37.HOLD) for t in unknown];targets=[(t,y) for t,y in targets if t in lookup]
            direct,cpu,n=h36.direct_parent(c,[norm[t][1] for t,y in targets],[y for t,y in targets]);m["direct_parent"]=direct;m["teacher_cpu"]=cpu;m["teacher_pairs"]=n
        results[name]=m
    ck(results["33"]["parent"]["facts_correct"]==3 and results["34"]["parent"]["facts_correct"]==4 and results["36"]["parent"]["facts_correct"]==3,"PARENT_REPLAY")
    fresh=results["fresh"];d=rec["fresh"]["h37"]["direct"];dp=fresh["direct_parent"]
    def safe(z):return z["fact_errors"]==z["unknown_accepted"]==z["pairs_wrong"]==z["parent_correct_lost"]==0
    def direct_safe(z):return z["opposite_false_support"]==z["other_attribute_false_support"]==z["unknown_false_support"]==0
    passed=all(safe(m["h37_standalone"]) and safe(m["h37_same_candidate"]) for m in results.values()) and all(direct_safe(x["h37"]["direct"]) for x in [rec["fresh"],*rec["regression"].values()]) and fresh["h37_standalone"]["facts_correct"]>fresh["parent"]["facts_correct"] and fresh["h37_same_candidate"]["facts_correct"]>fresh["parent"]["facts_correct"] and fresh["h37_standalone"]["pairs_correct"]>=12 and d["true_support"]>dp["true_support"]
    report={"schema":"r84h37.result.v1","offline_receipt":rec,"comparison":results,"technical_independence":"PASS_BOUNDED_OFFLINE","quality_gate":"PASS_SHADOW" if passed else "HOLD","production":False,"scope":"Teacher-free standalone training/inference in isolated jobs; parent comparison in separate job; bounded six-property authored tests only, no general autonomy proof. Same-candidate and standalone results distinguished."}
    write(out/"result.json",report);log("R84H37_RESULT",report)
if __name__=="__main__":compare(*map(Path,sys.argv[1:]))
