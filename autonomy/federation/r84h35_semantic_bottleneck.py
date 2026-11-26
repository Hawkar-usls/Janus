"""H35 diagnostic only. Disclosed exam, no fitting or promotion."""
import sys,time,math
from collections import Counter
from pathlib import Path
import torch
import r84h34_semantic_crystal_expansion as h34
import r84h33_json_crystallization as h33
import r84h32_json_semantic_compiler as h32
import r84h31_semantic_evolution as h31
import r84h19_exploration as h19
from r84h30_semantic_inheritance import history
STATES=h32.STATES;HOLD=h32.HOLD
ck=h32.ck;read=h32.read;write=h32.write;log=h32.log;digest=h32.digest
def rank(scores):return sorted(range(len(scores)),key=lambda i:(-scores[i],i))
def metric(rows,selector):
    result={"known":0,"correct":0,"wrong":0,"unknown":0,"unknown_accepted":0,"abstained_known":0}
    for z in rows:
        y=z["truth"];p=selector(z)
        if y==HOLD:
            result["unknown"]+=1;result["unknown_accepted"]+=p is not None
        else:
            result["known"]+=1;result["correct"]+=p==y;result["wrong"]+=p is not None and p!=y;result["abstained_known"]+=p is None
    return result
def diagnose(demi,source,trained,h34results,out):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);ctx=h34.Context(demi,source,trained);policy=ctx.policy
    prior=read(h34results/"result.json");ck(prior["schema"]=="r84h34.result.v1" and prior["safety"]=="HOLD" and not prior["gate_enabled"] and not prior["production"] and prior["dev"]["weight_sha"]==ctx.z["sha"] and prior["dev"]["dictionary_sha"]==ctx.rec["dictionary_sha"],"H34_PIN")
    ex=h19.load_exam("R84H34_FROZEN_EXAM.json",h34.EXAM);norm=h34.corpus(ex)
    rows=[{"text":z["text"],"truth":STATES.index((z["fact"]["attribute"],z["fact"]["value"])),"fact":z["fact"]} for z in ex["statements"]]+[{"text":t,"truth":HOLD,"fact":None} for t in ex["unknown"]]
    valid=[z for z in rows if norm[z["text"]] is not None];texts=[norm[z["text"]][1] for z in valid]
    net,ncpu=h33.probabilities(ctx.model,texts);lex=[ctx.d.search(t) for t in texts];ev=h33.evidence(ctx.sem,texts,net,lex)
    full,lcpu=h32.lexical(ctx.tr,texts)
    pairs=[];mapping=[]
    for i,z in enumerate(valid):
        n=rank(net[i])[0];l=rank(lex[i][0])[0];f=rank(full[i])[0]
        # All candidate classes and gold (known only), even if prior route
        # never called teacher. Two correlated hypotheses per class.
        for label in sorted({n,l,f,z["truth"]}-{HOLD}):
            a,v=STATES[label];mapping.append((i,label))
            for h in (0,1):
                pairs.extend([(texts[i],h31.HYP[a][v][h].format(o="entity")),(texts[i],h31.HYP[a][1-v][h].format(o="entity"))])
    start=time.process_time();probs=ctx.sem.pairs(pairs) if pairs else [];tcpu=time.process_time()-start
    evidence={}
    for j,(i,label) in enumerate(mapping):
        scores=[]
        for k in (0,1):
            s,o=probs[4*j+2*k:4*j+2*k+2];scores.append({"support":s[1],"opposite_contradiction":o[0],"opposite_entailment":o[1]})
        evidence.setdefault(i,{})[label]=scores
    def semantic_fail(e):
        if e is None:return ["semantic_not_evaluated"]
        return [k for k,test in (("semantic_support",e["support"]<policy["semantic"]),("semantic_opposite",e["opposite_contradiction"]<policy["opposite"]),("semantic_opposite_entailment",e["opposite_entailment"]>.20)) if test]
    known_labels=Counter(z["label"] for z in ctx.tr)
    for z in rows:
        if norm[z["text"]] is None:z.update({"baseline":None,"first_block":"subject","failures":["subject"],"ablation_eligible":False});continue
    for i,z in enumerate(valid):
        y=z["truth"];t=texts[i];n,c,g=h32.candidate(net[i]);l,s,lm=h32.candidate(lex[i][0]);f,fs,fm=h32.candidate(full[i]);base,proof=h33.select(t,net[i],lex[i],ev.get(i),policy)
        failures=[]
        if n==HOLD:failures.append("neural_hold")
        if n!=l:failures.append("disagreement")
        if lex[i][1][l] is None:failures.append("dictionary_pointer")
        if not h33.scope(t):failures.append("scope")
        if c<policy["confidence"]:failures.append("neural_confidence")
        if g<policy["margin"]:failures.append("neural_margin")
        if s<policy["lexical"]:failures.append("lexical_score")
        if n==l and n!=HOLD and lex[i][1][l] is not None and h33.scope(t):
            failures+=semantic_fail(ev.get(i))
        else:failures.append("semantic_not_reached")
        ck((base is not None)==(not failures),"REPLAY_GATE")
        gold=None
        if y!=HOLD:
            gold={"neural_rank":rank(net[i]).index(y)+1,"neural_score":net[i][y],"neural_attribute_correct":n!=HOLD and STATES[n][0]==STATES[y][0],"dictionary_rank":rank(lex[i][0]).index(y)+1,"dictionary_score":lex[i][0][y],"dictionary_pointer":lex[i][1][y],"dictionary_class_documents":known_labels[y],"full_dictionary_rank":rank(full[i]).index(y)+1,"full_dictionary_score":full[i][y],"teacher":evidence.get(i,{}).get(y)}
            ck(gold["teacher"] is not None,"GOLD_NLI")
        z.update({"baseline":base,"first_block":failures[0] if failures else "admitted","failures":failures,"neural":n,"lexical":l,"full_lexical":f,"neural_probabilities":net[i],"retrieval_scores":lex[i][0],"retrieval_pointers":lex[i][1],"full_dictionary_scores":full[i],"teacher_scores":evidence.get(i,{}),"gold":gold,"ablation_eligible":True,"scope":h33.scope(t),"proof":proof})
    baseline=metric(rows,lambda z:z["baseline"]);b=prior["fresh"]["baseline"]
    ck(baseline["correct"]==b["facts_exact"]==4 and baseline["wrong"]==b["fact_errors"]==0 and baseline["unknown_accepted"]==b["unknown_accepted"]==0 and baseline["known"]==36 and baseline["unknown"]==24,"H34_REPLAY")
    held=[z for z in rows if z["truth"]!=HOLD and z["baseline"] is None];ck(len(held)==32,"HOLD_COUNT")
    counts=Counter();families={}
    for z in held:
        y=z["truth"];gold=z.get("gold");a=STATES[y][0];f=families.setdefault(a,Counter());f["held"]+=1
        if not gold:counts["subject_invalid"]+=1;continue
        tests={"neural_wrong_class":z["neural"]!=y,"neural_wrong_attribute":not gold["neural_attribute_correct"],"neural_right_attribute_wrong_value":gold["neural_attribute_correct"] and z["neural"]!=y,"dictionary_wrong_top":z["lexical"]!=y,"gold_pointer_missing_top8":gold["dictionary_pointer"] is None,"gold_class_absent_corpus":gold["dictionary_class_documents"]==0,"full_dictionary_recovers_top":z["lexical"]!=y and z["full_lexical"]==y,"both_extract_correct":z["neural"]==z["lexical"]==y,"teacher_gold_first_fails":bool(semantic_fail(gold["teacher"][0])),"teacher_gold_both_fail":all(semantic_fail(e) for e in gold["teacher"]),"teacher_gold_first_pass_second_fails":not semantic_fail(gold["teacher"][0]) and bool(semantic_fail(gold["teacher"][1])),"teacher_gold_second_pass_first_fails":bool(semantic_fail(gold["teacher"][0])) and not semantic_fail(gold["teacher"][1]),"teacher_not_reached": "semantic_not_reached" in z["failures"]}
        for k,v in tests.items():counts[k]+=int(v);f[k]+=int(v)
    # Diagnostic ablations on already disclosed labels. No threshold selection
    # and no enabling. Report all wrong/unknown consequences.
    ablations={}
    for gate in ("neural_confidence","neural_margin","lexical_score","semantic_support","semantic_opposite","semantic_opposite_entailment","all_semantic"):
        drop={gate} if gate!="all_semantic" else {"semantic_support","semantic_opposite","semantic_opposite_entailment"}
        def selector(z):
            if not z["ablation_eligible"] or set(z["failures"])-drop:return None
            return z["neural"]
        ablations[gate]=metric(rows,selector)
    # Full-dictionary top vs retrieval top is a comparison, not a new route.
    report={"schema":"r84h35.result.v1","source_exam_blob":h34.EXAM,"source_weight_sha":ctx.z["sha"],"dictionary_sha":ctx.rec["dictionary_sha"],"baseline":baseline,"held_n":len(held),"first_block":dict(Counter(z["first_block"] for z in held)),"overlapping_gate_failures":dict(Counter(k for z in held for k in z["failures"])),"component_diagnostics":dict(counts),"per_attribute":{a:dict(f) for a,f in families.items()},"ablation_diagnostic_only":ablations,"teacher_pairs":len(pairs),"neural_cpu":ncpu,"full_dictionary_cpu":lcpu,"teacher_diagnostic_cpu":tcpu,"dictionary_documents":len(ctx.tr),"production":False,"training_performed":False,"scope":"Disclosed H34 exam, 36 known/24 unknown; gold NLI diagnostic is label-informed and not a runtime route. First blocker depends on gate order, overlapping failures not additive. Full lexical scan uses same corpus but different retrieval scoring; cannot infer missing semantic paraphrase solely from class presence. No independent generalization or threshold change."}
    write(out/"result.json",report);write(out/"cases.json",rows);log("R84H35_RESULT",report)
if __name__=="__main__":{"diagnose":diagnose,"history":history}[sys.argv[1]](*map(Path,sys.argv[2:]))
