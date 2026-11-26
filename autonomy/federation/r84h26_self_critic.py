"""R84H26 bounded independent-job strategy critic, conservative candidate admission."""
import sys,json,time
from pathlib import Path
import r84h25_metacognition as h25
import r84h19_exploration as h19
import r84h18_fog as h18
import r8_learning as r
HERE=Path(__file__).parent
EXAM="7df3898166dc1ec3817e5291c793a094e344de47"
EXPERIENCE="054b679922f9e4b903c64673246f6afdbb93d7d5464c34fbd64fcfd8743e314c"
MOTTO="Мы должны узнать"
def ck(v,m):h25.ck(v,m)
def read(p):return h25.read(p)
def write(p,x):h25.write(p,x)
def digest(x):return h25.digest(x)
def log(t,x):h25.log(t,x)
def loss(q):return .5*q["correct_flagged"]/max(q["correct_n"],1)+.5*(1-q["unknown_flagged"]/max(q["unknown_n"],1))
DOMAINS={
    "storage":("cache_hit","cache_miss","write_through","write_back","replication","sharding"),
    "concurrency":("mutex","semaphore","deadlock","starvation","idempotency","atomicity"),
    "security":("checksum","encryption","authentication","authorization"),
    "operations":("rate_limit","backpressure","rollback","rollforward")}
def domain(z,inv):
    name=inv.get(z["neural"],"")
    return next((k for k,vs in DOMAINS.items() if name in {"h12_"+s for s in vs}),"other")
def measures(rows,select):
    q=[z for z in rows if z["kind"]=="known"]
    out={"n":len(q),"base":0,"correct":0,"switches":0,"disagreements":0,"rescues":0,"losses":0,"agreement":0,"agreement_wrong":0}
    for z in q:
        use=bool(select(z));a=z["lexical"] if use else z["neural"];good=a==z["truth"];base=z["neural_correct"];dis=z["neural"]!=z["lexical"]
        out["base"]+=base;out["correct"]+=good;out["switches"]+=use;out["disagreements"]+=use and dis;out["rescues"]+=good and not base;out["losses"]+=base and not good;out["agreement"]+=not dis;out["agreement_wrong"]+=not dis and not base
    return out
def accept(fit,val):
    return fit["disagreements"]>=3 and fit["rescues"]>fit["losses"] and val["disagreements"]>=3 and val["rescues"]>=2 and val["losses"]==0
def inputs(ctx,exp,feedback):
    p=read(exp/"policy.json");old=read(exp/"experience.json");val=read(feedback/"feedback.json");result=read(feedback/"result.json")
    ck(p["experience_sha256"]==EXPERIENCE==digest(old) and len(old)==773 and not p["exam_read"] and not p["production"],"PARENT_EXPERIENCE")
    h25.restore(exp/"experience.sqlite",old);h25.restore(feedback/"feedback.sqlite",val)
    ck(result["science"]=="HOLD" and result["lost"]==6 and result["rescued"]==0 and len(val)==79 and result["historical_parity"]==671 and not result["production"],"PARENT_FAILURE")
    ck(not {h18.clean(z["text"]) for z in old}&{h18.clean(z["text"]) for z in val},"VALIDATION_OVERLAP")
    # Recompute actual historical-exam predictions/feature scores from the
    # frozen models; validation labels are never fitted here.
    ex=h19.load_exam("R84H25_FROZEN_EXAM.json",h25.EXAM)
    refs={h18.clean(t) for t,_ in ctx.fit+ctx.dev+ctx.oldknown}|{h18.clean(t) for t in ctx.oldunknown}
    known=[z for z in ex["known"] if h18.clean(z[0]) not in refs];unknown=[t for t in ex["unknown"] if h18.clean(t) not in refs]
    replay,cost=ctx.observations(known,unknown,"FROZEN_EVAL")
    ck(all(all(z[k]==v[k] for k in z) for z,v in zip(replay,val)) and len(replay)==len(val),"VALIDATION_REPLAY")
    return p,old,val,result,cost
def gate(*args):
    *paths,exp,feedback,out=args;ctx=h25.Context(*paths);out.mkdir(parents=True,exist_ok=False)
    p,old,val,result,cost=inputs(ctx,exp,feedback);inv={int(v):k for k,v in ctx.labels.items()}
    globalfit=measures(old,lambda z:h25.route(z,p)=="LEXICAL");globalval=measures(val,lambda z:h25.route(z,p)=="LEXICAL")
    ck(globalval["losses"]==6 and globalval["rescues"]==0 and globalval["correct"]==24,"FAILED_STRATEGY_REPLAY")
    # Domain identity comes only from the model's predicted class, not truth.
    # A candidate must show paired gain on disclosed fit and fresh H25 validation.
    domains={}
    for d in (*DOMAINS,"other"):
        f=[z for z in old if z["kind"]=="known" and domain(z,inv)==d];v=[z for z in val if z["kind"]=="known" and domain(z,inv)==d]
        fm=measures(f,lambda z:z["neural"]!=z["lexical"]);vm=measures(v,lambda z:z["neural"]!=z["lexical"])
        domains[d]={"fit":fm,"validation":vm,"accepted":accept(fm,vm)}
    globalaccepted=accept(globalfit,globalval)
    baseflags=h25.flag_metrics(val,"support",ctx.fixed["known_at_least"]);newflags=h25.flag_metrics(val,p["mode"],p["threshold"])
    referralaccepted=loss(newflags)<=loss(baseflags) and newflags["correct_flagged"]<=baseflags["correct_flagged"] and newflags["unknown_flagged"]>=baseflags["unknown_flagged"] and loss(newflags)<loss(baseflags)
    controls={
        "failed_global_rejected":not globalaccepted,
        "empty_rejected":not accept({"disagreements":0,"rescues":0,"losses":0},{"disagreements":0,"rescues":0,"losses":0}),
        "harm_rejected":not accept({"disagreements":20,"rescues":10,"losses":1},{"disagreements":8,"rescues":5,"losses":1}),
        "no_gain_rejected":not accept({"disagreements":20,"rescues":10,"losses":1},{"disagreements":8,"rescues":0,"losses":0}),
        "positive_control":accept({"disagreements":20,"rescues":10,"losses":1},{"disagreements":8,"rescues":3,"losses":0}),
        "referral_regression_rejected":not referralaccepted}
    ck(all(controls.values()),"CRITIC_CONTROLS")
    failures=[]
    for z in val:
        if z["kind"]=="known" and z["neural_correct"] and not z["selected_correct"]:
            failures.append({"id":digest(["POLICY_REGRESSION",z["id"]]),"phase":"DISCLOSED_H25_VALIDATION","kind":"POLICY_REGRESSION","domain":domain(z,inv),"case_id":z["id"],"text":z["text"],"neural":z["neural"],"candidate":z["selected_prediction"],"truth":z["truth"],"reason":"verified loss caused by route switch","production":False})
    ck(len(failures)==6,"FAILURE_LEDGER")
    h25.ledger(out/"policy_failures.sqlite",failures);h25.restore(out/"policy_failures.sqlite",failures);write(out/"policy_failures.json",failures)
    policy={"schema":"r84h26.critic.v1","experience_sha256":digest(old),"validation_sha256":digest(val),"h25_policy_sha256":digest(p),"global_candidate":{"fit":globalfit,"validation":globalval,"accepted":globalaccepted},"domains":domains,"referral":{"base":baseflags,"candidate":newflags,"accepted":referralaccepted,"mode":p["mode"] if referralaccepted else "support","threshold":p["threshold"] if referralaccepted else ctx.fixed["known_at_least"]},"controls":controls,"failure_sha256":digest(failures),"exam_read":False,"production":False,"motto":MOTTO}
    write(out/"policy.json",policy);log("R84H26_CRITIC",policy);log("R84H26_VALIDATION_COST",cost)
def evaluate(*args):
    *paths,exp,feedback,gateout,out=args;ctx=h25.Context(*paths);out.mkdir(parents=True,exist_ok=False)
    p=read(exp/"policy.json");old=read(exp/"experience.json");val=read(feedback/"feedback.json");policy=read(gateout/"policy.json");failures=read(gateout/"policy_failures.json")
    ck(policy["experience_sha256"]==EXPERIENCE==digest(old) and policy["validation_sha256"]==digest(val) and policy["h25_policy_sha256"]==digest(p) and policy["failure_sha256"]==digest(failures) and not policy["exam_read"] and not policy["production"] and all(policy["controls"].values()),"CRITIC_PIN")
    h25.restore(gateout/"policy_failures.sqlite",failures)
    raw=(HERE/"R84H26_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN");ex=json.loads(raw)
    ref={h18.clean(t) for t,_ in ctx.fit+ctx.dev+ctx.oldknown}|{h18.clean(t) for t in ctx.oldunknown}|{h18.clean(z["text"]) for z in val}
    known=[z for z in ex["known"] if h18.clean(z[0]) not in ref];unknown=[t for t in ex["unknown"] if h18.clean(t) not in ref]
    ck(len(known)>=55 and len(unknown)>=35 and len({h18.clean(t) for t,_ in known}|{h18.clean(t) for t in unknown})==len(known)+len(unknown),"EXAM_OVERLAP")
    rows,cost=ctx.observations(known,unknown,"H26_FRESH_EVAL");inv={int(v):k for k,v in ctx.labels.items()}
    select=lambda z:z["neural"]!=z["lexical"] and policy["domains"][domain(z,inv)]["accepted"]
    base=measures(rows,lambda z:False);oldroute=measures(rows,lambda z:h25.route(z,p)=="LEXICAL");guarded=measures(rows,select)
    per={d:measures([z for z in rows if z["kind"]=="known" and domain(z,inv)==d],select) for d in (*DOMAINS,"other")}
    fixed=h25.flag_metrics(rows,"support",ctx.fixed["known_at_least"]);referral=h25.flag_metrics(rows,policy["referral"]["mode"],policy["referral"]["threshold"])
    pred,routes,histcpu=ctx.neural(ctx.hist);parity=sum(ctx.prev[i]["answer"]==int(pred[i]) and ctx.prev[i]["route"]==int(routes[i]) for i in range(671));ck(parity==671,"HISTORY")
    plans={};policyerrors=[]
    for z in rows:
        wrong=False;dis=False;reason=None
        if z["kind"]=="known":
            alt=select(z);z["domain"]=domain(z,inv);z["selected_route"]="LEXICAL" if alt else "NEURAL";z["selected_prediction"]=z["lexical"] if alt else z["neural"];z["selected_correct"]=z["selected_prediction"]==z["truth"];wrong=not z["selected_correct"];dis=z["lexical"]!=z["neural"]
            if z["neural_correct"] and wrong:reason="POLICY_REGRESSION"
            elif wrong:reason="VERIFIED_CLASSIFICATION_ERROR"
            elif dis:reason="MECHANISM_DISAGREEMENT"
            if z["neural_correct"] and wrong:
                policyerrors.append({"id":digest(["H26_POLICY_REGRESSION",z["id"]]),"phase":"H26_FRESH_EVAL","case_id":z["id"],"kind":"POLICY_REGRESSION","domain":z["domain"],"text":z["text"],"production":False})
        if reason is None and z["scores"][policy["referral"]["mode"]]<policy["referral"]["threshold"]:reason="LOW_FAMILIARITY"
        if reason:
            a=h19.study(z["text"],"FRONTIER",z["scores"],{"mode":policy["referral"]["mode"],"unknown_below":policy["referral"]["threshold"]});a["reason"]=reason;plans[a["id"]]=a
    # Fresh exam never becomes training material in this run. Preserve failed
    # strategy cases from both stages as separately tagged feedback records.
    combined=failures+policyerrors;h25.ledger(out/"policy_failures.sqlite",combined);h25.restore(out/"policy_failures.sqlite",combined);write(out/"policy_failures.json",combined)
    order={"POLICY_REGRESSION":0,"VERIFIED_CLASSIFICATION_ERROR":1,"MECHANISM_DISAGREEMENT":2,"LOW_FAMILIARITY":3}
    queue=sorted(plans.values(),key=lambda a:(order[a["reason"]],a["score"],a["id"]))[:12];ck(h19.save_queue(out/"exploration.sqlite",queue)==len(queue) and h19.save_queue(out/"exploration.sqlite",queue)==0,"QUEUE")
    h25.ledger(out/"feedback.sqlite",rows);h25.restore(out/"feedback.sqlite",rows);write(out/"feedback.json",rows);write(out/"plans.json",queue)
    safe=guarded["losses"]==0 and guarded["correct"]>=base["correct"] and loss(referral)<=loss(fixed)
    improved=guarded["correct"]>base["correct"] or loss(referral)<loss(fixed)
    report={"schema":"r84h26.result.v1","motto":MOTTO,"known_n":len(known),"unknown_n":len(unknown),"overlap_excluded":100-len(known)-len(unknown),"base":base,"h25_candidate":oldroute,"guarded":guarded,"domain_metrics":per,"accepted_domains":[d for d,v in policy["domains"].items() if v["accepted"]],"global_candidate_accepted":policy["global_candidate"]["accepted"],"referral_candidate_accepted":policy["referral"]["accepted"],"fixed_referral":fixed,"guarded_referral":referral,"fixed_referral_loss":loss(fixed),"guarded_referral_loss":loss(referral),"historical_parity":parity,"critic_controls_pass":sum(policy["controls"].values()),"known_failed_policy_cases_saved":len(failures),"fresh_policy_regressions":len(policyerrors),"queue_n":len(queue),"research_candidates":len(plans),"ledger_replay":True,"cost":cost,"historical_batch_cpu_seconds":histcpu,"safety_gate":"PASS_SHADOW" if safe else "HOLD","improvement_gate":"PASS_SHADOW" if safe and improved else "HOLD_NO_PROVEN_GAIN","production":False,"scope":"same-owner separate fit/validation/exam jobs; groups derived from neural predicted software class; agreement is not truth; validation H25 already exposed; new exam synthetic; no new weights, live dispatch or production strategy change"}
    write(out/"result.json",report);log("R84H26_RESULT",report)
if __name__=="__main__":{"gate":gate,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
