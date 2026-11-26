"""H25 bounded learned arbitration from disclosed feedback; frozen fresh evaluation."""
import sys,json,sqlite3,time,math
from pathlib import Path
import numpy as np
import r84h19_exploration as h19
import r84h18_fog as h18
import r84h15_world as h15
import r84h14_partition as h14
import r84h13_streaming as h13
import r84h12_evolution as h12
import r84h16_chunk as h16
import r84e_neural as e
import r8_learning as r
HERE=Path(__file__).parent
EXAM="b2093cd4651985296c0ad547a58ac22728e17bc0"
MOTTO="Мы должны узнать"
def ck(v,m):e.ck(v,m)
def log(t,x):e.log(t,x)
def read(p):return e.read(p)
def write(p,x):e.write(p,x)
def digest(x):return h16.sha(h16.canonical(x).encode())
def ledger(path,rows):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS experience(id TEXT PRIMARY KEY,body TEXT NOT NULL,sha256 TEXT NOT NULL)")
        n=0
        for z in rows:
            body=h16.canonical(z);sha=h16.sha(body.encode());old=db.execute("SELECT body,sha256 FROM experience WHERE id=?",(z["id"],)).fetchone()
            if old:ck(old==(body,sha),"IMMUTABLE_FEEDBACK")
            else:db.execute("INSERT INTO experience VALUES(?,?,?)",(z["id"],body,sha));n+=1
        return n
def restore(path,rows):
    with sqlite3.connect(path) as db:
        q=db.execute("SELECT id,body,sha256 FROM experience").fetchall()
        ck(len(q)==len(rows) and all(h16.sha(b.encode())==s and json.loads(b)["id"]==i for i,b,s in q),"LEDGER_HASH")
        ck({i:json.loads(b) for i,b,_ in q}=={z["id"]:z for z in rows},"LEDGER_CONTENT")
    ck(ledger(path,rows)==0,"LEDGER_REPLAY")
class Context:
    def __init__(self,prep,rootpath,bundlepath,h10path,h11path,h12path,h19results,h24results):
        self.prep=prep;self.rootpath=rootpath;self.bundlepath=bundlepath;self.h10path=h10path
        self.L,self.dd,self.root,self.d,self.mapping,self.labels=h15.base_data(prep)
        f,n=h12.banks(self.d,h15.SEED,10,self.mapping,"fit");v,w=h12.banks(self.d,h15.SEED,10,self.mapping,"dev")
        self.fit=h18.uniq(f+n);self.dev=h18.uniq(v+w);ck(len(self.fit)==553 and len(self.dev)==200,"SPLIT")
        self.det=h19.Evidence(self.L,self.fit);self.cells,_,_,_,_=h13.catalog(self.L,self.d,self.mapping,self.labels,h15.SEED,0,h11path,h12path)
        self.h19=read(h19results/"result.json");self.h24=read(h24results/"result.json")
        ck(self.h19["scientific"]=="PASS_SHADOW" and self.h24["scientific"]=="PASS_SHADOW" and self.h24["historical_parity"]==671 and not self.h24["production"],"PARENT")
        self.fixed=self.h19["policy"];ck(self.fixed["mode"]=="support" and abs(self.fixed["known_at_least"]-.07922915220260619)<1e-12,"FIXED_POLICY")
        groups,_=h14.exams(prep,self.dd,self.d);self.hist=sum(groups,[]);ck(len(self.hist)==671,"HIST")
        ex19=h19.load_exam("R84H19_FROZEN_EXAM.json",h19.EXAM)
        ex24=h19.load_exam("R84H24_FROZEN_EXAM.json","fbda534b26226e3a4e0e16f897eddda1fe6003fa")
        self.oldknown=self.hist+ex19["freshKnown"];self.oldunknown=[z["text"] for z in read(h19results/"cases.json") if z["cohort"] in ("prior_unknown","fresh_unknown")]+ex24["unknown"]
        ck(len(self.oldknown)==691 and len(self.oldunknown)==82,"EXPERIENCE_SIZE")
        self.prev={z["i"]:z for z in read(h19results/"cases.json") if z["cohort"]=="historical"}
    def neural(self,rows):
        start=time.process_time()
        base,br,_=h13.persistent(self.L,self.root,self.d,h15.SEED,0,self.rootpath,self.bundlepath,self.h10path,self.mapping,rows)
        pred,routes,_=h13.eager(self.L,self.cells,rows,base,br)
        return pred,routes,time.process_time()-start
    def features(self,t):
        start=time.process_time();s=self.det.score(t);score_cpu=time.process_time()-start
        start=time.process_time();v=self.det.base.lex.vector(h18.terms(t));sim=self.det.base.ref@v
        idx=int(np.argmax(sim));nearest=int(self.labels[self.fit[idx][1]])
        lex_cpu=time.process_time()-start
        return s,nearest,score_cpu,lex_cpu
    def observations(self,known,unknown,phase):
        pred,routes,neural_cpu=self.neural(known);rows=[];sc=lc=0.
        for i,(t,label) in enumerate(known):
            score,near,a,b=self.features(t);sc+=a;lc+=b
            truth=int(self.labels[label]);z={"phase":phase,"kind":"known","text":t,"scores":score,"neural":int(pred[i]),"route":int(routes[i]),"lexical":near,"truth":truth,"neural_correct":int(pred[i])==truth,"lexical_correct":near==truth,"feedback":"FROZEN_LABEL"}
            z["id"]=digest([phase,"known",i,t]);rows.append(z)
        for i,t in enumerate(unknown):
            score,_,a,b=self.features(t);sc+=a;lc+=b
            z={"phase":phase,"kind":"unknown","text":t,"scores":score,"feedback":"OUTSIDE_LABELED_SCOPE_NOT_FACTUAL_VERDICT"};z["id"]=digest([phase,"unknown",i,t]);rows.append(z)
        return rows,{"neural_batch_cpu_seconds":neural_cpu,"score_cpu_seconds":sc,"lexical_cpu_seconds":lc,"known_n":len(known),"unknown_n":len(unknown)}
def flag_metrics(rows,mode,threshold):
    good=[z for z in rows if z["kind"]=="known" and z["neural_correct"]];bad=[z for z in rows if z["kind"]=="known" and not z["neural_correct"]];u=[z for z in rows if z["kind"]=="unknown"]
    flag=lambda z:z["scores"][mode]<threshold
    return {"correct_n":len(good),"correct_flagged":sum(flag(z) for z in good),"error_n":len(bad),"errors_flagged":sum(flag(z) for z in bad),"unknown_n":len(u),"unknown_flagged":sum(flag(z) for z in u)}
def learn(rows,fixed):
    # Learned referral rule: equal weights for unfamiliar recall and avoiding
    # false flags on correctly answered cases; error recall is a secondary tie.
    options=[]
    for mode in ("support","cosine","coverage","joint"):
        vals=sorted({z["scores"][mode] for z in rows});thresholds=[0.]+[(a+b)/2 for a,b in zip(vals,vals[1:])]+[1.0000001]
        for t in thresholds:
            q=flag_metrics(rows,mode,t);loss=.5*q["correct_flagged"]/max(1,q["correct_n"])+.5*(1-q["unknown_flagged"]/max(1,q["unknown_n"]))
            options.append((loss,-q["errors_flagged"],mode,t,q))
    loss,_,mode,t,metrics=min(options,key=lambda x:x[:4])
    # Fallback learned from paired, externally labeled past predictions; no
    # inference-time answer labels. Three fixed familiarity bins; require n>=20.
    bins=[]
    for lo,hi in ((0.,fixed["unknown_below"]),(fixed["unknown_below"],fixed["known_at_least"]),(fixed["known_at_least"],1.000001)):
        q=[z for z in rows if z["kind"]=="known" and lo<=z["scores"]["support"]<hi];nn=sum(z["neural_correct"] for z in q);ll=sum(z["lexical_correct"] for z in q)
        use=len(q)>=20 and ll-nn>=max(3,math.ceil(.05*len(q)))
        bins.append({"lo":lo,"hi":hi,"n":len(q),"neural_correct":nn,"lexical_correct":ll,"route":"LEXICAL" if use else "NEURAL"})
    return {"mode":mode,"threshold":t,"experience_loss":loss,"experience_metrics":metrics,"fixed_experience":flag_metrics(rows,"support",fixed["known_at_least"]),"fallback_bins":bins,"production":False}
def route(z,p):
    return next(b["route"] for b in p["fallback_bins"] if b["lo"]<=z["scores"]["support"]<b["hi"])
def train(*args):
    *paths,out=args;ctx=Context(*paths);out.mkdir(parents=True,exist_ok=False)
    rows,cost=ctx.observations(ctx.oldknown,ctx.oldunknown,"DISCLOSED_EXPERIENCE")
    ck(all(ctx.prev[i]["answer"]==z["neural"] and ctx.prev[i]["route"]==z["route"] and ctx.prev[i]["correct"]==z["neural_correct"] for i,z in enumerate(rows[:671])),"HISTORICAL_FEEDBACK_PIN")
    p=learn(rows,ctx.fixed);p.update({"schema":"r84h25.policy.v1","experience_sha256":digest(rows),"exam_read":False,"motto":MOTTO})
    ck(ledger(out/"experience.sqlite",rows)==len(rows),"LEDGER_WRITE");restore(out/"experience.sqlite",rows)
    write(out/"experience.json",rows);write(out/"policy.json",p);write(out/"cost.json",cost)
    log("R84H25_LEARN",{"experience_n":len(rows),"policy":p,"cost":cost,"production":False})
def evaluate(*args):
    *paths,trained,out=args;ctx=Context(*paths);out.mkdir(parents=True,exist_ok=False)
    raw=(HERE/"R84H25_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN");ex=json.loads(raw);p=read(trained/"policy.json");old=read(trained/"experience.json")
    ck(p["experience_sha256"]==digest(old) and not p["exam_read"] and not p["production"],"POLICY_PIN");restore(trained/"experience.sqlite",old)
    ref={h18.clean(t) for t,_ in ctx.fit+ctx.dev+ctx.oldknown}|{h18.clean(t) for t in ctx.oldunknown}
    known=[z for z in ex["known"] if h18.clean(z[0]) not in ref];unknown=[t for t in ex["unknown"] if h18.clean(t) not in ref]
    ck(len(known)>=35 and len(unknown)>=35 and len({h18.clean(t) for t,_ in known}|{h18.clean(t) for t in unknown})==len(known)+len(unknown),"EXAM_OVERLAP")
    rows,cost=ctx.observations(known,unknown,"FROZEN_EVAL")
    fixed=flag_metrics(rows,"support",ctx.fixed["known_at_least"]);adaptive=flag_metrics(rows,p["mode"],p["threshold"])
    knownrows=[z for z in rows if z["kind"]=="known"];fallback=0;correct=0;rescued=lost=0
    for z in knownrows:
        selected=route(z,p);z["selected_route"]=selected;z["selected_prediction"]=z["lexical"] if selected=="LEXICAL" else z["neural"];z["selected_correct"]=z["selected_prediction"]==z["truth"]
        fallback+=selected=="LEXICAL";correct+=z["selected_correct"];rescued+=z["selected_correct"] and not z["neural_correct"];lost+=z["neural_correct"] and not z["selected_correct"]
    # Historical answers are always the compatibility path; recompute actual
    # legacy classification and routes on the exact historical requests.
    pred,routes,histcpu=ctx.neural(ctx.hist)
    parity=sum(ctx.prev[i]["answer"]==int(pred[i]) and ctx.prev[i]["route"]==int(routes[i]) for i in range(671));ck(parity==671,"HISTORICAL_PARITY")
    # Only externally labeled errors or uncertain out-of-scope prompts become
    # local study proposals. Fresh labels are not consumed by policy learning.
    plans={}
    for z in rows:
        wrong=z["kind"]=="known" and not z["selected_correct"];flag=z["scores"][p["mode"]]<p["threshold"]
        if wrong or flag:
            a=h19.study(z["text"],"FRONTIER",z["scores"],{"mode":p["mode"],"unknown_below":p["threshold"]});a["reason"]="VERIFIED_CLASSIFICATION_ERROR" if wrong else "LOW_FAMILIARITY";plans[a["id"]]=a
    queue=sorted(plans.values(),key=lambda x:(x["reason"]!="VERIFIED_CLASSIFICATION_ERROR",x["score"],x["id"]))[:12];ck(h19.save_queue(out/"exploration.sqlite",queue)==len(queue) and h19.save_queue(out/"exploration.sqlite",queue)==0,"QUEUE")
    ck(ledger(out/"feedback.sqlite",rows)==len(rows),"FEEDBACK_WRITE");restore(out/"feedback.sqlite",rows)
    write(out/"feedback.json",rows);write(out/"plans.json",queue)
    def loss(q):return .5*q["correct_flagged"]/max(q["correct_n"],1)+.5*(1-q["unknown_flagged"]/max(q["unknown_n"],1))
    basecorrect=sum(z["neural_correct"] for z in knownrows)
    report={"schema":"r84h25.result.v1","motto":MOTTO,"known_n":len(known),"unknown_n":len(unknown),"overlap_excluded":80-len(known)-len(unknown),"historical_parity":parity,"experience_n":len(old),"fixed":fixed,"adaptive":adaptive,"fixed_loss":loss(fixed),"adaptive_loss":loss(adaptive),"neural_correct":basecorrect,"lexical_correct":sum(z["lexical_correct"] for z in knownrows),"adaptive_correct":correct,"fallback_selected":fallback,"rescued":rescued,"lost":lost,"cost":cost,"historical_batch_cpu_seconds":histcpu,"H24_historical_route_cost_receipt":{"run":45790325413,"exact_cpu_seconds":ctx.h24["exact_only_cpu_seconds"],"always_shadow_cpu_seconds":ctx.h24["always_shadow_cpu_seconds"],"memory_cpu_seconds":ctx.h24["memory_cpu_seconds"],"memory_n":ctx.h24["memory_n"]},"new_research_candidates":len(plans),"queue_n":len(queue),"verified_error_plans":sum(x["reason"]=="VERIFIED_CLASSIFICATION_ERROR" for x in queue),"ledger_replay":True,"science":"PASS_SHADOW" if loss(adaptive)<loss(fixed) and lost==0 and correct>=basecorrect else "HOLD","production":False,"scope":"same-owner fresh synthetic classification/novelty; learned referral threshold and fallback bins from disclosed prior labels; historical predictions unchanged by policy; route costs measured in batches, not online service; H24 graph/memory costs historical, not rerun; unknowns outside labeled scope, not verified false answers; no self-labeled learning, no external dispatch or promotion"}
    write(out/"result.json",report);log("R84H25_RESULT",report)
if __name__=="__main__":
    {"train":train,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
