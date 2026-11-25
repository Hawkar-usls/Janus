"""R84H11 finite append-only neural expert and competence gate experiment."""
import sys,random,math,time,json
from pathlib import Path
import r84h10_resonance as h10
import r84h9_synapse as h9
import r84h8_neurogenesis as h8
import r84g_adaptive as a
import r84e_neural as e
import r8_learning as r
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="7d1c8a25d8523c060814f9254fa5de8db3671fff"
PINS=("34144067ddc6f41d1625dc42a4358aa798ece0de97f4024878154bbdeabb038f","8a5eaf94030325df6b97d23f6363ed2a7a27dab3a4342af1b90260bf166149a5","45d553b20139aa240a4ea27e529424fb897ce92d21e359a0cf2b0b1557dc1e00")
FIT=[
{"h11_confounder":["confounder causes both treatment and outcome","common antecedent drives exposure and response","third factor confounds causal estimate","adjust common cause of exposure and disease","shared upstream variable creates misleading association","confounding variable affects predictor and outcome independently"],"h11_mediator":["mediator lies along causal chain","intermediate variable transmits causal effect","treatment affects mediator then outcome","indirect effect passes through intervening mechanism","mediating process links intervention to response","causal pathway contains intermediate step"]},
{"h11_collider":["collider is common effect of two causes","conditioning on common consequence induces bias","two arrows point into collider variable","selection bias through shared downstream effect","collider opens path when conditioned upon","common outcome connects otherwise independent causes"],"h11_modifier":["effect modifier changes treatment effect size","interaction means causal effect varies by subgroup","effect heterogeneity across levels of covariate","treatment benefit depends on patient characteristic","moderator alters strength of causal relationship","effect modification across population strata"]},
{"h11_itt":["intention to treat uses original assignment","analyze randomized groups irrespective of compliance","keep assigned groups after treatment crossover","include nonadherent subjects by initial allocation","estimate effect of randomized assignment","intention to treat preserves allocation"],"h11_protocol":["per protocol analysis includes adherent participants","restrict trial analysis to protocol compliant subjects","exclude major treatment deviations from analysis","analyze only participants following assigned regimen","estimate treatment effect in compliant trial population","per protocol considers adherence to intervention"]},
{"h11_censor":["censoring records incomplete event time","right censored subjects followed without observed failure","retain observation with unknown exact survival time","event occurs after end of follow up","censored record provides partial time information","survival outcome known only beyond last contact"],"h11_truncate":["truncation excludes observations outside sampling bounds","truncated sample omits entire out of range cases","only event times within window are included","left truncation excludes earlier failures","selection requires event time within limits","truncation removes records rather than partial measurement"]},
{"h11_stationary":["stationarity means time invariant statistical properties","stationary process has constant mean and covariance structure","distribution unchanged under shift of time index","time series properties stable over time","stationary fluctuations have no changing distribution","statistical law independent of time origin"],"h11_ergodic":["ergodicity equates time and ensemble averages","single long trajectory represents ensemble behavior","ergodic process time mean converges to population mean","long run average equals expectation over realizations","ergodicity permits inference from one long realization","trajectory averages reproduce ensemble statistics"]}]
DEV=[
{"h11_confounder":["factor jointly determines exposure and disease risk","third variable makes treatment association misleading","shared cause biases estimated relationship"],"h11_mediator":["intervening step conveys treatment influence","mechanism through which exposure changes outcome","indirect causal pathway contains intermediate variable"]},
{"h11_collider":["shared consequence of independent predictors","selection on common effect induces dependence","conditioning on variable with two incoming causal arrows"],"h11_modifier":["response to treatment depends on subgroup","covariate changes magnitude of intervention benefit","causal effect heterogeneous across patient groups"]},
{"h11_itt":["retain participants in their randomized arms despite noncompliance","estimate assignment rather than received treatment effect","original allocation determines analysis group"],"h11_protocol":["analyze patients who complete prescribed regimen","limit trial comparison to compliant subjects","participants deviating from regimen excluded"]},
{"h11_censor":["incomplete survival time retained as lower bound","followup stops while subject still event free","outcome time unobserved but subject retained"],"h11_truncate":["exclude cases outside eligibility time range","records absent when event times fall beyond bounds","sample restricted by event occurrence window"]},
{"h11_stationary":["probability law does not vary with time origin","constant statistical moments in time series","time shift leaves joint distribution unchanged"],"h11_ergodic":["long trajectory average approaches ensemble mean","time mean equals average across realizations","one extended path sufficient for estimating expectation"]}]
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def labels(d):
    out=dict(d["labels"]);next_id=max(out.values())+1
    for group in FIT:
        for y in sorted(group):out[y]=next_id;next_id+=1
    return out
def banks(d,seed,cycle,role):
    # h10 split requires the original H8 routing map.
    mapping={y:0 for y in d["labels"]}
    for i,grp in enumerate(h8.TRAIN,1):
        for y in grp:mapping[y]=i
    f,v=h10.sets(d,mapping,seed)
    base=f if role=="fit" else v
    source=FIT if role=="fit" else DEV
    old=base+[row for grp in source[:cycle-1] for row in h10.extra(grp)]
    new=h10.extra(source[cycle-1]);return old,new
def score_gate(L,x,ref,nold):
    torch,_=L;sim=x@ref.T
    return (1+sim[:,nold:].max(-1).values-sim[:,:nold].max(-1).values)/2
def choose(L,p,nold):
    torch,_=L;best=None
    for t in sorted(set([1.01]+[float(z) for z in p[nold:]])):
        wrong=int((p[:nold]>=t).sum());hit=int((p[nold:]>=t).sum())
        if wrong:continue
        key=(hit,t)
        if best is None or key>best[0]:best=(key,{"threshold":t,"old_n":nold,"new_n":len(p)-nold,"old_intercepts":wrong,"new_hits":hit})
    ck(best is not None,"GATE_CALIBRATION");return best[1]
def model(L,dim,seed):
    torch,_=L;torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(dim,32),torch.nn.Tanh(),torch.nn.Linear(32,2))
def fit_module(L,x,y,seed):
    torch,_=L;m=model(L,x.shape[1],seed);opt=torch.optim.AdamW(m.parameters(),lr=.005,weight_decay=.01)
    groups=[[i for i,t in enumerate(y.tolist()) if t==k] for k in range(2)];rng=random.Random(seed)
    for step in range(240):
        ids=sum([rng.choices(grp,k=6) for grp in groups],[])
        loss=torch.nn.functional.cross_entropy(m(x[ids]),y[ids]);ck(math.isfinite(float(loss.detach())),"LOSS")
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return m
def load_h10(L,d,path,seed,i):
    torch,_=L;rec=e.read(path/"receipt.json");p=path/"shadow.pt"
    ck(rec["file_sha256"]==e.sha(p.read_bytes()) and rec["data_sha256"]==g.digest(d) and rec["model_sha256"]==PINS[i] and rec["production"] is False,"H10_PIN")
    f,_=h10.sets(d,{y:(next((j for j,grp in enumerate(h8.TRAIN,1) if y in grp),0)) for y in d["labels"]},seed)
    ck(rec["lex"]==h10.lexical_fit(f),"H10_LEX");m=torch.nn.Linear(len(rec["lex"]["keys"]),3);m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),strict=True)
    ck(e.state_hash(m)==PINS[i],"H10_HASH");return rec,f
def load_module(L,d,path,seed,cycle,last,basepin):
    torch,_=L;rec=e.read(path/"receipt.json");p=path/"shadow.pt"
    ck(rec["file_sha256"]==e.sha(p.read_bytes()) and rec["data_sha256"]==g.digest(d) and rec["production"] is False and rec["parent_sha256"]==last and rec["base_sha256"]==basepin and rec["cycle"]==cycle and rec["seed"]==seed,"MODULE_LINEAGE")
    old,new=banks(d,seed,cycle,"fit");ck(rec["ref_sha256"]==g.digest(old+new) and rec["lex"]==h10.lexical_fit(old+new),"REF")
    m=model(L,len(rec["lex"]["keys"]),seed+cycle*1000);m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),strict=True)
    ck(e.state_hash(m)==rec["model_sha256"],"MODULE_HASH");m.eval();return m,rec,old,new
def train(prep,root_path,bundle_path,h10path,out):
    L=e.lib(prep);torch,_=L;*_,ds=h8.data(prep);root,d=ds[0],ds[-1];out.mkdir(parents=True,exist_ok=False)
    for cycle in range(1,6):
        for i,seed in enumerate(e.SEEDS):
            start=time.process_time();core,b=h9.source(L,root,d,seed,i,root_path,bundle_path);load_h10(L,d,h10path/str(seed),seed,i)
            last=PINS[i];previous=[]
            for k in range(1,cycle):
                q,qr,_,_=load_module(L,d,out/str(k)/str(seed),seed,k,last,PINS[i]);last=qr["model_sha256"];previous.append((q,last))
            old,new=banks(d,seed,cycle,"fit");vo,vn=banks(d,seed,cycle,"dev")
            ck(not {a.norm(t) for t,_ in old+new}&{a.norm(t) for t,_ in vo+vn},"FIT_DEV")
            lex=h10.lexical_fit(old+new);x=h10.lexical(L,lex,old+new);v=h10.lexical(L,lex,vo+vn)
            ys=sorted(FIT[cycle-1]);y=torch.tensor([ys.index(t) for _,t in new])
            m=fit_module(L,x[len(old):],y,seed+cycle*1000);p=score_gate(L,v,x,len(old));gate=choose(L,p,len(vo))
            ck(e.state_hash(core)==h9.ROOTS[i] and e.state_hash(b)==h9.BUNDLES[i] and all(e.state_hash(q)==sha for q,sha in previous),"FROZEN")
            rec={"seed":seed,"cycle":cycle,"parent_sha256":last,"base_sha256":PINS[i],"ref_sha256":g.digest(old+new),"lex":lex,"gate":gate,"steps":240,"presentations":2880,"fit_pos":len(new),"fit_neg":len(old),"previous_modules_verified":len(previous),"parameters":sum(p.numel() for p in m.parameters()),"cpu_seconds":time.process_time()-start}
            e.save(L,m,d,out/str(cycle)/str(seed),rec);log("R84H11_TRAIN",{k:v for k,v in rec.items() if k!="lex"})
def exam(prep,dd):
    groups,_=h10.exams(prep,dd);ck(sum(map(len,groups))==479,"PRIOR")
    raw=(HERE/"R84H11_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN")
    added=[r for grp in FIT+DEV for r in h10.extra(grp)]
    d=dd[-1][-1];train=d["old"]+d["new"]+h10.extra(h10.FIT)+h10.extra(h10.DEV)+added+sum(d["monitor"],[])
    norm={a.norm(t) for t,_ in train}
    ck(not {a.norm(t) for t,_ in added}&{a.norm(t) for grp in groups for t,_ in grp},"HISTORIC_COLLISION")
    norm.update(a.norm(t) for grp in groups for t,_ in grp);fresh=[]
    for grp in json.loads(raw)["cohorts"]:
        clean=[]
        for t,y in grp:
            n=a.norm(t)
            if n in norm:continue
            norm.add(n);clean.append([t,y])
        ck(len(clean)>=4,"FRESH");fresh.append(clean)
    log("R84H11_EXAM",{"prior_n":479,"fresh_n":list(map(len,fresh))})
    return groups+fresh,fresh
def evaluate(prep,root_path,bundle_path,h10path,trained,out):
    L=e.lib(prep);torch,_=L;dd=h8.data(prep);ds=dd[-1];root,d=ds[0],ds[-1];groups,fresh=exam(prep,dd);rows=sum(groups,[]);ls=labels(d)
    mapping=h8.routes(root,ds,3)
    for k,grp in enumerate(FIT,4):
        for y in grp:mapping[y]=k
    y=torch.tensor([mapping[t] for _,t in rows]);actual=torch.tensor([ls[t] for _,t in rows]);old=y<=3;results=[]
    for i,seed in enumerate(e.SEEDS):
        core,b=h9.source(L,root,d,seed,i,root_path,bundle_path);_,_,_,cand=h9.inputs(L,core,b,root,d,rows)
        hr,f=load_h10(L,d,h10path/str(seed),seed,i);lex=hr["lex"];xx=h10.lexical(L,lex,rows);fx=h10.lexical(L,lex,f)
        fy=torch.tensor([mapping[t] for _,t in f]);p=h10.prototype(L,xx,fx,fy);z=h9.route(L,p,hr["thresholds"]["prototype"]["threshold"])
        base=cand[torch.arange(len(rows)),z];baseok=base==actual
        oracle=torch.full_like(base,-1);oracle[old]=cand[torch.arange(len(rows))[old],y[old]]
        prev={"guarded":base.clone(),"unguarded":base.clone()};pred={k:v.clone() for k,v in prev.items()}
        routes={k:z.clone() for k in prev};last=PINS[i];cumcpu=0.;prev_params=0;frozen=[]
        for cycle in range(1,6):
            start=time.process_time();m,rec,fo,fn=load_module(L,d,trained/str(cycle)/str(seed),seed,cycle,last,PINS[i]);last=rec["model_sha256"];frozen.append((m,last))
            lex=rec["lex"];xx=h10.lexical(L,lex,rows);fx=h10.lexical(L,lex,fo+fn);p=score_gate(L,xx,fx,len(fo))
            with torch.no_grad():
                ids=torch.tensor([ls[t] for t in sorted(FIT[cycle-1])]);answer=ids[m(xx).argmax(-1)]
            routeid=cycle+3;now=y==routeid;oracle[now]=answer[now]
            cumcpu+=rec["cpu_seconds"];prev_params+=rec["parameters"]
            for mode,threshold in [("guarded",rec["gate"]["threshold"]),("unguarded",.5)]:
                selected=p>=threshold;pred[mode][selected]=answer[selected];routes[mode][selected]=routeid
                correct=pred[mode]==actual;before=prev[mode]==actual;active=y<=routeid;known_new=(y>3)&active
                lost=int((before&~correct).sum());dep=int(correct.sum());gain=dep-int(before.sum())
                row={"mode":mode,"seed":seed,"cycle":cycle,"n":len(rows),"before":int(before.sum()),"dep":dep,"lost":lost,"root_before":int(baseok.sum()),"root_lost":int((baseok&~correct).sum()),"net_gain":gain,"cpu_seconds":rec["cpu_seconds"],"cumulative_cpu_seconds":cumcpu,"profit_cpu":gain/rec["cpu_seconds"],"current_route_correct":int((selected&now).sum()),"current_n":int(now.sum()),"new_route_correct":int(((routes[mode]==y)&known_new).sum()),"new_n":int(known_new.sum()),"old_intercepts":int((selected&(y<routeid)).sum()),"prior_n":int((y<routeid).sum()),"oracle_dep":int((oracle==actual).sum()),"additional_parameters":prev_params,"threshold":threshold,"eval_cpu_seconds":time.process_time()-start,"core_unchanged":e.state_hash(core)==h9.ROOTS[i],"previous_modules_unchanged":e.state_hash(b)==h9.BUNDLES[i] and all(e.state_hash(q)==sha for q,sha in frozen),"model_sha256":last}
                results.append(row);log("R84H11_RESULT",row);prev[mode]=pred[mode].clone()
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84h11.result.v1","results":results,"production":False,"external_validation":False,"scope":"five append-only two-class MLPs and independently fitted contrastive lexical gates. Existing H10 prototype and experts pinned. Guarded zero dev intercept vs unguarded fixed 0.5. Latest admitted gate takes precedence. Synthetic text collision screening, not semantic independence. No auto promotion."})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:]])
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
