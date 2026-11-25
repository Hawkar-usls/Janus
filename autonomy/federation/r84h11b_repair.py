"""R84H11B: finite, frozen-expert gate repair; no exam access during calibration."""
import sys,json,time
from pathlib import Path
import r84h11_cortex as h11
import r84h10_resonance as h10
import r84h9_synapse as h9
import r84h8_neurogenesis as h8
import r84e_neural as e
import r84g_adaptive as a
import r8_learning as r
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="b9ddf35690a55d514f89f49d7d344d2cdb4a66de"
PINS=[
("f1880da45a19a10327bca2815a3df79e4b2ee31bda2413c1abc066ab8dad33b1","f6ac70c77aa6e7111a6b43621b81bfd3c9e99016bd8d02987617ed00e3f4815f","89caa74e18d42d49602961f5e4138fff0fbad88bcb73975520b95a22a14a1138"),
("80d685320b87d433ae56c1484b688b6221e7f759b9e760675f3db947c28ab027","1ceeb94528536cc57fa6efd2741760d3cc55d68f6139f6f492d1049a63667c51","6673d8e7389c2f7dc2eeffe601f59043bbe341797554e28e8087e1b4bd1f0194"),
("1374b82bf83ed0cd6804dd58a1556db54bfaa787e96e356184152bd43bb03517","a5bb129d14e00edabea0775c45747deb77c9fbcad648932a00a18589c1796c34","259f6f1872780bdb0c793aff70b8bfc971d9a8fe14c4d79b36259c13f98e1ba4"),
("2fb4766a17d6df2a3e0a09b09aa9dabb143cfba209954de4fef181f294a90a64","aa4d222d6130cea093b089a82a0115e2e22f91da15573b309d3368631e86f13f","b6fba97da24371b38f00e88004e275d9a61dbc5215b8ef947193ba1fcf42eace"),
("4fa0d873b9b0e9bf3dbe1634ef7ecc9303e9abe7598144a684d7fc00366fe357","521c043563823b3e41498bf32faeaa2ac54a1041262653d9c966bdec48fdca7b","e9dd14845b477d6cceddfb2f0d8c58c30169a6d35d5606c37ea981dec9e45e62")]
FIT={
"h11_confounder":["common risk factor causes exposure and disease","shared precursor confounds observed effect"],
"h11_mediator":["causal intermediate explains indirect impact","mechanistic link conveys treatment effect"],
"h11_collider":["conditioning on joint outcome associates its independent determinants","two distinct causal inputs meet at an effect node","common downstream consequence causes selection distortion","dependence induced by stratifying on shared effect"],
"h11_modifier":["effect size depends on a third characteristic","interaction between intervention and covariate changes response","benefit differs across patient strata","conditional causal effects vary between groups"],
"h11_itt":["compare assigned arms even if medication not received","random allocation defines comparison regardless of deviation"],
"h11_protocol":["compare subjects meeting trial adherence requirements","exclude noncompliant participants from analysis"],
"h11_censor":["retain subject when exact event time unavailable","partially observed failure time bounded by last followup"],
"h11_truncate":["sample excludes subjects outside event eligibility range","observation absent due to time cutoff"],
"h11_stationary":["time translation preserves probability structure","unchanging process distribution over time","stable mean and autocovariance at fixed lag","statistical behavior independent of clock position"],
"h11_ergodic":["temporal mean represents expected ensemble value","one extended sample path gives representative population averages","long trajectory average converges to probabilistic expectation","time average substitutes for averaging independent realizations"]}
DEV={
"h11_confounder":["uncontrolled common determinant creates biased association","third causal variable affects both measured factors"],
"h11_mediator":["effect passes through an intermediate mechanism","variable transmits part of the causal influence"],
"h11_collider":["select on a consequence shared by unrelated causes","conditioning on downstream convergence induces dependence","observing common outcome connects independent causal parents"],
"h11_modifier":["subgroup membership changes causal response magnitude","intervention has heterogeneous effect according to covariate","another characteristic alters treatment effect strength"],
"h11_itt":["compare according to initially allocated intervention","keep randomized groups despite adherence differences"],
"h11_protocol":["compare only participants adhering to allocated regimen","restrict estimate to protocol followers"],
"h11_censor":["exact failure time missing but known to exceed study exit","incomplete time to event kept in dataset"],
"h11_truncate":["records outside time limits are missing entirely","sampling rule removes individuals beyond event boundary"],
"h11_stationary":["process law invariant to translation of time","moments remain stable across temporal origin","probability structure unchanged with shifted time index"],
"h11_ergodic":["time averages of long paths match ensemble statistics","average over a single realization estimates expectation","long trajectory reproduces population mean"]}
MODES=("guarded","unguarded","recalibrated","augmented")
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def maps(root,ds):
    mapping=h8.routes(root,ds,3)
    for k,grp in enumerate(h11.FIT,4):
        for y in grp:mapping[y]=k
    return mapping
def data(prep):
    dd=h8.data(prep);ds=dd[-1];return dd,ds,ds[0],ds[-1],maps(ds[0],ds)
def refs(d,seed,mapping):
    f,v=h10.sets(d,{y:mapping[y] for y in d["labels"]},seed)
    f=f+[row for grp in h11.FIT for row in h10.extra(grp)]+h10.extra(FIT)
    v=v+[row for grp in h11.DEV for row in h10.extra(grp)]+h10.extra(DEV)
    ck(not {a.norm(t) for t,_ in f}&{a.norm(t) for t,_ in v},"FIT_DEV_COLLISION");return f,v
def features(L,root,d,seed,i,mapping,rows,rootpath,bundlepath,h10path,h11path):
    torch,_=L;ls=h11.labels(d);core,b=h9.source(L,root,d,seed,i,rootpath,bundlepath)
    _,_,_,cand=h9.inputs(L,core,b,root,d,rows);hr,f=h11.load_h10(L,d,h10path/str(seed),seed,i)
    lex=hr["lex"];p=h10.prototype(L,h10.lexical(L,lex,rows),h10.lexical(L,lex,f),torch.tensor([mapping[y] for _,y in f]))
    z=h9.route(L,p,hr["thresholds"]["prototype"]["threshold"]);base=cand[torch.arange(len(rows)),z]
    allf,_=refs(d,seed,mapping);last=h11.PINS[i];scores=[];aug=[];answers=[];thresholds=[];pins=[]
    for k in range(1,6):
        m,rec,old,new=h11.load_module(L,d,h11path/str(k)/str(seed),seed,k,last,h11.PINS[i]);last=rec["model_sha256"];ck(last==PINS[k-1][i],"EXACT_MODULE");pins.append(last)
        lex=rec["lex"];x=h10.lexical(L,lex,rows);fx=h10.lexical(L,lex,old+new)
        scores.append(h11.score_gate(L,x,fx,len(old)))
        ids=torch.tensor([ls[y] for y in sorted(h11.FIT[k-1])])
        with torch.no_grad():answers.append(ids[m(x).argmax(-1)])
        neg=[row for row in allf if mapping[row[1]]!=k+3];pos=[row for row in allf if mapping[row[1]]==k+3]
        lx=h10.lexical_fit(neg+pos);aug.append(h11.score_gate(L,h10.lexical(L,lx,rows),h10.lexical(L,lx,neg+pos),len(neg)))
        thresholds.append(rec["gate"]["threshold"])
        ck(e.state_hash(m)==last,"WEIGHTS")
    ck(e.state_hash(core)==h9.ROOTS[i] and e.state_hash(b)==h9.BUNDLES[i],"BASE_WEIGHTS")
    return base,z,cand,scores,aug,answers,thresholds,pins
def select(L,score,answer,prev,y,actual,k):
    torch,_=L;old=y<k;now=y==k;correct=prev==actual;keep=correct&old
    best=None
    for th in sorted(set([1.01]+[float(v) for v in score[now]])):
        take=score>=th;pred=torch.where(take,answer,prev);lost=int((keep&(pred!=actual)).sum())
        if lost:continue
        hit=int((take&now&(answer==actual)).sum());bad=int((take&~now).sum());route=int((take&now).sum())
        key=(hit,-bad,route,th)
        if best is None or key>best[0]:best=(key,{"threshold":th,"old_correct":int(keep.sum()),"old_lost":lost,"new_correct":hit,"new_n":int(now.sum()),"new_route_correct":route,"other_intercepts":bad},pred)
    ck(best is not None,"CALIBRATE");return best[1],best[2]
def train(prep,rootpath,bundlepath,h10path,h11path,out):
    L=e.lib(prep);torch,_=L;dd,ds,root,d,mapping=data(prep);ls=h11.labels(d);out.mkdir(parents=True,exist_ok=False)
    for i,seed in enumerate(e.SEEDS):
        start=time.process_time();f,v=refs(d,seed,mapping);y=torch.tensor([mapping[t] for _,t in v]);actual=torch.tensor([ls[t] for _,t in v])
        base,z,cand,scores,aug,answers,thresholds,pins=features(L,root,d,seed,i,mapping,v,rootpath,bundlepath,h10path,h11path)
        policies={}
        for mode,ss in [("recalibrated",scores),("augmented",aug)]:
            prev=base.clone();policies[mode]=[]
            for k,(score,answer) in enumerate(zip(ss,answers),4):
                rec,prev=select(L,score,answer,prev,y,actual,k);policies[mode].append(rec)
        rec={"seed":seed,"schema":"r84h11b.calibration.v1","fit_sha256":g.digest(f),"dev_sha256":g.digest(v),"fit_n":len(f),"dev_n":len(v),"pins":pins,"base_pin":h11.PINS[i],"policies":policies,"cpu_seconds":time.process_time()-start,"production":False}
        e.write(out/(str(seed)+".json"),rec);log("R84H11B_TRAIN",rec)
def exams(prep,dd,d):
    groups,_=h11.exam(prep,dd);ck(sum(map(len,groups))==509,"PRIOR");raw=(HERE/"R84H11B_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN")
    added=h10.extra(FIT)+h10.extra(DEV);historic={a.norm(t) for grp in groups for t,_ in grp};ck(not historic&{a.norm(t) for t,_ in added},"HISTORIC_COLLISION")
    used=d["old"]+d["new"]+h10.extra(h10.FIT)+h10.extra(h10.DEV)+[row for grp in h11.FIT+h11.DEV for row in h10.extra(grp)]+added+sum(d["monitor"],[])
    norm=historic|{a.norm(t) for t,_ in used};fresh=[]
    for grp in json.loads(raw)["cohorts"]:
        clean=[]
        for t,y in grp:
            n=a.norm(t)
            if n in norm:continue
            norm.add(n);clean.append([t,y])
        ck(len(clean)>=4,"FRESH");fresh.append(clean)
    log("R84H11B_EXAM",{"prior_n":509,"fresh_n":list(map(len,fresh))});return groups+fresh,sum(map(len,fresh))
def evaluate(prep,rootpath,bundlepath,h10path,h11path,trained,out):
    L=e.lib(prep);torch,_=L;dd,ds,root,d,mapping=data(prep);ls=h11.labels(d);groups,nf=exams(prep,dd,d);rows=sum(groups,[])
    y=torch.tensor([mapping[t] for _,t in rows]);actual=torch.tensor([ls[t] for _,t in rows]);results=[]
    for i,seed in enumerate(e.SEEDS):
        f,v=refs(d,seed,mapping);rec=e.read(trained/(str(seed)+".json"));ck(rec["fit_sha256"]==g.digest(f) and rec["dev_sha256"]==g.digest(v) and rec["seed"]==seed and rec["base_pin"]==h11.PINS[i],"CALIBRATION_PIN")
        base,z,cand,scores,aug,answers,thresholds,pins=features(L,root,d,seed,i,mapping,rows,rootpath,bundlepath,h10path,h11path);ck(rec["pins"]==pins,"PINS")
        baseok=base==actual;oracle=torch.full_like(base,-1);old=y<=3;oracle[old]=cand[torch.arange(len(rows))[old],y[old]]
        for mode in MODES:
            prev=base.clone();route=z.clone();oracle_local=oracle.clone()
            for cycle in range(1,6):
                k=cycle+3;score=aug[cycle-1] if mode=="augmented" else scores[cycle-1];answer=answers[cycle-1];now=y==k;oracle_local[now]=answer[now]
                th=thresholds[cycle-1] if mode=="guarded" else .5 if mode=="unguarded" else rec["policies"][mode][cycle-1]["threshold"]
                take=score>=th;pred=torch.where(take,answer,prev);route=torch.where(take,torch.full_like(route,k),route)
                correct=pred==actual;before=prev==actual;known=(y>3)&(y<=k);fresh=correct[-nf:];fknown=known[-nf:]
                row={"seed":seed,"mode":mode,"cycle":cycle,"n":len(rows),"before":int(before.sum()),"dep":int(correct.sum()),"lost":int((before&~correct).sum()),"root_before":int(baseok.sum()),"root_lost":int((baseok&~correct).sum()),"new_route_correct":int(((route==y)&known).sum()),"new_n":int(known.sum()),"current_route_correct":int((take&now).sum()),"current_n":int(now.sum()),"fresh_dep":int(fresh.sum()),"fresh_n":nf,"fresh_new_route_correct":int(((route==y)[-nf:]&fknown).sum()),"fresh_new_n":int(fknown.sum()),"old_intercepts":int((take&(y<k)).sum()),"oracle_dep":int((oracle_local==actual).sum()),"threshold":th,"calibration_cpu_seconds":rec["cpu_seconds"],"weights_unchanged":True}
                results.append(row);log("R84H11B_RESULT",row);prev=pred
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84h11b.result.v1","results":results,"production":False,"external_validation":False,"scope":"retrospective fixed five-expert routing repair; original gates vs recalibration vs enriched contrastive banks; calibration uses known classes across all five generations, not prospective lifelong learning; 509 historical + fresh synthetic prompts; no weight changes or automatic promotion"})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:]])
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
