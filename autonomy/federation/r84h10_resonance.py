"""R84H10 finite competence routing with frozen experts and fresh lexical supervision."""
import sys,random,math,time,json,re
from collections import Counter
from pathlib import Path
import r84h9_synapse as h9
import r84h8_neurogenesis as h8
import r84g_adaptive as a
import r84e_neural as e
import r8_learning as r
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="5a50396c908723f0a1ded484318dcb4ba78aa5b0"
PINS=("020d216e5cdde7b030ed15f80f6fabc9da34a43c177a11a569de0bb76e9a0bd3","e32c617add908042ae6bf83ef708228009bf6b469e6fed85291076cee078eb03","5e214cb3f7cb380a1ee30d6cd74cc8c9fe203f6618c07bcbce3d9f5070ba549d")
FIT={
"h8_sensitivity":["recall of truly present target events","ratio of successful detections to true cases","detect disease in affected population","share of real targets found by detector","ability to capture existing positives"],
"h8_specificity":["reject non target samples correctly","rate of correctly classified healthy controls","avoid false positives among genuine negatives","correct negative test among unaffected patients","fraction of absent targets correctly excluded"],
"h8_precision":["purity of predicted positive set","fraction of reported hits that are genuine","trustworthiness of positive alerts","positive predictive value for issued alarms","proportion of retrieved positives that are relevant"],
"h8_f1":["harmonic combination of completeness and correctness","f measure from recall and precision","combine retrieval quality with coverage via harmonic mean","balanced harmonic score of predictive value and sensitivity","classification f measure balances both positive metrics"],
"h8_calibration":["reliability diagram compares predicted probabilities to actual rates","probability forecasts should agree with empirical proportions","measure agreement of predicted risk and real frequency","assess forecast confidence against observed outcomes","probability estimates reflect long term occurrence rates"],
"h8_discrimination":["ranking quality between target and background","separate classes using their score distributions","order genuine cases above controls","ability to distinguish targets from non targets","ranking performance across positive and negative samples"],
"h7_bias":["persistent offset of measuring device relative to reference","measurement accuracy impaired by fixed calibration error"],
"h7_randomerror":["unpredictable scatter of replicate instrument measurements","random variation in measurements rather than fixed offset"],
"h7_selection":["recruitment preferentially includes particular population segments","sample inclusion mechanism biases population representation"],
"h7_nonresponse":["missing survey replies affect sample representativeness","respondents and silent invitees differ in composition"],
"h6_harmonic":["reciprocal average for rates over equal lengths","average inverse values and invert result"],
"h5_variance":["mean squared distance of observations from center","measure spread in square units of original measurement"]}
DEV={
"h8_sensitivity":["share of genuine target objects successfully identified","how completely are real cases found by screening"],
"h8_specificity":["how often does a test return negative for truly unaffected individuals","fraction of background samples excluded by classifier"],
"h8_precision":["among affirmative classifications count the valid hits","measure reliability of issued positive alarms"],
"h8_f1":["harmonic score combining recall and positive predictive value","summarize correctness and coverage with harmonic f measure"],
"h8_calibration":["probabilistic forecasts correspond to measured incidence","confidence estimates agree with realized proportions"],
"h8_discrimination":["rank real cases ahead of background observations","score based separation between classes"],
"h7_bias":["instrument has a stable error in one direction"],
"h7_randomerror":["measurement scatter varies unpredictably from trial to trial"],
"h7_selection":["sampling process preferentially recruits one population type"],
"h7_nonresponse":["nonparticipants and respondents differ systematically"],
"h7_reproduce":["agreement across measurement facilities"],
"h7_repeat":["agreement under unchanged measurement conditions"],
"h6_harmonic":["mean for positive rates based on inverse quantities"],
"h6_geometric":["average multiplicative factors using product roots"]}
MODES=("core","h9_consensus","prototype","balanced","resonance")
STOP=set("gate practice exam the a an of for to and in is are be by with from on as at how what which does do can using measure report evaluate assess calculate quantify identify compare compute".split())
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def extra(grp):return [["gate practice "+t,y] for y,ts in grp.items() for t in ts]
def terms(t):
    ws=[w for w in re.findall(r"[a-z]+|\d+",a.norm(t)) if w not in STOP]
    return Counter(["w:"+w for w in ws]+["b:"+x+"_"+y for x,y in zip(ws,ws[1:])]+["c:"+w[i:i+3] for w in ws for i in range(max(0,len(w)-2))])
def lexical_fit(rows):
    counts=Counter(k for t,_ in rows for k in terms(t))
    keys=sorted(k for k,n in counts.items() if n>=2);ck(len(keys)<=12000,"LEXICAL_CAP")
    df=Counter(k for t,_ in rows for k in terms(t))
    return {"keys":keys,"idf":[math.log((1+len(rows))/(1+df[k]))+1 for k in keys]}
def lexical(L,lex,rows):
    torch,_=L;keys={k:i for i,k in enumerate(lex["keys"])};x=torch.zeros(len(rows),len(keys))
    for i,(t,_) in enumerate(rows):
        for k,v in terms(t).items():
            if k in keys:
                j=keys[k];x[i,j]=(1+math.log(v))*lex["idf"][j]*(.6 if k.startswith("c:") else 1.)
    return torch.nn.functional.normalize(x,p=2,dim=1)
def load_h9(L,dim,d,folder,seed,i):
    torch,_=L;rec=e.read(folder/"receipt.json");p=folder/"shadow.pt"
    ck(rec["file_sha256"]==e.sha(p.read_bytes()) and rec["data_sha256"]==g.digest(d) and rec["model_sha256"]==PINS[i] and rec["production"] is False,"H9_RECEIPT")
    m=h9.model(L,dim,seed);m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),strict=True);ck(e.state_hash(m)==PINS[i],"H9_HASH")
    return m,rec
def prototype(L,x,ref,y):
    torch,_=L;sim=x@ref.T
    vals=[]
    for k in range(1,4):
        pos=sim[:,y==k].max(-1).values;neg=sim[:,y!=k].max(-1).values
        vals.append((1+pos-neg)/2)
    return torch.stack(vals,dim=1)
def fit(L,x,y,seed):
    torch,_=L;torch.manual_seed(seed+841000);m=torch.nn.Linear(x.shape[1],3)
    opt=torch.optim.AdamW(m.parameters(),lr=.015,weight_decay=.02)
    rng=random.Random(seed+841001);groups=[[i for i,v in enumerate(y.tolist()) if v==k] for k in range(4)]
    ck(all(groups),"FIT_GROUPS");target=(y[:,None]==torch.arange(1,4)[None,:]).float()
    for step in range(240):
        ids=[]
        for grp in groups:ids+=rng.choices(grp,k=8)
        z=m(x[ids]);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,target[ids])
        ck(math.isfinite(float(loss.detach())),"LOSS");opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return m
def probs(L,m,x,ref,y):
    torch,_=L;m.eval()
    with torch.no_grad():
        p=prototype(L,x,ref,y);b=m(x).sigmoid()
    return {"prototype":p,"balanced":b,"resonance":(p+b)/2}
def calibrate(L,p,y,cand,actual):
    torch,_=L;old=y==0;new=y>0;bc=(cand[:,0]==actual)&old;n=int(bc.sum());ck(n>0 and int(new.sum())>0,"DEV")
    best=None
    for th in sorted(set([0.,1.01]+[float(v) for v in p.max(-1).values])):
        z=h9.route(L,p,th);pred=cand[torch.arange(len(y)),z];lost=int((bc&(pred!=actual)).sum())
        if lost/n>.01:continue
        hit=int(((z==y)&new).sum());mis=int(((z!=0)&old).sum())
        key=(hit,-lost,-mis,th)
        if best is None or key>best[0]:best=(key,{"threshold":th,"old_correct":n,"lost":lost,"new_n":int(new.sum()),"new_route_correct":hit,"old_misroute":mis})
    ck(best is not None,"CALIBRATION");return best[1]
def sets(d,mapping,seed):
    base=d["old"]+d["new"];fi,di=h9.split(base,mapping,seed);f=[base[i] for i in fi]+extra(FIT);v=[base[i] for i in di]+extra(DEV)
    norm=lambda rows:{a.norm(t) for t,_ in rows}
    ck(not norm(f)&norm(v),"FIT_DEV_COLLISION")
    return f,v
def train(prep,root_path,bundle_path,h9path,out):
    L=e.lib(prep);torch,_=L;*_,ds=h8.data(prep);root,d=ds[0],ds[-1];mapping=h8.routes(root,ds,3)
    out.mkdir(parents=True,exist_ok=False)
    for i,seed in enumerate(e.SEEDS):
        start=time.process_time();core,b=h9.source(L,root,d,seed,i,root_path,bundle_path);f,v=sets(d,mapping,seed)
        lex=lexical_fit(f);x=lexical(L,lex,f);xx=lexical(L,lex,v);y=torch.tensor([mapping[t] for _,t in f]);vy=torch.tensor([mapping[t] for _,t in v]);actual=torch.tensor([d["labels"][t] for _,t in v])
        _,_,_,cand=h9.inputs(L,core,b,root,d,v)
        m=fit(L,x,y,seed);pr=probs(L,m,xx,x,y);th={mode:calibrate(L,p,vy,cand,actual) for mode,p in pr.items()}
        ck(e.state_hash(core)==h9.ROOTS[i] and e.state_hash(b)==h9.BUNDLES[i],"FROZEN")
        rec={"seed":seed,"fit_n":len(f),"dev_n":len(v),"added_fit":len(extra(FIT)),"added_dev":len(extra(DEV)),"lex":lex,"thresholds":th,"steps":240,"presentations":7680,"core_sha256":h9.ROOTS[i],"bundle_sha256":h9.BUNDLES[i],"cpu_seconds":time.process_time()-start}
        e.save(L,m,d,out/str(seed),rec);log("R84H10_TRAIN",{k:v for k,v in rec.items() if k!="lex"})
def exams(prep,dd):
    groups,fresh=h9.exam_groups(prep,dd);groups+=fresh;ck(sum(map(len,groups))==449,"PRIOR")
    d=dd[-1][-1];raw=(HERE/"R84H10_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM")
    additions=extra(FIT)+extra(DEV);nn={a.norm(t) for t,_ in additions}
    ck(not nn&{a.norm(t) for grp in groups for t,_ in grp},"HISTORIC_TEXT_COLLISION")
    used=d["old"]+d["new"]+additions+sum(d["monitor"],[])+sum(groups,[])
    norm={a.norm(t) for t,_ in used};enc={tuple(e.encode(d,t)) for t,_ in used};new=[]
    for grp in json.loads(raw)["cohorts"]:
        clean=[]
        for t,y in grp:
            key=tuple(e.encode(d,t));s=a.norm(t)
            if s in norm or key in enc:continue
            clean.append([t,y]);norm.add(s);enc.add(key)
        ck(len(clean)>=3,"FRESH");new.append(clean)
    log("R84H10_EXAM",{"prior_n":449,"fresh_n":list(map(len,new))})
    return groups+new,sum(map(len,new))
def evaluate(prep,root_path,bundle_path,h9path,trained,out):
    L=e.lib(prep);torch,_=L;dd=h8.data(prep);ds=dd[-1];root,d=ds[0],ds[-1];groups,nf=exams(prep,dd);rows=sum(groups,[]);mapping=h8.routes(root,ds,3)
    y=torch.tensor([mapping[t] for _,t in rows]);actual=torch.tensor([d["labels"][t] for _,t in rows]);old=y==0;new=y>0;results=[]
    for i,seed in enumerate(e.SEEDS):
        core,b=h9.source(L,root,d,seed,i,root_path,bundle_path);x,xx,orig,cand=h9.inputs(L,core,b,root,d,rows);base=cand[:,0];oracle=cand[torch.arange(len(rows)),y]
        h,hr=load_h9(L,x.shape[1],d,h9path/str(seed),seed,i)
        hz=h9.route(L,h9.probabilities(L,h,x,xx,orig)["consensus"],hr["thresholds"]["consensus"]["threshold"])
        rec=e.read(trained/str(seed)/"receipt.json");p=trained/str(seed)/"shadow.pt"
        ck(rec["file_sha256"]==e.sha(p.read_bytes()) and rec["data_sha256"]==g.digest(d) and rec["core_sha256"]==h9.ROOTS[i] and rec["bundle_sha256"]==h9.BUNDLES[i] and rec["production"] is False,"RECEIPT")
        f,v=sets(d,mapping,seed);lex=rec["lex"];ck(lex==lexical_fit(f),"LEX_PIN")
        fx=lexical(L,lex,f);ex=lexical(L,lex,rows);fy=torch.tensor([mapping[t] for _,t in f])
        m=torch.nn.Linear(fx.shape[1],3);m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),strict=True);ck(e.state_hash(m)==rec["model_sha256"],"HASH")
        pr=probs(L,m,ex,fx,fy)
        for mode in MODES:
            z=torch.zeros_like(y) if mode=="core" else hz if mode=="h9_consensus" else h9.route(L,pr[mode],rec["thresholds"][mode]["threshold"])
            pred=cand[torch.arange(len(rows)),z];correct=pred==actual;bc=(base==actual)&old
            by=[{"route":k,"n":int((y==k).sum()),"correct":int(((z==y)&(y==k)).sum())} for k in range(4)]
            row={"mode":mode,"seed":seed,"n":len(rows),"dep":int(correct.sum()),"old_base_correct":int(bc.sum()),"root_lost":int((bc&~correct).sum()),"old_n":int(old.sum()),"new_n":int(new.sum()),"old_misroute":int(((z!=0)&old).sum()),"new_route_correct":int(((z==y)&new).sum()),"new_correct":int((correct&new).sum()),"route_correct":int((z==y).sum()),"routes":by,"oracle_dep":int((oracle==actual).sum()),"fresh_n":nf,"fresh_dep":int(correct[-nf:].sum()),"fresh_new_n":int(new[-nf:].sum()),"fresh_new_route_correct":int(((z==y)&new)[-nf:].sum()),"core_unchanged":e.state_hash(core)==h9.ROOTS[i],"modules_unchanged":e.state_hash(b)==h9.BUNDLES[i]}
            results.append(row);log("R84H10_RESULT",row)
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84h10.result.v1","results":results,"production":False,"external_validation":False,"scope":"frozen experts; lexical prototype, balanced learned competence, mean fusion; separate synthetic fit/dev additions; dev old-correct loss cap 1%; no test-based selection; no autonomous promotion"})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:]])
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
