"""R84H12: ten bounded prospective append-only neural modules per seed."""
import sys,json,time
from pathlib import Path
import r84h11b_repair as repair
import r84h11_cortex as h11
import r84h10_resonance as h10
import r84h9_synapse as h9
import r84h8_neurogenesis as h8
import r84e_neural as e
import r84g_adaptive as a
import r8_learning as r
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="f82994e2ca9993fee86f51325ac2bebd109f995e"
FIT=[
{"cache_hit":["cache hit satisfies request locally","cached object found during lookup","read hits existing cache entry","serve available data directly from cache"],"cache_miss":["cache miss requires backend fetch","lookup cannot locate object in cache","missing cached record triggers storage read","requested entry absent from local cache"]},
{"write_through":["write through cache updates main store immediately","synchronous cache and disk write","each cache write reaches persistent store","write through waits for underlying storage update"],"write_back":["write back postpones disk updates","dirty cache pages later flushed to disk","asynchronous persistence of cached writes","write back buffers modifications before persistence"]},
{"mutex":["mutex ensures exclusive critical section","only one thread holds mutual exclusion lock","protect shared resource with exclusive ownership","single lock owner enters critical region"],"semaphore":["semaphore counts available resource permits","counting semaphore allows bounded parallel access","limit concurrency using permit counter","multiple holders allowed within resource capacity"]},
{"deadlock":["deadlock arises from circular lock waiting","threads mutually block while holding required locks","cycle of dependencies prevents progress","processes wait forever for each others resources"],"starvation":["starvation means repeated resource denial","unfair scheduler never selects a waiting task","one process cannot progress despite others running","low priority task indefinitely postponed"]},
{"idempotency":["idempotent operation safe to repeat","retries leave final state identical","repeated calls have no additional effect","duplicate request does not alter completed result"],"atomicity":["atomic transaction all or nothing","indivisible set of changes commits together","failure leaves no partial transaction effects","atomic update cannot be partly applied"]},
{"replication":["replication copies same dataset across servers","maintain redundant database copies","mirror records to additional nodes","identical data stored on several machines"],"sharding":["sharding divides database by key","split records into independent partitions","horizontal partitioning distributes distinct data subsets","each shard owns separate part of dataset"]},
{"checksum":["checksum detects accidental bit errors","integrity check compares calculated summary","verify data corruption using check value","detect transmission damage with checksum"],"encryption":["encryption converts plaintext to ciphertext","secret key protects message confidentiality","encrypted content requires decryption","cryptographic transformation hides readable data"]},
{"authentication":["authentication verifies user identity","validate credentials proving identity","confirm account holder through login","identify principal using authentication proof"],"authorization":["authorization checks permission to access resource","determine permitted actions for user","access control enforces privileges","check whether authenticated principal may perform operation"]},
{"rate_limit":["rate limiting caps requests per second","throttle clients exceeding request quota","limit frequency of API calls","enforce maximum traffic rate"],"backpressure":["backpressure signals producer to slow down","consumer capacity controls upstream throughput","full queue causes sender to reduce production","flow control prevents downstream overload"]},
{"rollback":["rollback reverts deployment to previous version","undo release and restore known good state","return service to earlier revision","restore prior configuration after failure"],"rollforward":["roll forward deploys corrected successor","recover from faulty deployment by applying fix","advance to patched release rather than reverting","resolve failure with newer software revision"]}
]
DEV=[
{"cache_hit":["requested value exists in cache so backend is bypassed","cached lookup returns stored object"],"cache_miss":["requested value absent in cache and read from origin","cache lookup needs external fetch"]},
{"write_through":["writes are immediately committed to cache and storage","cache changes synchronously reach durable medium"],"write_back":["modified cached blocks persist at a later time","deferred flush writes dirty data to disk"]},
{"mutex":["exclusive ownership protects critical region from concurrent tasks","one holder permitted inside shared critical section"],"semaphore":["counter of available permits regulates parallel workers","bounded simultaneous access through permit accounting"]},
{"deadlock":["mutually dependent lock requests block every participant","resource wait graph contains a blocking cycle"],"starvation":["unfair resource scheduling prevents one worker from ever proceeding","a task is repeatedly bypassed while others make progress"]},
{"idempotency":["reissuing completed request does not change resulting state","multiple identical requests have effect of one"],"atomicity":["transaction is committed in full or entirely aborted","no intermediate partially committed changes survive"]},
{"replication":["copies of same records maintained on independent hosts","redundant equivalent databases improve availability"],"sharding":["distinct key partitions distributed among database hosts","split dataset into disjoint storage segments"]},
{"checksum":["detect damaged bytes by verifying a computed check number","integrity summary changes when data corrupted"],"encryption":["key based transformation prevents unauthorized reading","ciphertext conceals original message contents"]},
{"authentication":["establish identity by validating credentials","prove claimed identity before service access"],"authorization":["decide allowed operations based on access policy","verify privileges for an identified principal"]},
{"rate_limit":["cap accepted requests within a fixed window","deny calls exceeding client frequency allowance"],"backpressure":["overloaded receiver requests reduced sending pace","upstream production adapts to downstream capacity"]},
{"rollback":["recover service by restoring older working release","revert unsuccessful changes to earlier configuration"],"rollforward":["correct broken release using subsequent patched deployment","recover by moving ahead to a fixed version"]}
]
FIT=[{"h12_"+k:v for k,v in grp.items()} for grp in FIT]
DEV=[{"h12_"+k:v for k,v in grp.items()} for grp in DEV]
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def maps(root,ds,d):
    mapping=repair.maps(root,ds);labels=h11.labels(d);n=max(labels.values())+1
    for cy,grp in enumerate(FIT,9):
        for y in sorted(grp):mapping[y]=cy;labels[y]=n;n+=1
    return mapping,labels
def banks(d,seed,cycle,mapping,role):
    # Only previous classes and current module: no future examples or exam access.
    f,v=repair.refs(d,seed,mapping);src=FIT if role=="fit" else DEV
    old=(f if role=="fit" else v)+[row for grp in src[:cycle-1] for row in h10.extra(grp)]
    return old,h10.extra(src[cycle-1])
def base_load(L,root,d,seed,i,rootpath,bundlepath,h10path,h11path,mapping):
    torch,_=L;core,b=h9.source(L,root,d,seed,i,rootpath,bundlepath)
    hr,f=h11.load_h10(L,d,h10path/str(seed),seed,i);lex=hr["lex"];fx=h10.lexical(L,lex,f);fy=torch.tensor([mapping[y] for _,y in f])
    old=[];last=h11.PINS[i]
    for k in range(1,6):
        m,rec,fo,fn=h11.load_module(L,d,h11path/str(k)/str(seed),seed,k,last,h11.PINS[i]);last=rec["model_sha256"];ck(last==repair.PINS[k-1][i],"H11_PIN")
        old.append((m,rec,h10.lexical(L,rec["lex"],fo+fn),len(fo),last))
    return core,b,hr,fx,fy,old,last
def base_predict(L,root,d,rows,state,labels):
    torch,_=L;core,b,hr,fx,fy,old,last=state
    _,_,_,cand=h9.inputs(L,core,b,root,d,rows)
    p=h10.prototype(L,h10.lexical(L,hr["lex"],rows),fx,fy);z=h9.route(L,p,hr["thresholds"]["prototype"]["threshold"])
    pred=cand[torch.arange(len(rows)),z]
    oracle_cand=[cand[:,i] for i in range(4)]
    for j,(m,rec,ref,nold,sha) in enumerate(old):
        xx=h10.lexical(L,rec["lex"],rows);p=h11.score_gate(L,xx,ref,nold)
        ids=torch.tensor([labels[y] for y in sorted(h11.FIT[j])])
        with torch.no_grad():ans=ids[m(xx).argmax(-1)]
        oracle_cand.append(ans);take=p>=.5;pred=torch.where(take,ans,pred);z=torch.where(take,torch.full_like(z,j+4),z)
    return pred,z,torch.stack(oracle_cand,dim=1)
def load(L,d,seed,cycle,folder,parent,mapping):
    torch,_=L;rec=e.read(folder/"receipt.json");p=folder/"shadow.pt";fo,fn=banks(d,seed,cycle,mapping,"fit")
    ck(rec["file_sha256"]==e.sha(p.read_bytes()) and rec["data_sha256"]==g.digest(d) and rec["parent_sha256"]==parent and rec["seed"]==seed and rec["cycle"]==cycle and rec["production"] is False,"LINEAGE")
    ck(rec["ref_sha256"]==g.digest(fo+fn) and rec["lex"]==h10.lexical_fit(fo+fn),"REFERENCE")
    m=h11.model(L,len(rec["lex"]["keys"]),seed+12000+cycle);m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),strict=True);m.eval()
    ck(e.state_hash(m)==rec["model_sha256"],"HASH")
    return m,rec,fo,fn
def train(prep,rootpath,bundlepath,h10path,h11path,out):
    L=e.lib(prep);torch,_=L;dd=h8.data(prep);ds=dd[-1];root,d=ds[0],ds[-1];mapping,labels=maps(root,ds,d);out.mkdir(parents=True,exist_ok=False)
    for i,seed in enumerate(e.SEEDS):
        state=base_load(L,root,d,seed,i,rootpath,bundlepath,h10path,h11path,mapping);parent=state[-1];prior=[]
        for cycle in range(1,11):
            start=time.process_time();fo,fn=banks(d,seed,cycle,mapping,"fit");vo,vn=banks(d,seed,cycle,mapping,"dev")
            ck(not {a.norm(t) for t,_ in fo+fn}&{a.norm(t) for t,_ in vo+vn},"FIT_DEV")
            lex=h10.lexical_fit(fo+fn);x=h10.lexical(L,lex,fo+fn);v=h10.lexical(L,lex,vo+vn);ys=sorted(FIT[cycle-1]);y=torch.tensor([ys.index(t) for _,t in fn])
            m=h11.fit_module(L,x[len(fo):],y,seed+12000+cycle);gate=h11.choose(L,h11.score_gate(L,v,x,len(fo)),len(vo))
            ck(all(e.state_hash(q)==sha for q,sha in prior) and all(e.state_hash(q)==sha for q,_,_,_,sha in state[-2]) and e.state_hash(state[0])==h9.ROOTS[i] and e.state_hash(state[1])==h9.BUNDLES[i],"FROZEN")
            rec={"seed":seed,"cycle":cycle,"parent_sha256":parent,"ref_sha256":g.digest(fo+fn),"lex":lex,"gate":gate,"steps":240,"presentations":2880,"fit_pos":len(fn),"fit_neg":len(fo),"parameters":sum(p.numel() for p in m.parameters()),"cpu_seconds":time.process_time()-start}
            e.save(L,m,d,out/str(cycle)/str(seed),rec);parent=rec["model_sha256"];prior.append((m,parent));log("R84H12_TRAIN",{k:v for k,v in rec.items() if k!="lex"})
def exams(prep,dd,d):
    groups,_=repair.exams(prep,dd,d);ck(sum(map(len,groups))==551,"PRIOR");raw=(HERE/"R84H12_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN")
    prior=d["old"]+d["new"]+h10.extra(h10.FIT)+h10.extra(h10.DEV)+[row for grp in h11.FIT+h11.DEV for row in h10.extra(grp)]+h10.extra(repair.FIT)+h10.extra(repair.DEV)+sum(d["monitor"],[])
    added=[row for grp in FIT+DEV for row in h10.extra(grp)];historic={a.norm(t) for grp in groups for t,_ in grp};ck(not historic&{a.norm(t) for t,_ in added},"HISTORIC_COLLISION")
    norm=historic|{a.norm(t) for t,_ in prior+added};fresh=[]
    for grp in json.loads(raw)["cohorts"]:
        clean=[]
        for t,y in grp:
            n=a.norm(t)
            if n in norm:continue
            norm.add(n);clean.append([t,y])
        ck(len(clean)>=4,"FRESH");fresh.append(clean)
    log("R84H12_EXAM",{"prior_n":551,"fresh_n":list(map(len,fresh))});return groups+fresh,sum(map(len,fresh))
def evaluate(prep,rootpath,bundlepath,h10path,h11path,trained,out):
    L=e.lib(prep);torch,_=L;dd=h8.data(prep);ds=dd[-1];root,d=ds[0],ds[-1];mapping,labels=maps(root,ds,d);groups,nf=exams(prep,dd,d);rows=sum(groups,[])
    y=torch.tensor([mapping[t] for _,t in rows]);actual=torch.tensor([labels[t] for _,t in rows]);results=[]
    for i,seed in enumerate(e.SEEDS):
        state=base_load(L,root,d,seed,i,rootpath,bundlepath,h10path,h11path,mapping);parent=state[-1]
        start=time.process_time();base,z,cand=base_predict(L,root,d,rows,state,labels);basecpu=time.process_time()-start
        old=y<=8;oracle=torch.full_like(base,-1);oracle[old]=cand[torch.arange(len(rows))[old],y[old]]
        pred={k:base.clone() for k in ("fixed","guarded")};route={k:z.clone() for k in pred};baseok=base==actual;totalcpu=basecpu;params=0;traincpu=0.;prior=[]
        # Exact H11B unguarded regression on the original 551 prompts (before new additions).
        log("R84H12_ROOT",{"seed":seed,"historical_dep":int(baseok[:-nf].sum()),"base_cpu_seconds":basecpu,"base_dep":int(baseok.sum())})
        for cycle in range(1,11):
            m,rec,fo,fn=load(L,d,seed,cycle,trained/str(cycle)/str(seed),parent,mapping);parent=rec["model_sha256"];prior.append((m,parent));lex=rec["lex"];ref=h10.lexical(L,lex,fo+fn)
            start=time.process_time();xx=h10.lexical(L,lex,rows);p=h11.score_gate(L,xx,ref,len(fo));ids=torch.tensor([labels[t] for t in sorted(FIT[cycle-1])])
            with torch.no_grad():ans=ids[m(xx).argmax(-1)]
            evalcpu=time.process_time()-start;totalcpu+=evalcpu;params+=rec["parameters"];traincpu+=rec["cpu_seconds"];k=cycle+8;now=y==k;oracle[now]=ans[now]
            ck(all(e.state_hash(q)==sha for q,sha in prior) and all(e.state_hash(q)==sha for q,_,_,_,sha in state[-2]) and e.state_hash(state[0])==h9.ROOTS[i] and e.state_hash(state[1])==h9.BUNDLES[i],"FROZEN")
            for mode,th in [("fixed",.5),("guarded",rec["gate"]["threshold"])]:
                before=pred[mode]==actual;take=p>=th;pred[mode]=torch.where(take,ans,pred[mode]);route[mode]=torch.where(take,torch.full_like(route[mode],k),route[mode]);correct=pred[mode]==actual;known=(y>8)&(y<=k);gain=int(correct.sum())-int(before.sum())
                row={"seed":seed,"mode":mode,"cycle":cycle,"n":len(rows),"before":int(before.sum()),"dep":int(correct.sum()),"lost":int((before&~correct).sum()),"root_before":int(baseok.sum()),"root_lost":int((baseok&~correct).sum()),"net_gain":gain,"cpu_seconds":rec["cpu_seconds"],"cumulative_train_cpu_seconds":traincpu,"profit_cpu":gain/rec["cpu_seconds"],"current_route_correct":int((take&now).sum()),"current_n":int(now.sum()),"new_route_correct":int(((route[mode]==y)&known).sum()),"new_n":int(known.sum()),"old_intercepts":int((take&(y<k)).sum()),"oracle_dep":int((oracle==actual).sum()),"fresh_dep":int(correct[-nf:].sum()),"fresh_n":nf,"historical_dep":int(correct[:-nf].sum()),"additional_parameters":params,"inference_feature_cpu_seconds":totalcpu,"incremental_inference_feature_cpu_seconds":evalcpu,"threshold":th,"model_sha256":parent,"weights_unchanged":True}
                results.append(row);log("R84H12_RESULT",row)
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84h12.result.v1","results":results,"production":False,"external_validation":False,"scope":"ten prospective two-class modules per seed; fixed 0.5 versus zero-intercept dev gate; H11B unguarded base; new classes introduced sequentially; CPU inference feature timing on one batch, cached references, excludes model loads and routing application; synthetic classification only; no promotion"})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:]])
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
