"""H29 bounded frozen semantic comparison; no model training, no promotion."""
import sys,json,time,re,hashlib,resource
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import r84h28_evidence_mastery as h28
import r84h27_trinity_critic as h27
import r84h25_metacognition as h25
import r84h19_exploration as h19
import r84h18_fog as h18
HERE=Path(__file__).parent
EXAM="0baa3e4b450c8dc759b729f1ad0b8d73976a33e9"
PROTO="fdeccaec90c400f29db906d00edbfc0b9c14f7c2"
MOTTO="Мы должны узнать"
ck=h25.ck;read=h25.read;write=h25.write;log=h25.log;digest=h25.digest
def protocol():return h19.load_exam("R84H29_PROTOCOL.json",PROTO)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
class Semantic:
    def __init__(self,demi,source):
        p=protocol();self.p=p;self.ret=h28.Retrieval(demi,source)
        self.hyp=p["hypotheses"];self.labels=sorted(self.hyp);self.texts={};self.pointers={}
        ck(len(self.ret.active)==5,"FIVE_SOURCE_CARDS_REQUIRED")
        # Native Cortex provides authenticated source pointers; expand this
        # finite five-card catalog once, then rank all identical candidates.
        for z in self.ret.active.values():
            label=z["label"];card=self.ret.cards[label];q=self.ret.m.query(self.ret.root,card["text"][:120],limit=5);self.ret.m.checked(q)
            hit=next((x for x in q["citations"] if x["sha256"]==z["card_sha256"]),None);ck(hit is not None and hit["content_is_untrusted_data"],"NATIVE_POINTER")
            raw=self.ret.m.read_object(self.ret.root,hit["sha256"],hit["verified_bytes"]);ck(raw==card["text"].encode() and hit["origin"]["provider_sha256"]==z["provider_sha256"],"SOURCE_HASH")
            self.texts[label]=raw.decode();self.pointers[label]={"sha256":hit["sha256"],"url":z["url"],"provider_sha256":z["provider_sha256"]}
        ck(set(self.labels)==set(self.texts),"CATALOG")
        start=time.monotonic()
        path=Path(snapshot_download(repo_id=p["model"],revision=p["revision"],allow_patterns=["config.json","model.safetensors","tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.json","merges.txt","README.md"],max_workers=2))
        ck(path.name==p["revision"] and (path/"model.safetensors").is_file(),"MODEL_PIN")
        self.hashes={x.name:sha(x) for x in sorted(path.iterdir()) if x.is_file()}
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForSequenceClassification.from_pretrained(str(path),local_files_only=True,trust_remote_code=False,use_safetensors=True).cpu().eval()
        ck(self.model.config.num_labels==3,"NLI_LABEL_COUNT")
        labels=[str(self.model.config.id2label[i]).lower() for i in range(3)]
        ck(labels==p["label_order"] or labels==["label_0","label_1","label_2"],"NLI_LABEL_ORDER")
        self.load_wall=time.monotonic()-start;self.cpu=0.;self.pairs_n=0;self.cache={}
    def pairs(self,pairs):
        result=[];start=time.process_time()
        with torch.inference_mode():
            for i in range(0,len(pairs),16):
                batch=pairs[i:i+16];x=self.tokenizer([a for a,b in batch],[b for a,b in batch],padding=True,truncation=True,max_length=192,return_tensors="pt")
                y=self.model(**x).logits.softmax(-1).tolist();ck(all(len(z)==3 and abs(sum(z)-1)<1e-5 for z in y),"NLI_OUTPUT");result+=y
        self.cpu+=time.process_time()-start;self.pairs_n+=len(pairs);return result
    def prepare(self,texts):
        ts=list(dict.fromkeys(t for t in texts if t not in self.cache))
        pairs=[(t,self.hyp[l]) for t in ts for l in self.labels];values=self.pairs(pairs)
        for i,t in enumerate(ts):
            rows=[]
            for j,l in enumerate(self.labels):
                c,e,n=values[i*len(self.labels)+j];rows.append({"label":l,"entailment":e,"contradiction":c,"neutral":n,"lexical":h28.cosine(h28.features(t),h28.features(self.texts[l])),**self.pointers[l]})
            self.cache[t]=sorted(rows,key=lambda z:(-z["entailment"],z["label"]))
    def candidate(self,t):
        v=self.cache[t];a=v[0];gap=a["entailment"]-v[1]["entailment"]
        return a,gap
def scope(t):
    # Scope guard applied to both accepted H27/H28 evidence and new
    # semantic overrides on new cases. Do not treat all negative words as false.
    return h28.scope(t) and not re.search(r"\b(?:might or might not|unspecified|not stated|does not assert|mutually exclusive|at the same moment|compare|explicitly does not|it is not stated)\b",t.lower())
def base(z,c,enabled,ret,labels,h28policy):
    a,_,used,_,supported=h28.choose(z,c,enabled,ret,labels,h28policy)
    return a,used or supported
def decide(z,c,enabled,sem,labels,h28policy,policy):
    t=z["text"];a,support=base(z,c,enabled,sem.ret,labels,h28policy);best,gap=sem.candidate(t)
    gate=scope(t);picked=False;b=a
    # Retain baseline classification candidate even when withholding its
    # evidence verdict. Never call a held classification verified.
    if gate and not support and best["label"] in policy:
        p=policy[best["label"]]
        if best["entailment"]>=p["min_entailment"] and best["contradiction"]<=.10 and gap>=p["margin"]:
            b=int(labels[best["label"]]);picked=True
    return b,a,picked,gate and (support or picked),support,best,gap
def measure(rows,c,enabled,sem,labels,h28policy,policy):
    m={"known_n":0,"base_correct":0,"correct":0,"rescued":0,"lost":0,"semantic_used":0,"semantic_wrong":0,"unknown_n":0,"unknown_supported":0,"base_unknown_supported":0,"target_n":0,"semantic_top1":0,"lexical_top1":0,"joint_wrong":0,"joint_wrong_correct":0};details=[]
    for z in rows:
        b,a,picked,supported,old_support,best,gap=decide(z,c,enabled,sem,labels,h28policy,policy)
        d={"id":z["id"],"kind":z["kind"],"text":z["text"],"semantic_used":picked,"evidence_supported":supported,"base_supported":old_support,"best":best,"margin":gap,"candidates":sem.cache[z["text"]]}
        if z["kind"]=="known":
            good=b==z["truth"];old=a==z["truth"];joint=z["neural"]==z["lexical"] and z["neural"]!=z["truth"]
            m["known_n"]+=1;m["base_correct"]+=old;m["correct"]+=good;m["rescued"]+=good and not old;m["lost"]+=old and not good;m["semantic_used"]+=picked;m["semantic_wrong"]+=picked and not good;m["joint_wrong"]+=joint;m["joint_wrong_correct"]+=joint and good
            if z["truth"] in {int(labels[l]) for l in sem.labels}:
                m["target_n"]+=1;m["semantic_top1"]+=int(labels[best["label"]])==z["truth"];lex=max(sem.cache[z["text"]],key=lambda x:(x["lexical"],x["label"]));m["lexical_top1"]+=int(labels[lex["label"]])==z["truth"]
            d.update({"truth":z["truth"],"answer":b,"base_answer":a,"correct":good})
        else:m["unknown_n"]+=1;m["unknown_supported"]+=supported;m["base_unknown_supported"]+=old_support
        details.append(d)
    return m,details
def previous(ctx,h25folder,h26folder,h27gate,h27result,h28gate,h28result):
    a,b,c,enabled=h28.past(ctx,h25folder,h26folder,h27gate,h27result)
    hp=read(h28gate/"gate.json");hr=read(h28result/"result.json")
    ck(hr["safety"]=="HOLD" and hr["fresh_h28"]["correct"]==21 and hr["prior_joint_correct"]==8 and not hr["production"],"H28_RECEIPT")
    ex=h19.load_exam("R84H28_FROZEN_EXAM.json",h28.EXAM);d,_=ctx.observations(ex["known"],ex["unknown"],"H28_FRESH_EVAL")
    return a,b,c,d,enabled,hp["retrieval_policy"],ex["conflict"]
def gate(*args):
    *paths,h25folder,h26folder,h27gate,h27result,h28gate,h28result,source,demi,out=args
    torch.set_num_threads(2);ctx=h25.Context(*paths);out.mkdir(parents=True,exist_ok=False);sem=Semantic(demi,source)
    a,b,c,d,enabled,hp,conf=previous(ctx,h25folder,h26folder,h27gate,h27result,h28gate,h28result)
    rows=a+b+c+d;crit=h27.Critic(ctx.labels);sem.prepare([z["text"] for z in rows]+conf)
    old,_=measure(d,crit,enabled,sem,ctx.labels,hp,{});ck(old["base_correct"]==21,"H28_REPLAY")
    policy={};validation={}
    for label in sem.labels:
        options=[]
        for e in (.65,.75,.85,.90,.95):
            for gap in (.10,.20,.35):
                p={label:{"min_entailment":e,"margin":gap}}
                m,_=measure(rows,crit,enabled,sem,ctx.labels,hp,p)
                # A new semantic route must rescue at least one labeled case
                # and introduce zero wrong overrides and no extra false supports.
                if m["rescued"]>=1 and m["lost"]==0 and m["semantic_wrong"]==0 and m["unknown_supported"]<=m["base_unknown_supported"] and all(not decide({"text":t,"kind":"unknown"},crit,enabled,sem,ctx.labels,hp,p)[2] for t in conf):
                    options.append((m["rescued"],e,gap,m))
        if options:
            _,e,gap,m=max(options,key=lambda x:x[:3]);policy[label]={"min_entailment":e,"margin":gap};validation[label]=m
    prior,_=measure(b,crit,enabled,sem,ctx.labels,hp,policy);disclosed,_=measure(d,crit,enabled,sem,ctx.labels,hp,policy)
    rec={"schema":"r84h29.gate.v1","protocol":PROTO,"source_sha":digest(sem.ret.rec),"model_hashes":sem.hashes,"policy":policy,"validation":validation,"prior_h26":prior,"disclosed_h28":disclosed,"enabled_h27":sorted(enabled),"h28_policy":hp,"model_parameters":sum(p.numel() for p in sem.model.parameters()),"model_load_wall_seconds":sem.load_wall,"nli_cpu_seconds":sem.cpu,"nli_pairs":sem.pairs_n,"exam_read":False,"production":False}
    write(out/"gate.json",rec);log("R84H29_GATE",rec)
def evaluate(*args):
    *paths,h25folder,h26folder,h27gate,h27result,h28gate,h28result,source,demi,gateout,out=args
    torch.set_num_threads(2);ctx=h25.Context(*paths);out.mkdir(parents=True,exist_ok=False);sem=Semantic(demi,source);p=read(gateout/"gate.json")
    ck(p["protocol"]==PROTO and p["source_sha"]==digest(sem.ret.rec) and p["model_hashes"]==sem.hashes and not p["exam_read"] and not p["production"],"GATE_PIN")
    a,b,c,d,enabled,hp,conf=previous(ctx,h25folder,h26folder,h27gate,h27result,h28gate,h28result);ck(p["enabled_h27"]==sorted(enabled) and p["h28_policy"]==hp,"BASELINE_PIN")
    ex=h19.load_exam("R84H29_FROZEN_EXAM.json",EXAM);refs={h18.clean(t) for t,_ in ctx.fit+ctx.dev+ctx.oldknown}|{h18.clean(t) for t in ctx.oldunknown}|{h18.clean(z["text"]) for z in a+b+c+d}
    known=[z for z in ex["known"] if h18.clean(z[0]) not in refs];unknown=[t for t in ex["unknown"] if h18.clean(t) not in refs];ck(len(known)>=30 and len(unknown)>=18,"EXAM_OVERLAP")
    rows,cost=ctx.observations(known,unknown,"H29_FRESH_EVAL");crit=h27.Critic(ctx.labels)
    sem.prepare([z["text"] for z in rows]+ex["conflict"]);m,details=measure(rows,crit,enabled,sem,ctx.labels,hp,p["policy"])
    conflicts=[]
    for t in ex["conflict"]:
        ans,baseanswer,picked,supported,old,best,gap=decide({"text":t,"kind":"unknown"},crit,enabled,sem,ctx.labels,hp,p["policy"])
        conflicts.append({"text":t,"supported":supported,"baseline_supported":old,"semantic_used":picked,"top":best,"margin":gap})
    pairs=ex["pairs"];probs=sem.pairs([(a,b) for a,b,_ in pairs]);mapping=protocol()["label_order"];classes={k:{"n":0,"correct":0} for k in mapping};pairrows=[]
    for (a,b,y),v in zip(pairs,probs):
        pred=mapping[max(range(3),key=lambda i:v[i])];classes[y]["n"]+=1;classes[y]["correct"]+=pred==y
        pairrows.append({"premise":a,"hypothesis":b,"truth":y,"prediction":pred,"probabilities":dict(zip(mapping,v)),"lexical_similarity":h28.cosine(h28.features(a),h28.features(b))})
    # Pairwise discrimination between matched entailment and contradiction
    # for the same premise: no invented contradiction class for lexical score.
    def ranks(key):
        return sum(pairrows[i][key]>pairrows[i+1][key] for i in range(0,len(pairrows),3))
    lexrank=ranks("lexical_similarity");nlirank=sum(probs[i][1]>probs[i+1][1] for i in range(0,len(probs),3))
    pred,routes,histcpu=ctx.neural(ctx.hist);parity=sum(ctx.prev[i]["answer"]==int(pred[i]) and ctx.prev[i]["route"]==int(routes[i]) for i in range(671));ck(parity==671,"HISTORY")
    held=sum(not z["supported"] for z in conflicts);safe=m["lost"]==0 and m["semantic_wrong"]==0 and m["unknown_supported"]==0 and held==len(conflicts)
    improved=safe and m["rescued"]>0
    report={"schema":"r84h29.result.v1","motto":MOTTO,"fresh":m,"prior_h26":p["prior_h26"],"disclosed_h28":p["disclosed_h28"],"enabled":sorted(p["policy"]),"pair_classes":classes,"pair_n":len(pairs),"lexical_entailment_over_contradiction":lexrank,"nli_entailment_over_contradiction":nlirank,"pair_groups":len(pairs)//3,"conflict_held":held,"conflict_n":len(conflicts),"baseline_conflict_held":sum(not z["baseline_supported"] for z in conflicts),"overlap_excluded":55-len(known)-len(unknown),"history_parity":parity,"model_parameters":p["model_parameters"],"model_load_wall_seconds":sem.load_wall,"nli_cpu_seconds":sem.cpu,"nli_pairs":sem.pairs_n,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"legacy_cost":cost,"historical_cpu_seconds":histcpu,"safety":"PASS_SHADOW" if safe else "HOLD","improvement":"PASS_SHADOW" if improved else "HOLD_NO_PROVEN_GAIN","production":False,"scope":"frozen external NLI cross-encoder, same-owner scenario and pair exam, native Cortex five verified authored cards, no proof of factual truth; model downloaded on CPU, no JANUS weight update or production change; historical replay separate"}
    write(out/"result.json",report);write(out/"cases.json",details);write(out/"pairs.json",pairrows);write(out/"conflicts.json",conflicts);log("R84H29_RESULT",report)
if __name__=="__main__":{"gate":gate,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
