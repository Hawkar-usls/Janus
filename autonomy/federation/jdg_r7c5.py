"""R7C5: bounded research shadow trial; no production writes."""
import hashlib,json,math,os,random,re,sys,tempfile,urllib.request
from pathlib import Path
import jdg_r7c4 as school
import jdg_r7c3 as upstream
import jfp_gateway as g

EXAM="84c033c7bf38824249255cdc90f44db2a475f1af"
SEED=75
STEPS=300
SOURCE="https://openstax.org/books/prealgebra-2e/pages/1-2-add-whole-numbers"
CHECK=school.check
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,o):Path(p).write_bytes(g.canonical(o))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pair(a,b):return tuple(sorted((a,b)))
def exam():
    raw=(HERE/"R7C5_FROZEN_INTERNAL_EXAM.json").read_bytes()
    CHECK(upstream.blob(raw)==EXAM,"EXAM_PIN")
    r=json.loads(raw)["cases"]
    CHECK(len(r)==48 and len({pair(a,b) for a,b,_,_ in r})==48 and sum(c for _,_,_,c in r)==24,"EXAM_SIZE")
    CHECK(all(a+b==y and c==int(a%10+b%10>=10) for a,b,y,c in r),"EXAM_REFERENCE")
    return r
def oracle(demi):
    sys.path.insert(0,str(demi))
    from tools import trinity_r14_verified_oracle as o
    CHECK(upstream.blob(Path(o.__file__).read_bytes())=="7792bae07705e040504052bcf673342b661858f7","ORACLE_PIN")
    return o
def question(a,b):return f"Compute {a} + {b}. Reply with the integer only."
def prepare(demi,lapis,genesis,out):
    out.mkdir(parents=True,exist_ok=False)
    o=oracle(demi)
    # Real bounded Beetle scout on disposable state; search leads are not truth or training text.
    sys.path.insert(0,str(lapis/"tools"))
    import universal_beetle as beetle
    CHECK(upstream.blob(Path(beetle.__file__).read_bytes())=="cdeb4777fdd1df49090f60e36a995951fe214924","BEETLE_PIN")
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        src=school.get_sources(root);p=school.profile(src);route=school.route(root,src,p)[0]
        state=beetle.State(root/"scout")
        direction={**route,"enabled":True,"min_score":1.0,"keywords":["addition","arithmetic","place value","carry","regrouping"],
          "sources":[{"kind":"arxiv","query":'all:"addition" AND all:"arithmetic" AND (all:"carry" OR all:"place value")',"max_results":5}]}
        scout=beetle.run_scout({"global_budget":{"max_requests_per_run":1,"max_new_candidates_per_run":5,"max_metadata_bytes_per_run":1000000},"directions":[direction]},state,None)
        from beetle_shard_store import iter_lines
        leads=[json.loads(x) for x in iter_lines(state.root,"candidates")]
        # Teacher method is separately retrieved from a fixed primary educational source.
        # Never ingest arbitrary discovered text, links or answer strings.
        req=urllib.request.Request(SOURCE,headers={"User-Agent":"JANUS educational research/1.0"})
        with urllib.request.urlopen(req,timeout=30) as f:
            CHECK(f.geturl().startswith("https://openstax.org/"),"SOURCE_ORIGIN")
            material=f.read(4000001)
        CHECK(0<len(material)<=4000000,"SOURCE_SIZE")
        txt=re.sub(r"<[^>]+>"," ",material.decode("utf-8",errors="replace")).lower()
        CHECK("place value" in txt and "carry" in txt and "add" in txt,"MATERIAL_METHOD_NOT_VERIFIED")
        lib=root/"upstream";lib.mkdir();upstream.obtain(lib);sys.path.insert(0,str(lib))
        from evoagentx.optimizers.map_elites_optimizer import MapElitesOptimizer
        from evoagentx.optimizers.engine.registry import ParamRegistry
        prog=school.Program();reg=ParamRegistry();reg.track(prog,"carry",name="carry")
        opt=MapElitesOptimizer(registry=reg,evaluator=lambda c:school.metrics(p,c),program=prog.run,
          search_space=school.SPACE,feature_dimensions=["carry"],feature_ranges={"carry":(20.,80.)},
          feature_bins=5,n_iterations=school.BUDGET,exploration_ratio=.6,random_seed=school.SEED)
        cfg,res=opt.optimize();hist=json.loads(g.canonical(res["history"]))
        chosen,summary=school.scores(p,hist);CHECK(school.metrics(p,cfg)==school.metrics(p,chosen),"OPTIMIZER")
        rows=school.tasks(chosen)
        for t in rows:
            a,b=t["a"],t["b"]
            q=f"Calculate {a} + {b}. Answer with one integer."
            CHECK(o.verify(q,str(t["answer"]))["verdict"]=="APPROVE","TRINITY")
            CHECK(o.verify(q,str(t["answer"]+1))["verdict"]=="VETO_WRONG","TRINITY_NEGATIVE")
        # Old Genesis regression suites and R14 fifth exam are used only as exclusion audit.
        old=[]
        for path,pin in [
          ("research/essence/v55_functional/FROZEN_CASES.json","e76dca95402ccaf9deb5d0d8174757e4417c4c7f"),
          ("research/essence/g9/FROZEN_FUNCTIONAL_CASES.json","5098f73ffae7ee1c643322eac639ceaca92c7602")]:
            raw=(genesis/path).read_bytes();CHECK(upstream.blob(raw)==pin,"OLD_EXAM_PIN");old+=json.loads(raw)["items"]
        fifth=demi/o.FIFTH
        raw=fifth.read_bytes();CHECK(upstream.blob(raw)==o.FIFTH_BLOB,"FIFTH_PIN")
        old+=json.loads(raw)["cases"]
        protected={pair(a,b) for a,b,_,_ in exam()}
        for t in old:
            for a,b in re.findall(r"(?<!\d)(\d+)\s*(?:\+|plus|added to)\s*(\d+)(?!\d)",t.get("prompt",""),re.I):
                protected.add(pair(int(a),int(b)))
        train={pair(t["a"],t["b"]) for t in rows}
        CHECK(len(train)==100 and not train&protected,"EXAM_OVERLAP")
        receipt={"schema":"janus.jdg.r7c5.train.v1","train_count":100,"train_sha256":g.digest(rows),"profile":p,
          "allocation":chosen,"optimizer":summary,"beetle_scout":scout,"lead_count":len(leads),
          "source":SOURCE,"source_sha256":hashlib.sha256(material).hexdigest(),
          "source_method":"place-value addition; independently generated examples, no source exercise copied",
          "trinity_checked":100,"negative_controls":100,"overlap_audit":"new exam and addition patterns in 128 archived cases",
          "old_cases":len(old),"exam_blob":EXAM,"promotion":False}
        write(out/"train.json",rows);write(out/"receipt.json",receipt)
        # Metadata, not training text.
        write(out/"leads.json",[{"url":x["url"],"title":x["title"],"candidate_id":x["candidate_id"]} for x in leads])
        print("R7C5_PREPARED",json.dumps(receipt,sort_keys=True),flush=True)
def train(demi,prepared,out,arm):
    import torch
    sys.path.insert(0,str(demi))
    from janus_model.model import ByteTokenizer
    from janus_model.train_registry import load_checkpoint,save_checkpoint,sha256_file
    rows=read(prepared/"train.json");r=read(prepared/"receipt.json")
    CHECK(r["train_sha256"]==g.digest(rows) and len(rows)==100 and r["trinity_checked"]==100 and r["promotion"] is False,"TRAIN_RECEIPT")
    CHECK(arm in ("adaptive","control"),"ARM")
    base=demi/"janus_model/checkpoints/promoted.pt"
    torch.set_num_threads(4);torch.manual_seed(SEED)
    model,_=load_checkpoint(base,expected_sha256=school.CKPT)
    CHECK(model.config.context_length==128,"BASE_CONTEXT")
    out.mkdir(parents=True,exist_ok=False)
    order=list(rows);random.Random(SEED).shuffle(order)
    if arm=="adaptive":
        # Curriculum order from easier no-carry to carry; only the first pass
        # uses a generic place-value hint (no answers or operand-specific computation).
        order.sort(key=lambda t:(t["carry"],t["a"]+t["b"]))
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    losses=[]
    for step in range(STEPS):
        t=order[step%100];q=question(t["a"],t["b"])
        if arm=="adaptive" and step<100:q="Add units first; carry if needed. "+q
        prefix=ByteTokenizer.encode("Q:"+q+"\nJANUS:",bos=True)
        answer=ByteTokenizer.encode(str(t["answer"]),eos=True)
        seq=prefix+answer
        CHECK(len(seq)-1<=128,"CONTEXT")
        x=torch.tensor([seq[:-1]],dtype=torch.long)
        y=torch.tensor([[-100]*(len(prefix)-1)+answer],dtype=torch.long)
        model.train();opt.zero_grad(set_to_none=True);_,loss=model(x,y)
        CHECK(math.isfinite(loss.item()),"LOSS")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();losses.append(float(loss.item()))
    target=out/"shadow.pt";save_checkpoint(target,model,{"shadow_only":True,"arm":arm,"production_promotion":False})
    CHECK(sha256_file(base)==school.CKPT and sha256_file(target)!=school.CKPT,"CHECKPOINT")
    result={"arm":arm,"base_sha256":school.CKPT,"shadow_sha256":sha256_file(target),"steps":STEPS,
      "train_sha256":r["train_sha256"],"prepared_receipt_sha256":sha(prepared/"receipt.json"),
      "first_loss":losses[0],"last_loss":losses[-1],"seed":SEED,"promotion":False}
    write(out/"receipt.json",result);print("R7C5_TRAINED",json.dumps(result),flush=True)
def evaluate(demi,prepared,control,adaptive,out):
    import torch
    sys.path.insert(0,str(demi))
    from janus_model.train_registry import load_checkpoint
    from tools.trinity_parental_cpu import student_answer
    torch.set_num_threads(4)
    base=demi/"janus_model/checkpoints/promoted.pt"
    rows=exam();train_rows=read(prepared/"train.json")
    CHECK(not {pair(x["a"],x["b"]) for x in train_rows}&{pair(a,b) for a,b,_,_ in rows},"OVERLAP")
    models={"original":(base,school.CKPT)}
    for arm,folder in (("control",control),("adaptive",adaptive)):
        r=read(folder/"receipt.json")
        CHECK(r["arm"]==arm and r["base_sha256"]==school.CKPT and r["steps"]==STEPS and r["train_sha256"]==g.digest(train_rows) and r["prepared_receipt_sha256"]==sha(prepared/"receipt.json") and r["promotion"] is False,"ARM_BINDING")
        models[arm]=(folder/"shadow.pt",r["shadow_sha256"])
    results={}
    for name,(path,pin) in models.items():
        model,_=load_checkpoint(path,expected_sha256=pin);model.eval()
        correct=[];answers=[]
        for a,b,y,c in rows:
            torch.manual_seed(SEED)
            v=student_answer(model,question(a,b));answers.append(v);correct.append(v==str(y))
        results[name]={"exact":sum(correct),"carry_exact":sum(x for x,(_,_,_,c) in zip(correct,rows) if c),
          "no_carry_exact":sum(x for x,(_,_,_,c) in zip(correct,rows) if not c),"correct":correct,"answers":answers}
    a=results["adaptive"]["correct"];b=results["original"]["correct"];c=results["control"]["correct"]
    lost=sum(x and not y for x,y in zip(b,a));gain=sum(y and not x for x,y in zip(b,a))
    lost_control=sum(x and not y for x,y in zip(c,a))
    status="INTERNAL_GAIN_ONLY__REGRESSION_AND_EXTERNAL_HOLD" if sum(a)>max(sum(b),sum(c)) and lost==0 and lost_control==0 else "HOLD_NO_ADAPTIVE_SUPERIORITY"
    report={"status":status,"exam_blob":EXAM,"cases":48,"results":results,"new_vs_original":gain,
      "lost_vs_original":lost,"lost_vs_control":lost_control,"production_promotion":False,
      "broader_regression_test":"NOT_RUN","external_validation":False}
    out.mkdir(parents=True,exist_ok=False);write(out/"result.json",report)
    print("R7C5_EVALUATED",json.dumps({**report,"results":{k:{x:y for x,y in v.items() if x not in ("correct","answers")} for k,v in results.items()}},sort_keys=True),flush=True)
if __name__=="__main__":
    mode=sys.argv[1];args=[Path(x).resolve() for x in sys.argv[2:]]
    if mode=="prepare":prepare(*args)
    elif mode=="train":train(*args[:3],sys.argv[5])
    elif mode=="evaluate":evaluate(*args)
    else:raise ValueError("MODE")
