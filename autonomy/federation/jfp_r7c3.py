"""JFP R7C3: real upstream evaluator, bounded search, Beetle student helper."""
import base64,hashlib,importlib.util,itertools,json,math,os,subprocess,sys,tempfile
from pathlib import Path
import jfp_gateway as g
P=json.loads(Path(__file__).with_name("JFP_R7C3_CAPABILITY.json").read_text())
def req(ok,why):
    if not ok:raise ValueError(why)
def sha(v):return hashlib.sha256(v).hexdigest()
def blob(v):return hashlib.sha1(b"blob "+str(len(v)).encode()+b"\0"+v).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_bytes(g.canonical(v)+b"\n")
def run(args,cwd=None):
    env={"PATH":"/usr/local/bin:/usr/bin:/bin","LANG":"C.UTF-8","HOME":str(Path(cwd or ".").resolve()),"PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1"}
    r=subprocess.run(args,cwd=cwd,env=env,capture_output=True,timeout=30)
    req(r.returncode==0,"COMMAND_FAILED:"+str(args[0])+":"+r.stderr.decode(errors="replace")[-1000:])
    return r.stdout
def pin(root,pins,commit):
    root=Path(root).resolve()
    req(run(["git","-C",str(root),"rev-parse","HEAD"]).decode().strip()==commit,"SOURCE_COMMIT")
    for p,h in pins.items():req(blob((root/p).read_bytes())==h,"SOURCE_BLOB")
    return root
def coefficients(degree,scale):return [scale*((-1)**k)/math.factorial(2*k+1) for k in range((degree+1)//2)]
def candidate_source(c):
    return "def approximate(x):\n    z=x*x\n    a=0.0\n    for b in "+repr(list(reversed(c)))+":\n        a=a*z+b\n    return x*a\n"
def dense(c):
    errors=[]
    for i in range(P["holdout_n"]):
        x=-math.pi+2*math.pi*(i+0.5)/P["holdout_n"]
        a=0.0
        for b in reversed(c):a=a*x*x+b
        errors.append(abs(x*a-math.sin(x)))
    return {"rmse":math.sqrt(sum(x*x for x in errors)/len(errors)),"max_abs_error":max(errors),"n":len(errors)}
def module(path):
    spec=importlib.util.spec_from_file_location("jdg_evaluation_results",path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def worker(upstream,out):
    u=pin(upstream,P["source_files"],P["upstream_commit"]);o=Path(out).resolve();o.mkdir(parents=True,exist_ok=False)
    v=module(u/"shinka/core/evaluation_results.py")
    rows=[]
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);program=root/"program.py"
        for n,(degree,scale) in enumerate(itertools.product(P["candidate_degrees"],P["scales"])):
            c=coefficients(degree,scale);program.write_text(candidate_source(c));dest=root/str(n)
            run([sys.executable,str(u/"examples/sine_approx_headless/evaluate.py"),"--program_path",str(program),"--results_dir",str(dest)],root)
            result=v.validate_evaluation_results({"correct":load(dest/"correct.json"),"metrics":load(dest/"metrics.json")},exit_status=0)
            req(result.correct and math.isfinite(result.combined_score),"UPSTREAM_EVALUATION")
            rows.append({"id":f"p{degree}-s{scale}","degree":degree,"scale":scale,"coefficients":c,"program_sha256":sha(program.read_bytes()),"upstream_score":result.combined_score,"upstream_metrics":result.public_metrics,"dense":dense(c)})
        bad=root/"bad";program.write_text("def approximate(x):\n    return float('nan')\n")
        run([sys.executable,str(u/"examples/sine_approx_headless/evaluate.py"),"--program_path",str(program),"--results_dir",str(bad)],root)
        b=v.validate_evaluation_results({"correct":load(bad/"correct.json"),"metrics":load(bad/"metrics.json")},exit_status=0)
        req(not b.correct,"NONFINITE_ACCEPTED")
        try:v.validate_evaluation_results({"correct":{"correct":True},"metrics":{"combined_score":float("nan")}},exit_status=0)
        except ValueError:pass
        else:raise ValueError("FALSE_SUCCESS_ACCEPTED")
        winner=max(rows,key=lambda r:r["upstream_score"]);base=next(r for r in rows if r["degree"]==1 and r["scale"]==1)
        req(winner["dense"]["rmse"]<base["dense"]["rmse"],"NO_DENSE_IMPROVEMENT")
        report={"schema":"janus.jfp.algorithm_result.v1","capability":P["capability"],"provider":P["provider"],"upstream_commit":P["upstream_commit"],"source_files":P["source_files"],"run_id":os.environ["GITHUB_RUN_ID"],"run_attempt":os.environ["GITHUB_RUN_ATTEMPT"],"rows":rows,"selected":winner["id"],"baseline":base["id"],"negative_controls_rejected":2,"external_peer":False,"external_compute":False,"llm_calls":False,"full_evolution":False,"training_authority":False,"execution_authority":False}
        k,pub=g.keys(root,"worker")
        (o/"RESULT.json").write_bytes(g.canonical(report)+b"\n");(o/"SIGNATURE.bin").write_bytes(g.crypto(k,g.canonical(report)));(o/"PUBLIC.pem").write_bytes(pub.read_bytes())
        with open(os.environ["GITHUB_OUTPUT"],"a") as f:f.write("key_sha="+sha(pub.read_bytes())+"\n")
    print("JFP_R7C3_WORKER "+json.dumps({"n":len(rows),"selected":winner["id"],"baseline_dense_rmse":base["dense"]["rmse"],"selected_dense_rmse":winner["dense"]["rmse"],"selected_score":winner["upstream_score"],"controls":2},sort_keys=True))
def verify(folder,key_sha):
    f=Path(folder);r=load(f/"RESULT.json");req(sha((f/"PUBLIC.pem").read_bytes())==key_sha,"KEY_PIN")
    g.crypto(f/"PUBLIC.pem",g.canonical(r),(f/"SIGNATURE.bin").read_bytes())
    req(r["provider"]==P["provider"] and r["upstream_commit"]==P["upstream_commit"] and r["source_files"]==P["source_files"] and r["capability"]==P["capability"] and r["run_id"]==os.environ["GITHUB_RUN_ID"] and r["run_attempt"]==os.environ["GITHUB_RUN_ATTEMPT"],"RESULT_BINDING")
    req(all(r[k] is False for k in ("external_peer","external_compute","llm_calls","full_evolution","training_authority","execution_authority")),"AUTHORITY")
    rows=r["rows"];req(len(rows)==len(P["candidate_degrees"])*len(P["scales"]) and r["negative_controls_rejected"]==2,"COVERAGE")
    for x,(degree,scale) in zip(rows,itertools.product(P["candidate_degrees"],P["scales"])):
        c=coefficients(degree,scale)
        req(x["degree"]==degree and x["scale"]==scale and x["id"]==f"p{degree}-s{scale}" and x["coefficients"]==c and x["program_sha256"]==sha(candidate_source(c).encode()),"CANDIDATE")
        fresh=dense(c)
        req(fresh["n"]==x["dense"]["n"] and all(math.isclose(fresh[k],x["dense"][k],rel_tol=1e-12,abs_tol=1e-15) for k in ("rmse","max_abs_error")),"DENSE_REPLAY")
        # Independently recreate upstream public grid and score without executing candidate code.
        es=[]
        for i in range(161):
            z=max(-math.pi,min(math.pi,-math.pi+2*math.pi*i/160+0.007*math.sin(i*1.61803398875)))
            a=0.
            for b in reversed(c):a=a*z*z+b
            es.append(abs(z*a-math.sin(z)))
        score=1/(1+4*math.sqrt(sum(y*y for y in es)/len(es))+max(es))
        req(math.isclose(score,x["upstream_score"],rel_tol=1e-12,abs_tol=1e-15),"SCORE_REPLAY")
    req(r["selected"]==max(rows,key=lambda x:x["upstream_score"])["id"],"SELECTION")
    return r
def beetle(folder,key_sha,lapis,out):
    r=verify(folder,key_sha)
    lp=pin(lapis,{"tools/beetle_context_projector.py":P["beetle_projector_blob"]},P["beetle_commit"]);o=Path(out).resolve();o.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);state=root/"state";state.mkdir();cfg=root/"config.json"
        result_hash=sha(g.canonical(r));cid=result_hash[:24]
        metadata={"candidate_id":cid,"title":"ShinkaEvolve algorithm evaluation mathematics","summary":"Verified polynomial approximation error comparison exercise","tags":["math","algorithm","comparison"],"source":"JFP","url":"https://github.com/"+P["provider"]+"/tree/"+P["upstream_commit"]}
        unrelated={"candidate_id":"irrelevant-control","title":"unrelated weather record","summary":"rainfall forecast","tags":[]}
        (state/"candidates.jsonl").write_text("\n".join(json.dumps(x) for x in (metadata,unrelated))+"\n")
        write(cfg,{"directions":[{"id":"janus-student-jfp-math","title":"JANUS student mathematics","keywords":["algorithm evaluation","error comparison"],"priority_keywords":["mathematics"],"min_score":3,"target_repositories":["Hawkar-usls/Janus-Demiurge"]}]})
        args=[sys.executable,str(lp/"tools/beetle_context_projector.py"),"--state-dir",str(state),"--effective-config",str(cfg),"--max-source-candidates","2"]
        run(args,root);first=load(state/"latest_context_projection.json")
        run(args,root);second=load(state/"latest_context_projection.json")
        req(first["new_contexts"]==1 and second["new_contexts"]==0,"BEETLE_ROUTE_OR_DEDUP")
        contexts=[json.loads(line) for p in sorted((state/"contexts.parts").glob("part-*.jsonl")) for line in p.read_text().splitlines()]
        req(len(contexts)==1 and contexts[0]["candidate_id"]==cid,"BEETLE_CONTEXT")
        ctx=contexts[0];write(o/"BEETLE_CONTEXT.json",ctx)
    # Small exact integer comparisons; use measured errors, not upstream authored prompts.
    tasks=[];gold=[];seen=set()
    for a,b in itertools.combinations(r["rows"],2):
        for metric in ("rmse","max_abs_error"):
            x=int(round(a["dense"][metric]*1000000));y=int(round(b["dense"][metric]*1000000))
            if x==y or (x,y) in seen or (y,x) in seen:continue
            seen.add((x,y));tid=sha(g.canonical([result_hash,a["id"],b["id"],metric]))[:24]
            # A lower error is better. Alternate order to avoid a positional answer shortcut.
            if len(tasks)%2:x,y=y,x
            tasks.append({"id":tid,"prompt":f"Which integer is smaller? A={x} B={y}. Reply A or B.","source_result_sha256":result_hash,"beetle_context_id":ctx["context_id"]})
            gold.append({"id":tid,"answer":"A" if x<y else "B"})
            if len(tasks)>=P["task_limit"]:break
        if len(tasks)>=P["task_limit"]:break
    req(len(tasks)==P["task_limit"] and len(set(x["answer"] for x in gold))==2,"TASK_COVERAGE")
    write(o/"STUDENT_TASKS.json",{"schema":"janus.trinity.public_development_candidates.v1","tasks":tasks,"training_authority":False,"exam_authority":False})
    write(o/"REVIEW_GOLD.json",{"schema":"janus.trinity.review_answers.v1","answers":gold,"public_development_only":True})
    summary={"schema":"janus.jfp.beetle_delivery_receipt.v1","source_result_sha256":result_hash,"beetle_commit":P["beetle_commit"],"projected":first["new_contexts"],"duplicates_on_repeat":second["new_contexts"],"irrelevant_rejected":True,"tasks":len(tasks),"task_sha256":sha((o/"STUDENT_TASKS.json").read_bytes()),"review_gold_sha256":sha((o/"REVIEW_GOLD.json").read_bytes()),"student_inference":False,"training_performed":False,"external_peer":False,"training_authority":False}
    write(o/"DELIVERY_RECEIPT.json",summary)
    print("JFP_R7C3_BEETLE "+json.dumps(summary,sort_keys=True))
if __name__=="__main__":
    if sys.argv[1]=="worker":worker(sys.argv[2],sys.argv[3])
    elif sys.argv[1]=="beetle":beetle(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
    else:raise ValueError("MODE")
