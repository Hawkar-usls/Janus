"""R84B bounded incremental internal learning; no production or general packet execution."""
import copy,hashlib,json,os,re,subprocess,sys,tempfile,time,urllib.error,urllib.request
from pathlib import Path
import jfp_gateway as g
import r8_learning as r
import r84_loop as prior
J=r.J;M=r.M
BRANCH="research/r84b-live-intake-20261125"
FIX="c8ec992f88f74cf48d80a0db85cd19250d408fa2"
FBLOB="0daebe2816c82c20ccac1304bc535e7e8801b193"
EXAM="ecc0a3b7f57cc22e5f0ea03de7cc8009031929c6"
OLD_MODEL="52d33cd1cde244ffcdd0e7a4532596778b641667868b81070a47e198b9d24927"
DIR=Path("data/r84b-learning")
STATE=DIR/"STATE.json"
TEACH=DIR/"TEACHING.json"
RESULT=DIR/"RESULT.json"
LABELS=[{"calibration","replication"},{"latency","memory"}]
MAX_TRAIN=2
MAX_WAKE=3
def check(ok,why):g.require(ok,why)
def log(tag,o):print(tag,json.dumps(o,sort_keys=True),flush=True)
def remote(repo,ref,path,pin=None):
    check(repo in (J,M) and re.fullmatch("[0-9a-f]{40}",ref) and path in ("data/r84b/TEACHING_FIXTURE.json",str(TEACH),str(RESULT),str(STATE)),"REMOTE_SCOPE")
    return prior.get(repo,ref,path,pin)
def find(repo,path,wait=150):
    end=time.monotonic()+wait
    while time.monotonic()<end:
        try:
            s=prior.cmd(["git","ls-remote","--heads","https://github.com/"+repo+".git","refs/heads/"+BRANCH],10).split()
            if len(s)==2 and s[1]=="refs/heads/"+BRANCH:return remote(repo,s[0],str(path)),s[0]
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError,subprocess.TimeoutExpired):pass
        time.sleep(3)
    raise g.Reject("MAILBOX_TIMEOUT")
def fixture():
    f=remote(M,FIX,"data/r84b/TEACHING_FIXTURE.json",FBLOB)
    check(set(f)=={"schema","packets"} and f["schema"]=="r84b.fixture.v1" and len(f["packets"])==2,"FIXTURE")
    alltext=set()
    for i,p in enumerate(f["packets"]):
        check(set(p)=={"schema","author","license","purpose","sequence","examples"} and p["schema"]=="r84b.training.v1" and p["author"]=="Hawkar-usls/internal-r84b-synthetic" and p["license"]=="CC0-1.0" and p["purpose"]=="research request routing only" and p["sequence"]==i+1,"PROVENANCE")
        check(len(p["examples"])==10 and {y for t,y in p["examples"]}==LABELS[i],"LABELS")
        for t,y in p["examples"]:
            r.tokens(t);check(t not in alltext,"DUPLICATE_ROW");alltext.add(t)
    return f["packets"]
def publish(repo,paths):
    check(os.environ.get("GITHUB_REPOSITORY")==repo and os.environ.get("R84B_HEAD")==BRANCH and os.environ.get("GITHUB_RUN_ATTEMPT")=="1","WRITE_SCOPE")
    check(all(p in (TEACH,STATE,RESULT) for p in paths),"PATH")
    check(prior.cmd(["git","remote","get-url","origin"]) in ("https://github.com/"+repo,"https://github.com/"+repo+".git"),"ORIGIN")
    prior.cmd(["git","add","--"]+list(map(str,paths)))
    check(set(prior.cmd(["git","diff","--cached","--name-only"]).splitlines())==set(map(str,paths)),"STAGED_SCOPE")
    prior.cmd(["git","-c","user.name=JANUS R84B research","-c","user.email=41898282+github-actions[bot]@users.noreply.github.com","commit","-m","research(R84B): bounded learning checkpoint"])
    prior.cmd(["git","push","origin","HEAD:refs/heads/"+BRANCH],40)
    ref=prior.cmd(["git","rev-parse","HEAD"])
    with open(os.environ["GITHUB_OUTPUT"],"a") as f:f.write("checkpoint="+ref+"\n")
    return ref
def teacher():
    f=fixture();DIR.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        k,pub=g.keys(Path(d),"teacher")
        packets=[{"public_key":pub.read_text(),"packet":r.envelope("TEACHING_R84B",M,J,FIX,x,k)} for x in f]
    g.write_json(TEACH,{"schema":"r84b.mailbox.v1","packets":packets});ref=publish(M,[TEACH])
    log("R84B_TEACH",{"commit":ref,"datasets":[b["packet"]["payload_sha256"] for b in packets],"examples":20,"internal":True})
def admit(bundle,at=None):
    check(type(bundle) is dict and set(bundle)=={"schema","packets"} and bundle["schema"]=="r84b.mailbox.v1" and len(bundle["packets"])==2,"BUNDLE")
    b=bundle["packets"];f=fixture()
    check(b[0]["public_key"]==b[1]["public_key"],"KEY_PIN")
    p=[r.verify(x,M,J,"TEACHING_R84B",at) for x in b]
    check(p[0]["run"]==p[1]["run"] and all(x["source_commit"]==FIX and x["payload"]==y for x,y in zip(p,f)),"SOURCE")
    check(len({x["payload_sha256"] for x in p})==2,"DUPLICATE_DATASET")
    return p
def initial():
    a=prior.archive();old=r.OLD+r.fixture()["examples"];m=r.train(old)
    check(g.digest(m)==OLD_MODEL,"PRIOR_SKILL_HASH")
    return {"schema":"r84b.state.v1","archive":a,"teacher_commit":None,"teacher_bundle":None,"accepted":[],"events":[],"model":m,"training":0,"wakes":0,"production":False}
def rows(s,p):
    return r.OLD+r.fixture()["examples"]+[row for i in range(s["training"]) for row in p[i]["payload"]["examples"]]
def validate(s,at=None):
    check(s["schema"]=="r84b.state.v1" and s["archive"]==prior.archive() and s["production"] is False,"STATE_SCOPE")
    check(type(s["training"]) is int and type(s["wakes"]) is int and 0<=s["training"]<=MAX_TRAIN and s["training"]<=s["wakes"]<=MAX_WAKE and len(s["events"])==s["wakes"],"BUDGET")
    p=[] if s["teacher_bundle"] is None else admit(s["teacher_bundle"],at)
    check(s["training"]<=len(p) and s["accepted"]==[x["payload_sha256"] for x in p[:s["training"]]],"QUEUE")
    check(g.digest(s["model"])==g.digest(r.train(rows(s,p))),"MODEL")
    prev="0"*64
    for i,e in enumerate(s["events"],1):
        check(e["index"]==i and e["prev"]==prev and e["sha256"]==g.digest({k:v for k,v in e.items() if k!="sha256"}),"CHAIN")
        check(e["decision"]==("TRAIN_VERIFIED" if i<=s["training"] else "DUPLICATE_NO_TRAIN"),"EVENT")
        prev=e["sha256"]
    return p
def next_candidate(s,bundle,ref,sequence):
    validate(s)
    p=admit(bundle)
    check(s["wakes"]<MAX_WAKE and s["training"]<MAX_TRAIN and sequence==s["training"]+1,"BUDGET")
    check(s["teacher_commit"] in (None,ref) and s["teacher_bundle"] in (None,bundle),"TEACHER_PIN")
    d=p[sequence-1]["payload_sha256"];check(d not in s["accepted"] and d!=prior.DATA,"DUPLICATE")
    c=copy.deepcopy(s);c["teacher_commit"]=ref;c["teacher_bundle"]=bundle;c["accepted"].append(d);c["training"]+=1
    c["model"]=r.train(rows(c,p));return c
def learn(out,seq):
    check(os.environ["GITHUB_REPOSITORY"]==J,"REPO")
    s=g.read_json(STATE) if STATE.exists() else initial();check(s["training"]==seq-1 and s["wakes"]==seq-1,"RESUME")
    b,ref=find(M,TEACH) if seq==1 else (s["teacher_bundle"],s["teacher_commit"])
    c=next_candidate(s,b,ref,seq);out.mkdir(parents=True,exist_ok=False)
    g.write_json(out/"candidate.json",c);g.write_json(out/"parent.json",{"sha256":g.digest(s)})
    log("R84B_TRAIN",{"sequence":seq,"prior_model":g.digest(s["model"]),"new_model":g.digest(c["model"]),"dataset":c["accepted"][-1],"training":c["training"],"teacher_commit":ref})
def exams():
    raw=(Path(__file__).parent/"R84B_FROZEN_EXAM.json").read_bytes();check(r.blob(raw)==EXAM,"EXAM_PIN")
    e=json.loads(raw);o=r.exam();return [o["old"],o["new"],e["cohort1"],e["cohort2"]]
def exact(m,rows):return [r.predict(m,t)==y for t,y in rows]
def append_event(s,decision,dataset,metrics):
    e={"index":s["wakes"]+1,"prev":s["events"][-1]["sha256"] if s["events"] else "0"*64,"decision":decision,"dataset":dataset,"metrics":metrics,"model_sha256":g.digest(s["model"])}
    e["sha256"]=g.digest(e);s["events"].append(e);s["wakes"]+=1
def audit(inp,seq):
    s=g.read_json(STATE) if STATE.exists() else initial();check(g.read_json(inp/"parent.json")["sha256"]==g.digest(s),"PARENT")
    c=g.read_json(inp/"candidate.json");check(c==next_candidate(s,c["teacher_bundle"],c["teacher_commit"],seq),"RECOMPUTE")
    p=admit(c["teacher_bundle"]);ex=exams();seen={t for t,y in rows(c,p)};check(not seen&{t for group in ex for t,y in group},"EXAM_OVERLAP")
    before=[exact(s["model"],group) for group in ex];after=[exact(c["model"],group) for group in ex]
    for i in range(seq+1):check(all(before[i]) and all(after[i]),"REGRESSION")
    check(sum(before[seq+1])==0 and sum(after[seq+1])>=7,"GAIN")
    metrics={"old":[sum(x) for x in before[:seq+1]],"retained":[sum(x) for x in after[:seq+1]],"new_before":sum(before[seq+1]),"new_after":sum(after[seq+1]),"new_n":len(after[seq+1])}
    # Rebuild pre-packet state on revocation: exact hash equality; not production.
    rollback=r.train(rows(s,p));check(rollback==s["model"],"ROLLBACK")
    append_event(c,"TRAIN_VERIFIED",c["accepted"][-1],metrics);validate(c)
    DIR.mkdir(parents=True,exist_ok=True)
    if STATE.exists():STATE.unlink()
    g.write_json(STATE,c);ref=publish(J,[STATE])
    log("R84B_AUDIT",{"sequence":seq,"commit":ref,"metrics":metrics,"rollback_exact":True,"state_sha256":g.digest(c)})
def replay():
    s=g.read_json(STATE);p=validate(s);check(s["training"]==2 and s["wakes"]==2,"REPLAY_SCOPE")
    negatives=[]
    for name,fn in (("budget",lambda:next_candidate(s,s["teacher_bundle"],s["teacher_commit"],2)),("tamper",lambda:validate({**s,"model":{"bad":1}})),("unknown",lambda:validate({**s,"accepted":s["accepted"]+["0"*64]})),("key_substitution",lambda:admit({**s["teacher_bundle"],"packets":[s["teacher_bundle"]["packets"][0],{**s["teacher_bundle"]["packets"][1],"public_key":"invalid"}]}))):
        try:fn()
        except (g.Reject,ValueError) as e:negatives.append([name,str(e)])
        else:raise g.Reject("NEGATIVE_ACCEPTED:"+name)
    check(all(x["payload_sha256"] in s["accepted"] for x in p),"DEDUPE")
    append_event(s,"DUPLICATE_NO_TRAIN",s["accepted"][-1],{"negative_controls":negatives,"duplicate_packets":2});validate(s)
    ex=exams();scores=[sum(exact(s["model"],x)) for x in ex];check(scores[:2]==[6,16] and min(scores[2:])>=7,"FINAL_REGRESSION")
    STATE.unlink();g.write_json(STATE,s)
    result={"schema":"r84b.result.v1","teacher_commit":s["teacher_commit"],"datasets":s["accepted"],"state_sha256":g.digest(s),"model_sha256":g.digest(s["model"]),"scores":scores,"wakes":s["wakes"],"training":s["training"],"production":False}
    with tempfile.TemporaryDirectory() as d:
        k,pub=g.keys(Path(d),"janus");q=r.envelope("SKILL_R84B",J,M,prior.cmd(["git","rev-parse","HEAD"]),result,k);g.write_json(RESULT,{"public_key":pub.read_text(),"packet":q})
    ref=publish(J,[STATE,RESULT]);log("R84B_RETURN",{"commit":ref,**result,"negative_controls":negatives})
def witness():
    b,ref=find(J,RESULT,210);q=r.verify(b,J,M,"SKILL_R84B");z=q["payload"]
    s=remote(J,ref,str(STATE));validate(s);check(s["wakes"]==3 and s["training"]==2 and z["state_sha256"]==g.digest(s) and z["model_sha256"]==g.digest(s["model"]) and z["datasets"]==s["accepted"] and z["teacher_commit"]==s["teacher_commit"] and z["production"] is False,"RETURN")
    own=remote(M,s["teacher_commit"],str(TEACH));check(own==s["teacher_bundle"],"TEACHER_BIND")
    check(z["scores"][:2]==[6,16] and min(z["scores"][2:])>=7,"SCORES")
    log("R84B_WITNESS",{"commit":ref,"state_sha256":g.digest(s),"wakes":3,"training":2,"duplicate_packets":2,"scores":z["scores"],"same_owner":True,"persistent_keys":False,"schedule":False,"production":False})
if __name__=="__main__":
    mode=sys.argv[1]
    if mode=="teacher":teacher()
    elif mode=="learn":learn(Path(sys.argv[2]),int(sys.argv[3]))
    elif mode=="audit":audit(Path(sys.argv[2]),int(sys.argv[3]))
    elif mode=="replay":replay()
    elif mode=="witness":witness()
    else:raise ValueError("MODE")
