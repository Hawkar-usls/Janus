"""R8.4 bounded archival replay and durable checkpoint pilot; no live learning authority."""
import hashlib,json,os,re,subprocess,sys,tempfile,time,urllib.request,urllib.error,copy
from pathlib import Path
import jfp_gateway as g
import r8_learning as r
J=r.J;M=r.M
BRANCH="research/r84-bounded-loop-20261125"
ROOT=Path("data/r84-loop")
STATE=ROOT/"STATE.json"
TREF="68df0112a994d36295bcf07fd0afffad3a5b7269"
RREF="a4a46ba619beb9e936e615d84e07e302b5b8f6de"
TBLOB="d4c9dafe1141076dad3f982032a92bba28891017"
RBLOB="d245054e2b48fd9f44ba7eddafeb29a476ce90c0"
PACKET="83d7b187547a67915d17e7cbbc9cb71cbab820103e3a0820db798b27b8626735"
DATA="604b8c48d0003e742208f06309b888d9f150067945bf52c179a6dc40dab71c8c"
RECEIPT="7fa71ca9d281ceb1f1de18d159053a4f28c287b904a4693718bfdd1729698cf4"
POLICY={"schema":"r84.policy.v1","nodes":[J,M],"max_cycles":2,"max_new_training":0,"max_queue":4,"allow_external":False,"production":False,"enabled":True,"persistent_signing_keys":False,"mode":"ARCHIVE_REPLAY_ONLY"}
def check(ok,why):g.require(ok,why)
def log(tag,obj):print(tag,json.dumps(obj,sort_keys=True),flush=True)
def cmd(args,timeout=30):
    x=subprocess.run(args,capture_output=True,text=True,timeout=timeout);check(x.returncode==0,"COMMAND:"+args[0]+":"+args[1]);return x.stdout.strip()
def get(repo,ref,path,pin=None):
    check(repo in (J,M) and re.fullmatch("[0-9a-f]{40}",ref),"REF")
    u="https://raw.githubusercontent.com/"+repo+"/"+ref+"/"+path
    with urllib.request.urlopen(u,timeout=10) as f:
        check(f.geturl()==u and f.status==200,"URL");raw=f.read(g.MAX_PACKET+1)
    check(len(raw)<=g.MAX_PACKET,"SIZE")
    if pin:check(r.blob(raw)==pin,"BLOB_PIN")
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"raw.json";p.write_bytes(raw);return g.read_json(p)
def archive():
    t=get(M,TREF,"data/r8-mailbox/TEACHING.json",TBLOB);v=get(J,RREF,"data/r8-mailbox/RESULT.json",RBLOB)
    at=v["packet"]["created"]
    p=r.verify(t,M,J,"TEACHING",at);q=r.verify(v,J,M,"SKILL_RECEIPT",at)
    z=q["payload"]
    check(g.digest(p)==PACKET and p["payload_sha256"]==DATA and g.digest(q)==RECEIPT,"ARCHIVE_HASH")
    check(p["payload"]==r.fixture() and z["teacher_commit"]==TREF and z["packet_sha256"]==PACKET and z["dataset_sha256"]==DATA and z["gate"] is True and z["production"] is False,"BINDING")
    check(z["new_after"]==16 and z["old_after"]==6 and z["rollback_exact"] is True,"SKILL")
    return {"dataset_sha256":DATA,"packet_sha256":PACKET,"receipt_sha256":RECEIPT,"teacher_commit":TREF,"learner_commit":RREF,"historical_keys":{M:hashlib.sha256(t["public_key"].encode()).hexdigest(),J:hashlib.sha256(v["public_key"].encode()).hexdigest()},"archive_only":True}
def fresh(a):
    return {"schema":"r84.state.v1","policy":POLICY,"archive":a,"queue":{DATA:{"status":"HISTORICALLY_VERIFIED","receipt":RECEIPT}},"cycles":0,"training":0,"events":[]}
def validate(s,a):
    check(s["schema"]=="r84.state.v1" and s["policy"]==POLICY and s["archive"]==a,"STATE_SCOPE")
    check(s["queue"]=={DATA:{"status":"HISTORICALLY_VERIFIED","receipt":RECEIPT}},"QUEUE")
    check(type(s["cycles"]) is int and 0<=s["cycles"]<=POLICY["max_cycles"] and s["training"]==0 and len(s["events"])==s["cycles"],"BUDGET")
    prev="0"*64
    for i,e in enumerate(s["events"],1):
        check(e["index"]==i and e["prev"]==prev and e["dataset"]==DATA and e["decision"]=="DUPLICATE_NO_TRAIN" and e["training"]==0,"EVENT")
        check(e["sha256"]==g.digest({k:v for k,v in e.items() if k!="sha256"}),"EVENT_HASH")
        prev=e["sha256"]
def wake(s,a,enabled=True):
    validate(s,a);check(enabled and POLICY["enabled"],"STOPPED")
    check(s["cycles"]<POLICY["max_cycles"],"CYCLE_BUDGET")
    check(DATA in s["queue"],"UNAPPROVED_DATA")
    n=copy.deepcopy(s);e={"index":s["cycles"]+1,"prev":s["events"][-1]["sha256"] if s["events"] else "0"*64,"dataset":DATA,"decision":"DUPLICATE_NO_TRAIN","training":0}
    e["sha256"]=g.digest(e);n["events"].append(e);n["cycles"]+=1;validate(n,a);return n
def negative(s,a):
    cases=[]
    for name,fn in (("stop",lambda:wake(s,a,False)),("changed_archive",lambda:validate(s,{**a,"receipt_sha256":"0"*64})),("unknown_queue",lambda:validate({**s,"queue":{**s["queue"],"0"*64:{"status":"PENDING"}}},a)),("training_budget",lambda:validate({**s,"training":1},a)),("cycle_budget",lambda:wake(s,a))):
        try:fn()
        except g.Reject as e:cases.append([name,str(e)])
        else:raise g.Reject("NEGATIVE_ACCEPTED:"+name)
    return cases
def publish(s):
    check(os.environ.get("GITHUB_REPOSITORY")==J and os.environ.get("R84_HEAD")==BRANCH and os.environ.get("GITHUB_RUN_ATTEMPT")=="1","WRITE_SCOPE")
    check(cmd(["git","remote","get-url","origin"]) in ("https://github.com/"+J,"https://github.com/"+J+".git"),"ORIGIN")
    ROOT.mkdir(parents=True,exist_ok=True);tmp=ROOT/"STATE.tmp";check(not tmp.exists(),"TEMP")
    g.write_json(tmp,s);tmp.replace(STATE)
    cmd(["git","add","--",str(STATE)]);check(cmd(["git","diff","--cached","--name-only"])==str(STATE),"STAGED_SCOPE")
    cmd(["git","-c","user.name=JANUS R84 research","-c","user.email=41898282+github-actions[bot]@users.noreply.github.com","commit","-m","research(R84): durable bounded wake "+str(s["cycles"])])
    cmd(["git","push","origin","HEAD:refs/heads/"+BRANCH],45)
    ref=cmd(["git","rev-parse","HEAD"])
    with open(os.environ["GITHUB_OUTPUT"],"a") as f:f.write("checkpoint="+ref+"\n")
    log("R84_CHECKPOINT",{"commit":ref,"cycles":s["cycles"],"training":s["training"],"state_sha256":g.digest(s)})
def bootstrap():
    check(not STATE.exists(),"ALREADY_BOOTSTRAPPED");a=archive();s=wake(fresh(a),a);publish(s)
def resume():
    a=archive();s=g.read_json(STATE);validate(s,a);check(s["cycles"]==1,"RESUME_SCOPE")
    n=wake(s,a);cases=negative(n,a);log("R84_NEGATIVE",cases);publish(n)
def witness():
    check(os.environ.get("GITHUB_REPOSITORY")==J,"REPO_SCOPE")
    a=archive();end=time.monotonic()+180
    while time.monotonic()<end:
        try:
            ref=cmd(["git","ls-remote","--heads","https://github.com/"+J+".git","refs/heads/"+BRANCH],10).split()
            if len(ref)==2:
                s=get(J,ref[0],str(STATE));validate(s,a)
                if s["cycles"]==2:
                    cases=negative(s,a);log("R84_WITNESS",{"commit":ref[0],"cycles":2,"training":0,"duplicate_noop":2,"negative_controls":cases,"state_sha256":g.digest(s),"archive_only":True,"production":False,"live_schedule":False,"persistent_keys":False});return
        except (urllib.error.HTTPError,urllib.error.URLError,subprocess.TimeoutExpired):pass
        time.sleep(3)
    raise g.Reject("WITNESS_TIMEOUT")
if __name__=="__main__":
    {"bootstrap":bootstrap,"resume":resume,"witness":witness}[sys.argv[1]]()
