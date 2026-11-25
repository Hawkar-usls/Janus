"""Bounded, explicitly allowed same-owner JFP learning pilot. No incoming code execution."""
import base64, collections, hashlib, json, math, os, re, sqlite3, subprocess, sys, tempfile, time, urllib.request, urllib.error
from pathlib import Path
import jfp_gateway as g
J="Hawkar-usls/Janus"
M="Hawkar-usls/janus-meta-registry"
BRANCH="research/r8-live-learning-20261125"
PILOT="R8-INTERNAL-20261125-01"
FIXTURE_COMMIT="ea4098071f0a265897f8876f64cb5ecbfd0bb33d"
FIXTURE_BLOB="d1b4f5ec9220e25eef58b3f0cbac2c1f299d81b0"
EXAM_BLOB="453b018c0c425cee7ecd792bfbc6b8ffce0a972e"
DIR=Path("data/r8-mailbox")
LABELS={"provenance","license","regression","rollback"}
OLD=[["inspect workflow run status","status"],["show job status progress","status"],["report pipeline run progress","status"],["find repository source file","locate"],["locate directory path","locate"],["search project file location","locate"]]
FIELDS={"schema","pilot","kind","sender","recipient","run","created","source_commit","payload","payload_sha256","signature"}
def check(v,s):g.require(v,s)
def blob(raw):return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(p):return g.read_json(p)
def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);g.write_json(p,obj)
def cmd(args,cwd=None,timeout=30):
    r=subprocess.run(args,cwd=cwd,capture_output=True,text=True,timeout=timeout);check(r.returncode==0,"COMMAND:"+args[0]+":"+args[1]);return r.stdout.strip()
def context(repo):
    check(os.environ.get("GITHUB_REPOSITORY")==repo and os.environ.get("R8_HEAD")==BRANCH,"REPO_SCOPE")
    check(os.environ.get("GITHUB_RUN_ATTEMPT")=="1","ONE_SHOT")
def raw(repo,ref,path):
    check(repo in (J,M) and re.fullmatch("[0-9a-f]{40}",ref) and path in ("data/r8/TEACHING_FIXTURE.json",str(DIR/"TEACHING.json"),str(DIR/"RESULT.json")),"REMOTE_SCOPE")
    u="https://raw.githubusercontent.com/"+repo+"/"+ref+"/"+path
    with urllib.request.urlopen(u,timeout=10) as f:
        check(f.geturl()==u and f.status==200,"URL")
        b=f.read(g.MAX_PACKET+1)
    check(len(b)<=g.MAX_PACKET,"SIZE")
    return b
def remote(repo,name,wait=180):
    check(repo in (J,M) and name in ("TEACHING.json","RESULT.json"),"REMOTE_SCOPE")
    end=time.monotonic()+wait
    while time.monotonic()<end:
        try:
            with tempfile.TemporaryDirectory() as d:
                s=cmd(["git","ls-remote","--heads","https://github.com/"+repo+".git","refs/heads/"+BRANCH],cwd=d,timeout=10).split()
            if len(s)==2 and s[1]=="refs/heads/"+BRANCH:
                b=raw(repo,s[0],str(DIR/name))
                with tempfile.TemporaryDirectory() as d:
                    p=Path(d)/"b.json";p.write_bytes(b);obj=read(p)
                return obj,s[0]
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError,subprocess.TimeoutExpired):
            pass
        time.sleep(3)
    raise g.Reject("MAILBOX_TIMEOUT:"+name)
def publish(repo,name):
    context(repo);check(name in ("TEACHING.json","RESULT.json"),"WRITE_SCOPE")
    origin=cmd(["git","remote","get-url","origin"]);check(origin in ("https://github.com/"+repo,"https://github.com/"+repo+".git"),"ORIGIN")
    p=str(DIR/name);cmd(["git","add","--",p])
    check(cmd(["git","diff","--cached","--name-only"])==p,"STAGED_SCOPE")
    cmd(["git","-c","user.name=JANUS R8 pilot","-c","user.email=41898282+github-actions[bot]@users.noreply.github.com","commit","-m","research(R8): publish "+name])
    cmd(["git","push","origin","HEAD:refs/heads/"+BRANCH],timeout=40)
    return cmd(["git","rev-parse","HEAD"])
def envelope(kind,sender,recipient,source,payload,key):
    p={"schema":"janus.jfp.learning.v1","pilot":PILOT,"kind":kind,"sender":sender,"recipient":recipient,"run":os.environ["GITHUB_RUN_ID"],"created":int(time.time()),"source_commit":source,"payload":payload,"payload_sha256":g.digest(payload)}
    p["signature"]=base64.b64encode(g.crypto(key,g.canonical(p))).decode("ascii")
    return p
def verify(b,sender,recipient,kind,now=None):
    # Ephemeral key authenticated by the explicitly allowlisted repository branch, NOT independent operator identity.
    check(type(b) is dict and set(b)=={"public_key","packet"} and type(b["public_key"]) is str and len(b["public_key"])<256,"BUNDLE")
    p=b["packet"];check(type(p) is dict and set(p)==FIELDS,"ENVELOPE")
    check(p["schema"]=="janus.jfp.learning.v1" and p["pilot"]==PILOT and p["sender"]==sender and p["recipient"]==recipient and p["kind"]==kind,"SCOPE")
    at=int(time.time()) if now is None else now
    check(type(p["created"]) is int and 0<=at-p["created"]<=900 and re.fullmatch("[0-9]{1,30}",p["run"]) and re.fullmatch("[0-9a-f]{40}",p["source_commit"]),"FRESHNESS")
    check(p["payload_sha256"]==g.digest(p["payload"]),"PAYLOAD_HASH")
    sig=base64.b64decode(p["signature"],validate=True);check(len(sig)==64,"SIG")
    with tempfile.TemporaryDirectory() as d:
        pub=Path(d)/"pub.pem";pub.write_text(b["public_key"]);g.crypto(pub,g.canonical(g.body(p)),sig)
    return p
def tokens(t):
    check(type(t) is str and len(t)<=160 and re.fullmatch("[a-z ]+",t),"TEXT")
    return re.findall("[a-z]+",t)
def fixture():
    rawbytes=raw(M,FIXTURE_COMMIT,"data/r8/TEACHING_FIXTURE.json");check(blob(rawbytes)==FIXTURE_BLOB,"FIXTURE_PIN")
    x=json.loads(rawbytes)
    check(set(x)=={"schema","author","license","purpose","examples"} and x["schema"]=="janus.r8.training_fixture.v1" and x["author"]=="Hawkar-usls/internal-r8-synthetic" and x["license"]=="CC0-1.0" and x["purpose"]=="research request routing only","LICENSE_OR_PROVENANCE")
    check(len(x["examples"])==20 and len({t for t,y in x["examples"]})==20 and collections.Counter(y for t,y in x["examples"])=={k:5 for k in LABELS},"DATA_SCHEMA")
    for t,y in x["examples"]:tokens(t)
    return x
def teacher():
    context(M);x=fixture()
    with tempfile.TemporaryDirectory() as d:
        key,pub=g.keys(Path(d),"teacher");p=envelope("TEACHING",M,J,FIXTURE_COMMIT,x,key)
        write(DIR/"TEACHING.json",{"public_key":pub.read_text(),"packet":p})
    c=publish(M,"TEACHING.json");print("R8_TEACHER_PUBLISHED",json.dumps({"commit":c,"packet_sha256":g.digest(p),"examples":20,"internal":True}),flush=True)
def admit(b,ref,db,revoked=()):
    p=verify(b,M,J,"TEACHING");check(p["source_commit"]==FIXTURE_COMMIT,"PROVENANCE");check(p["payload"]==fixture(),"SOURCE_DATA")
    h=p["payload_sha256"];check(h not in revoked,"REVOKED")
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE IF NOT EXISTS seen(hash TEXT PRIMARY KEY)")
        try:c.execute("INSERT INTO seen VALUES (?)",(h,))
        except sqlite3.IntegrityError:raise g.Reject("DUPLICATE")
    return p
def train(rows):
    counts={};docs={};vocab=set()
    for t,y in rows:
        docs[y]=docs.get(y,0)+1;c=counts.setdefault(y,{})
        for w in tokens(t):c[w]=c.get(w,0)+1;vocab.add(w)
    return {"schema":"r8.multinomial_nb.v1","counts":counts,"docs":docs,"vocab":sorted(vocab)}
def predict(model,t):
    v=set(model["vocab"]);scores={}
    for y,c in model["counts"].items():
        total=sum(c.values())+len(v)
        scores[y]=math.log(model["docs"][y]/sum(model["docs"].values()))+sum(math.log((c.get(w,0)+1)/total) for w in tokens(t) if w in v)
    return max(sorted(scores),key=lambda y:scores[y])
def learner(out):
    check(os.environ["GITHUB_REPOSITORY"]==J,"REPO_SCOPE")
    b,ref=remote(M,"TEACHING.json")
    out.mkdir(parents=True,exist_ok=False);write(out/"quarantine.json",b)
    db=out/"dedup.sqlite";p=admit(b,ref,db);rejections=[]
    for tag,x,rev in (("duplicate",b,()),("tamper",{**b,"packet":{**b["packet"],"payload_sha256":"0"*64}},()),("revoked",b,(p["payload_sha256"],)),("unknown_peer",{**b,"packet":{**b["packet"],"sender":"unapproved"}},())):
        try:admit(x,ref,db,rev)
        except (g.Reject,ValueError) as e:rejections.append({"case":tag,"reason":str(e)})
        else:raise g.Reject("NEGATIVE_ACCEPTED:"+tag)
    check(len(rejections)==4,"NEGATIVES")
    base=train(OLD);candidate=train(OLD+p["payload"]["examples"])
    write(out/"baseline.json",base);write(out/"candidate.json",candidate)
    write(out/"receipt.json",{"schema":"r8.training_receipt.v1","teacher_commit":ref,"packet_sha256":g.digest(p),"dataset_sha256":p["payload_sha256"],"baseline_sha256":g.digest(base),"candidate_sha256":g.digest(candidate),"examples":20,"negative_controls":rejections,"promotion":False})
    print("R8_LEARNER",json.dumps({"teacher_commit":ref,"new_examples":20,"negative_controls":rejections,"candidate_sha256":g.digest(candidate)}),flush=True)
def exam():
    rawbytes=(Path(__file__).parent/"R8_FROZEN_SKILL_EXAM.json").read_bytes();check(blob(rawbytes)==EXAM_BLOB,"EXAM_PIN");return json.loads(rawbytes)
def audit(inp,out):
    check(os.environ["GITHUB_REPOSITORY"]==J,"REPO_SCOPE")
    r=read(inp/"receipt.json");b=read(inp/"quarantine.json");p=verify(b,M,J,"TEACHING")
    check(p["source_commit"]==FIXTURE_COMMIT and p["payload"]==fixture() and g.digest(p)==r["packet_sha256"],"PROVENANCE")
    baseline=read(inp/"baseline.json");candidate=read(inp/"candidate.json")
    check(baseline==train(OLD) and candidate==train(OLD+p["payload"]["examples"]) and g.digest(candidate)==r["candidate_sha256"],"MODEL_INTEGRITY")
    ex=exam();seen={t for t,y in OLD+p["payload"]["examples"]};check(not seen&{t for t,y in ex["old"]+ex["new"]},"OVERLAP")
    def score(model,rows):return [predict(model,t)==y for t,y in rows]
    old0=score(baseline,ex["old"]);old1=score(candidate,ex["old"]);new0=score(baseline,ex["new"]);new1=score(candidate,ex["new"])
    # Simulated revocation, rebuild from retained pre-packet data (not a production rollback).
    revoked=train(OLD);check(g.digest(revoked)==g.digest(baseline) and score(revoked,ex["old"])==old0,"ROLLBACK")
    gate=sum(new1)>=12 and sum(new1)>sum(new0) and all(not a or b for a,b in zip(old0,old1)) and sum(old0)==len(old0)
    result={"schema":"r8.verified_skill.v1","teacher_commit":r["teacher_commit"],"packet_sha256":r["packet_sha256"],"dataset_sha256":r["dataset_sha256"],"candidate_sha256":g.digest(candidate),"exam_blob":EXAM_BLOB,"old_before":sum(old0),"old_after":sum(old1),"old_n":len(old0),"new_before":sum(new0),"new_after":sum(new1),"new_n":len(new0),"rollback_exact":True,"negative_controls":r["negative_controls"],"gate":gate,"production":False,"independent_operator":False}
    out.mkdir(parents=True,exist_ok=False);write(out/"audit.json",result)
    print("R8_AUDIT",json.dumps(result),flush=True);check(gate,"SKILL_GATE")
def return_skill(inp):
    context(J);r=read(inp/"audit.json");check(r["gate"] is True and r["production"] is False,"GATE")
    with tempfile.TemporaryDirectory() as d:
        key,pub=g.keys(Path(d),"learner");p=envelope("SKILL_RECEIPT",J,M,cmd(["git","rev-parse","HEAD"]),r,key)
        write(DIR/"RESULT.json",{"public_key":pub.read_text(),"packet":p})
    c=publish(J,"RESULT.json");print("R8_RETURN",json.dumps({"commit":c,"receipt_sha256":g.digest(p),"gate":True}),flush=True)
def teacher_audit():
    check(os.environ["GITHUB_REPOSITORY"]==M,"REPO_SCOPE")
    own,ownref=remote(M,"TEACHING.json",15);p=verify(own,M,J,"TEACHING")
    b,ref=remote(J,"RESULT.json",240);q=verify(b,J,M,"SKILL_RECEIPT")
    r=q["payload"];check(r["schema"]=="r8.verified_skill.v1" and r["packet_sha256"]==g.digest(p) and r["teacher_commit"]==ownref and r["dataset_sha256"]==p["payload_sha256"] and r["exam_blob"]==EXAM_BLOB and r["gate"] is True and r["production"] is False and r["old_before"]==r["old_after"]==r["old_n"] and r["new_after"]>=12 and r["new_before"]==0,"RECEIPT")
    print("R8_CLOSED_LOOP",json.dumps({"teacher_commit":ownref,"learner_commit":ref,"receipt_sha256":g.digest(q),"new":r["new_after"],"old":r["old_after"],"same_owner":True,"production":False}),flush=True)
if __name__=="__main__":
    mode=sys.argv[1];args=[Path(x) for x in sys.argv[2:]]
    {"teacher":teacher,"learner":learner,"audit":audit,"return":return_skill,"teacher_audit":teacher_audit}[mode](*args)
