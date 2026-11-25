"""JDG R7C.1: one fixed, real upstream compiler capability. Not a remote peer."""
import base64,json,os,hashlib,subprocess,sys,tempfile
from pathlib import Path
import jfp_gateway as g
HERE=Path(__file__).parent
S=json.loads((HERE/"JDG_R7C1_CAPABILITY.json").read_text())
F=HERE/"JDG_R7C1_FIXTURE.md"
def sha(b):return hashlib.sha256(b).hexdigest()
def req(ok,msg):
    if not ok:raise ValueError(msg)
def request():
    return {"schema":"janus.jdg.request.v1","capability":S["capability"],"provider":S["provider"],"version":S["version"],"fixture_sha256":sha(F.read_bytes()),"binary_sha256":S["binary_sha256"],"run_id":os.environ["GITHUB_RUN_ID"],"run_attempt":os.environ["GITHUB_RUN_ATTEMPT"],"head":os.environ["GITHUB_SHA"],"action_authority":False}
def call(args,cwd,env):
    return subprocess.run(args,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=S["limits"]["timeout_seconds"],check=False)
def worker(binary,out):
    binary=Path(binary).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    raw=binary.read_bytes();req(len(raw)==S["binary_bytes"] and sha(raw)==S["binary_sha256"],"UPSTREAM_BINARY_PIN")
    q=request()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);home=root/"home";home.mkdir();work=root/"work";work.mkdir()
        env={"PATH":"/usr/local/bin:/usr/bin:/bin","HOME":str(home),"LANG":"C.UTF-8","CI":"true","GH_PROMPT_DISABLED":"1","GIT_CONFIG_NOSYSTEM":"1","GH_NO_UPDATE_NOTIFIER":"1"}
        p=call(["git","init","-q"],work,env);req(p.returncode==0,"TEMP_REPO")
        p=call(["git","remote","add","origin","https://github.com/Hawkar-usls/Janus.git"],work,env);req(p.returncode==0,"TEMP_REMOTE")
        w=work/".github/workflows";w.mkdir(parents=True);f=w/"jdg-fixture.md";f.write_bytes(F.read_bytes())
        args=[str(binary),"compile","jdg-fixture","--strict","--no-check-update"]
        p=call(args,work,env);print("JDG_COMPILER_LOG "+p.stdout.decode(errors="replace")[-7000:],flush=True)
        req(p.returncode==0,"UPSTREAM_COMPILE_FAILED")
        lock=w/"jdg-fixture.lock.yml";req(lock.is_file(),"MISSING_LOCK")
        generated=lock.read_bytes();req(0<len(generated)<=S["limits"]["output_bytes"],"LOCK_SIZE")
        (out/"COMPILED_LOCK.txt").write_bytes(generated)
        # Negative control: identical source but an explicitly forbidden write permission.
        lock.unlink();f.write_text(F.read_text().replace("contents: read","contents: write"))
        bad=call(args,work,env);print("JDG_NEGATIVE_LOG "+bad.stdout.decode(errors="replace")[-3000:],flush=True)
        req(bad.returncode!=0,"UNSAFE_CONTROL_ACCEPTED")
        k,pub=g.keys(root,"ephemeral")
        report={"schema":"janus.jdg.result.v1","request":q,"request_sha256":g.digest(q),"compiler_sha256":sha(raw),"compiled_sha256":sha(generated),"compiled_bytes":len(generated),"positive_exit":p.returncode,"negative_exit":bad.returncode,"runtime_execution":False,"engine_calls":False,"external_peer":False,"action_authority":False}
        signature=base64.b64encode(g.crypto(k,g.canonical(report))).decode()
        g.write_json(out/"RESULT.json",{"report":report,"signature":signature})
        (out/"PUBLIC.pem").write_bytes(pub.read_bytes())
        with open(os.environ["GITHUB_OUTPUT"],"a") as f:f.write("key_sha="+sha(pub.read_bytes())+"\n")
        print("JDG_WORKER "+json.dumps(report,sort_keys=True),flush=True)
def verify(folder,key_sha):
    folder=Path(folder);pub=folder/"PUBLIC.pem";req(sha(pub.read_bytes())==key_sha,"KEY_PIN")
    data=g.read_json(folder/"RESULT.json");r=data["report"]
    req(set(data)=={"report","signature"},"ENVELOPE")
    g.crypto(pub,g.canonical(r),base64.b64decode(data["signature"],validate=True))
    req(r["request"]==request() and r["request_sha256"]==g.digest(request()),"REQUEST_BINDING")
    lock=(folder/"COMPILED_LOCK.txt").read_bytes()
    req(0<len(lock)<=S["limits"]["output_bytes"] and sha(lock)==r["compiled_sha256"] and len(lock)==r["compiled_bytes"],"LOCK_HASH")
    req(r["compiler_sha256"]==S["binary_sha256"] and r["positive_exit"]==0 and r["negative_exit"]!=0,"COMPILER_RESULT")
    req(all(r[k] is False for k in ("runtime_execution","engine_calls","external_peer","action_authority")),"AUTHORITY")
    text=lock.decode("utf-8")
    req("jobs:" in text and "workflow_dispatch:" in text,"LOCK_SHAPE")
    print("JDG_R7C1_VERIFIED "+json.dumps({"status":"PASS_REAL_UPSTREAM_COMPILER","capability":S["capability"],"provider":S["provider"],"version":S["version"],"compiled_sha256":r["compiled_sha256"],"compiled_bytes":r["compiled_bytes"],"unsafe_control_rejected":True,"independent_verifier_job":True,"permanent_identity":False,"external_peers":0,"external_compute":False,"engine_calls":False,"action_authority":False},sort_keys=True))
if __name__=="__main__":
    if sys.argv[1]=="worker":worker(sys.argv[2],sys.argv[3])
    elif sys.argv[1]=="verify":verify(sys.argv[2],sys.argv[3])
    else:raise ValueError("MODE")
