"""Bounded, typed event graph; supplied facts only, no text extraction."""
import json,sys,re,os,socket,subprocess,hashlib
from pathlib import Path
from collections import deque
from verified_capabilities import bfs
KINDS={"BEFORE","AFTER","CAUSES","REQUIRES","PREVENTS"}
def ck(x,msg):
    if not x:raise RuntimeError(msg)
def witness():
    current=os.readlink("/proc/self/ns/net");parent=os.environ.get("JANUS_PARENT_NETNS","")
    interfaces=[name for _,name in socket.if_nameindex()]
    routes=[json.loads(subprocess.run(["ip","-j",v,"route","show","table","all"],capture_output=True,text=True,check=True,timeout=5).stdout) for v in ("-4","-6")]
    w={"parent_namespace":parent,"process_namespace":current,"uid":os.geteuid(),"interfaces":interfaces,"ipv4_routes":routes[0],"ipv6_routes":routes[1]}
    print("R84H38C_NETWORK",json.dumps(w,sort_keys=True),flush=True)
    ck(bool(re.fullmatch(r"net:\[\d+\]",parent)) and current!=parent and os.geteuid()!=0 and interfaces==["lo"],"ISOLATION")
    ck(all(z.get("dev")=="lo" and z.get("dst") not in ("default","0.0.0.0/0","::/0") and "gateway" not in z for group in routes for z in group),"EXTERNAL_ROUTE")
    def deny(*a,**kw):raise RuntimeError("NETWORK_DISABLED")
    socket.create_connection=deny;socket.socket.connect=deny;socket.socket.connect_ex=deny
def ident(s):return type(s) is str and bool(re.fullmatch(r"[A-Za-z0-9_-]{1,48}",s))
def atom(x):
    return type(x) is dict and set(x)=={"left","rel","right","positive"} and ident(x["left"]) and ident(x["right"]) and type(x["rel"]) is str and x["rel"] in KINDS and type(x["positive"]) is bool
def canon(x):
    a,r,b=x["left"],x["rel"],x["right"]
    return (b,"BEFORE",a) if r=="AFTER" else (a,r,b)
def answer(status,proof=(),reason=""):
    return {"status":status,"proof":list(proof),"reason":reason}
def solve(q,direct=False):
    if type(q) is not dict or set(q)!={"schema","facts","claim"} or q["schema"]!="janus.event.graph.v1" or type(q["facts"]) is not list or len(q["facts"])>96 or not atom(q["claim"]):
        return answer("HOLD_INVALID",reason="request")
    fs=q["facts"];ids=set();pos={};neg={};nodes={q["claim"]["left"],q["claim"]["right"]}
    for f in fs:
        if type(f) is not dict or set(f)!={"id","left","rel","right","positive"} or not ident(f["id"]) or f["id"] in ids or not atom({k:v for k,v in f.items() if k!="id"}):
            return answer("HOLD_INVALID",reason="fact")
        ids.add(f["id"]);nodes.update((f["left"],f["right"]));k=canon(f)
        (pos if f["positive"] else neg).setdefault(k,f["id"])
    if len(nodes)>24:return answer("HOLD_INVALID",reason="node_limit")
    for k in sorted(pos.keys()&neg.keys()):
        return answer("HOLD_CONFLICT",[pos[k],neg[k]],"explicit_polarity")
    adj={n:[] for n in nodes}
    for (a,r,b),i in sorted(pos.items()):
        if r=="BEFORE":adj[a].append((b,i))
    def path(a,b):
        queue=deque([(a,[])]);seen={a}
        while queue:
            u,p=queue.popleft()
            for v,i in adj[u]:
                if v==b:return p+[i]
                if v not in seen:seen.add(v);queue.append((v,p+[i]))
        return None
    # Reject cyclic strict time globally, even for unrelated claims.
    for n in sorted(nodes):
        p=path(n,n)
        if p:return answer("HOLD_CONFLICT",p,"temporal_cycle")
    for (a,r,b),i in sorted(neg.items()):
        if r=="BEFORE":
            p=path(a,b)
            if p:return answer("HOLD_CONFLICT",p+[i],"negative_temporal_conflict")
    a,r,b=canon(q["claim"]);k=(a,r,b);positive=q["claim"]["positive"]
    yes=[pos[k]] if k in pos else None
    no=[neg[k]] if k in neg else None
    if r=="BEFORE" and not direct:
        yes=path(a,b)
        if a!=b:
            reverse=path(b,a)
            if reverse:no=reverse
        # Reuse H23 exact graph reachability as independent cross-check
        # for distinct endpoints, not as causal authority.
        if a!=b:
            ns=sorted(nodes);ix={n:i for i,n in enumerate(ns)}
            mat=[0]*(len(ns)**2)
            for u in ns:
                for v,_ in adj[u]:mat[ix[u]*len(ns)+ix[v]]=1
            roots=[int(n==a) for n in ns]
            ck(bool(bfs({"roots":roots,"edges":mat,"target":ix[b]}))==bool(yes),"BFS_PARITY")
    if r=="BEFORE" and a==b and not direct:
        return answer("CONTRADICTED" if positive else "ENTAILED",reason="strict_irreflexivity")
    if yes:return answer("ENTAILED" if positive else "CONTRADICTED",yes,"explicit_or_temporal_path")
    if no:return answer("CONTRADICTED" if positive else "ENTAILED",no,"explicit_negative_or_reverse_time")
    return answer("UNKNOWN",reason="insufficient_evidence")
def main(exam,out):
    witness()
    raw=exam.read_bytes();data=json.loads(raw);out.mkdir(parents=True,exist_ok=False)
    cases=data["cases"];ck(len({z["id"] for z in cases})==len(cases),"CASE_IDS")
    details=[];metrics={}
    for z in cases:
        g=solve(z["request"]);base=solve(z["request"],True)
        # Every proof references supplied fact IDs. No invented evidence.
        fids={f.get("id") for f in z["request"].get("facts",[]) if type(f) is dict}
        ck(set(g["proof"])<=fids,"PROOF_REFERENCE")
        details.append({"id":z["id"],"cohort":z["cohort"],"expected":z["expected"],"graph":g,"direct":base})
    for mode in ("graph","direct"):
        m={"correct":0,"wrong":0,"false_certainty":0,"by_cohort":{}}
        for z in details:
            ok=z[mode]["status"]==z["expected"]
            m["correct"]+=ok;m["wrong"]+=not ok
            m["false_certainty"]+=z[mode]["status"] in ("ENTAILED","CONTRADICTED") and not ok
            c=m["by_cohort"].setdefault(z["cohort"],{"n":0,"correct":0});c["n"]+=1;c["correct"]+=ok
        metrics[mode]=m
    # Reordering facts must not change results.
    for z,d in zip(cases,details):
        q=z["request"];rev={**q,"facts":list(reversed(q["facts"]))}
        ck(solve(rev)["status"]==d["graph"]["status"],"ORDER_INVARIANCE")
    report={"schema":"h38c.result.v1","n":len(cases),"exam_sha256":hashlib.sha256(raw).hexdigest(),"metrics":metrics,"order_invariance":len(cases),"production":False,"quality_gate":"PASS_TYPED_SHADOW" if metrics["graph"]["wrong"]==0 else "HOLD","scope":"Same-author bounded typed event graphs; temporal closure only; causal relations explicit, no natural-language extraction, no autonomous knowledge admission, no MiniLM or full regression."}
    for name,obj in [("result.json",report),("cases.json",details)]:(out/name).write_text(json.dumps(obj,sort_keys=True)+"\n")
    print("R84H38C_RESULT",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":main(*map(Path,sys.argv[1:]))
