"""H38D: bounded explicit English clauses -> frozen H38C graph, shadow only."""
import hashlib,json,re,sys
from pathlib import Path
import r84h38c_semantic_world_graph as graph

# H38B's role/order/negation approach, implemented as a separate clause
# front end. H38B's neural classifier is NOT used or modified here.
NAME=r"([a-z][a-z0-9_-]{0,39})"
RULES=[
    (NAME+r" (?:occurred|happened|took place) before "+NAME,"BEFORE",True,False),
    (NAME+r" (?:occurred|happened|took place) after "+NAME,"BEFORE",True,True),
    (NAME+r" precedes "+NAME,"BEFORE",True,False),
    (NAME+r" follows "+NAME,"BEFORE",True,True),
    (r"before "+NAME+r", "+NAME+r" (?:occurred|happened)","BEFORE",True,True),
    (r"after "+NAME+r", "+NAME+r" (?:occurred|happened)","BEFORE",True,False),
    (r"only after "+NAME+r" did "+NAME+r" (?:occur|happen)","BEFORE",True,False),
    (NAME+r" did not (?:occur|happen) before "+NAME,"BEFORE",False,False),
    (NAME+r" did not (?:occur|happen) after "+NAME,"BEFORE",False,True),
    (NAME+r" (?:caused|causes) "+NAME,"CAUSES",True,False),
    (NAME+r" was caused by "+NAME,"CAUSES",True,True),
    (NAME+r" did not cause "+NAME,"CAUSES",False,False),
    (NAME+r" (?:requires|required) "+NAME,"REQUIRES",True,False),
    (NAME+r" is required by "+NAME,"REQUIRES",True,True),
    (NAME+r" does not require "+NAME,"REQUIRES",False,False),
    (NAME+r" (?:prevents|prevented) "+NAME,"PREVENTS",True,False),
    (NAME+r" was prevented by "+NAME,"PREVENTS",True,True),
    (NAME+r" did not prevent "+NAME,"PREVENTS",False,False),
]
RESERVED={"it","he","she","they","this","that","there","something","someone","nothing","anything","unknown","before","after","not","never","maybe","none","all","either","both","no","yes"}
def parse(text):
    if type(text) is not str or not 1<=len(text)<=220:return None
    # Full-match only, one explicit relation per sentence; no conditional,
    # reported speech, coreference, conjunction or speculative assertion.
    t=" ".join(text.strip().lower().split())
    if t.endswith("."):t=t[:-1].rstrip()
    found=[]
    for pattern,rel,positive,swap in RULES:
        m=re.fullmatch(pattern,t)
        if m:
            a,b=m.groups()
            if a in RESERVED or b in RESERVED:return None
            if swap:a,b=b,a
            found.append({"left":a,"rel":rel,"right":b,"positive":positive})
    return found[0] if len(found)==1 else None
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def compile_text(sentences,claim,direct=False):
    if type(sentences) is not list or len(sentences)>96 or not all(type(t) is str and 1<=len(t)<=220 for t in sentences) or type(claim) is not str or not 1<=len(claim)<=220:
        return {"status":"HOLD_INVALID","proof":[],"facts":[],"unparsed":[],"claim":None}
    facts=[];unparsed=[];source={}
    for i,t in enumerate(sentences,1):
        a=parse(t);sid="s"+str(i)
        if a is None:unparsed.append(sid);continue
        facts.append({"id":sid,**a})
        source[sid]={"id":sid,"text":t,"sha256":digest(t)}
    q=parse(claim)
    # Do not silently discard unsupported evidence and then claim certainty.
    if unparsed or q is None:
        return {"status":"HOLD_UNPARSED","proof":[],"facts":facts,"unparsed":unparsed,"claim":q,"sources":source}
    result=graph.solve({"schema":"janus.event.graph.v1","facts":facts,"claim":q},direct=direct)
    return {**result,"facts":facts,"unparsed":[],"claim":q,"sources":source,"proof_sources":[source[i] for i in result["proof"]]}
def main(exam,prior,out):
    graph.witness()
    raw=exam.read_bytes();data=json.loads(raw);out.mkdir(parents=True,exist_ok=False)
    old=json.loads(prior.read_text())["cases"]
    graph.ck(len(old)==76 and all(graph.solve(z["request"])["status"]==z["expected"] for z in old),"H38C_REGRESSION")
    totals={k:{"correct":0,"wrong":0,"false_certainty":0,"by_cohort":{}} for k in ("text_graph","text_direct","oracle_graph")}
    extraction={"n":0,"exact":0,"expected_relations":0,"correct_relations":0,"false_extractions":0,"expected_abstentions":0,"correct_abstentions":0}
    details=[];order=0;proof_checks=0
    for z in data["cases"]:
        s=z["sentences"];q=z["claim"];targets=z["atoms"];qc=z["claim_atom"]
        graph.ck(len(s)==len(targets),"TARGET_LENGTH")
        for t,expected in list(zip(s,targets))+[(q,qc)]:
            p=parse(t);extraction["n"]+=1;extraction["exact"]+=p==expected
            if expected is None:
                extraction["expected_abstentions"]+=1;extraction["correct_abstentions"]+=p is None
            else:
                extraction["expected_relations"]+=1;extraction["correct_relations"]+=p==expected
            extraction["false_extractions"]+=p is not None and p!=expected
        g=compile_text(s,q);d=compile_text(s,q,True)
        if qc is None or any(x is None for x in targets):o={"status":"HOLD_UNPARSED"}
        else:o=graph.solve({"schema":"janus.event.graph.v1","facts":[{"id":"s"+str(i),**a} for i,a in enumerate(targets,1)],"claim":qc})
        graph.ck(o["status"]==z["expected"],"ORACLE_LABEL_"+z["id"])
        for k,x in (("text_graph",g),("text_direct",d),("oracle_graph",o)):
            m=totals[k];ok=x["status"]==z["expected"];m["correct"]+=ok;m["wrong"]+=not ok
            m["false_certainty"]+=x["status"] in ("ENTAILED","CONTRADICTED") and not ok
            c=m["by_cohort"].setdefault(z["cohort"],{"n":0,"correct":0});c["n"]+=1;c["correct"]+=ok
        graph.ck(compile_text(list(reversed(s)),q)["status"]==g["status"],"ORDER_INVARIANCE");order+=1
        for p in g.get("proof_sources",[]):
            i=int(p["id"][1:])-1;graph.ck(p["text"]==s[i] and p["sha256"]==digest(s[i]),"PROVENANCE");proof_checks+=1
        details.append({"id":z["id"],"cohort":z["cohort"],"expected":z["expected"],"graph":g,"direct_status":d["status"],"oracle_status":o["status"]})
    report={"schema":"h38d.result.v1","exam_sha256":hashlib.sha256(raw).hexdigest(),"n":len(details),"extraction":extraction,"metrics":totals,"h38c_regression":len(old),"order_invariance":order,"proof_source_checks":proof_checks,"quality_gate":"PASS_BOUNDED_TEXT_SHADOW" if totals["text_graph"]["wrong"]==0 and extraction["exact"]==extraction["n"] else "HOLD","production":False,"scope":"Same-author explicit single-clause English statements with supplied one-token event names; deterministic grammar inspired by H38B, immutable H38C, no neural training, no broad language, no external exam, no production integration."}
    for name,obj in [("result.json",report),("cases.json",details)]:(out/name).write_text(json.dumps(obj,sort_keys=True)+"\n")
    print("R84H38D_RESULT",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":main(*map(Path,sys.argv[1:]))
