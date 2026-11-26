"""H38M diagnostic and diverse, bounded, independently checked graph curriculum.
Real source witnesses anchor hypothetical tasks; only operational provenance is
real. No article claims or new model promotion.
"""
import sys,json,hashlib,copy
from pathlib import Path
from collections import Counter
import torch
import r84h38l_knowledge_crystallization as l
import r84h38j_fractal_self_learning as j
import r84h38i_fractal_reconstruction as i
import r84h38h_semantic_kaleidoscope as h
import r84h38e_semantic_invariance as e
import r84h38d_world_understanding as d
import r84h38c_semantic_world_graph as g
ck=j.ck;read=j.read;write=j.write;sha=j.sha
REL=("BEFORE","CAUSES","REQUIRES","PREVENTS")
def atom(a,r,b,p=True):return {"left":a,"rel":r,"right":b,"positive":p}
def request(facts,claim):return {"schema":"janus.event.graph.v1","facts":[{"id":"s"+str(k),**a} for k,a in enumerate(facts)],"claim":claim}
def independent(q):
    # Separate finite relational reference, using a Boolean closure matrix,
    # not H38C's BFS or the learned model. Scope is generated valid atoms.
    pos=set();neg=set();nodes=set()
    for a in q["facts"]:
        t=(a["left"],a["rel"],a["right"]);(pos if a["positive"] else neg).add(t);nodes.update((t[0],t[2]))
    c=q["claim"];nodes.update((c["left"],c["right"]));ns=sorted(nodes);idx={v:k for k,v in enumerate(ns)};n=len(ns)
    mat=[[False]*n for _ in ns]
    for a,r,b in pos:
        if r=="BEFORE":mat[idx[a]][idx[b]]=True
    for k in range(n):
        for x in range(n):
            for y in range(n):mat[x][y]|=mat[x][k] and mat[k][y]
    if pos&neg or any(mat[x][x] for x in range(n)) or any(r=="BEFORE" and mat[idx[a]][idx[b]] for a,r,b in neg):return "HOLD_CONFLICT"
    a,r,b=c["left"],c["rel"],c["right"];key=(a,r,b);yes=key in pos;no=key in neg
    if r=="BEFORE":
        yes=mat[idx[a]][idx[b]]
        no=no or a==b or mat[idx[b]][idx[a]]
    if yes:return "ENTAILED" if c["positive"] else "CONTRADICTED"
    if no:return "CONTRADICTED" if c["positive"] else "ENTAILED"
    return "UNKNOWN"
def verify_task(q,expected):
    oracle=independent(q);actual=g.solve(q)
    ck(oracle==expected==actual["status"],"TRINITY_DIVERGENCE")
    ck(set(actual["proof"])<={z["id"] for z in q["facts"]},"PROOF_REF")
    ck(g.solve({**q,"facts":list(reversed(q["facts"]))})["status"]==expected,"ORDER")
    return actual
def scenarios(p):
    a=p["derived"]["left"];c=p["derived"]["right"];b=p["atoms"][0]["right"];x="unlinked_"+p["witness"]["source_sha256"][:16]
    # Five explicit lenses: forward, reversal, negation, composition, unrelated.
    # Only BEFORE has transitive/reverse-time inference. Others are hypothetical.
    for r in REL:
        ab=atom(a,r,b);bc=atom(b,r,c);ac=atom(a,r,c)
        fs=[ab,bc];origin="REAL_OPERATIONAL_PATH" if r=="BEFORE" else "HYPOTHETICAL_GRAPH_NOT_SOURCE_CLAIM"
        for lens,claim,expected in [
            ("forward",ab,"ENTAILED"),
            ("reverse",atom(b,r,a),"CONTRADICTED" if r=="BEFORE" else "UNKNOWN"),
            ("negation",atom(a,r,b,False),"CONTRADICTED"),
            ("composition",ac,"ENTAILED" if r=="BEFORE" else "UNKNOWN"),
            ("unrelated",atom(a,r,x),"UNKNOWN")]:
            yield r,lens,origin,request(fs,claim),expected
        yield r,"conflict","HYPOTHETICAL_GRAPH_NOT_SOURCE_CLAIM",request(fs+[atom(a,r,b,False)],ab),"HOLD_CONFLICT"
        # Explicit negative facts are synthetic, and are not inferred from absence.
        yield r,"negative_fact","HYPOTHETICAL_GRAPH_NOT_SOURCE_CLAIM",request([atom(a,r,b,False)],atom(a,r,b,False)),"ENTAILED"
def frame(q,r,lens,proof):
    # Recursive four-branch return: facts, claim, relation, verdict.
    facts=[i.packet("fact",a,[]) for a in q["facts"]]
    branch=[i.packet("facts",len(facts),[],children=facts),
            i.packet("claim",q["claim"],[]),
            i.packet("relation",r,[]),
            i.packet("verdict",proof["status"],[],children=[i.packet("proof",proof["proof"],[])])]
    out=i.packet("scenario",lens,[],children=branch);ck(i.verify(out),"FRACTAL_RETURN");return out
def diag(hdir,gdir,idir,edir,ldir,demi,results,exam,out):
    g.witness();torch.set_num_threads(2);out.mkdir();hr,bank,ns=i.load_h(hdir);gm=i.load_g(gdir,hr["bank_sha"]);ir,im=j.load_i(idir);lr=read(ldir/"receipt.json")
    er=read(edir/"receipt.json");eb=read(edir/"bank.json");ck(sha(edir/"bank.json")==er["bank_sha"],"E_BANK")
    em=[]
    for z in er["runs"]:
        p=edir/f"model-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"E_WEIGHT");net=e.Net();net.load_state_dict(j.load_file(str(p)));em.append(net.eval())
    di=e.Dictionary(e.cortex(demi),edir/"cortex",eb["train"]);active=j.load_models(ldir,lr["active"]);rule=lr["rule"]
    cases=read(results/"l-results/cases.json");report=read(results/"l-results/comparison.json");ex=read(exam)
    ck(report["l_candidate"]["exam_sha"]==sha(exam) and len(cases)==len(ex["rows"]),"EXAM_PIN")
    errors=[]
    for z,orig in zip(cases,ex["rows"]):
        ck(all(z[k]==orig[k] for k in ("text","names","gold","y")),"CASE_BIND")
        pred=z["pred"]["stack"]
        if pred is None or pred==z["gold"]:continue
        n=e.norm(z["text"],z["names"]);ck(n is not None,"NORM")
        t,events=n;base=d.parse(z["text"]);ej=e.pick(e.probs(em,[t])[0],di.search(t),er["rules"]["agreement"],"agreement")
        traces={"frozen":i.trace(t,hr,ns,bank,gm),"parent_tile":i.trace(t,hr,ns,bank,im,rule),"active":i.trace(t,hr,ns,bank,active,rule)}
        route="grammar" if base is not None else "h38e" if ej is not None else next((k for k,v in traces.items() if v["value"] is not None),None)
        got=base if route=="grammar" else e.decode(ej,events) if route=="h38e" else e.decode(traces[route]["value"],events)
        ck(got==pred,"ROUTE_REPLAY")
        gold=z["gold"];diff=[]
        if gold is None:diff=["certainty"]
        else:
            if got["rel"]!=gold["rel"]:diff.append("relation")
            if got["positive"]!=gold["positive"]:diff.append("polarity")
            if (got["left"],got["right"])!=(gold["left"],gold["right"]):diff.append("direction")
        errors.append({"text":z["text"],"gold":gold,"pred":got,"route":route,"failed_slots":diff,"traces":traces})
    ck(len(errors)==1,"EXPECTED_ONE_ERROR")
    write(out/"diagnostic.json",errors)
    print("R84H38M_DIAGNOSTIC",json.dumps({"errors":errors,"source_exam_sha":sha(exam)},sort_keys=True),flush=True)
def diversify(kcode,queue,source,diagnostic,out):
    g.witness();torch.set_num_threads(2);out.mkdir();k=l.load_k(kcode);packets=list(l.source_packets(k,queue,source));ck(bool(packets),"NO_PACKETS")
    seen={};rows=[];counts=Counter();lenses=Counter()
    for dg,p in packets[:8]:
        for r,lens,origin,q,expected in scenarios(p):
            proof=verify_task(q,expected);ret=frame(q,r,lens,proof)
            # Canonical scenario family, independent of source-specific event IDs.
            key=r+":"+lens
            if key not in seen:seen[key]=0
            seen[key]+=1;counts[origin]+=1;lenses[lens]+=1
            rows.append({"source_sha":dg,"source_receipt":p["receipt_sha256"],"family":key,"origin":origin,"request":q,"expected":expected,"proof":proof,"return":ret,"scientific_truth":False,"training_status":"RESEARCH_CANDIDATE_NOT_ADMITTED"})
    sample=rows[0];bad=copy.deepcopy(sample["return"]);bad["children"][0]["value"]=99;ck(not i.verify(bad),"TAMPER")
    ck(independent(sample["request"])!="UNKNOWN","CONTROL")
    # Wrong expected label must be vetoed, independent of model confidence.
    veto=0
    for row in rows[:7]:
        wrong="UNKNOWN" if row["expected"]!="UNKNOWN" else "ENTAILED"
        try:verify_task(row["request"],wrong)
        except RuntimeError:veto+=1
    ck(veto==7,"NEGATIVE_CONTROLS")
    errors=read(diagnostic/"diagnostic.json");slots=sorted({x for a in errors for x in a["failed_slots"]})
    # Allowlisted topics, never use arbitrary source/diagnostic text as commands.
    topics={"relation":["relation extraction","semantic parsing"],"polarity":["negation scope","natural language inference"],"direction":["temporal relation extraction","semantic role labeling"],"certainty":["selective prediction","uncertainty calibration"]}
    terms=sorted({t for slot in slots for t in topics[slot]})
    ck(bool(terms),"GAP_EMPTY")
    req={"schema":"janus.h38m.gap.v1","source_exam":"H38L_DISCLOSED_DIAGNOSTIC","failed_slots":slots,"terms":terms,"purpose":"SEARCH_METADATA_ONLY_NOT_TRUTH","query":" OR ".join('all:"'+t+'"' for t in terms),"max_results":6}
    write(out/"gap_request.json",req);write(out/"tasks.json",rows)
    result={"schema":"h38m.diversify.v1","source_packets":min(len(packets),8),"rendered_graph_tasks":len(rows),"unique_families":len(seen),"families":seen,"origin_counts":dict(counts),"lenses":dict(lenses),"reference_agreement":len(rows),"negative_controls_vetoed":veto,"tampered_return_rejected":True,"gap":req,"scientific_claims_admitted":0,"neural_training_admitted":0,"production":False}
    write(out/"result.json",result);print("R84H38M_DIVERSIFY",json.dumps(result,sort_keys=True),flush=True)
if __name__=="__main__":{"diag":diag,"diversify":diversify}[sys.argv[1]](*map(Path,sys.argv[2:]))
