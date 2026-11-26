"""Opt-in deterministic typed capability. No model or external execution."""
import json,sys
def valid(q):
    if type(q) is not dict or set(q)!={"schema","edges","roots","target"} or q["schema"]!="janus.graph.reachability.v1":return False
    roots=q["roots"];edges=q["edges"];target=q["target"]
    if type(roots) is not list or not 1<=len(roots)<=24 or type(edges) is not list or len(edges)!=len(roots)**2:return False
    return type(target) is int and 0<=target<len(roots) and all(type(x) is int and x in (0,1) for x in roots+edges)
def bfs(z):
    n=len(z["roots"]);visited={i for i,v in enumerate(z["roots"]) if v};pending=list(visited)
    while pending:
        u=pending.pop()
        for v in range(n):
            if z["edges"][u*n+v] and v not in visited:visited.add(v);pending.append(v)
    return int(z["target"] in visited)

def relation(q):
    if type(q) is not dict or set(q)!={"schema","facts","claim"} or q["schema"]!="janus.relation.v1":return "HOLD_INVALID"
    facts=q["facts"];claim=q["claim"]
    def valid(x):
        return type(x) is dict and set(x)=={"subject","attribute","value"} and all(type(x[k]) is str and 0<len(x[k])<=80 for k in ("subject","attribute")) and type(x["value"]) is int and x["value"] in (0,1)
    if type(facts) is not list or not 1<=len(facts)<=32 or not valid(claim) or not all(valid(x) for x in facts):return "HOLD_INVALID"
    seen={}
    for x in facts:
        k=(x["subject"],x["attribute"])
        if k in seen and seen[k]!=x["value"]:return "HOLD_CONFLICT"
        seen[k]=x["value"]
    k=(claim["subject"],claim["attribute"])
    return "neutral" if k not in seen else "entailment" if seen[k]==claim["value"] else "contradiction"

def dispatch(request,legacy=None):
    if type(request) is str:
        return legacy(request) if legacy is not None else {"status":"LEGACY_UNCHANGED","answer":None}
    if type(request) is dict and request.get("schema")=="janus.relation.v1":
        answer=relation(request)
        return {"status":answer if answer.startswith("HOLD_") else "RELATION","answer":None if answer.startswith("HOLD_") else answer,"route":"exact_relation","authority":"supplied_structured_facts_only"}
    if not valid(request):
        return {"status":"REJECT","answer":None}
    return {"status":"GRAPH","answer":bfs(request),"route":"exact_bfs","authority":"bounded_typed_reachability"}
def main():
    raw=sys.stdin.buffer.read(16385)
    if len(raw)>16384:raise SystemExit("REQUEST_TOO_LARGE")
    try:q=json.loads(raw)
    except (ValueError,UnicodeError):q=None
    print(json.dumps(dispatch(q),sort_keys=True))
if __name__=="__main__":main()
