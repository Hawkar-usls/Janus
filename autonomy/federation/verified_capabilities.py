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

def dispatch(request,legacy=None):
    if type(request) is str:
        return legacy(request) if legacy is not None else {"status":"LEGACY_UNCHANGED","answer":None}
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
