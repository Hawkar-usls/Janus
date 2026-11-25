"""R84H17 finite lexical-semantic biome router; immutable H16 world, bounded inference."""
import sys,json,time,random,re,gc
from pathlib import Path
import r84h16_chunk as h16
import r84h15_world as h15
import r84h14_partition as h14
import r84h13_streaming as h13
import r84h10_resonance as h10
import r84e_neural as e
import r8_learning as r
HERE=Path(__file__).parent
EXAM="63a598e24fbf1e73f1e6157b8964cb49819c9205"
COLORS=("amber","cobalt","jade","silver","ivory","coral","violet")
COLOR_ALIASES=("golden","ultramarine","emerald","argent","cream","salmon","purple")
NOUNS=("ledger","relay","archive","queue","catalog","buffer","vault","index","journal","registry","mirror","channel","broker","cache","terminal")
NOUN_ALIASES=("account book","repeater","repository","waiting line","inventory","staging area","strongbox","lookup table","logbook","register","replica","conduit","intermediary","temporary store","console")
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
class Router:
    """Finite ontology built from registry names and authored synonym vocabulary.
    A token trie recognizes canonical names and aliases, without scanning chunks.
    No answer labels, exam patterns, or weights are consulted during address lookup.
    """
    def __init__(self):
        self.trie={};self.aliases=0
        for i,color in enumerate(COLORS):
            for j,noun in enumerate(NOUNS):
                domain=color+" "+noun;ck(domain==h15.DOMAINS[i*15+j],"ONTOLOGY")
                for c in (color,COLOR_ALIASES[i]):
                    for n in (noun,NOUN_ALIASES[j]):
                        node=self.trie
                        for word in (c+" "+n).split():node=node.setdefault(word,{})
                        ck("$" not in node,"ALIAS_CONFLICT");node["$"]=domain;self.aliases+=1
    def mentions(self,text):
        tok=list(re.finditer(r"[a-z]+",text.lower()));out=[];i=0
        while i<len(tok):
            node=self.trie;last=None;j=i
            while j<len(tok) and tok[j].group() in node:
                node=node[tok[j].group()];j+=1
                if "$" in node:last=(j,node["$"])
            if last:
                j,name=last;out.append((tok[i].start(),tok[j-1].end(),name));i=j
            else:i+=1
        return out
    def plan(self,text):
        # Only explicit scope cues are supported for open-world abstention.
        # Generic unfamiliar language falls through to the original legacy system.
        if re.search(r"\b(?:unexplored|unmapped)\b",text.lower()):return "FOG",[]
        if not self.mentions(text):return "LEGACY",[]
        clauses=[t.strip() for t in re.split(r";|\band then\b",text,flags=re.I) if t.strip()]
        if len(clauses)>3:return "AMBIGUOUS",[]
        plan=[]
        for clause in clauses:
            mentions=self.mentions(clause);unique={x[2] for x in mentions}
            if len(unique)!=1:return "AMBIGUOUS",[]
            # Multiple mentions of the same domain remain a single scope.
            name=next(iter(unique));normalized=clause
            for start,end,domain in reversed(mentions):normalized=normalized[:start]+domain+normalized[end:]
            plan.append({"domain":name,"address":"fixture/"+name,"text":normalized})
        return ("SINGLE" if len(plan)==1 else "CROSS"),plan
def evaluate(prep,rootpath,bundlepath,h10path,h11path,h12path,fixtures,snapshot,out):
    raw=(HERE/"R84H17_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN");ex=json.loads(raw)
    receipt=e.read(snapshot/"snapshot.json");path=snapshot/"world.sqlite";ck(receipt["size"]==100000 and receipt["world"]==h16.WORLD and receipt["production"] is False and h16.sha(path.read_bytes())==receipt["db_sha256"],"WORLD_PIN")
    s=h16.Store(path,True);ck(s.count()==100000,"WORLD_COUNT");router=Router();L,dd,root,d,mapping,labels=h15.base_data(prep)
    cells,coarse,mat,offsets,indexbytes=h13.catalog(L,d,mapping,labels,h15.SEED,0,h11path,h12path)
    for i,c in enumerate(cells):
        rec=s.get("legacy/cell-%02d"%i);ck(rec["payload"]=={"route":c.route,"model_sha256":c.pin},"LEGACY_PIN")
    atlas=h14.Atlas(cells);groups,_=h14.exams(prep,dd,d);hist=sum(groups,[]);ck(len(hist)==671,"HIST")
    actual=[labels[y] for _,y in hist];base,br,persistent=h13.persistent(L,root,d,h15.SEED,0,rootpath,bundlepath,h10path,mapping,hist)
    ep,er,_=h13.eager(L,cells,hist,base,br);ck(sum(x==y for x,y in zip(ep,actual))==386,"BASELINE")
    raw=(HERE/"R84H15_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==h15.EXAM,"ORIGINAL_FIXTURE_PIN");old=json.loads(raw)
    ck(old["domains"]==h15.DOMAINS and len(old["rows"])==210,"FIXTURE_ROWS")
    original=[{"text":t,"expected":[{"domain":h15.DOMAINS[i],"op":k}]} for t,i,k in old["rows"]]
    req=[("historical",{"text":t},i) for i,(t,_) in enumerate(hist)]
    for cohort,rows in [("original",original)]+[(k,ex[k]) for k in ("fresh","cross","ambiguous","unknown","mixed","openUnknown")]:
        req.extend((cohort,row,j) for j,row in enumerate(rows))
    ck(len({x[1]["text"].strip().lower() for x in req if x[0]!="historical"})==len(req)-len(hist),"EXAM_DUPLICATES")
    random.Random(17001).shuffle(req)
    cache=h13.Cache(3);lazy=h16.Lazy(L,d,mapping,labels,fixtures);first=max(labels.values())+1;stats={k:0 for k in ("gate_checks","boundary_checks","sprite_checks","sprite_rejected","region_checks","regions_rejected","region_cells_rejected")}
    results=[];lookup=0;routercpu=0.;gc.collect();start=time.process_time();wall=time.perf_counter()
    for cohort,row,ix in req:
        t=row["text"];ts=time.process_time();status,plan=router.plan(t);routercpu+=time.process_time()-ts
        domains=[x["domain"] for x in plan];pred=[];routes=[];gate_rejects=0
        if status=="LEGACY" and cohort=="historical":
            y,z=base[ix],br[ix];terms=h10.terms(t)
            for j in atlas.visible(terms,"sprite",stats):
                c=cells[j]
                if c.qualifies(terms,stats):
                    y=c.predict(cache.get(c),terms);z=c.route;break
            pred=[y];routes=[z]
        elif status in ("SINGLE","CROSS"):
            # Resolve the whole plan first; no partial outputs on a bad scope.
            records=[]
            for p in plan:
                rec=s.get(p["address"]);lookup+=1
                if rec is None or rec["kind"]!="fixture_ready":status="FOG";break
                records.append(rec)
            if status!="FOG":
                for p,rec in zip(plan,records):
                    c=lazy.get(rec);terms=h10.terms(p["text"])
                    if not c.qualifies(terms,stats):
                        pred.append(None);routes.append(None);gate_rejects+=1
                    else:
                        pred.append(c.predict(cache.get(c),terms));routes.append(c.route)
        expected=row.get("expected",[]);expected_domains=[p["domain"] for p in expected]
        expected_pred=[first+2*h15.DOMAINS.index(p["domain"])+p["op"] for p in expected]
        result={"cohort":cohort,"i":ix,"status":status,"domains":domains,"pred":pred,"routes":routes,"gate_rejects":gate_rejects}
        if cohort=="historical":
            result.update({"parity":status=="LEGACY" and pred==[ep[ix]],"route_parity":status=="LEGACY" and routes==[er[ix]],"correct":pred==[actual[ix]]})
        elif expected:
            wanted="SINGLE" if len(expected)==1 else "CROSS"
            result.update({"address_correct":status==wanted and domains==expected_domains,"answer_correct":status==wanted and domains==expected_domains and pred==expected_pred,"expected_domains":expected_domains})
        elif cohort!="openUnknown":result["abstain_correct"]=status==row["status"] and not pred
        else:result["open_unknown_detected"]=status=="FOG"
        results.append(result)
    elapsed=time.process_time()-start;elapsedwall=time.perf_counter()-wall
    metrics={}
    for cohort in ("historical","original","fresh","cross","ambiguous","unknown","mixed","openUnknown"):
        rs=[x for x in results if x["cohort"]==cohort];metric={"n":len(rs),"statuses":{v:sum(x["status"]==v for x in rs) for v in sorted({x["status"] for x in rs})}}
        for k in ("parity","route_parity","correct","address_correct","answer_correct","abstain_correct","open_unknown_detected","gate_rejects"):
            if any(k in x for x in rs):metric[k]=sum(int(x.get(k,0)) for x in rs)
        metrics[cohort]=metric
    ck(metrics["historical"]["n"]==671 and metrics["historical"]["parity"]==671 and metrics["historical"]["route_parity"]==671 and metrics["historical"]["correct"]==386,"LEGACY_BORDER")
    ck(metrics["original"]["address_correct"]==210 and metrics["original"]["answer_correct"]==210,"ORIGINAL_ROUTING")
    ck(all(metrics[k]["abstain_correct"]==metrics[k]["n"] for k in ("ambiguous","unknown","mixed")),"SCOPED_ABSTENTION")
    ck(cache.max_cells<=3 and lazy.max_cells<=3,"CAP")
    s.close();ck(h16.sha(path.read_bytes())==receipt["db_sha256"],"WORLD_UNCHANGED")
    row={"schema":"r84h17.result.v1","metrics":metrics,"world_size":100000,"ready_fixture_cells":105,"legacy_cells":15,"biomes":7,"aliases":router.aliases,"address_lookups":lookup,"router_cpu_seconds":routercpu,"stream_cpu_seconds":elapsed,"stream_wall_seconds":elapsedwall,"rss_peak_bytes":h15.peak(),"max_lazy_metadata_cells":lazy.max_cells,"lazy_metadata_loads":lazy.loads,**cache.stats(),**stats,"production":False,"external_validation":False,"scope":"single seed111; explicit artificial domain names or authored aliases extracted from text, no supplied address/label to router; split up to three same-ontology clauses; scoped fog cue, unmarked unknown diagnostic only; no general semantic ontology or mixed legacy/fixture composition; exact historical parity and canonical fixture regression hard gates; novel alias answer and composition diagnostic; no weight updates or promotion"}
    out.mkdir(parents=True,exist_ok=True);e.write(out/"result.json",row);e.write(out/"cases.json",results);log("R84H17_RESULT",row)
    log("R84H17_GATE",{"legacy_parity":True,"original_routing":True,"scoped_abstention":True,"fresh_full_pass":metrics["fresh"]["answer_correct"]==metrics["fresh"]["n"],"cross_full_pass":metrics["cross"]["answer_correct"]==metrics["cross"]["n"],"open_unknown_supported":False,"production":False})
    cache.close()
if __name__=="__main__":evaluate(*map(Path,sys.argv[1:]))
