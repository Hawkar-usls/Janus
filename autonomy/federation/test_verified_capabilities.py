import json,random,unittest
from pathlib import Path
import verified_capabilities as v
HERE=Path(__file__).parent
class Integration(unittest.TestCase):
    def test_frozen(self):
        rows=json.loads((HERE/"R84H23_FROZEN_EXAM.json").read_text())["rows"]
        self.assertEqual(len(rows),1836)
        for z in rows:
            q={"schema":"janus.graph.reachability.v1",**{k:z[k] for k in ("edges","roots","target")}}
            self.assertEqual(v.dispatch(q)["answer"],z["expected"])
            n=len(q["roots"]);reachable=q["roots"][:]
            for _ in range(n):reachable=[int(reachable[j] or any(reachable[i] and q["edges"][i*n+j] for i in range(n))) for j in range(n)]
            self.assertEqual(v.dispatch(q)["answer"],reachable[q["target"]])
    def test_legacy(self):
        for t in ("hello","graph reachability","cache hit",""):
            self.assertEqual(v.dispatch(t,lambda s:("legacy",s)),("legacy",t))
            self.assertEqual(v.dispatch(t)["status"],"LEGACY_UNCHANGED")
    def test_fail_closed(self):
        good={"schema":"janus.graph.reachability.v1","edges":[0],"roots":[1],"target":0}
        self.assertEqual(v.dispatch(good)["answer"],1)
        for q in (None,[],{},{"schema":"unknown"},{**good,"target":True},{**good,"edges":[2]},{**good,"roots":[True]},{**good,"expected":1},{**good,"target":1},{**good,"edges":[]},{**good,"roots":[0]*25,"edges":[0]*625}):
            self.assertEqual(v.dispatch(q)["status"],"REJECT")
    def test_relations(self):
        count=0
        for i in range(64):
            for attribute in ("presence","sequence","access"):
                for a in (0,1):
                    for b in (0,1):
                        for same in (False,True):
                            subject="holdout-%d"%i
                            fact={"subject":subject,"attribute":attribute,"value":a}
                            claim={"subject":subject if same else "other-"+subject,"attribute":attribute,"value":b}
                            q={"schema":"janus.relation.v1","facts":[fact],"claim":claim}
                            expected="neutral" if not same else "entailment" if a==b else "contradiction"
                            self.assertEqual(v.dispatch(json.loads(json.dumps(q)))["answer"],expected)
                            count+=1
        self.assertEqual(count,1536)
    def test_relation_boundaries(self):
        f={"subject":"s","attribute":"a","value":1}
        q={"schema":"janus.relation.v1","facts":[f],"claim":f}
        self.assertEqual(v.dispatch(q)["answer"],"entailment")
        self.assertEqual(v.dispatch({**q,"facts":[f,{**f,"value":0}]})["status"],"HOLD_CONFLICT")
        self.assertEqual(v.dispatch({**q,"facts":[f,f]})["answer"],"entailment")
        self.assertEqual(v.dispatch({**q,"claim":{**f,"attribute":"b"}})["answer"],"neutral")
        for bad in ({**q,"facts":[]},{**q,"facts":[f]*33},{**q,"claim":{**f,"value":True}},{**q,"facts":[{**f,"subject":""}]},{**q,"claim":{**f,"source":"x"}},{**q,"extra":1}):
            self.assertEqual(v.dispatch(bad)["status"],"HOLD_INVALID")
        self.assertEqual(v.dispatch("some prose",lambda s:("legacy",s)),("legacy","some prose"))
if __name__=="__main__":unittest.main()
