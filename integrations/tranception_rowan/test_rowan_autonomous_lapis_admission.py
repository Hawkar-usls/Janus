"""No-network controls for the real Rowan/Lapis source admission boundary."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from rowan_autonomous_lapis_admission import validate_collector,validate_lapis,check,Hold,FROZEN,SOURCE_SHA,SOURCE_COMMIT

def source():
    return {
        "schema_id":"janus.rowan.bbc1.stratified_8_mutant_pilot.v1",
        "source_sha256":SOURCE_SHA,"source_commit":SOURCE_COMMIT,
        "pilot_cases":[{"quantile":q,"mutation":m,"provider_state":"PROVIDER_RESULT_RETURNED",
                        "ptm":0.90,"avg_lddt":0.96,"credits_charged":3}
                       for q,m in FROZEN],
        "wildtype":{"state":"PROVIDER_RESULT_RETURNED","ptm":0.91,"avg_lddt":0.97},
        "all_eight_ready_for_descriptive_association":True,
        "topa_scientific_review":"OPEN",
        "protein_stability_or_fitness_inferred":False
    }

def lapis(src):
    return {"candidate.json":{
        "facts":{"source_class":"ROWAN_BBC1_STRATIFIED","ready_count":8,
                 "samples":[{k:v for k,v in item.items() if k in ("mutation","ptm","avg_lddt")}
                            for item in src["pilot_cases"]]},
        "for_fundamentum":{"proof_or_exact_sat_witness":False,"P_VS_NP":"OPEN"}},
        "algorithm_ir.json":{"algorithm":{"mode":"explicit_contract","steps":[{"operator":"ANNOTATE"}]}},
        "manifest.json":{"schema":"test"}
    }

class AdmissionTests(unittest.TestCase):
    def test_frozen_eight_accepted(self):
        self.assertTrue(validate_collector(source()))

    def test_partial_collection_holds(self):
        s=source();s["pilot_cases"][1]["provider_state"]="PENDING_OR_FAILED"
        with self.assertRaisesRegex(Hold,"INCOMPLETE"):validate_collector(s)

    def test_over_cap_holds(self):
        s=source();s["pilot_cases"][0]["credits_charged"]=12
        with self.assertRaisesRegex(Hold,"CREDIT_CAP"):validate_collector(s)

    def test_mutation_swap_holds(self):
        s=source();s["pilot_cases"][0]["mutation"]="WRONG"
        with self.assertRaisesRegex(Hold,"MUTATION"):validate_collector(s)

    def test_scientific_authority_holds(self):
        s=source();s["protein_stability_or_fitness_inferred"]=True
        with self.assertRaisesRegex(Hold,"PROMOTION"):validate_collector(s)

    def test_unverified_corpus_holds(self):
        s=source();s["source_sha256"]="0"*64
        with self.assertRaisesRegex(Hold,"IDENTITY"):validate_collector(s)

    def test_lapis_projection_and_original_data(self):
        s=source()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for fn,data in lapis(s).items():(p/fn).write_text(json.dumps(data))
            self.assertTrue(validate_lapis(s,p)[0]["facts"]["ready_count"]==8)
            src=p/"source.json";src.write_text(json.dumps(s))
            r=check(src,p,40638479917)
            self.assertEqual(r["provider_source_run_id"],40638479917)
            self.assertEqual(r["rowan_provider_jobs_started"],0)

    def test_lapis_tamper_holds(self):
        s=source()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);v=lapis(s)
            v["candidate.json"]["facts"]["samples"][0]["ptm"]=0.1
            for fn,item in v.items():(p/fn).write_text(json.dumps(item))
            with self.assertRaisesRegex(Hold,"DATA_CHANGED"):validate_lapis(s,p)

    def test_lapis_proof_escalation_holds(self):
        s=source()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);v=lapis(s)
            v["candidate.json"]["for_fundamentum"]["proof_or_exact_sat_witness"]=True
            for fn,item in v.items():(p/fn).write_text(json.dumps(item))
            with self.assertRaisesRegex(Hold,"PROOF_ESCALATION"):validate_lapis(s,p)

if __name__=="__main__":
    unittest.main()
