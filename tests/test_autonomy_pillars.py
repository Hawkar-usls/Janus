"""Pillars R2: 13 pinned references, 16 gaps and non-authoritative organ evolution."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from autonomy.pillars import engine, verifier, upstream_watch

ROOT=Path(__file__).resolve().parents[1]
CAT=engine.read_json(ROOT/"autonomy/pillars/CATALOG.json")
NEEDS=engine.read_json(ROOT/"autonomy/pillars/NEEDS.json")
POLICY=engine.read_json(ROOT/"autonomy/policy.json")


class PillarContracts(unittest.TestCase):
    def test_all_thirteen_sources_pinned_without_execution(self):
        pins=engine.validate_catalog(CAT)
        self.assertEqual(len(pins),13)
        self.assertEqual(len({p["repo"] for p in pins.values()}),13)
        self.assertEqual(sum(p["import_policy"].endswith("LICENSE_OR_POLICY")
             for p in pins.values()),4)
        self.assertEqual(sum(p["license_metadata_spdx"] in engine.ALLOWED
             for p in pins.values()),9)
        self.assertTrue(all(not p["license_bytes_verified"] and
             not p["action_authority"] for p in pins.values()))

    def test_exact_sixteen_gaps_link_to_valid_sources(self):
        n=engine.validate_needs(NEEDS,engine.validate_catalog(CAT))
        self.assertEqual(len(n),16)
        self.assertEqual(NEEDS["source_evidence"]["observed_health"],"HOLD")
        self.assertEqual(NEEDS["source_evidence"]["autonomy_phase"],"P0.3")

    def test_complete_plan_is_stable_limited_and_inert(self):
        a=engine.plan(CAT,NEEDS,POLICY)
        b=engine.plan(CAT,NEEDS,POLICY)
        self.assertEqual(a,b)
        self.assertEqual(a["status"],"OFFLINE_PROPOSALS_GENERATED__ACTIVATION_AND_AUTONOMY_HOLD")
        self.assertEqual(len(a["selected_blueprints"]),4)
        self.assertEqual(len(a["gene_capsule_events"]),4)
        self.assertEqual(a["fixed_ranked_gap_count"],16)
        self.assertEqual(a["not_selected_count"],12)
        self.assertEqual(a["receipt_sha256"],engine.sha({k:v for k,v in a.items() if k!="receipt_sha256"}))
        for organ in a["selected_blueprints"]:
            self.assertEqual(organ["kind"],"MODULE_PROPOSAL_ONLY")
            self.assertTrue(organ["module_path"].startswith("modules/candidates/"))
            self.assertFalse(organ["execution_allowed"])
            self.assertFalse(organ["fork_allowed"])
            self.assertFalse(organ["production_promotion"])
            self.assertFalse(organ["review_is_authenticated"])
            self.assertEqual(organ["state"],"CATALOGUED_NOT_BORN")
        v=verifier.verify(a,CAT,NEEDS,POLICY)
        self.assertEqual(v["proposals_checked"],4)
        self.assertFalse(v["may_activate_organs"])
        self.assertFalse(v["external_witness_attested"])

    def test_hard_kill_switch_is_respected(self):
        p=copy.deepcopy(POLICY)
        p["kill_switch"]=True
        with self.assertRaisesRegex(ValueError,"PILLARS_EFFECT_BOUNDARY"):
            engine.plan(CAT,NEEDS,p)

    def test_no_repository_creation_or_extra_proposals(self):
        for field,value in (("max_new_repositories",1),("max_proposals",5)):
            p=copy.deepcopy(POLICY)
            p["limits"][field]=value
            with self.assertRaises(ValueError):
                engine.plan(CAT,NEEDS,p)

    def test_license_claim_or_source_rebinding_fails_closed(self):
        for key,val in (("commit_sha","f"*40),("license_bytes_verified",True),
                        ("action_authority",True),("import_policy","COPY_CODE_NOW")):
            c=copy.deepcopy(CAT)
            c["pillars"][0][key]=val
            if key=="commit_sha":
                # A valid different SHA is still merely a new version; it is
                # flagged by the watcher, not accepted as established evidence.
                self.assertNotEqual(engine.plan(c,NEEDS,POLICY)["catalog_digest"],
                                    engine.plan(CAT,NEEDS,POLICY)["catalog_digest"])
            else:
                with self.assertRaises(ValueError):
                    engine.plan(c,NEEDS,POLICY)

    def test_untrusted_gap_inflation_cannot_override_budgets(self):
        for key,value in (("impact",True),("risk",-1),("urgency",100),
                          ("requires_human_activation",False)):
            n=copy.deepcopy(NEEDS)
            n["needs"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                engine.plan(CAT,n,POLICY)

    def test_genome_and_plan_rebinding_rejected_separately(self):
        p=engine.plan(CAT,NEEDS,POLICY)
        for section,key,value in (("proposal","execution_allowed",True),
                                  ("proposal","source_map_git_commit","a"*40),
                                  ("event","head","0"*64)):
            x=copy.deepcopy(p)
            if section=="proposal":x["selected_blueprints"][0][key]=value
            else:x["gene_capsule_events"][0][key]=value
            x["receipt_sha256"]=verifier.seal({k:v for k,v in x.items() if k!="receipt_sha256"})
            with self.subTest(section=section,key=key),self.assertRaises(ValueError):
                verifier.verify(x,CAT,NEEDS,POLICY)

    def test_held_health_can_be_classified_without_becoming_pass(self):
        h={"schema":"janus.autonomy.health.v1",
           "snapshot_sha256":"c"*64,"verified_at":"2026-11-23T00:00:00Z",
           "integrity_gate":"PASS","observation_gate":"HOLD",
           "integrity_reasons":[],
           "observation_reasons":["Hawkar-usls/AIFC:CHECKS_STALE",
                                  "Hawkar-usls/Hrain:MEMBER_LINK_UNVERIFIED",
                                  "Hawkar-usls/x:UNKNOWN_ERROR"],
           "operational_gate":"NOT_ASSESSED_REQUIRES_SERVICE_SPECIFIC_PROBES",
           "independent_admission_gate":"HOLD_EXTERNAL_REVIEW_REQUIRED",
           "execution_allowed":False,"full_autonomy_proven":False,"authority_delta":0}
        h["health_sha256"]=hashlib.sha256(json.dumps(h,sort_keys=True,
            separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        out=engine.health_triage(h)
        self.assertEqual(out["status"],"HEALTH_HOLD_TRIAGED_NOT_REPAIRED")
        self.assertEqual(out["reason_counts"]["CHECKS_STALE"],1)
        self.assertEqual(out["unclassified_reasons"],1)
        self.assertFalse(out["admission_allowed"])
        h["observation_reasons"].append("new")
        with self.assertRaisesRegex(ValueError,"HEALTH_DIGEST_MISMATCH"):
            engine.health_triage(h)


class WatcherContracts(unittest.TestCase):
    def test_pinned_live_metadata_comparison_is_never_import_permission(self):
        p=CAT["pillars"][0]
        info={"full_name":p["repo"],"default_branch":p["default_branch"],
              "private":False,"archived":False,"disabled":False,
              "license":{"spdx_id":p["license_metadata_spdx"]}}
        branch={"commit":{"sha":p["commit_sha"]}}
        result=upstream_watch.compare(p,info,branch)
        self.assertTrue(result["status"].startswith("PIN_MATCH"))
        self.assertFalse(result["license_bytes_verified"])
        self.assertFalse(result["auto_fork"])
        branch["commit"]["sha"]="a"*40
        self.assertEqual(upstream_watch.compare(p,info,branch)["status"],
                         "HOLD_METADATA_DRIFT_OR_UNVERIFIED")

    def test_watcher_is_fixed_to_exact_public_sources(self):
        calls=[]
        by_repo={p["repo"]:p for p in CAT["pillars"]}
        def fake(path,token):
            calls.append(path)
            upstream=next((p for name,p in by_repo.items()
                         if path.startswith("/repos/"+name) and
                         (len(path)==len("/repos/"+name) or path[len("/repos/"+name)]=="/")),None)
            if upstream is None:raise ValueError("unexpected source")
            if "/branches/" in path:
                return {"commit":{"sha":upstream["commit_sha"]}}
            return {"full_name":upstream["repo"],"default_branch":upstream["default_branch"],
                    "private":False,"disabled":False,"archived":False,
                    "license":{"spdx_id":upstream["license_metadata_spdx"]}}
        result=upstream_watch.watch(CAT,fetch=fake)
        self.assertEqual(result["pin_matches"],13)
        self.assertEqual(result["requests_used"],26)
        self.assertEqual(len(calls),26)
        self.assertFalse(result["auto_promoted"])
        self.assertEqual(result["forks_created"],0)

    def test_source_unavailable_cannot_be_mislabeled_health_pass(self):
        result=upstream_watch.watch(CAT,fetch=lambda p,t: (_ for _ in ()).throw(OSError("offline")))
        self.assertEqual(result["pin_matches"],0)
        self.assertEqual(result["held_for_review"],13)
        self.assertTrue(all(x["status"]=="HOLD_PUBLIC_SOURCE_UNAVAILABLE"
                            for x in result["results"]))
        self.assertEqual(result["requests_used"],0)

    def test_no_arbitrary_urls_or_redirects(self):
        for p in ("https://evil.example/x","/users/current",
                  "/repos/Hawkar-usls/SkinGPT",
                  "/repos/github/gh-aw/releases/latest",
                  "/repos/evil/repo/branches/../../admin"):
            with self.subTest(path=p),self.assertRaises(ValueError):
                upstream_watch.github_get(p)

if __name__=="__main__":
    unittest.main()
