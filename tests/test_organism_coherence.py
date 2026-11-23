"""JANUS organism topology and P0.2 source-cohort decision tests, offline."""
import base64
import copy
import json
from pathlib import Path
import unittest
from urllib.error import HTTPError

from organism import coherence as c

ROOT=Path(__file__).resolve().parents[1]


def matched():
    meta={"schema_id":"janus.rowan.observer_state.v1",
          "latest_durable_external_run":991,
          "last_completed_mutants":8,
          "underlying_state_is_derived_from":"a"*64}
    aura={"source":{"upstream_run":991,"completed_mutants":8,
                    "state_sha256":c.sha256_obj(meta),
                    "upstream_artifact_sha256":"a"*64}}
    council={"sources":{"meta_registry":{"canonical_run":991,
                            "sha256":c.sha256_obj(meta)},
                        "aura_oracle":{"packet_sha256":c.sha256_obj(aura)}}}
    return {"canonical":meta,"aura":aura,"council":council,
            "coordination":{"meta_canonical_run":991},
            "cosmos":{"canonical_rowan_run":991}}


class OrganismCoherenceTests(unittest.TestCase):
    def test_topology_uses_canonical_plus_reviewed_delta_only(self):
        t=c.topology(ROOT)
        self.assertEqual(t["canonical_slots"],22)
        self.assertEqual(t["delta_admissions"],1)
        self.assertEqual(len(t["public_admitted"]),22)
        self.assertEqual(t["private_slots"],1)
        self.assertEqual(t["public_admitted"]["Hawkar-usls/Simptomat"]["organ_key"],
                         "diagnostic_reasoning")
        self.assertNotIn("somatosensory_skin",t["public_admitted"])

    def test_canonical_link_good_does_not_grant_write_power(self):
        t=c.topology(ROOT)["public_admitted"]
        repo="Hawkar-usls/Janus"
        entry=t[repo]
        link={"schema":"janus.organism.member_link.v1",
              "organism_id":c.ID,"member_repository":repo,
              "organ_key":entry["organ_key"],"role":entry["role"],
              "authority":entry["authority"],
              "cross_repo_write_default":"DENY","authority_delta":0}
        self.assertEqual(c.evaluate_link(repo,link,entry),"ADMITTED_LINK_MATCH")
        link["authority_delta"]=1
        self.assertEqual(c.evaluate_link(repo,link,entry),"INVALID_LINK")
        link["authority_delta"]=0
        link["member_repository"]="Hawkar-usls/another"
        self.assertEqual(c.evaluate_link(repo,link,entry),"INVALID_LINK")

    def test_extra_declared_organ_is_not_automatically_admitted(self):
        repo="Hawkar-usls/some-public-auxiliary"
        link={"schema":"janus.organism.member_link.v1",
              "organism_id":c.ID,"member_repository":repo,
              "cross_repo_write_default":"DENY","authority_delta":0}
        self.assertEqual(c.evaluate_link(repo,link,None),"EXTRA_DECLARATION_NO_ADMISSION")
        self.assertEqual(c.evaluate_link(repo,None,None),"TOOL_OR_UNCLASSIFIED")

    def test_cohort_holds_on_missing_or_invalid_sources(self):
        self.assertEqual(c.next_cohort_action({})["status"],"HOLD_UPSTREAM_UNKNOWN")
        source=matched()
        source["canonical"]["schema_id"]="false.source"
        self.assertEqual(c.next_cohort_action(source)["status"],"HOLD_CANONICAL_INVALID")

    def test_cohort_requires_aura_first(self):
        source=matched()
        source["aura"]["source"]["upstream_run"]=991-1
        got=c.next_cohort_action(source)
        self.assertEqual(got["repair"]["organ"],"AURA")
        self.assertEqual(got["status"],"HOLD_SOURCE_COHORT_DRIFT")
        self.assertFalse(got["repair"]["automatically_dispatched"])
        self.assertFalse(got["full_autonomy_proven"])

    def test_cohort_requires_council_after_aura(self):
        source=matched()
        source["council"]["sources"]["meta_registry"]["canonical_run"]=990
        self.assertEqual(c.next_cohort_action(source)["repair"]["organ"],"COUNCIL")

    def test_cohort_requires_coordination_after_council(self):
        source=matched()
        source["coordination"]["meta_canonical_run"]=990
        self.assertEqual(c.next_cohort_action(source)["repair"]["organ"],"COORDINATION")

    def test_cohort_requires_cosmos_last(self):
        source=matched()
        source["cosmos"]["canonical_rowan_run"]=990
        self.assertEqual(c.next_cohort_action(source)["repair"]["organ"],"COSMOS")

    def test_matched_cohort_is_not_empirical_validation(self):
        source=matched()
        result=c.next_cohort_action(source)
        self.assertEqual(result["status"],
                         "MATCHED_COHORT_IDS_ONLY__INDEPENDENT_WITNESSES_REQUIRED")
        self.assertEqual(result["repair"]["organ"],"WITNESS")
        self.assertFalse(result["full_autonomy_proven"])

    def test_manifest_authority_escape_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            dst=Path(root)/"organism"
            dst.mkdir()
            name="JANUS_ORGANISM_v1.json"
            original=json.loads((ROOT/"organism"/name).read_text())
            original["design"]["cross_repo_write_default"]="ALLOW"
            (dst/name).write_text(json.dumps(original))
            other="JANUS_ORGANISM_DELTA_2026-08-26_SIMPTOMAT_v1.json"
            (dst/other).write_bytes((ROOT/"organism"/other).read_bytes())
            with self.assertRaisesRegex(ValueError,"AUTHORITY_ESCAPE"):
                c.topology(root)

    def test_pinned_content_must_match_git_blob_and_must_be_object(self):
        raw=b'{"x":1}'
        blob=c.git_sha1(raw)
        class Fake(c.PublicGithub):
            def get(self,path):
                return {"type":"file","encoding":"base64",
                        "content":base64.b64encode(raw).decode(),"sha":blob}
        api=Fake()
        self.assertEqual(api.source("Hawkar-usls/Janus","1"*40,"test.json"),{"x":1})
        with self.assertRaisesRegex(ValueError,"UNPINNED_READ"):
            api.source("Hawkar-usls/Janus","main","test.json")
        class Tamper(Fake):
            def get(self,path):
                o=super().get(path)
                o["sha"]="0"*40
                return o
        with self.assertRaisesRegex(ValueError,"BLOB_INTEGRITY"):
            Tamper().source("Hawkar-usls/Janus","1"*40,"test.json")

    def test_public_inventory_must_never_include_private_names(self):
        class Fake(c.PublicGithub):
            def get(self,path):
                if "page=1" in path:
                    return [{"full_name":"Hawkar-usls/Janus",
                             "default_branch":"main","private":False,
                             "visibility":"public"}]
                return []
        self.assertEqual(Fake().public_repos()[0]["repository"],"Hawkar-usls/Janus")
        class Secret(Fake):
            def get(self,path):
                if "page=1" in path:
                    return [{"full_name":"Hawkar-usls/secret-placeholder",
                             "default_branch":"main","private":True,
                             "visibility":"private"}]
                return []
        with self.assertRaisesRegex(ValueError,"INVALID_PUBLIC"):
            Secret().public_repos()

    def test_no_private_repo_is_generated_in_report(self):
        class FakeAPI:
            def public_repos(self):
                return [{"repository":"Hawkar-usls/Janus","default_branch":"main"}]
            def pin(self,repo,branch):
                return "f"*40
            def source(self,repo,sha,path):
                if path.endswith("JANUS_ORGANISM_LINK.json"):
                    v=c.topology(ROOT)["public_admitted"].get(repo)
                    if not v:return None
                    return {"schema":"janus.organism.member_link.v1",
                       "organism_id":c.ID,"member_repository":repo,
                       "organ_key":v["organ_key"],"role":v["role"],
                       "authority":v["authority"],
                       "cross_repo_write_default":"DENY","authority_delta":0}
                if path.endswith("essence/manifest.json"):
                    return {"schema":"janus.essence.microkernel/0.1","enabled":False}
                return None
        report=c.perform(ROOT,FakeAPI())
        self.assertEqual(report["summary"]["admitted_link_matches"],1)
        self.assertTrue(report["private_repository_names_disclosed"] is False)
        self.assertFalse(report["cross_repository_dispatch_performed"])
        self.assertFalse(report["full_autonomy_proven"])
        self.assertEqual(report["sha256"],c.sha256_obj({k:v for k,v in report.items() if k!="sha256"}))


if __name__=="__main__":
    unittest.main()
