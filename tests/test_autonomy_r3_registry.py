"""JANUS R3 Meta Registry Navigator — anchored JSON bootstrap, no authority."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import tempfile
import unittest

from autonomy.registry import navigator as n

MAIN=Path(os.environ.get("JANUS_R3_MAIN_ROOT","pinned-meta-main"))
MAP=Path(os.environ.get("JANUS_R3_MAP_ROOT","pinned-meta-map"))


def inputs():
    index=n.strict_json(n.safe_bytes(MAIN,n.INDEX,n.INDEX_BLOB))
    state=n.strict_json(n.safe_bytes(MAP,n.STATE,n.STATE_BLOB))
    contract=n.strict_json(n.safe_bytes(MAP,n.CONTRACT,n.CONTRACT_BLOB))
    health=n.strict_json(n.safe_bytes(MAP,n.HEALTH,n.HEALTH_BLOB))
    return index,state,contract,health


@unittest.skipUnless(
    os.environ.get("JANUS_R3_REQUIRE_PINNED_REGISTRY")=="1",
    "R3 two-repo pinned Meta Registry checkout not present in generic R1 offline suite")
class NavigatorTests(unittest.TestCase):
    def test_autonomy_bootstrap_finds_expected_organs_and_review_gates(self):
        x=n.boot(MAIN,MAP,n.MAIN_COMMIT,n.MAP_COMMIT,"JANUS")
        self.assertEqual(x["status"],
                         "BOOT_CONTEXT_READY_FROM_PINNED_DRAFT_JSON__AUTONOMY_ADMISSION_HOLD")
        self.assertEqual(x["historical_repositories_observed"],49)
        self.assertFalse(x["indexed_in_main_project_list"])
        self.assertEqual(len(x["observed_controllers"]),8)
        self.assertEqual(len(x["next_review_gates"]),7)
        self.assertEqual(len(x["operating_cycle"]),10)
        self.assertEqual(x["historical_health_observation_gate"],"HOLD")
        self.assertFalse(x["capability_limits"]["production_writeback"])
        self.assertFalse(x["capability_limits"]["autonomous_external_forks"])
        self.assertEqual(x["receipt_sha256"],n.sha256({k:v for k,v in x.items()
                                                     if k!="receipt_sha256"}))

    def test_index_understands_cosmos_and_fundamentum_without_fake_loaded_files(self):
        idx,state,contract,health=inputs()
        for q,p in [("COSMOS","COSMOS_SETI"),("радиопоиск","COSMOS_SETI"),
                    ("A3","TRUMP"),("Fundamentum","TRUMP")]:
            x=n.resolve(idx,state,contract,health,q)
            self.assertEqual(x["project_id"],p)
            self.assertFalse(x["trusted_project_current_state_attested"])
            self.assertFalse(x["production_writeback_allowed"])
            self.assertEqual(x["status"],"INDEX_PROJECT_RESOLVED__LIVE_STATE_NOT_FETCHED__HOLD")

    def test_unknown_does_not_guess_a_project_or_grant_authority(self):
        idx,state,contract,health=inputs()
        x=n.resolve(idx,state,contract,health,"undocumented legacy object")
        self.assertEqual(x["status"],"PROJECT_NOT_INDEXED__DO_NOT_GUESS_OR_EXECUTE")
        self.assertIsNone(x["project_id"])
        self.assertFalse(x["production_writeback_allowed"])

    def test_main_and_draft_commit_pin_must_match(self):
        for m,d in (("0"*40,n.MAP_COMMIT),(n.MAIN_COMMIT,"0"*40)):
            with self.subTest(m=m,d=d),self.assertRaisesRegex(n.RegistryHold,"UNPINNED_CHECKOUT"):
                n.boot(MAIN,MAP,m,d,"JANUS")

    def test_tampered_source_bytes_are_rejected_before_json(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/n.INDEX
            p.parent.mkdir(parents=True)
            p.write_bytes(n.safe_bytes(MAIN,n.INDEX,n.INDEX_BLOB)+b" ")
            with self.assertRaisesRegex(n.RegistryHold,"SOURCE_BLOB_MISMATCH"):
                n.safe_bytes(td,n.INDEX,n.INDEX_BLOB)

    def test_path_escape_and_repo_and_pointer_injection_rejected(self):
        for bad_path in ("../../private", "/tmp/secret", "data/../secret", "README.md"):
            with self.assertRaisesRegex(n.RegistryHold,"UNREVIEWED_PATH"):
                n.safe_bytes(MAIN,bad_path,n.INDEX_BLOB)
        index,_,_,_=inputs()
        for path in ("../../secrets", ".github/workflows/run.yml", "data/../../../etc"):
            bad=copy.deepcopy(index)
            bad["projects"][0]["state"]["path"]=path
            with self.assertRaisesRegex(n.RegistryHold,"UNTRUSTED_PROJECT_ROUTE"):
                n.validate_index(bad)

    def test_fake_approval_cannot_be_imported_from_map_json(self):
        idx,state,contract,health=inputs()
        for key,value in (("action_authority",True),("weights_changed",True),
                          ("actual_forks_created",1)):
            x=copy.deepcopy(state)
            x["boundaries"][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(n.RegistryHold,"MAP_HEALTH_AUTHORITY"):
                n.resolve(idx,x,contract,health,"Janus")

    def test_fake_index_alias_authority_and_duplicate_key(self):
        idx,_,_,_=inputs()
        forged=copy.deepcopy(idx)
        forged["projects"][0]["aliases"].append("TRUMP")
        with self.assertRaisesRegex(n.RegistryHold,"AMBIGUOUS_ALIAS"):
            n.validate_index(forged)
        forged=copy.deepcopy(idx)
        forged["projects"][0]["state"]["repo"]="outsider/other"
        with self.assertRaisesRegex(n.RegistryHold,"UNTRUSTED_PROJECT_ROUTE"):
            n.validate_index(forged)
        with self.assertRaisesRegex(n.RegistryHold,"DUPLICATE_JSON_KEY"):
            n.strict_json(b'{"schema":"a","schema":"b"}')

    def test_index_cannot_override_operating_contract(self):
        idx,state,contract,health=inputs()
        bad=copy.deepcopy(contract)
        bad["effect_gate"]["current_r1_effect_authority"]=True
        with self.assertRaisesRegex(n.RegistryHold,"OPERATING_CONTRACT"):
            n.resolve(idx,state,bad,health,"autonomy")
        bad=copy.deepcopy(health)
        bad["observation_gate"]="PASS"
        with self.assertRaises(n.RegistryHold):
            n.resolve(idx,state,contract,bad,"autonomy")

    def test_traversal_symlink_and_unknown_named_source_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/n.STATE
            p.parent.mkdir(parents=True)
            p.symlink_to(MAP.resolve()/n.STATE)
            with self.assertRaisesRegex(n.RegistryHold,"UNSAFE_OR_MISSING_SOURCE"):
                n.safe_bytes(td,n.STATE,n.STATE_BLOB)
        with self.assertRaisesRegex(n.RegistryHold,"UNREVIEWED_PATH"):
            n.safe_bytes(MAP,"data/unknown.json",n.STATE_BLOB)

    def test_nonfinite_numbers_and_query_budget_rejected(self):
        with self.assertRaises(n.RegistryHold):
            n.strict_json(b'{"x":NaN}')
        idx,state,contract,health=inputs()
        with self.assertRaisesRegex(n.RegistryHold,"QUERY_BUDGET"):
            n.resolve(idx,state,contract,health,"x"*81)

if __name__=="__main__":
    unittest.main()
