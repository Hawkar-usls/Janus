"""Verifier controls use independently constructed metadata; no exams/models."""
import copy
import hashlib
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autonomy.health import SOURCE_PATHS, digest, verify

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 23, 18, tzinfo=timezone.utc)


def fixture():
    base = json.loads((ROOT / "organism/JANUS_ORGANISM_v1.json").read_text(encoding="utf-8"))
    delta = json.loads((ROOT / "organism/JANUS_ORGANISM_DELTA_2026-08-26_SIMPTOMAT_v1.json").read_text(encoding="utf-8"))
    members = {key: {"repository": value["repo"], "class": value["class"], "authority": value["authority"]} for key, value in base["members"].items()}
    member = delta["admit_member"]
    members[member["organ_key"]] = {"repository": member["repo"], "class": member["class"], "authority": member["authority"]}
    ecology = json.loads((ROOT / "organism/JANUS_REPOSITORY_ECOLOGY_v1.3.json").read_text(encoding="utf-8"))
    classifications = {value["repo"]: value["class"] for group in ("public_subtissues", "legacy_archives", "toolchain_dependencies", "external_research_instruments") for value in ecology[group].values()}
    classifications[ecology["profile_navigation"]["repo"]] = ecology["profile_navigation"]["class"]
    classifications.update({value["repository"]: "ORGAN" for value in members.values() if value["repository"]})
    private_slots = [{"slot": key, "class": value["class"]} for key, value in members.items() if value["repository"] is None]
    private_slots += [{"slot": key, "class": value["class"]} for key, value in ecology["private_subtissues"].items()]
    rows = [{"repository": value["repository"], "visibility": "public", "discovery_state": "DECLARED", "declared_class": "ORGAN",
             "declared_organ": key, "head_sha": "a" * 40,
             "member_link": {"state": "PRESENT", "repository": value["repository"], "commit_sha": "a" * 40,
                             "organ_key": key, "schema": "janus.organism.member_link.v1", "authority_delta": 0,
                             "member_write_authority": False, "cross_repo_write_default": "DENY"},
             "checks": {"state": "OBSERVED", "head_sha": "a" * 40, "runs": [
                 {"name": "contract", "status": "completed", "conclusion": "success", "completed_at": NOW.isoformat()}]},
             "actions": {"state": "OBSERVED", "head_sha": "a" * 40, "status": "completed",
                         "conclusion": "success", "updated_at": NOW.isoformat()}}
            for key, value in members.items() if value["repository"]]
    report = {"schema": "janus.autonomy.inventory.v1", "owner": "Hawkar-usls", "observed_at": NOW.isoformat(),
              "action_authority": False, "declared_map": {"members": members, "flows": base["flows"] + delta["add_flows"],
                  "runtime_integration_verified": False,
                  "classifications": classifications, "private_slots": private_slots,
                  "refined_routes": json.loads((ROOT / "organism/JANUS_ORGANISM_ROUTE_REFINEMENT_2026-08-26_SIMPTOMAT_v1.json").read_text(encoding="utf-8"))["effective_routes"]},
              "sources": [{"path": path, "repository": "Hawkar-usls/Janus", "commit_sha": "b" * 40,
                           "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()} for path in sorted(SOURCE_PATHS)],
              "repositories": rows, "completeness": {"owner_inventory": True, "repository_observations": True,
                                                       "errors": [], "repository_count": len(rows)}}
    return seal(report)


def seal(report):
    report["snapshot_sha256"] = digest({k: v for k, v in report.items() if k != "snapshot_sha256"})
    return report


class HealthTests(unittest.TestCase):
    def check(self, report):
        return verify(report, trusted_root=ROOT, expected_source_commit="b" * 40, now=NOW)

    def test_metadata_pass_never_grants_runtime_or_admission(self):
        result = self.check(fixture())
        self.assertEqual(result["integrity_gate"], "PASS")
        self.assertEqual(result["observation_gate"], "PASS")
        self.assertFalse(result["execution_allowed"])
        self.assertFalse(result["full_autonomy_proven"])
        self.assertTrue(result["independent_admission_gate"].startswith("HOLD"))

    def test_digest_and_authority_tamper(self):
        for key, value in (("action_authority", True), ("owner", "attacker")):
            r = fixture(); r[key] = value
            self.assertEqual(self.check(r)["integrity_gate"], "HOLD")

    def test_stale_or_future_snapshot_cannot_pass(self):
        for offset in (-25, 1):
            r = fixture(); r["observed_at"] = (NOW + timedelta(hours=offset)).isoformat()
            self.assertEqual(self.check(seal(r))["integrity_gate"], "HOLD")

    def test_old_success_does_not_cover_current_pending_or_new_head(self):
        for changes in ({"status": "in_progress", "conclusion": None}, {"head_sha": "c" * 40},
                        {"updated_at": (NOW - timedelta(days=3)).isoformat()}):
            r = fixture(); r["repositories"][0]["actions"].update(changes)
            self.assertEqual(self.check(seal(r))["observation_gate"], "HOLD")

    def test_one_green_workflow_cannot_hide_failed_check_or_escalated_member_link(self):
        r = fixture(); r["repositories"][0]["checks"]["runs"][0]["conclusion"] = "failure"
        self.assertEqual(self.check(seal(r))["observation_gate"], "HOLD")
        r = fixture(); r["repositories"][0]["member_link"]["member_write_authority"] = True
        self.assertEqual(self.check(seal(r))["observation_gate"], "HOLD")
        r = fixture(); r["declared_map"]["refined_routes"] = []
        self.assertEqual(self.check(seal(r))["integrity_gate"], "HOLD")

    def test_partial_and_type_confused_completeness(self):
        for value in (False, 1, "true"):
            r = fixture(); r["completeness"]["owner_inventory"] = value
            self.assertEqual(self.check(seal(r))["observation_gate"], "HOLD")

    def test_source_traversal_and_wrong_content(self):
        for field, value in (("path", "../../secret"), ("sha256", "0" * 64)):
            r = fixture(); r["sources"][0][field] = value
            self.assertEqual(self.check(seal(r))["integrity_gate"], "HOLD")

    def test_graph_authority_or_unknown_endpoint(self):
        for field, value in (("authority_delta", True), ("to", ["unknown"])):
            r = fixture(); r["declared_map"]["flows"][0][field] = value
            self.assertEqual(self.check(seal(r))["integrity_gate"], "HOLD")

    def test_membership_cannot_be_rewritten_by_collector(self):
        r = fixture(); r["declared_map"]["members"]["gateway"]["repository"] = "Hawkar-usls/another"
        self.assertEqual(self.check(seal(r))["integrity_gate"], "HOLD")

    def test_unknown_discovery_has_no_organ_rights(self):
        r = fixture(); r["repositories"][0]["discovery_state"] = "CATALOG_ONLY"
        self.assertEqual(self.check(seal(r))["integrity_gate"], "HOLD")
        r = fixture(); r["declared_map"]["members"]["gateway"]["authority"] = "ADMIN"
        self.assertEqual(self.check(seal(r))["integrity_gate"], "HOLD")
        r = fixture(); r["declared_map"]["classifications"]["Hawkar-usls/player"] = "ORGAN"
        self.assertEqual(self.check(seal(r))["integrity_gate"], "HOLD")

    def test_missing_declared_repo_remains_hold(self):
        r = fixture(); r["repositories"].pop(); r["completeness"]["repository_count"] -= 1
        self.assertEqual(self.check(seal(r))["observation_gate"], "HOLD")

    def test_duplicate_and_private_rows_rejected(self):
        r = fixture(); r["repositories"].append(copy.deepcopy(r["repositories"][0]))
        self.assertEqual(self.check(seal(r))["integrity_gate"], "HOLD")

    def test_forged_collector_cannot_remove_graph_or_skip_member_health(self):
        for mutate in (lambda r: r["declared_map"].update(flows=[]),
                       lambda r: r["repositories"][0].update(declared_organ=False),
                       lambda r: r["sources"][0].update(commit_sha="0" * 40)):
            r = fixture(); mutate(r)
            self.assertEqual(self.check(seal(r))["integrity_gate"], "HOLD")
        r = fixture(); r["repositories"][0]["visibility"] = "private"
        self.assertEqual(self.check(seal(r))["integrity_gate"], "HOLD")

    def test_malformed_types_produce_hold_receipt(self):
        for mutate in (lambda r: r["sources"][0].update(path=[]),
                       lambda r: r.update(declared_map=True),
                       lambda r: r["declared_map"].update(flows=None),
                       lambda r: r["declared_map"]["flows"][0].update({"from": []})):
            r = fixture(); mutate(r)
            self.assertEqual(self.check(seal(r))["integrity_gate"], "HOLD")


if __name__ == "__main__":
    unittest.main()
