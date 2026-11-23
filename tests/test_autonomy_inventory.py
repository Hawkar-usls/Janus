"""Adversarial offline contracts for the public metadata collector."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import unittest
from urllib.error import HTTPError

from autonomy import inventory as inv

HEAD = "a" * 40
NEW_HEAD = "b" * 40
NOW = "2026-11-23T16:00:00Z"
ROOT = Path(__file__).resolve().parents[1]


def metadata(name="Janus", **kw):
    return {"full_name": inv.OWNER + "/" + name, "private": False,
            "owner": {"login": inv.OWNER}, "default_branch": "main",
            "archived": False, "fork": False, **kw}


def content(link=None):
    link = link or {"schema": "janus.organism.member_link.v1",
                    "member_repository": inv.OWNER + "/Janus", "organ_key": "gateway",
                    "authority_delta": 0, "member_write_authority": False,
                    "cross_repo_write_default": "DENY"}
    raw = json.dumps(link).encode()
    return {"encoding": "base64", "content": base64.b64encode(raw).decode(),
            "sha": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()}


class FakeGitHub:
    def __init__(self, rows=None, head=HEAD):
        self.rows = rows or [metadata()]
        self.head = head
        self.requests_used = 0
        self.routes = []
        self.override = {}
        self.link = content()
        self.actions = {"workflow_runs": [{"id": 42, "head_sha": head, "status": "completed",
                                          "conclusion": "success", "updated_at": NOW}]}
        self.checks = {"total_count": 1, "check_runs": [{"name": "safety", "head_sha": head,
                                                       "status": "completed", "conclusion": "success",
                                                       "completed_at": NOW}]}

    def get(self, route):
        self.requests_used += 1
        self.routes.append(route)
        if route in self.override:
            value = self.override[route]
            if isinstance(value, Exception):
                raise value
            return value
        if route.startswith("/users/"):
            return self.rows
        if "/git/ref/heads/" in route:
            return {"object": {"sha": self.head, "type": "commit"}}
        if "/contents/" in route:
            return self.link
        if "/actions/runs?" in route:
            return self.actions
        if "/check-runs?" in route:
            return self.checks
        raise AssertionError(route)


class Response:
    def __init__(self, url, raw=b"{}"):
        self.url, self.raw = url, raw
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def geturl(self): return self.url
    def read(self, limit): return self.raw[:limit]


class Opener:
    def __init__(self, raw=b"{}", error=None):
        self.raw, self.error, self.requests = raw, error, []
    def open(self, request, timeout):
        self.requests.append(request)
        if self.error:
            raise self.error
        return Response(request.full_url, self.raw)


class InventoryTests(unittest.TestCase):
    def snapshot(self, client=None, previous=None):
        return inv.build_snapshot(client or FakeGitHub(), source_commit=HEAD,
                                  observed_at=NOW, previous=previous)

    def test_public_scope_and_exact_pinned_blob_observation(self):
        client = FakeGitHub()
        report = self.snapshot(client)
        self.assertTrue(report["completeness"]["owner_inventory"])
        self.assertTrue(report["completeness"]["repository_observations"])
        row = report["repositories"][0]
        self.assertEqual(row["declared_organ"], "gateway")
        self.assertEqual(row["member_link"]["commit_sha"], HEAD)
        self.assertEqual(row["member_link"]["state"], "PRESENT")
        self.assertTrue(any("?ref=" + HEAD in route for route in client.routes))
        self.assertFalse(report["action_authority"])
        self.assertEqual(report["snapshot_sha256"], inv.digest({k: v for k, v in report.items() if k != "snapshot_sha256"}))

    def test_new_repository_is_catalog_only_despite_claimed_member_link(self):
        report = self.snapshot(FakeGitHub(rows=[metadata("brand-new")]))
        row = report["repositories"][0]
        self.assertEqual(row["discovery_state"], "CATALOG_ONLY")
        self.assertIsNone(row["declared_organ"])
        self.assertIsNone(row["declared_class"])
        self.assertNotIn("brand-new", report["declared_map"]["members"])

    def test_private_and_foreign_owner_fail_closed_without_echoing_name(self):
        for row in (metadata("do-not-publish", private=True),
                    metadata("do-not-publish", owner={"login": "attacker"})):
            report = self.snapshot(FakeGitHub(rows=[row]))
            self.assertFalse(report["completeness"]["owner_inventory"])
            self.assertEqual(report["repositories"], [])
            self.assertNotIn("do-not-publish", json.dumps(report))

    def test_canonical_delta_promotes_only_preexisting_reviewed_declaration(self):
        declared, sources = inv.load_declared_map(source_commit=HEAD)
        self.assertEqual(declared["classifications"][inv.OWNER + "/Simptomat"], "ORGAN")
        self.assertEqual(declared["members"]["diagnostic_reasoning"]["repository"], inv.OWNER + "/Simptomat")
        self.assertEqual(len(sources), 5)
        self.assertTrue(declared["refined_routes"])
        self.assertFalse(declared["runtime_integration_verified"])
        self.assertIsNone(declared["members"]["somatosensory_skin"]["repository"])
        self.assertNotIn("locator_env", json.dumps(declared))

    def test_pinned_blob_hash_mismatch_unknown(self):
        client = FakeGitHub()
        client.link["sha"] = NEW_HEAD
        result = self.snapshot(client)["repositories"][0]["member_link"]
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertEqual(result["reason"], "MEMBER_BLOB_HASH_MISMATCH")

    def test_head_must_be_a_commit_and_controls_are_removed(self):
        client = FakeGitHub()
        client.override[f"/repos/{inv.OWNER}/Janus/git/ref/heads/main"] = {"object": {"sha": HEAD, "type": "tree"}}
        row = self.snapshot(client)["repositories"][0]
        self.assertIsNone(row["head_sha"])
        self.assertEqual(row["head_error"], "HEAD_NOT_COMMIT")
        self.assertEqual(inv.safe_text("A\x1b\nB\x7f"), "AB")

    def test_missing_link_preserved_and_not_replaced_by_latest_branch_file(self):
        client = FakeGitHub()
        client.override[f"/repos/{inv.OWNER}/Janus/contents/{inv.MEMBER_PATH}?ref={HEAD}"] = inv.ObservationError("HTTP_404")
        row = self.snapshot(client)["repositories"][0]
        self.assertEqual(row["member_link"]["state"], "MISSING")
        self.assertEqual(row["member_link"]["commit_sha"], HEAD)

    def test_latest_failure_preserved_without_searching_older_success(self):
        client = FakeGitHub()
        client.actions["workflow_runs"][0]["conclusion"] = "failure"
        row = self.snapshot(client)["repositories"][0]
        self.assertEqual(row["actions"]["state"], "OBSERVED")
        self.assertEqual(row["actions"]["conclusion"], "failure")
        self.assertEqual(sum("/actions/runs?" in route for route in client.routes), 1)
        self.assertTrue(all("per_page=1" in route for route in client.routes if "/actions/runs?" in route))

    def test_wrong_head_and_stale_success_unknown(self):
        for change in ({"head_sha": NEW_HEAD}, {"updated_at": "2026-10-01T00:00:00Z"}):
            client = FakeGitHub()
            client.actions["workflow_runs"][0].update(change)
            row = self.snapshot(client)["repositories"][0]
            self.assertEqual(row["actions"]["state"], "UNKNOWN")
            self.assertFalse(self.snapshot(client)["completeness"]["repository_observations"])

    def test_check_scope_and_pagination_fail_closed(self):
        for change in ({"total_count": 101}, {"check_runs": []},
                       {"check_runs": [{"head_sha": NEW_HEAD}]}):
            client = FakeGitHub()
            client.checks.update(change)
            self.assertEqual(self.snapshot(client)["repositories"][0]["checks"]["state"], "UNKNOWN")

    def test_duplicate_inventory_rejected(self):
        report = self.snapshot(FakeGitHub(rows=[metadata(), metadata()]))
        self.assertFalse(report["completeness"]["owner_inventory"])
        self.assertIn("DUPLICATE_REPOSITORY", report["completeness"]["errors"])

    def test_drift_hash_binds_added_removed_and_changed(self):
        before = self.snapshot()
        client = FakeGitHub(rows=[metadata(), metadata("new-module")], head=NEW_HEAD)
        after = self.snapshot(client, previous=before)
        self.assertEqual(after["drift"]["added"], [inv.OWNER + "/new-module"])
        self.assertEqual(after["drift"]["head_changed"][0]["before"], HEAD)
        self.assertEqual(after["drift"]["baseline_sha256"], before["snapshot_sha256"])
        tampered = copy.deepcopy(before)
        tampered["repositories"] = []
        with self.assertRaisesRegex(ValueError, "BASELINE_HASH_OR_SCOPE_INVALID"):
            self.snapshot(previous=tampered)

    def test_partial_listing_does_not_report_false_mass_deletion(self):
        before = self.snapshot()
        client = FakeGitHub()
        client.override[f"/users/{inv.OWNER}/repos?per_page=100&type=owner&sort=full_name&direction=asc&page=1"] = inv.ObservationError("HTTP_403")
        after = self.snapshot(client, previous=before)
        self.assertFalse(after["completeness"]["owner_inventory"])
        self.assertEqual(after["drift"]["removed"], [])

    def test_endpoint_grammar_denies_arbitrary_host_path_and_query(self):
        denied = ("https://evil.test/x", "/user", "/repos/attacker/Janus/actions/runs?branch=main&per_page=1",
                  "/repos/Hawkar-usls/../actions/runs?branch=main&per_page=1",
                  "/repos/Hawkar-usls/Janus/contents/secret?ref=" + HEAD,
                  "/users/Hawkar-usls/repos?per_page=100&type=owner&sort=full_name&direction=asc&page=4")
        client = inv.PublicGitHub(opener=Opener())
        for route in denied:
            with self.subTest(route=route), self.assertRaisesRegex(inv.ObservationError, "ENDPOINT_DENIED"):
                client.get(route)
        self.assertEqual(client.requests_used, 0)

    def test_client_is_get_only_unauthenticated_by_default_and_bounded(self):
        opener = Opener()
        client = inv.PublicGitHub(opener=opener, max_requests=1)
        route = "/repos/Hawkar-usls/Janus/git/ref/heads/main"
        client.get(route)
        request = opener.requests[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(request.full_url, inv.API_PREFIX + route)
        with self.assertRaisesRegex(inv.ObservationError, "BUDGET_EXHAUSTED"):
            client.get(route)

    def test_redirect_and_response_overflow_denied(self):
        with self.assertRaisesRegex(inv.ObservationError, "REDIRECT_DENIED"):
            inv.NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.test")
        client = inv.PublicGitHub(opener=Opener(raw=b"x" * (inv.MAX_BYTES + 1)))
        with self.assertRaisesRegex(inv.ObservationError, "RESPONSE_TOO_LARGE"):
            client.get("/repos/Hawkar-usls/Janus/git/ref/heads/main")

    def test_rate_limit_stops_further_network_calls_and_suppresses_body(self):
        opener = Opener(error=HTTPError("https://api.github.com/secret", 403, "secret", {}, None))
        client = inv.PublicGitHub(opener=opener)
        route = "/repos/Hawkar-usls/Janus/git/ref/heads/main"
        with self.assertRaisesRegex(inv.ObservationError, "HTTP_403"):
            client.get(route)
        with self.assertRaisesRegex(inv.ObservationError, "BUDGET_EXHAUSTED"):
            client.get(route)
        self.assertEqual(len(opener.requests), 1)

    def test_deadline_and_invalid_json_fail_closed(self):
        times = iter([0, 301])
        client = inv.PublicGitHub(opener=Opener(), clock=lambda: next(times))
        with self.assertRaisesRegex(inv.ObservationError, "BUDGET_EXHAUSTED"):
            client.get("/repos/Hawkar-usls/Janus/git/ref/heads/main")
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'not json'):
            client = inv.PublicGitHub(opener=Opener(raw=raw))
            with self.assertRaises(inv.ObservationError):
                client.get("/repos/Hawkar-usls/Janus/git/ref/heads/main")


if __name__ == "__main__":
    unittest.main()
