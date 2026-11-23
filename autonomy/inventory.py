"""Observe public JANUS repository metadata without running repository code.

The collector is deliberately separate from the health evaluator. A complete
inventory is an observation, never admission of an organ or runtime authority.
Only five fixed GitHub GET endpoint shapes are reachable; redirects, arbitrary
hosts, external code downloads, shell commands and GitHub mutations are absent.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

OWNER = "Hawkar-usls"
SCHEMA = "janus.autonomy.inventory.v1"
ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"[0-9a-f]{40}\Z")
REPO = re.compile(r"Hawkar-usls/[A-Za-z0-9_.-]{1,100}\Z")
SOURCE_PATHS = (
    "organism/JANUS_ORGANISM_v1.json",
    "organism/JANUS_REPOSITORY_ECOLOGY_v1.3.json",
    "organism/JANUS_REPOSITORY_ECOLOGY_v1.4.json",
    "organism/JANUS_ORGANISM_DELTA_2026-08-26_SIMPTOMAT_v1.json",
    "organism/JANUS_ORGANISM_ROUTE_REFINEMENT_2026-08-26_SIMPTOMAT_v1.json",
)
MEMBER_PATH = ".janus/JANUS_ORGANISM_LINK.json"
MAX_REPOS = 100
MAX_PAGES = 3
MAX_BYTES = 2_000_000
MAX_AGE_SECONDS = 7 * 86400
API_PREFIX = "https://api.github.com"


class ObservationError(ValueError):
    """A bounded observation is unavailable; callers must retain UNKNOWN."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ObservationError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def nonfinite(_value):
        raise ObservationError("NONFINITE_JSON")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value):
    if not isinstance(value, str):
        raise ObservationError("MISSING_TIMESTAMP")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ObservationError("NAIVE_TIMESTAMP")
    return result.astimezone(timezone.utc)


def safe_text(value, limit=160):
    # Public metadata is still untrusted: do not preserve terminal controls.
    return "".join(char for char in value[:limit] if char >= " " and char != "\x7f") if isinstance(value, str) else None


def unknown(reason, **evidence):
    return {"state": "UNKNOWN", "reason": reason, **evidence}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ObservationError("REDIRECT_DENIED")


def allowed_route(route):
    """Full URLs, credentials and unconstrained paths are never accepted."""
    name = r"[A-Za-z0-9_.-]{1,100}"
    ref = r"[A-Za-z0-9%_.~-]{1,500}"
    repo = rf"/repos/{OWNER}/{name}"
    patterns = (
        rf"/users/{OWNER}/repos\?per_page=100&type=owner&sort=full_name&direction=asc&page=[1-3]",
        rf"{repo}/git/ref/heads/{ref}",
        rf"{repo}/contents/\.janus/JANUS_ORGANISM_LINK\.json\?ref=[0-9a-f]{{40}}",
        rf"{repo}/actions/runs\?branch={ref}&per_page=1",
        rf"{repo}/commits/[0-9a-f]{{40}}/check-runs\?per_page=100",
    )
    return "/../" not in route and "/./" not in route and any(re.fullmatch(pattern, route) for pattern in patterns)


class PublicGitHub:
    """Fixed-host GET client with request, wall time and response byte budgets."""

    def __init__(self, *, token=None, max_requests=300, deadline_seconds=240,
                 opener=None, clock=time.monotonic):
        if type(max_requests) is not int or not 1 <= max_requests <= 500:
            raise ValueError("REQUEST_BUDGET_OUT_OF_RANGE")
        if not 1 <= deadline_seconds <= 300:
            raise ValueError("DEADLINE_OUT_OF_RANGE")
        self.max_requests = max_requests
        self.requests_used = 0
        self.clock = clock
        self.deadline = clock() + deadline_seconds
        self.opener = opener or build_opener(NoRedirect())
        self.token = token
        self.exhausted = False

    def get(self, route):
        if not isinstance(route, str) or not allowed_route(route):
            raise ObservationError("ENDPOINT_DENIED")
        remaining = self.deadline - self.clock()
        if self.exhausted or self.requests_used >= self.max_requests or remaining <= 0:
            raise ObservationError("OBSERVATION_BUDGET_EXHAUSTED")
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "JANUS-Inventory/1",
                   "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        self.requests_used += 1
        req = Request(API_PREFIX + route, headers=headers, method="GET")
        try:
            with self.opener.open(req, timeout=min(12, remaining)) as response:
                if response.geturl() != req.full_url:
                    raise ObservationError("RESPONSE_LOCATION_CHANGED")
                data = response.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise ObservationError("RESPONSE_TOO_LARGE")
                return strict_json(data.decode("utf-8"))
        except HTTPError as error:
            # Do not echo server bodies, URLs or headers into public receipts.
            if error.code in (401, 403, 429):
                self.exhausted = True
            raise ObservationError("HTTP_" + str(error.code)) from None
        except (URLError, TimeoutError, OSError):
            raise ObservationError("NETWORK_UNAVAILABLE") from None
        except (UnicodeError, json.JSONDecodeError):
            raise ObservationError("INVALID_JSON") from None


def load_declared_map(*, root=ROOT, source_commit="UNKNOWN"):
    if source_commit != "UNKNOWN" and not SHA.fullmatch(source_commit):
        raise ValueError("INVALID_SOURCE_COMMIT")
    sources, documents = [], []
    for path in SOURCE_PATHS:
        raw = (root / path).read_bytes()
        if len(raw) > MAX_BYTES:
            raise ValueError("CANONICAL_MAP_TOO_LARGE")
        documents.append(strict_json(raw.decode("utf-8")))
        sources.append({"repository": OWNER + "/Janus", "path": path,
                        "commit_sha": source_commit, "sha256": hashlib.sha256(raw).hexdigest()})
    topology, ecology, update, delta, refinement = documents
    if (topology.get("schema") != "janus.organism.v1" or
            ecology.get("version") != "1.3.0" or update.get("version") != "1.4.0" or
            update.get("supersedes") != SOURCE_PATHS[1] or
            update.get("admission_delta") != SOURCE_PATHS[3] or
            delta.get("status") != "ACTIVE_ADMISSION_DELTA"):
        raise ValueError("CANONICAL_LINEAGE_MISMATCH")
    members = dict(topology["members"])
    admitted = delta["admit_member"]
    members[admitted["organ_key"]] = admitted
    public_members = {}
    private_slots = []
    for key, member in members.items():
        repository = member.get("repo")
        if repository is not None and not REPO.fullmatch(repository):
            raise ValueError("INVALID_CANONICAL_REPOSITORY")
        public_members[key] = {"repository": repository, "class": member["class"],
                               "authority": member["authority"]}
        if repository is None:
            private_slots.append({"slot": key, "class": member["class"]})
    classifications = {}
    for category in ("public_subtissues", "legacy_archives", "toolchain_dependencies",
                     "external_research_instruments"):
        for row in ecology.get(category, {}).values():
            repository = row.get("repo")
            if repository and REPO.fullmatch(repository):
                classifications[repository] = row["class"]
    profile = ecology.get("profile_navigation", {})
    if profile.get("repo"):
        classifications[profile["repo"]] = profile["class"]
    for key, row in ecology.get("private_subtissues", {}).items():
        private_slots.append({"slot": key, "class": row["class"]})
    for member in public_members.values():
        if member["repository"]:
            classifications[member["repository"]] = "ORGAN"
    flows = topology["flows"] + delta.get("add_flows", [])
    for flow in flows:
        if flow.get("authority_delta") != 0:
            raise ValueError("FLOW_AUTHORITY_ESCALATION")
    return {"members": public_members, "classifications": classifications, "flows": flows,
            "refined_routes": refinement.get("effective_routes", []),
            "private_slots": private_slots, "runtime_integration_verified": False}, sources


def list_public_repositories(client):
    result, seen = [], set()
    for page in range(1, MAX_PAGES + 1):
        rows = client.get(f"/users/{OWNER}/repos?per_page=100&type=owner&sort=full_name&direction=asc&page={page}")
        if not isinstance(rows, list):
            raise ObservationError("INVALID_OWNER_INVENTORY")
        for row in rows:
            full_name = row.get("full_name") if isinstance(row, dict) else None
            if (not isinstance(full_name, str) or not REPO.fullmatch(full_name)
                    or row.get("private") is not False
                    or row.get("owner", {}).get("login") != OWNER):
                raise ObservationError("NONPUBLIC_OR_FOREIGN_REPOSITORY")
            if full_name in seen:
                raise ObservationError("DUPLICATE_REPOSITORY")
            seen.add(full_name)
            result.append(row)
            if len(result) > MAX_REPOS:
                raise ObservationError("REPOSITORY_BUDGET_EXCEEDED")
        if len(rows) < 100:
            return result
    raise ObservationError("PAGINATION_INCOMPLETE")


def member_link(client, repository, head_sha):
    try:
        data = client.get(f"/repos/{repository}/contents/{MEMBER_PATH}?ref={head_sha}")
        if not isinstance(data, dict) or data.get("encoding") != "base64":
            raise ObservationError("INVALID_MEMBER_CONTENT")
        raw = base64.b64decode(data.get("content", "").replace("\n", ""), validate=True)
        if len(raw) > 128_000:
            raise ObservationError("MEMBER_LINK_TOO_LARGE")
        blob_sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if data.get("sha") != blob_sha:
            raise ObservationError("MEMBER_BLOB_HASH_MISMATCH")
        link = strict_json(raw.decode("utf-8"))
        if not isinstance(link, dict):
            raise ObservationError("INVALID_MEMBER_JSON")
        return {"state": "PRESENT", "commit_sha": head_sha, "blob_sha": blob_sha,
                "sha256": hashlib.sha256(raw).hexdigest(), "repository": safe_text(link.get("member_repository")),
                "organ_key": safe_text(link.get("organ_key")), "schema": safe_text(link.get("schema")),
                "authority_delta": link.get("authority_delta") if type(link.get("authority_delta")) is int else None,
                "member_write_authority": link.get("member_write_authority") if type(link.get("member_write_authority")) is bool else None,
                "cross_repo_write_default": safe_text(link.get("cross_repo_write_default"))}
    except ObservationError as error:
        if str(error) == "HTTP_404":
            return {"state": "MISSING", "commit_sha": head_sha, "reason": "HTTP_404"}
        return unknown(str(error), commit_sha=head_sha)
    except (ValueError, TypeError, UnicodeError):
        return unknown("INVALID_MEMBER_JSON", commit_sha=head_sha)


def observe_actions(client, repository, branch, head_sha, observed_at):
    try:
        data = client.get(f"/repos/{repository}/actions/runs?branch={quote(branch, safe='')}&per_page=1")
        runs = data.get("workflow_runs") if isinstance(data, dict) else None
        if not isinstance(runs, list) or not runs:
            return unknown("NO_RUNS")
        run = runs[0]
        result = {"run_id": run.get("id") if type(run.get("id")) is int else None,
                  "head_sha": run.get("head_sha") if isinstance(run.get("head_sha"), str) and SHA.fullmatch(run["head_sha"]) else None,
                  "status": safe_text(run.get("status")), "conclusion": safe_text(run.get("conclusion")),
                  "updated_at": safe_text(run.get("updated_at")),
                  "url": f"https://github.com/{repository}/actions/runs/{run.get('id')}" if type(run.get("id")) is int else None}
        if result["head_sha"] != head_sha:
            return unknown("LATEST_RUN_HEAD_MISMATCH", **result)
        age = (parse_time(observed_at) - parse_time(result["updated_at"])).total_seconds()
        if age < -300 or age > MAX_AGE_SECONDS:
            return unknown("STALE_OR_FUTURE_RUN", **result)
        return {"state": "OBSERVED", **result}
    except ObservationError as error:
        return unknown(str(error))
    except (ValueError, TypeError, AttributeError):
        return unknown("INVALID_ACTIONS_RESPONSE")


def observe_checks(client, repository, head_sha):
    try:
        data = client.get(f"/repos/{repository}/commits/{head_sha}/check-runs?per_page=100")
        runs = data.get("check_runs") if isinstance(data, dict) else None
        if not isinstance(runs, list) or not runs:
            return unknown("NO_CHECK_RUNS", head_sha=head_sha)
        if (type(data.get("total_count")) is not int or data["total_count"] > 100
                or data["total_count"] != len(runs)):
            return unknown("CHECK_PAGINATION_INCOMPLETE", head_sha=head_sha)
        projected = []
        for run in runs:
            if run.get("head_sha") != head_sha:
                return unknown("CHECK_HEAD_MISMATCH", head_sha=head_sha)
            projected.append({"name": safe_text(run.get("name")), "status": safe_text(run.get("status")),
                              "conclusion": safe_text(run.get("conclusion")),
                              "completed_at": safe_text(run.get("completed_at"))})
        return {"state": "OBSERVED", "head_sha": head_sha, "runs": projected}
    except ObservationError as error:
        return unknown(str(error), head_sha=head_sha)
    except (TypeError, AttributeError):
        return unknown("INVALID_CHECKS_RESPONSE", head_sha=head_sha)


def observe_repository(client, metadata, declared, observed_at):
    repository = metadata["full_name"]
    branch = metadata.get("default_branch")
    organ_key = next((key for key, member in declared["members"].items()
                      if member["repository"] == repository), None)
    classification = declared["classifications"].get(repository)
    result = {"repository": repository, "visibility": "public", "declared_class": classification,
              "declared_organ": organ_key, "discovery_state": "DECLARED" if classification else "CATALOG_ONLY",
              "default_branch": safe_text(branch, 500), "head_sha": None,
              "archived": metadata.get("archived") is True, "fork": metadata.get("fork") is True,
              "member_link": unknown("HEAD_UNAVAILABLE"), "actions": unknown("HEAD_UNAVAILABLE"),
              "checks": unknown("HEAD_UNAVAILABLE")}
    if not isinstance(branch, str) or not branch or len(branch) > 500:
        result["head_error"] = "DEFAULT_BRANCH_UNAVAILABLE"
        return result
    try:
        head = client.get(f"/repos/{repository}/git/ref/heads/{quote(branch, safe='')}")
        head_object = head.get("object", {}) if isinstance(head, dict) else {}
        head_sha = head_object.get("sha")
        if not isinstance(head_sha, str) or not SHA.fullmatch(head_sha):
            raise ObservationError("INVALID_HEAD_SHA")
        if head_object.get("type") != "commit":
            raise ObservationError("HEAD_NOT_COMMIT")
        result["head_sha"] = head_sha
    except ObservationError as error:
        result["head_error"] = str(error)
        return result
    result["member_link"] = member_link(client, repository, head_sha)
    result["actions"] = observe_actions(client, repository, branch, head_sha, observed_at)
    result["checks"] = observe_checks(client, repository, head_sha)
    return result


def build_snapshot(client, *, source_commit="UNKNOWN", root=ROOT, previous=None, observed_at=None):
    observed_at = observed_at or utc_now()
    parse_time(observed_at)
    declared, sources = load_declared_map(root=root, source_commit=source_commit)
    errors, repositories = [], []
    owner_complete = False
    try:
        metadata = list_public_repositories(client)
        owner_complete = True
        for row in sorted(metadata, key=lambda x: x["full_name"].lower()):
            repositories.append(observe_repository(client, row, declared, observed_at))
    except ObservationError as error:
        errors.append(str(error))
    for row in repositories:
        if row.get("head_error"):
            errors.append(row["repository"] + ":" + row["head_error"])
        for field in ("member_link", "actions", "checks"):
            if row[field]["state"] == "UNKNOWN":
                errors.append(row["repository"] + ":" + field + ":" + row[field]["reason"])
    old = {}
    baseline = None
    if previous is not None:
        if (previous.get("schema") != SCHEMA or previous.get("owner") != OWNER or
                previous.get("snapshot_sha256") != digest({k: v for k, v in previous.items() if k != "snapshot_sha256"})):
            raise ValueError("BASELINE_HASH_OR_SCOPE_INVALID")
        old = {row["repository"]: row.get("head_sha") for row in previous["repositories"]}
        baseline = previous["snapshot_sha256"]
    current = {row["repository"]: row["head_sha"] for row in repositories}
    drift = {"baseline_sha256": baseline, "added": [], "removed": [], "head_changed": []}
    if baseline:
        # An incomplete owner listing must never look like mass deletion.
        if owner_complete:
            drift["added"] = sorted(set(current) - set(old))
            drift["removed"] = sorted(set(old) - set(current))
        drift["head_changed"] = [{"repository": repo, "before": old[repo], "after": current[repo]}
                                 for repo in sorted(set(old) & set(current))
                                 if old[repo] and current[repo] and old[repo] != current[repo]]
    report = {"schema": SCHEMA, "owner": OWNER, "observed_at": observed_at,
              "sources": sources, "declared_map": declared, "repositories": repositories,
              "completeness": {"owner_inventory": owner_complete,
                               "repository_observations": owner_complete and not errors,
                               "errors": errors[:500], "requests_used": client.requests_used,
                               "repository_count": len(repositories)},
              "drift": drift, "action_authority": False,
              "claim_boundary": "METADATA_OBSERVATION_ONLY__NO_RUNTIME_OR_SCIENTIFIC_VALIDATION"}
    report["snapshot_sha256"] = digest(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly permit public metadata GETs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--source-commit", default="UNKNOWN")
    parser.add_argument("--authenticated", action="store_true", help="Use GITHUB_TOKEN on the fixed GitHub host only")
    parser.add_argument("--max-requests", type=int, default=300)
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("--live is required; collection never starts implicitly")
    if args.out.exists():
        parser.error("output already exists; keep previous observations immutable")
    token = os.environ.get("GITHUB_TOKEN") if args.authenticated else None
    if args.authenticated and not token:
        parser.error("explicit authenticated mode requires GITHUB_TOKEN")
    previous = strict_json(args.previous.read_text(encoding="utf-8")) if args.previous else None
    client = PublicGitHub(token=token, max_requests=args.max_requests)
    report = build_snapshot(client, source_commit=args.source_commit, previous=previous)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps({"snapshot_sha256": report["snapshot_sha256"],
                      "owner_inventory": report["completeness"]["owner_inventory"],
                      "repository_count": len(report["repositories"]),
                      "error_count": len(report["completeness"]["errors"]),
                      "action_authority": False}, sort_keys=True))
    return 0  # The independent evaluator owns PASS/HOLD and its exit status.


if __name__ == "__main__":
    raise SystemExit(main())
