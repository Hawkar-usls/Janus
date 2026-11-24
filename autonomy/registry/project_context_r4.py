#!/usr/bin/env python3
"""R4 read-only project continuity reader. Original JANUS implementation.

Pinned first-party source bytes only; output is historical evidence, not a
task dispatcher, live state attestation, permission, or executable prompt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import unittest

META_COMMIT = "e85c19c31bdad729f06961d64de36d8f3a620483"
TRUMP_COMMIT = "941148371aac0a503d4f254c839683642fdf2f20"
META = "Hawkar-usls/janus-meta-registry"
FUND = "Hawkar-usls/Janus-Fundamentum"
INDEX = "data/JANUS-CHAT-CONTEXT-INDEX.json"
ROUTES = {
    "COSMOS_SETI": {
        "authority": "Hawkar-usls/Janus-Cosmos", "repo": META,
        "commit": META_COMMIT,
        "state": "data/JANUS-COSMOS-SETI-CURRENT-STATE.json",
        "state_blob": "13fa373545d163b25ad12f4bb1929bf47cb90da0",
        "journal": "data/JANUS-COSMOS-SETI-DEVELOPMENT-JOURNAL.json",
        "journal_blob": "862f66a4e26817a4ac0f1892251ca15a04595aba",
    },
    "TRUMP": {
        "authority": FUND, "repo": FUND, "commit": TRUMP_COMMIT,
        "state": "registry/TRUMP_CURRENT_STATE.json",
        "state_blob": "7c87873974b9994a8d5e02eb283c80ad5d8f797a",
        "journal": "docs/TRUMP_DEVELOPMENT_JOURNAL.md",
        "journal_blob": "f060f1096aaf7b5e67f74846146fbe3837439267",
    },
}
INDEX_BLOB = "c3d10c5ea67a36fc7a515b144e7338350b99bfc1"
MAX_BYTES = 2 * 1024 * 1024

class Hold(ValueError):
    pass

def require(condition, reason):
    if not condition:
        raise Hold(reason)

def blob(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()

def verified_bytes(root, rel, expected):
    root = Path(root).resolve(strict=True)
    parts = Path(rel).parts
    require(parts and not Path(rel).is_absolute() and ".." not in parts,
            "UNSAFE_PATH")
    p = root
    for part in parts:
        p = p / part
        require(not p.is_symlink(), "SYMLINK_REJECTED")
    require(p.is_file() and p.resolve().is_relative_to(root), "MISSING_SOURCE")
    with p.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    require(0 < len(raw) <= MAX_BYTES, "SOURCE_BUDGET_EXCEEDED")
    require(blob(raw) == expected, "SOURCE_BLOB_MISMATCH")
    return raw

def strict_object(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "DUPLICATE_JSON_KEY")
            out[key] = value
        return out
    def reject_constant(_):
        raise Hold("NONFINITE_JSON")
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                         parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise Hold("INVALID_JSON") from exc
    require(type(obj) is dict, "EXPECTED_OBJECT")
    return obj

def field_text(value, limit=2000):
    require(type(value) is str and 0 < len(value) <= limit, "INVALID_TEXT_FIELD")
    return value

def field_object(obj, key):
    value = obj.get(key)
    require(type(value) is dict, "INVALID_OBJECT_FIELD:" + key)
    return value

def route(index, project):
    require(project in ROUTES, "PROJECT_NOT_REVIEWED")
    require(index.get("schema") == "janus.chat_context.index.v1"
            and index.get("status") == "ACTIVE_CONTEXT_POINTER"
            and type(index.get("projects")) is list, "INDEX_SCHEMA_DRIFT")
    rows = [x for x in index["projects"] if type(x) is dict
            and x.get("id") == project]
    require(len(rows) == 1, "AMBIGUOUS_OR_MISSING_PROJECT")
    r = ROUTES[project]
    row = rows[0]
    require(row.get("authority") == r["authority"], "AUTHORITY_DRIFT")
    for kind in ("state", "journal"):
        require(row.get(kind) == {"repo": r["repo"], "path": r[kind]},
                "POINTER_DRIFT:" + kind)
    return r

def cosmos(state, journal):
    require(state.get("schema") == "janus.context_bridge.current_state.v1"
            and journal.get("schema") == "janus.cosmos.seti.development_journal.v1",
            "COSMOS_SCHEMA_DRIFT")
    current = field_object(state, "current")
    active = field_object(state, "active_experiment")
    jc = field_object(journal, "current")
    ja = field_object(journal, "active_experiment")
    events = journal.get("events")
    require(type(events) is list and 0 < len(events) <= 5000
            and all(type(e) is dict for e in events), "INVALID_EVENTS")
    gates = state.get("open_gates")
    require(type(gates) is list and len(gates) <= 200, "INVALID_GATES")
    stages = {
        "state_current": field_text(current.get("stage"), 120),
        "state_active": field_text(active.get("stage"), 120),
        "journal_current": field_text(jc.get("stage"), 120),
        "journal_active": field_text(ja.get("stage"), 120),
        "journal_last_listed_event": field_text(events[-1].get("stage"), 120),
    }
    return {
        "recorded_goal": field_text(state.get("goal")),
        "stage_candidates": stages,
        "stage_candidates_differ": len(set(stages.values())) > 1,
        "journal_event_count": len(events),
        "recorded_next_allowed": field_text(state.get("next_allowed")),
        "recorded_open_gates": [field_text(x) for x in gates],
        "recorded_release_gate": field_text(field_object(state, "release_gate").get("status")),
        "claim_ceiling": field_text(state.get("claim_ceiling")),
        "selection": "NO_AUTOMATIC_STAGE_SELECTION__CHECK_LATEST_PRIMARY_RECEIPTS",
    }

def trump(state, journal_text):
    require(state.get("artifact_id") == "JANUS-TRUMP-CURRENT-STATE-CONTINUITY-2026-09-14-v1.0"
            and state.get("repository_authority") == FUND
            and state.get("journal") == ROUTES["TRUMP"]["journal"]
            and journal_text.startswith("# TRUMP Development Journal"),
            "TRUMP_CONTINUITY_DRIFT")
    frontier = field_object(state, "current_frontier")
    authority = field_object(state, "latest_composite_authority")
    history = field_object(state, "historical_v1_1")
    old = state.get("must_not_repeat_as_new")
    require(type(old) is list and len(old) <= 200, "INVALID_HISTORY_LIST")
    limits = field_object(state, "permanent_firewalls")
    require(len(limits) <= 20, "INVALID_FIREWALLS")
    return {
        "recorded_frontier": {k: field_text(frontier.get(k))
            for k in ("gate", "status", "preregistration_status", "correct_question")},
        "recorded_scoped_authority": {k: field_text(authority.get(k))
            for k in ("version", "verdict", "scientific_seal_commit", "scope")},
        "recorded_historical_failure": field_text(history.get("verdict")),
        "must_not_repeat_as_new": [field_text(x) for x in old],
        "recorded_claim_limits": {k: field_text(v) for k, v in limits.items()},
        "journal_format": "MARKDOWN_BYTES_VERIFIED__SEMANTIC_REVIEW_STILL_REQUIRED",
        "selection": "NO_AUTOMATIC_STAGE_SELECTION__CHECK_LATEST_SCIENTIFIC_SEAL",
    }

def read_project(meta_root, fund_root, project):
    index = strict_object(verified_bytes(meta_root, INDEX, INDEX_BLOB))
    r = route(index, project)
    root = meta_root if r["repo"] == META else fund_root
    state = strict_object(verified_bytes(root, r["state"], r["state_blob"]))
    raw = verified_bytes(root, r["journal"], r["journal_blob"])
    fields = cosmos(state, strict_object(raw)) if project == "COSMOS_SETI" else trump(state, raw.decode("utf-8"))
    return {
        "schema": "janus.registry.project_context.r4.v1",
        "project": project,
        "source": r,
        "index_blob": INDEX_BLOB,
        "index_commit": META_COMMIT,
        "status": "PINNED_HISTORICAL_CONTEXT_ONLY__LIVE_AUTHORITY_HOLD",
        "untrusted_recorded_data": fields,
        "required_review": ["VERIFY_CURRENT_HEAD", "VERIFY_PRIMARY_RECEIPTS_AND_CI",
                            "CHECK_SUPERSESSION_AND_OPEN_GATES"],
        "action_authority": False,
        "live_state_verified": False,
        "production_writeback": False,
    }

class LocalTests(unittest.TestCase):
    def test_duplicate(self):
        with self.assertRaises(Hold):
            strict_object(b'{"x":1,"x":2}')
    def test_nonfinite(self):
        with self.assertRaises(Hold):
            strict_object(b'{"x":NaN}')
    def test_array(self):
        with self.assertRaises(Hold):
            strict_object(b'[]')
    def test_unknown(self):
        with self.assertRaises(Hold):
            route({}, "UNKNOWN")
    def test_pointer_drift(self):
        x = {"schema":"janus.chat_context.index.v1", "status":"ACTIVE_CONTEXT_POINTER",
             "projects":[{"id":"TRUMP","authority":FUND,"state":{"repo":FUND,"path":"../bad"}}]}
        with self.assertRaises(Hold):
            route(x, "TRUMP")
    def test_hash_and_symlink(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/"real").write_bytes(b"test")
            self.assertEqual(verified_bytes(p, "real", blob(b"test")), b"test")
            with self.assertRaises(Hold):
                verified_bytes(p, "real", blob(b"other"))
            (p/"link").symlink_to(p/"real")
            with self.assertRaises(Hold):
                verified_bytes(p, "link", blob(b"test"))
    def test_missing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Hold):
                verified_bytes(d, "missing", blob(b"test"))
    def test_traversal(self):
        with self.assertRaises(Hold):
            verified_bytes(".", "../bad", blob(b"test"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--meta-root", type=Path)
    p.add_argument("--fund-root", type=Path)
    p.add_argument("--project", choices=sorted(ROUTES))
    a = p.parse_args()
    if a.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LocalTests))
        return 0 if result.wasSuccessful() else 1
    if not all((a.meta_root, a.fund_root, a.project)):
        p.error("supply both roots and a reviewed project")
    try:
        report = read_project(a.meta_root, a.fund_root, a.project)
    except (Hold, OSError, UnicodeError) as exc:
        print(json.dumps({"status":"HOLD", "reason":str(exc), "action_authority":False}))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
