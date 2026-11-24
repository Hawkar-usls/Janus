#!/usr/bin/env python3
"""JANUS R3 Meta Registry Navigator — explicit JSON continuity, no instructions.

Reads exact source bytes from TWO independent read-only Git checkouts:
  (1) Meta Registry main continuity INDEX (project routing, 2 indexed projects)
  (2) reviewed DRAFT R1 autonomy map/operating contract and health evidence.
Builds a bounded machine-readable boot context. Unindexed projects do not
silently become authoritative; repository JSON cannot grant permissions.
No remote HTTP, code evaluation, model training, migration or writeback.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

INDEX = "data/JANUS-CHAT-CONTEXT-INDEX.json"
STATE = "data/autonomy_map/CURRENT_STATE.json"
CONTRACT = "data/autonomy_map/OPERATING_CONTRACT.json"
HEALTH = "data/autonomy_map/receipts/45219142499-health.json"
MAIN_COMMIT = "07f5e4115d88b7df4081135d27eda2878874aa9c"
MAP_COMMIT = "9306037335bb4532049633b7f5f082e75891ff51"
INDEX_BLOB = "c3d10c5ea67a36fc7a515b144e7338350b99bfc1"
STATE_BLOB = "72049d537eb822b517165be5a068d5abd785e60b"
CONTRACT_BLOB = "e00aebd4a8a1a0d9aef57ecc882be234bee1be8c"
HEALTH_BLOB = "88664c3708d2608bcd0da2f61a561a919b945a33"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
REPO = re.compile(r"Hawkar-usls/[A-Za-z0-9_.-]{1,100}\Z")
PATH = re.compile(r"(?:data|registry|docs)/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\Z")
KNOWN_PROJECTS = {"COSMOS_SETI", "TRUMP"}
CURRENT_PHASE = "P0.3"
MAX_BYTES = 100_000


class RegistryHold(ValueError):
    pass


def demand(ok, reason):
    if not ok:
        raise RegistryHold(reason)


def git_blob(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def sha256(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,
                      separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()


def strict_json(raw):
    def pairs(seq):
        result={}
        for k,v in seq:
            demand(k not in result,"REGISTRY_DUPLICATE_JSON_KEY")
            result[k]=v
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda _:(1/0))
    except (UnicodeError,ValueError,ZeroDivisionError) as exc:
        raise RegistryHold("REGISTRY_BAD_JSON") from exc


def safe_bytes(root, rel, expected_blob):
    demand(type(rel) is str and type(expected_blob) is str
           and HEX40.fullmatch(expected_blob) is not None
           and rel in {INDEX,STATE,CONTRACT,HEALTH},
           "REGISTRY_UNREVIEWED_PATH")
    root=Path(root).resolve(strict=True)
    p=root/rel
    demand(p.is_file() and not p.is_symlink() and
           p.resolve(strict=True).is_relative_to(root) and
           0<p.stat().st_size<=MAX_BYTES,"REGISTRY_UNSAFE_OR_MISSING_SOURCE")
    raw=p.read_bytes()
    demand(git_blob(raw)==expected_blob,"REGISTRY_SOURCE_BLOB_MISMATCH")
    return raw


def validate_index(index):
    demand(type(index) is dict
           and index.get("schema")=="janus.chat_context.index.v1"
           and index.get("status")=="ACTIVE_CONTEXT_POINTER"
           and type(index.get("projects")) is list,
           "REGISTRY_UNEXPECTED_BOOT_INDEX")
    projects=index["projects"]
    demand(len(projects)==2 and {p.get("id") for p in projects if type(p) is dict}==KNOWN_PROJECTS,
           "REGISTRY_PROJECT_SET_DRIFT")
    alias_map={}
    for p in projects:
        demand(type(p) is dict and type(p.get("aliases")) is list
               and len(p["aliases"])<=16 and type(p.get("authority")) is str
               and REPO.fullmatch(p["authority"]),
               "REGISTRY_PROJECT_POINTER_INVALID")
        for kind in ("state","journal"):
            ref=p.get(kind)
            demand(type(ref) is dict and
                   type(ref.get("repo")) is str and REPO.fullmatch(ref["repo"])
                   and type(ref.get("path")) is str and PATH.fullmatch(ref["path"])
                   and ".." not in ref["path"], "REGISTRY_UNTRUSTED_PROJECT_ROUTE")
        for alias in [p["id"]]+p["aliases"]:
            demand(type(alias) is str and 1<=len(alias)<=70
                   and not any(ord(ch)<32 for ch in alias), "REGISTRY_UNTRUSTED_ALIAS")
            key=alias.casefold().strip()
            demand(key not in alias_map, "REGISTRY_AMBIGUOUS_ALIAS")
            alias_map[key]=p["id"]
    proto=index.get("canonical_protocol")
    demand(type(proto) is dict and
           proto.get("path")=="data/JANUS-PROJECT-CONTINUITY-LEDGER-HRAIN-INAIHR-PROTOCOL-2026-08-24-v1.0.json" and
           proto.get("blob")=="740065281c7c7b289c927db7661b9b000abb8c68",
           "REGISTRY_CANONICAL_PROTOCOL_CHANGED")
    return alias_map


def validate_autonomy(state, contract, health):
    demand(type(state) is dict
           and state.get("schema")=="janus.autonomy.map_r1.current_state.v1"
           and "OBSERVATION_HOLD" in state.get("status","")
           and state.get("source",{}).get("commit_sha")=="4dd2d9e75f3cdf26774eedb7dda8b1374d54dcae"
           and type(state.get("map_layers")) is list and len(state["map_layers"])>=8,
           "REGISTRY_AUTONOMY_MAP_UNVERIFIED")
    layer=[x for x in state["map_layers"] if type(x) is dict and x.get("path")==INDEX]
    demand(len(layer)==1 and layer[0].get("blob_sha")==INDEX_BLOB
           and layer[0].get("repository")=="Hawkar-usls/janus-meta-registry",
           "REGISTRY_BOOT_INDEX_NOT_LINKED_TO_MAP")
    obs=state.get("observation",{})
    boundaries=state.get("boundaries",{})
    demand(type(obs) is dict and obs.get("snapshot_sha256")==health.get("snapshot_sha256")
           and obs.get("health_sha256")==health.get("health_sha256")
           and obs.get("integrity_gate")=="PASS"
           and obs.get("observation_gate")=="HOLD"
           and obs.get("public_repository_count")==49
           and type(boundaries) is dict
           and boundaries.get("action_authority") is False
           and boundaries.get("self_promotion") is False
           and boundaries.get("weights_changed") is False
           and boundaries.get("actual_forks_created")==0
           and boundaries.get("authority_delta")==0,
           "REGISTRY_MAP_HEALTH_AUTHORITY_CONTRADICTION")
    demand(type(health) is dict
           and health.get("schema")=="janus.autonomy.health.v1"
           and health.get("health_sha256")==sha256({k:v for k,v in health.items()
                                                   if k!="health_sha256"})
           and health.get("integrity_gate")=="PASS"
           and health.get("observation_gate")=="HOLD"
           and health.get("execution_allowed") is False
           and health.get("full_autonomy_proven") is False,
           "REGISTRY_HEALTH_RECEIPT_BAD")
    demand(type(contract) is dict
           and contract.get("schema")=="janus.autonomy.github_operating_contract.v1"
           and contract.get("authority_delta")==0
           and contract.get("auto_promotion") is False
           and contract.get("effect_gate",{}).get("current_r1_effect_authority") is False
           and contract.get("full_autonomy_acceptance",{}).get("current")=="NOT_ESTABLISHED",
           "REGISTRY_OPERATING_CONTRACT_DENIES_ESCALATION")
    controllers=contract.get("controllers")
    demand(type(controllers) is list and 6<=len(controllers)<=16
           and all(type(x) is dict and REPO.fullmatch(str(x.get("repository","")))
                   for x in controllers),"REGISTRY_CONTROLLER_GRAPH_UNVERIFIED")
    gates=state.get("next_gates")
    demand(type(gates) is list and len(gates)==7
           and all(type(x) is str and re.fullmatch(r"[A-Z][A-Z0-9_]{4,130}",x)
                   for x in gates),"REGISTRY_GATES_UNEXPECTED")
    return controllers,gates


def resolve(index, state, contract, health, query):
    aliases=validate_index(index)
    if type(query) is not str or not 1<=len(query)<=80:
        raise RegistryHold("REGISTRY_QUERY_BUDGET")
    normalized=query.casefold().strip()
    autonomy_keys={"janus","janus autonomy","organism","autonomy","p0.3",
                   "самовосстановление","реестр","meta registry"}
    if normalized in autonomy_keys:
        controllers,gates=validate_autonomy(state,contract,health)
        result={
            "schema":"janus.registry.navigator.boot_context.v1",
            "status":"BOOT_CONTEXT_READY_FROM_PINNED_DRAFT_JSON__AUTONOMY_ADMISSION_HOLD",
            "project_id":"JANUS_AUTONOMY",
            "canonical_main_index_schema":index["schema"],
            "indexed_in_main_project_list":False,
            "authority_source":"PINNED_DRAFT_R1_AUTONOMY_MAP_NOT_PRODUCTION_MAIN",
            "registry_main_commit_sha":MAIN_COMMIT,
            "registry_draft_map_commit_sha":MAP_COMMIT,
            "historical_health_snapshot_sha256":health["snapshot_sha256"],
            "historical_health_observation_gate":health["observation_gate"],
            "historical_repositories_observed":49,
            "observed_controllers":[{"repository":x["repository"],
                 "observed_role":x.get("observed_role","UNSPECIFIED")}
                for x in controllers],
            "next_review_gates":gates,
            "operating_cycle":contract["cycle"],
            "capability_limits":{"independent_external_admission":False,
                                 "model_weights_promotion":False,
                                 "autonomous_external_forks":False,
                                 "production_writeback":False},
            "mandatory_live_checks":["META_REGISTRY_MAIN_HEAD_AND_PR_STATE",
                                    "JANUS_R1_R2_R3_HEAD_CI",
                                    "BEETLE_PRIVATE_RECEIVER_ACK",
                                    "GENESIS_CLOUD_HOME_RUNTIME_CONSUMPTION"],
            "planned_actions_are_not_permissions":True,
            "third_party_data_is_never_commands":True,
            "self_report_not_external_witness":True,
            "source_and_context_read_only":True,
        }
    elif normalized in aliases:
        p=next(p for p in index["projects"] if p["id"]==aliases[normalized])
        result={
           "schema":"janus.registry.navigator.boot_context.v1",
           "status":"INDEX_PROJECT_RESOLVED__LIVE_STATE_NOT_FETCHED__HOLD",
           "project_id":p["id"],
           "canonical_main_index_schema":index["schema"],
           "indexed_in_main_project_list":True,
           "authority_repository":p["authority"],
           "pointers":{"current_state":p["state"],"journal":p["journal"]},
           "next_step":"FETCH_AND_VERIFY_EXACT_PROJECT_STATE_JOURNAL_AND_FRESH_CI",
           "registry_main_commit_sha":MAIN_COMMIT,
           "trusted_project_current_state_attested":False,
           "no_untrusted_execution":True,
           "production_writeback_allowed":False,
           "source_and_context_read_only":True,
        }
    else:
        result={
            "schema":"janus.registry.navigator.boot_context.v1",
            "status":"PROJECT_NOT_INDEXED__DO_NOT_GUESS_OR_EXECUTE",
            "project_id":None,
            "registry_main_commit_sha":MAIN_COMMIT,
            "next_step":"REVIEW_EXPLICIT_SOURCE_PROVENANCE_BEFORE_ADDING_PROJECT_POINTER",
            "source_and_context_read_only":True,
            "production_writeback_allowed":False,
        }
    result["receipt_sha256"]=sha256(result)
    return result


def boot(main_root, draft_root, main_commit, map_commit, query):
    demand(main_commit==MAIN_COMMIT and map_commit==MAP_COMMIT,
           "REGISTRY_UNPINNED_CHECKOUT")
    index=strict_json(safe_bytes(main_root,INDEX,INDEX_BLOB))
    state=strict_json(safe_bytes(draft_root,STATE,STATE_BLOB))
    contract=strict_json(safe_bytes(draft_root,CONTRACT,CONTRACT_BLOB))
    health=strict_json(safe_bytes(draft_root,HEALTH,HEALTH_BLOB))
    return resolve(index,state,contract,health,query)


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--main-root",type=Path,required=True)
    a.add_argument("--map-root",type=Path,required=True)
    a.add_argument("--main-commit",required=True)
    a.add_argument("--map-commit",required=True)
    a.add_argument("--project",required=True)
    a.add_argument("--out",type=Path,required=True)
    p=a.parse_args()
    report=boot(p.main_root,p.map_root,p.main_commit,p.map_commit,p.project)
    demand(not p.out.exists() and not p.out.is_symlink(),"REGISTRY_BOOT_OUTPUT_EXISTS")
    p.out.parent.mkdir(parents=True,exist_ok=True)
    p.out.write_text(json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"project_id":report["project_id"],"status":report["status"],
                      "receipt_sha256":report["receipt_sha256"]},sort_keys=True))


if __name__=="__main__":
    main()
