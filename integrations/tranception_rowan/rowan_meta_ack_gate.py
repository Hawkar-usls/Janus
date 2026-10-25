#!/usr/bin/env python3
"""Fail-closed cross-repository durability gate before any new Rowan spend.

Prohibit next provider submission unless the latest *previous successful*
JANUS stage-2 workflow has an archived source-run checkpoint in the
authoritative Meta Registry main branch. No write endpoints or Rowan key used.
"""
from __future__ import annotations
import json
import os
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError

RUNS_URL = ("https://api.github.com/repos/Hawkar-usls/Janus/actions/"
            "workflows/rowan-bbc1-stratified-pilot.yml/runs"
            "?per_page=30&branch=main&status=completed")
ACK_URL = ("https://raw.githubusercontent.com/Hawkar-usls/"
           "janus-meta-registry/main/data/rowan/CANONICAL_STATE.json")

class ArchiveNotConfirmed(ValueError):
    pass

def check(previous_successful_run_id: int | None, state: dict | None,
          current_run_id: int) -> str:
    if not isinstance(current_run_id,int) or current_run_id<=0:
        raise ArchiveNotConfirmed("INVALID_CURRENT_RUN_ID")
    if previous_successful_run_id is None:
        # First ever run: no already-completed source receipt to protect.
        return "NO_PREVIOUS_SUCCESSFUL_RUN"
    if not isinstance(previous_successful_run_id,int) or previous_successful_run_id>=current_run_id:
        raise ArchiveNotConfirmed("INVALID_PREVIOUS_RUN_ID")
    if not isinstance(state,dict) or state.get("schema_id")!="janus.rowan.observer_state.v1":
        raise ArchiveNotConfirmed("META_REGISTRY_ACK_MISSING_OR_INVALID")
    last=state.get("latest_durable_external_run")
    if not isinstance(last,int) or last < previous_successful_run_id or last >= current_run_id:
        raise ArchiveNotConfirmed("META_REGISTRY_BEHIND_LATEST_SUCCESSFUL_RUN")
    if state.get("new_unreviewed_studies")!="PROPOSAL_ONLY_NOT_AUTO_EXECUTED":
        raise ArchiveNotConfirmed("META_SCIENTIFIC_AUTHORITY_DRIFT")
    if state.get("autonomous_spend_authority")!="ONLY_FROZEN_EIGHT_MUTANT_S2_WITH_EXISTING_WEEKLY_BUDGET":
        raise ArchiveNotConfirmed("META_SPEND_AUTHORITY_DRIFT")
    alerts=state.get("budget_guard_alert_mutations",[])
    if not isinstance(alerts,list):
        raise ArchiveNotConfirmed("META_BUDGET_ALERT_FIELD_INVALID")
    if alerts:
        raise ArchiveNotConfirmed("BUDGET_ALERT_HARD_STOP")
    return "META_REGISTRY_ACK_VERIFIED"

def retrieve_json(url, *, token=None, max_bytes=300_000):
    headers={"Accept":"application/vnd.github+json","User-Agent":"JANUS-Rowan-Ack-Gate/1"}
    if token:
        headers["Authorization"]="Bearer "+token
    with urlopen(Request(url,headers=headers),timeout=18) as response:
        data=response.read(max_bytes+1)
    if len(data)>max_bytes:
        raise ArchiveNotConfirmed("GITHUB_RESPONSE_OVERSIZE")
    try:
        return json.loads(data)
    except (ValueError,UnicodeDecodeError) as exc:
        raise ArchiveNotConfirmed("GITHUB_JSON_INVALID") from exc

def evaluate_real(current_id):
    try:
        runs=retrieve_json(RUNS_URL,token=os.environ.get("GITHUB_TOKEN"))
        records=runs.get("workflow_runs")
        if not isinstance(records,list):
            raise ArchiveNotConfirmed("PRIOR_RUNS_NOT_RETURNED")
        qualifying=[r["id"] for r in records
            if r.get("status")=="completed" and r.get("conclusion")=="success"
            and r.get("head_branch")=="main"
            and r.get("event") in ("schedule","workflow_dispatch")
            and isinstance(r.get("id"),int) and r["id"]<current_id]
        if not qualifying:
            return check(None,None,current_id)
        newest=max(qualifying)
        state=retrieve_json(ACK_URL)  # Public meta; NEVER send Rowan or GitHub secret to this host.
        return check(newest,state,current_id)
    except ArchiveNotConfirmed:
        raise
    except (HTTPError,URLError,TimeoutError,OSError,ValueError) as exc:
        raise ArchiveNotConfirmed("ARCHIVE_OR_GITHUB_READ_UNAVAILABLE") from exc

def main():
    value=os.environ.get("GITHUB_RUN_ID","")
    if not value.isdecimal() or len(value)>20:
        print("ROWAN_ARCHIVE_ACK=GITHUB_RUN_ID_MISSING")
        return 2
    try:
        status=evaluate_real(int(value))
        print("ROWAN_ARCHIVE_ACK="+status)
        return 0
    except ArchiveNotConfirmed as exc:
        print("ROWAN_ARCHIVE_ACK_HOLD="+str(exc))
        return 2

if __name__=="__main__":
    raise SystemExit(main())
