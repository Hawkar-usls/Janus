#!/usr/bin/env python3
"""Zero-spend S2 completion freeze, upstream of *every* Rowan submit step.

Read-only GitHub public Meta Registry source. Never allows submission on a
missing/invalid/stale source. SUBMIT_ELIGIBLE only means the existing ACK,
provider matching, exact case freeze and credit gates MAY be considered.
It is not itself spending authorization or a scientific claim.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

URL="https://raw.githubusercontent.com/Hawkar-usls/janus-meta-registry/main/data/rowan/CANONICAL_STATE.json"
SCHEMA="janus.rowan.observer_state.v1"
FROZEN_CASES=8
MAX_RESPONSE_BYTES=4096
HEX64=re.compile(r"^[0-9a-f]{64}$")


class FreezeHold(ValueError):
    pass


def gate(state:object)->tuple[str,bool]:
    if not isinstance(state,dict) or state.get("schema_id")!=SCHEMA:
        return "HOLD_MISSING_OR_INVALID_CANONICAL",False
    run=state.get("latest_durable_external_run")
    n=state.get("last_completed_mutants")
    if type(run) is not int or run<=0 or type(n) is not int or n<0 or n>FROZEN_CASES:
        return "HOLD_INVALID_PILOT_PROGRESS",False
    if not isinstance(state.get("underlying_state_is_derived_from"),str) or not HEX64.fullmatch(
            state["underlying_state_is_derived_from"]):
        return "HOLD_UNATTESTED_SOURCE_DIGEST",False
    if (state.get("autonomous_spend_authority")!=
            "ONLY_FROZEN_EIGHT_MUTANT_S2_WITH_EXISTING_WEEKLY_BUDGET"
        or state.get("new_unreviewed_studies")!="PROPOSAL_ONLY_NOT_AUTO_EXECUTED"):
        return "HOLD_SCIENCE_OR_BUDGET_AUTHORITY_DRIFT",False
    alerts=state.get("budget_guard_alert_mutations")
    if not isinstance(alerts,list) or alerts:
        return "HOLD_EXISTING_BUDGET_ALERTS",False
    if n==FROZEN_CASES:
        return "FROZEN_COMPLETED_8_OF_8_NO_MORE_SUBMISSIONS",False
    return "INCOMPLETE_EXISTING_S2_ELIGIBLE_FOR_EXISTING_ACK_AND_CREDIT_GATES",True


def load_remote()->dict:
    with urlopen(Request(URL,headers={"User-Agent":"JANUS-Rowan-S2-Freeze-1",
                                    "Accept":"application/json"}),timeout=18) as stream:
        raw=stream.read(MAX_RESPONSE_BYTES+1)
    if not raw or len(raw)>MAX_RESPONSE_BYTES:
        raise FreezeHold("SOURCE_EMPTY_OR_OVERSIZED")
    try:
        obj=json.loads(raw.decode("utf-8"))
    except (ValueError,UnicodeDecodeError) as exc:
        raise FreezeHold("INVALID_CANONICAL_JSON") from exc
    if not isinstance(obj,dict):
        raise FreezeHold("INVALID_CANONICAL_TYPE")
    return obj


def main()->int:
    try:
        status,allowed=gate(load_remote())
    except (OSError,TimeoutError,URLError,HTTPError,FreezeHold):
        status,allowed="HOLD_SOURCE_UNAVAILABLE",False
    output=Path(os.environ.get("GITHUB_OUTPUT",""))
    if os.environ.get("GITHUB_OUTPUT"):
        with output.open("a",encoding="utf-8") as f:
            f.write("should_submit="+("true" if allowed else "false")+"\n")
            f.write("completion_gate_status="+status+"\n")
    print("ROWAN_S2_COMPLETION_GATE="+status)
    print("ROWAN_S2_NEW_SUBMISSION_ALLOWED_BY_COMPLETION_GATE="+str(allowed).upper())
    return 0


if __name__=="__main__":
    raise SystemExit(main())
