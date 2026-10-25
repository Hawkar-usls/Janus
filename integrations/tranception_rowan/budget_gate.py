#!/usr/bin/env python3
"""Read-only provider balance probe, exposing bounded bands not raw account data.

Never assert that a provider non-weekly credit is promotional instead of
purchased. Credit origin cannot be established using whoami alone.
"""
from __future__ import annotations
import importlib
import math
import os
from decimal import Decimal
from typing import Any

def finite_nonnegative(x: Any) -> float | None:
    if isinstance(x, bool) or type(x) not in (int, float, Decimal):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if math.isfinite(v) and 0 <= v <= 1e9 else None

def bands(prefix: str, amount: Any) -> dict[str, str]:
    n = finite_nonnegative(amount)
    return {
        prefix+"_PRESENT": "YES" if n is not None else "NO",
        prefix+"_GE_10": "YES" if n is not None and n >= 10 else "NO_OR_UNKNOWN",
        prefix+"_GE_20": "YES" if n is not None and n >= 20 else "NO_OR_UNKNOWN",
        prefix+"_GE_30": "YES" if n is not None and n >= 30 else "NO_OR_UNKNOWN",
        prefix+"_GE_100": "YES" if n is not None and n >= 100 else "NO_OR_UNKNOWN",
        prefix+"_GE_450": "YES" if n is not None and n >= 450 else "NO_OR_UNKNOWN",
        prefix+"_LE_500": "YES" if n is not None and n <= 500 else "NO_OR_UNKNOWN",
    }

def evaluate(user: Any) -> dict[str, str]:
    weekly = getattr(user,"weekly_credits",None)
    other = getattr(user,"credits",None)
    subscription = getattr(user,"individual_subscription",None)
    org = getattr(user,"organization",None)
    plan = getattr(subscription,"subscription_plan",None)
    plan_value = str(getattr(plan,"value",plan) or "").lower()
    # Only map a fixed list of known safe plan labels, not arbitrary untrusted strings.
    category = "EXPLICIT_FREE" if plan_value in {"free","free_tier","free-plan","free plan"} else (
        "NO_SUBSCRIPTION_OBJECT" if subscription is None else "SUBSCRIPTION_REQUIRES_REVIEW"
    )
    status={
        "account_class": "NO_ORGANIZATION" if org is None else "HAS_ORGANIZATION",
        "subscription_class": category,
        "signup_grant_provenance":"NOT_DISTINGUISHABLE_FROM_PURCHASED_IN_WHOAMI",
        "provider_computation":"NOT_ATTEMPTED",
        "credit_spending":"NOT_ATTEMPTED",
    }
    status.update(bands("PERSONAL_WEEKLY",weekly))
    status.update(bands("PERSONAL_NONWEEKLY",other))
    status.update(bands("ORG_WEEKLY",getattr(org,"weekly_credits",None) if org else None))
    status.update(bands("ORG_NONWEEKLY",getattr(org,"credits",None) if org else None))
    return status

def main()->int:
    if not os.environ.get("ROWAN_API_KEY"):
        print("JANUS_BUDGET_GATE=NO_KEY")
        return 2
    try:
        api=importlib.import_module("rowan")
        state=evaluate(api.whoami())
    except Exception:
        print("JANUS_BUDGET_GATE=READ_FAILED")
        return 3
    for k,v in state.items():
        print("JANUS_BUDGET_"+k+"="+v)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
