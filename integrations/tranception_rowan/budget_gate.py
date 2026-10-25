#!/usr/bin/env python3
"""Rowan budget gate: no credentials, balances, identity or purchases in logs.

Safety invariant: an authenticated account with grant-like non-weekly credits
is not, by itself, proof that every credit was granted rather than purchased.
This gate attests balances and account class without spending anything.
"""
from __future__ import annotations
import importlib
import math
import os
from typing import Any


def finite_nonnegative(x: Any) -> float | None:
    if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
        return None
    return float(x)


def evaluate(user: Any) -> dict[str, str]:
    weekly = finite_nonnegative(getattr(user, "weekly_credits", None))
    other = finite_nonnegative(getattr(user, "credits", None))
    subscription = getattr(user, "individual_subscription", None)
    org = getattr(user, "organization", None)
    # This is only a subscription/no-org attestation; the 500 sign-up grant
    # is not separately identified by Rowan's user model.
    free_account_candidate = subscription is None and org is None
    return {
        "account_class": "NO_PAID_SUBSCRIPTION_OR_ORG_DETECTED" if free_account_candidate else "ACCOUNT_CLASS_REQUIRES_REVIEW",
        "weekly_credits_ge_10": "YES" if weekly is not None and weekly >= 10 else "NO_OR_UNKNOWN",
        "weekly_credits_ge_20": "YES" if weekly is not None and weekly >= 20 else "NO_OR_UNKNOWN",
        "nonweekly_credits_ge_20": "YES" if other is not None and other >= 20 else "NO_OR_UNKNOWN",
        "nonweekly_credits_ge_100": "YES" if other is not None and other >= 100 else "NO_OR_UNKNOWN",
        "nonweekly_credits_le_500": "YES" if other is not None and other <= 500 else "NO_OR_UNKNOWN",
        "signup_grant_provenance": "NOT_DISTINGUISHABLE_FROM_PURCHASED_IN_WHOAMI",
        "provider_computation": "NOT_ATTEMPTED",
        "credit_spending": "NOT_ATTEMPTED",
    }


def main() -> int:
    if not os.getenv("ROWAN_API_KEY"):
        print("JANUS_BUDGET_GATE=NO_KEY")
        return 2
    api = importlib.import_module("rowan")
    try:
        status = evaluate(api.whoami())
    except Exception:
        print("JANUS_BUDGET_GATE=READ_FAILED")
        return 3
    for name, value in status.items():
        print("JANUS_BUDGET_" + name.upper() + "=" + value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
