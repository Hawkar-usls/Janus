#!/usr/bin/env python3
"""Scheduled, read-only Rowan connectivity gate for JANUS.

No writes, purchases, workflow submissions, user identity, balances, raw API
responses, or secret contents in output. Safe for public GitHub Actions logs.
"""
from __future__ import annotations

import importlib
import os

FOLDER_UUID = "70480946-24c3-4078-91d1-ad99009262bf"


def status_code(exc: Exception) -> int | None:
    value = getattr(getattr(exc, "response", None), "status_code", None)
    return value if type(value) is int and 100 <= value <= 599 else None


def safe_error(exc: Exception) -> str:
    code = status_code(exc)
    return "HTTP_" + str(code) if code is not None else "PROVIDER_REQUEST_FAILED"


def inspect_readonly(sdk, *, key_present: bool) -> dict[str, str]:
    """Never use a provider mutating method. Pure test doubles are supported."""
    state = {
        "secret": "CONFIGURED" if key_present else "MISSING",
        "auth": "NOT_CHECKED",
        "folder": "NOT_CHECKED",
        "provider_computation": "NOT_ATTEMPTED",
        "credit_spending": "NOT_ATTEMPTED",
    }
    if not key_present:
        return state
    try:
        sdk.whoami()
        state["auth"] = "VERIFIED"
    except Exception as exc:
        state["auth"] = safe_error(exc)
        return state
    try:
        folder = sdk.retrieve_folder(FOLDER_UUID)
        # A small bounded read to establish access; never log folder entries.
        folder.contents(size=10)
        state["folder"] = "VERIFIED_READ_ONLY"
    except Exception as exc:
        state["folder"] = safe_error(exc)
    return state


def main() -> int:
    key_present = bool(os.environ.get("ROWAN_API_KEY"))
    if key_present:
        api = importlib.import_module("rowan")
    else:
        api = None
    state = inspect_readonly(api, key_present=key_present)
    for field in ("secret", "auth", "folder", "provider_computation", "credit_spending"):
        print("JANUS_ROWAN_" + field.upper() + "=" + state[field])
    if not key_present:
        return 2
    return 0 if state["auth"] == "VERIFIED" and state["folder"] == "VERIFIED_READ_ONLY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
