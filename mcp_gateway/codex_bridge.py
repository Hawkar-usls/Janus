"""Human-operated SIWC client. Deliberately not registered as an MCP tool.

Credentials stay in the trusted local client; the model receives neither the
vault nor GitHub access. Commands are inert until explicitly invoked locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from .codex_handoff import CodexHandoffError, prepare_codex_cloud_task
from .codex_runtime import CodexRuntimeError, check_runtime, run_turn
from .siwc_auth import SiwcClient, SiwcError

SMOKE_PROMPT = "Reply with exactly JANUS_BRIDGE_OK and no other text."


def default_storage_dir() -> Path:
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            raise SiwcError("SIWC_LOCALAPPDATA_REQUIRED")
        return Path(local) / "JANUS" / "CodexBridge"
    state = os.environ.get("XDG_STATE_HOME")
    return (Path(state) if state else Path.home() / ".local" / "state") / "janus-codex-bridge"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="JANUS local Sign in with ChatGPT / isolated Codex bridge",
    )
    result.add_argument("--state-dir", type=Path, help="private client state directory, outside the repository")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="show sanitized connection state; no network or model call")
    commands.add_parser("accounts", help="list saved registrations without credentials")
    commands.add_parser("models", help="fetch current model choices for the active ChatGPT account; no inference")
    doctor = commands.add_parser("doctor", help="probe app-server compatibility locally with a synthetic credential")
    doctor.add_argument("--codex-binary", type=Path, help="explicit trusted official Codex executable")
    doctor.add_argument("--model", help="probe compatibility for this exact model, otherwise the default")
    login = commands.add_parser("login", help="Continue with ChatGPT in the system browser")
    login.add_argument("--profile", help="reauthorize this existing registration")
    login.add_argument("--timeout", type=int, default=180)
    select = commands.add_parser("select", help="select a saved registration")
    select.add_argument("profile")
    logout = commands.add_parser("logout", help="revoke the selected renewable session and clear its tokens")
    logout.add_argument("--profile")
    for name, help_text in (("smoke", "verify one real completed model turn"),
                            ("ask", "send one bounded text request with executable tools blocked")):
        call = commands.add_parser(name, help=help_text)
        call.add_argument("--model", help="model identifier; otherwise select the default catalog entry")
        call.add_argument("--codex-binary", type=Path, help="explicit trusted official Codex executable")
        call.add_argument("--timeout", type=int, default=120)
        call.add_argument("--allow-plan-usage", action="store_true",
                          help="authorize this request to use your ChatGPT plan; manage credit limits in ChatGPT")
        if name == "ask":
            call.add_argument("--prompt", required=True)
            call.add_argument("--thread", help="resume a thread previously created by this isolated bridge")
    return result


def execute(args: argparse.Namespace) -> dict:
    if args.command == "doctor":
        return check_runtime(codex_binary=args.codex_binary, model=args.model)
    # Reject missing authorization and unsafe text before even opening the vault.
    if args.command in {"smoke", "ask"}:
        if not args.allow_plan_usage:
            raise SiwcError("SIWC_EXPLICIT_PLAN_USAGE_REQUIRED")
        prompt = SMOKE_PROMPT if args.command == "smoke" else args.prompt
        prepare_codex_cloud_task(prompt)  # Reuse bounded text / secret-marker checks.
    directory = args.state_dir if args.state_dir is not None else default_storage_dir()
    client = SiwcClient(directory)
    if args.command == "models":
        return client.list_models()
    if args.command in {"status", "accounts"}:
        return {
            **client.status(),
            "codex_binary_available": shutil.which("codex") is not None,
            "model_called": False,
            "status_source": "local_client",
            "github_write_performed": False,
            "habitat_modified": False,
        }
    if args.command == "login":
        if not 10 <= args.timeout <= 600:
            raise SiwcError("SIWC_TIMEOUT_INVALID")
        print("Continue with ChatGPT. Complete account selection and consent in your browser. "
              "Eligible requests use your ChatGPT plan. Manage usage at https://chatgpt.com/settings/usage. "
              "Do not paste credentials into chat.", file=sys.stderr, flush=True)
        return client.login(profile_id=args.profile, timeout=args.timeout)
    if args.command == "select":
        return client.select_profile(args.profile)
    if args.command == "logout":
        return client.logout(profile_id=args.profile)
    if not 10 <= args.timeout <= 300:
        raise SiwcError("SIWC_TIMEOUT_INVALID")
    # A separate persistent home per account prevents inherited Codex history,
    # configuration and cross-account thread reuse. Only the runtime sees a token.
    status = client.status()
    active = status.get("active_profile_id")
    if not isinstance(active, str) or not active:
        raise SiwcError("SIWC_LOGIN_REQUIRED")
    runtime_dir = directory / "runtime" / hashlib.sha256(active.encode("utf-8")).hexdigest()
    result = run_turn(
        client.access_token(profile_id=active), args.model, prompt, timeout=args.timeout,
        thread_id=getattr(args, "thread", None), runtime_dir=runtime_dir,
        codex_binary=args.codex_binary,
    )
    if args.command == "smoke":
        text = result.get("text", "")
        result = {**result, "smoke_response_matches": text.strip() == "JANUS_BRIDGE_OK"}
        if not result["smoke_response_matches"]:
            return {**result, "status": "CODEX_SMOKE_RESPONSE_MISMATCH"}
    return {**result, "github_write_performed": False, "auto_merge": False,
            "production_authority_delta": 0}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = execute(args)
    except (SiwcError, CodexRuntimeError, CodexHandoffError) as exc:
        # These exception classes expose only fixed diagnostic codes.
        result = {"status": str(exc), "ok": False}
        code = 2
    except KeyboardInterrupt:
        result, code = {"status": "BRIDGE_INTERRUPTED", "ok": False}, 130
    except Exception:
        # Provider exceptions, URLs, JWTs and subprocess stderr must never escape.
        result, code = {"status": "BRIDGE_INTERNAL_ERROR", "ok": False}, 3
    else:
        code = 2 if (result.get("ok") is False or result.get("compatible") is False
                     or result.get("status") == "CODEX_SMOKE_RESPONSE_MISMATCH") else 0
    print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
