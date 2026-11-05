"""Local account-based provider pool. No separate API keys or paid fallback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from . import codex_bridge, google_gateway
from .codex_handoff import CodexHandoffError, prepare_codex_cloud_task
from .codex_runtime import CodexRuntimeError
from .siwc_auth import SiwcError

QUOTA_ERRORS = {"CODEX_PROVIDER_RATE_LIMIT", "CODEX_PROVIDER_USAGE_LIMIT",
                "CODEX_PROVIDER_SESSION_BUDGET", "CODEX_PROVIDER_HTTP_429"}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--chatgpt-state-dir", type=Path)
    result.add_argument("--codex-binary", type=Path)
    result.add_argument("--google-binary", type=Path)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    models = commands.add_parser("models")
    models.add_argument("--provider", choices=["chatgpt", "google"], required=True)
    for name in ("smoke", "ask"):
        call = commands.add_parser(name)
        call.add_argument("--provider", choices=["chatgpt", "google", "auto", "deepseek"], default="chatgpt")
        call.add_argument("--model")
        call.add_argument("--thread")
        call.add_argument("--timeout", type=int, default=120)
        call.add_argument("--allow-plan-usage", action="store_true")
        if name == "ask":
            call.add_argument("--prompt", required=True)
    return result


def _chatgpt(args, command, prompt=None):
    return codex_bridge.execute(argparse.Namespace(
        command=command, state_dir=args.chatgpt_state_dir, model=getattr(args, "model", None),
        codex_binary=args.codex_binary, prompt=prompt, thread=getattr(args, "thread", None),
        timeout=getattr(args, "timeout", 120), allow_plan_usage=getattr(args, "allow_plan_usage", False),
    ))


def execute(args):
    if args.command == "status":
        google = args.google_binary or (Path(shutil.which("agy")) if shutil.which("agy") else None)
        return {"providers": {
            "chatgpt": _chatgpt(args, "status"),
            "google": {"status": "CLI_AVAILABLE_AUTH_NOT_PROBED" if google and google.is_file() else "CLI_NOT_INSTALLED",
                       "model_called": False},
            "deepseek": {"status": "OFFICIAL_API_REQUIRES_KEY", "enabled": False,
                         "browser_mode": "HOST_OPERATED_ONLY", "model_called": False},
        }, "automatic_purchases": False, "paid_api_fallback": False}
    if args.command == "models":
        return _chatgpt(args, "models") if args.provider == "chatgpt" else google_gateway.list_models(args.google_binary)
    if not args.allow_plan_usage:
        raise SiwcError("ACCOUNT_GATEWAY_EXPLICIT_USAGE_REQUIRED")
    if not 10 <= args.timeout <= 300:
        raise SiwcError("ACCOUNT_GATEWAY_TIMEOUT_INVALID")
    if args.provider == "deepseek":
        return {"status": "DEEPSEEK_API_KEY_REQUIRED", "ok": False, "model_called": False}
    prompt = codex_bridge.SMOKE_PROMPT if args.command == "smoke" else args.prompt
    prepare_codex_cloud_task(prompt)
    provider = args.provider
    fallback = None
    if provider == "auto" and (args.model is not None or args.thread is not None):
        raise SiwcError("ACCOUNT_GATEWAY_AUTO_REQUIRES_PROVIDER_NEUTRAL_REQUEST")
    if provider in {"chatgpt", "auto"}:
        try:
            result = {**_chatgpt(args, args.command, prompt), "provider": "chatgpt"}
            if result.get("status") == "CODEX_SMOKE_RESPONSE_MISMATCH":
                result["ok"] = False
            return result
        except CodexRuntimeError as error:
            if provider != "auto" or str(error) not in QUOTA_ERRORS:
                raise
            fallback = str(error)
    result = google_gateway.run_turn(prompt, binary=args.google_binary,
                                    model=args.model, conversation=args.thread, timeout=args.timeout)
    if args.command == "smoke":
        result["smoke_response_matches"] = result["text"].strip() == "JANUS_BRIDGE_OK"
        if not result["smoke_response_matches"]:
            result.update(status="ACCOUNT_GATEWAY_SMOKE_RESPONSE_MISMATCH", ok=False)
    if fallback:
        result["fallback_reason"] = fallback
    return result


def main(argv=None):
    try:
        result = execute(parser().parse_args(argv))
    except (SiwcError, CodexRuntimeError, CodexHandoffError, google_gateway.GoogleGatewayError) as error:
        result = {"status": str(error), "ok": False}
    except KeyboardInterrupt:
        result = {"status": "ACCOUNT_GATEWAY_INTERRUPTED", "ok": False}
    except Exception:
        result = {"status": "ACCOUNT_GATEWAY_INTERNAL_ERROR", "ok": False}
    print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
    return 2 if result.get("ok") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
