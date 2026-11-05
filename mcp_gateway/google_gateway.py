"""Account-based Google provider through the official Antigravity CLI.

OAuth and keyring storage belong to Google's client. This module never reads
tokens and never enables API-key/Vertex billing or automatic sign-in.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from .codex_handoff import prepare_codex_cloud_task

MAX_BYTES = 2 * 1024 * 1024
DENIED_ACTIONS = ("read_file", "write_file", "command", "unsandboxed", "mcp", "read_url", "execute_url")
TEXT_SETTINGS = {"permissions": {"deny": [action + "(*)" for action in DENIED_ACTIONS]}}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
CONVERSATION = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


class GoogleGatewayError(ValueError):
    """Fixed diagnostic code; no native/provider free text is exposed."""


def _binary(path: Path | None) -> str:
    value = str(path) if path is not None else shutil.which("agy")
    if (not value or not Path(value).is_file() or not Path(value).is_absolute()
            or os.name == "nt" and Path(value).suffix.lower() != ".exe"):
        raise GoogleGatewayError("GOOGLE_ACCOUNT_CLI_NOT_INSTALLED")
    return value


def _environment() -> dict:
    # Google's official client uses its own OS keyring/profile, not API keys.
    allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "COMSPEC",
               "USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "TEMP", "TMP"}
    return {key: value for key, value in os.environ.items() if key.upper() in allowed}


def _run(binary: str, arguments: list[str], timeout: float) -> tuple[int, str, str]:
    if not 1 <= timeout <= 300:
        raise GoogleGatewayError("GOOGLE_TIMEOUT_INVALID")
    with tempfile.TemporaryDirectory(prefix="janus-google-text-") as directory:
        work = Path(directory)
        settings = work / "text-settings.json"
        settings.write_text(json.dumps(TEXT_SETTINGS), encoding="utf-8")
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                [binary, "--settings", str(settings), *arguments], cwd=work,
                env=_environment(), stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
                shell=False, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if os.fstat(output.fileno()).st_size + os.fstat(errors.fileno()).st_size > MAX_BYTES:
                        raise GoogleGatewayError("GOOGLE_OUTPUT_LIMIT")
                    if time.monotonic() >= deadline:
                        raise GoogleGatewayError("GOOGLE_TIMEOUT")
                    time.sleep(0.025)
                if os.fstat(output.fileno()).st_size + os.fstat(errors.fileno()).st_size > MAX_BYTES:
                    raise GoogleGatewayError("GOOGLE_OUTPUT_LIMIT")
                output.seek(0)
                errors.seek(0)
                return process.returncode, output.read().decode("utf-8", "replace"), errors.read().decode("utf-8", "replace")
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)


def _error(text: str) -> str:
    lowered = text.lower()
    if any(value in lowered for value in ("please sign in", "not signed in", "unauthenticated", "login required")):
        return "GOOGLE_ACCOUNT_LOGIN_REQUIRED"
    if any(value in lowered for value in ("quota", "rate limit", "resource_exhausted")):
        return "GOOGLE_ACCOUNT_QUOTA_EXHAUSTED"
    return "GOOGLE_ACCOUNT_REQUEST_FAILED"


def list_models(binary: Path | None = None) -> dict:
    code, output, errors = _run(_binary(binary), ["models"], 30)
    if code:
        raise GoogleGatewayError(_error(output + errors))
    models = []
    for line in output.splitlines():
        parts = re.split(r"\s{2,}|\t", line.strip(), maxsplit=1)
        if len(parts) == 2 and IDENTIFIER.fullmatch(parts[0]):
            name = parts[1]
            if len(name) <= 160 and not any(ord(c) < 32 for c in name):
                models.append({"slug": parts[0], "display_name": name})
    if not models or len(models) > 512:
        raise GoogleGatewayError("GOOGLE_ACCOUNT_CATALOG_INVALID")
    return {"provider": "google", "status": "AUTHENTICATED_CATALOG_AVAILABLE",
            "models": models, "model_called": False, "api_key_required": False}


def run_turn(prompt: str, *, binary: Path | None = None, model: str | None = None,
             conversation: str | None = None, timeout: int = 120) -> dict:
    prepare_codex_cloud_task(prompt)  # Common bounded text / secret-marker gate.
    if model is not None and (not isinstance(model, str) or not IDENTIFIER.fullmatch(model)):
        raise GoogleGatewayError("GOOGLE_MODEL_INVALID")
    if conversation is not None and (not isinstance(conversation, str) or not CONVERSATION.fullmatch(conversation)):
        raise GoogleGatewayError("GOOGLE_CONVERSATION_INVALID")
    arguments = ["--mode", "plan", "--disable-slash-commands", "--output-format", "json",
                 "--print-timeout", str(timeout) + "s", "-p", prompt]
    if model:
        arguments += ["--model", model]
    if conversation:
        arguments += ["--conversation", conversation]
    code, output, errors = _run(_binary(binary), arguments, timeout)
    try:
        result = json.loads(output)
    except ValueError:
        raise GoogleGatewayError(_error(errors)) from None
    if (code or not isinstance(result, dict) or result.get("status") != "SUCCESS"):
        raise GoogleGatewayError(_error(errors + (str(result.get("error", "")) if isinstance(result, dict) else "")))
    text, identifier = result.get("response"), result.get("conversation_id")
    if (not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 128 * 1024
            or not isinstance(identifier, str) or not CONVERSATION.fullmatch(identifier)):
        raise GoogleGatewayError("GOOGLE_ACCOUNT_RESPONSE_INVALID")
    return {"status": "MODEL_TURN_COMPLETED_ONLY", "provider": "google",
            "model": model or "provider-default", "text": text, "thread_id": identifier,
            "model_called": True, "permissions_denied": list(DENIED_ACTIONS),
            "github_write_performed": False, "production_authority_delta": 0}
