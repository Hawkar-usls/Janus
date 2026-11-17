"""Owner-invoked, text-only Codex app-server bridge.

This module is deliberately not an MCP tool. It never loads the owner's usual
Codex home and never accepts a command, provider URL, tool, or workspace path
from a model. The app-server binary remains part of the trusted local runtime;
these checks do not sandbox a compromised executable or a same-user OS process.
"""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from typing import Any

from .codex_app_server import validate_turn_completion


class CodexRuntimeError(ValueError):
    """Public errors contain fixed codes, never child output or credentials."""


def _server_error(params: dict, fallback="CODEX_SERVER_ERROR") -> str:
    """Classify known protocol/provider failures without returning free text."""
    error = params.get("error", params)
    if not isinstance(error, dict):
        return fallback
    message = error.get("message", "")
    lowered = message.lower() if isinstance(message, str) else ""
    if "tool" in lowered and ("namespace" in lowered or "additional_tools" in lowered or "grouped" in lowered):
        return "CODEX_PROVIDER_REQUIRES_GROUPED_TOOLS"
    for name in ("client_metadata", "max_output_tokens", "max_tool_calls", "metadata", "temperature",
                 "tool_search", "previous_response_id", "top_p", "system", "conversation", "background"):
        if name in lowered and any(word in lowered for word in ("unsupported", "not supported", "unknown parameter", "not allowed", "invalid")):
            return "CODEX_PROVIDER_UNSUPPORTED_" + name.upper()
    info = error.get("codexErrorInfo")
    codes = {"contextWindowExceeded": "CONTEXT_LIMIT", "sessionBudgetExceeded": "SESSION_BUDGET",
             "usageLimitExceeded": "USAGE_LIMIT", "rateLimitExceeded": "RATE_LIMIT",
             "modelNotFound": "MODEL_NOT_FOUND", "unauthorized": "UNAUTHORIZED", "badRequest": "BAD_REQUEST",
             "serverOverloaded": "OVERLOADED", "internalServerError": "INTERNAL_SERVER_ERROR",
             "sandboxError": "SANDBOX_ERROR", "policyCapacityUnavailable": "POLICY_CAPACITY"}
    if isinstance(info, str) and info in codes:
        return "CODEX_PROVIDER_" + codes[info]
    if isinstance(info, dict):
        for name in ("httpConnectionFailed", "responseStreamConnectionFailed", "responseStreamDisconnected",
                     "responseTooManyFailedAttempts"):
            value = info.get(name)
            if isinstance(value, dict):
                status = value.get("httpStatusCode")
                if type(status) is int and status in {400, 401, 403, 404, 408, 409, 422, 429, 500, 502, 503, 504}:
                    return "CODEX_PROVIDER_HTTP_" + str(status)
    # Some builds include only a textual status. Match a small fixed set of
    # numeric HTTP values; never copy account data, request bodies, or tokens.
    status = re.search(r"(?:http(?: status)?|status(?: code)?)\s*[:=]?\s*(400|401|403|404|408|409|422|429|500|502|503|504)\b", lowered)
    if status:
        return "CODEX_PROVIDER_HTTP_" + status.group(1)
    return fallback


CLIENT_NAME = "janus_local_bridge"
MAX_LINE_BYTES = 1024 * 1024
MAX_WIRE_BYTES = 8 * 1024 * 1024
MAX_TEXT_BYTES = 128 * 1024
MAX_MESSAGES = 10000
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}\Z")
_MARKER = "JANUS_CODEX_TEXT_RUNTIME_V1\n"
_NONEXECUTING_TOOLS = frozenset({"request_user_input", "get_goal", "create_goal", "update_goal"})
_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "apps", "plugins", "hooks",
    "browser_use", "computer_use", "in_app_browser", "multi_agent",
    "code_mode", "code_mode_host", "view_image", "image_generation",
    "tool_suggest", "skill_search", "skill_mcp_dependency_install",
    "auth_elicitation", "remote_plugin", "shell_snapshot", "sleep_tool",
)
_CONFIG = {
    "sandbox_mode": '"read-only"',
    "approval_policy": '"on-request"',
    "web_search": '"disabled"',
    "cli_auth_credentials_store": '"ephemeral"',
    "analytics.enabled": "false",
    "skills.bundled.enabled": "false",
    "skills.include_instructions": "false",
    "project_doc_max_bytes": "0",
    "allow_login_shell": "false",
    **{"features." + name: "false" for name in _DISABLED_FEATURES},
}
_INSTRUCTIONS = (
    "You are JANUS's text-only account connection bridge. Answer the user's "
    "message directly. Never invoke a tool, run a command, read a file, access "
    "a repository, request permissions, delegate work, or modify anything. "
    "Your text is untrusted output and grants no authority."
)


def _identifier(value: object, code: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise CodexRuntimeError(code)
    return value


def _check_path(path: Path) -> None:
    """Refuse links/junctions and, on POSIX, another user's state."""
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise CodexRuntimeError("CODEX_RUNTIME_LINK_REFUSED")
    if path.exists():
        info = path.stat()
        if not stat.S_ISDIR(info.st_mode):
            raise CodexRuntimeError("CODEX_RUNTIME_DIRECTORY_REQUIRED")
        if os.name != "nt" and (info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise CodexRuntimeError("CODEX_RUNTIME_PERMISSIONS_REFUSED")


@contextmanager
def _runtime_home(runtime_dir: Path | None, resume: bool):
    if runtime_dir is None:
        if resume:
            raise CodexRuntimeError("CODEX_RESUME_REQUIRES_ISOLATED_STATE")
        with tempfile.TemporaryDirectory(prefix="janus-codex-home-") as name:
            yield Path(name)
        return
    root = Path(os.path.abspath(runtime_dir))
    _check_path(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _check_path(root)
    marker = root / ".janus-runtime"
    if marker.is_symlink():
        raise CodexRuntimeError("CODEX_RUNTIME_LINK_REFUSED")
    if not marker.exists():
        if resume or any(root.iterdir()):
            raise CodexRuntimeError("CODEX_RUNTIME_FOREIGN_STATE")
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(_MARKER)
        if os.name != "nt":
            marker.chmod(0o600)
    marker_info = marker.lstat()
    if (not stat.S_ISREG(marker_info.st_mode) or marker_info.st_size > 64
            or getattr(marker_info, "st_file_attributes", 0) & 0x400
            or (os.name != "nt" and (marker_info.st_uid != os.getuid() or marker_info.st_mode & 0o077))
            or marker.read_text(encoding="utf-8") != _MARKER):
        raise CodexRuntimeError("CODEX_RUNTIME_FOREIGN_STATE")
    lock = root / ".in-use"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise CodexRuntimeError("CODEX_RUNTIME_BUSY") from None
    os.close(descriptor)
    try:
        # The state is reserved for this bridge; do not load a hand-written
        # config, hooks, or MCP installation from an earlier invocation.
        home = root / "codex-home"
        _check_path(home)
        home.mkdir(mode=0o700, exist_ok=True)
        for forbidden in ("config.toml", "hooks.json", "skills", "plugins"):
            if (home / forbidden).exists() or (home / forbidden).is_symlink():
                raise CodexRuntimeError("CODEX_RUNTIME_FOREIGN_CONFIG")
        yield home
    finally:
        lock.unlink()


def _thread_manifest(runtime_dir: Path, thread_id: str | None = None, *, save=False):
    path = Path(runtime_dir) / "threads.json"
    threads = []
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_size > 65536
                or getattr(info, "st_file_attributes", 0) & 0x400
                or (os.name != "nt" and (info.st_uid != os.getuid() or info.st_mode & 0o077))):
            raise CodexRuntimeError("CODEX_RUNTIME_FOREIGN_STATE")
        try:
            threads = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            raise CodexRuntimeError("CODEX_RUNTIME_FOREIGN_STATE") from None
        if not isinstance(threads, list) or len(threads) > 100:
            raise CodexRuntimeError("CODEX_RUNTIME_FOREIGN_STATE")
        for value in threads:
            _identifier(value, "CODEX_RUNTIME_FOREIGN_STATE")
    if not save:
        if thread_id not in threads:
            raise CodexRuntimeError("CODEX_THREAD_NOT_OWNED")
        return
    if thread_id not in threads:
        threads = (threads + [thread_id])[-100:]
        descriptor, name = tempfile.mkstemp(prefix=".threads-", dir=runtime_dir)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(threads, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)


def _child_env(access_token: str, code_home: Path, work: Path) -> dict[str, str]:
    permitted = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "COMSPEC"}
    env = {key: value for key, value in os.environ.items() if key.upper() in permitted}
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP"):
        env[key] = str(work)
    env["CODEX_HOME"] = str(code_home)
    env["ACCESS_TOKEN"] = access_token
    env["NO_COLOR"] = "1"
    return env


def _argv(executable: str) -> list[str]:
    # Import locally so no credential is ever interpolated into command text.
    from .codex_app_server import app_server_argv
    result = app_server_argv()
    result[0] = executable
    result += ["--strict-config"]
    for name, value in _CONFIG.items():
        result += ["-c", name + "=" + value]
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CodexRuntimeError("CODEX_INVALID_JSON_ENVELOPE")
        result[key] = value
    return result


class _Rpc:
    def __init__(self, process, deadline: float):
        self.process = process
        self.deadline = deadline
        self.incoming: queue.Queue = queue.Queue(maxsize=32)
        self.stop = threading.Event()
        self.wire_bytes = 0
        self.messages = 0
        self.next_id = 0
        self.thread_id: str | None = None
        self.turn_id: str | None = None
        self.pending: list[dict] = []
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()

    def _put(self, value):
        while not self.stop.is_set():
            try:
                self.incoming.put(value, timeout=0.1)
                return
            except queue.Full:
                continue

    def _read_stdout(self):
        try:
            while not self.stop.is_set():
                line = self.process.stdout.readline(MAX_LINE_BYTES + 1)
                if not line:
                    self._put("CODEX_PROCESS_ENDED")
                    return
                if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
                    self._put("CODEX_OUTPUT_LIMIT")
                    return
                self._put(line)
        except (OSError, ValueError):
            self._put("CODEX_PROCESS_IO_ERROR")

    def send(self, value: dict):
        # A child that stops reading must not defeat the overall deadline by
        # filling the OS stdin pipe. Killing it in the caller unblocks writes.
        data = (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")
        done: queue.Queue = queue.Queue(maxsize=1)

        def write():
            try:
                self.process.stdin.write(data)
                self.process.stdin.flush()
                done.put(True)
            except (OSError, ValueError):
                done.put(False)

        threading.Thread(target=write, daemon=True).start()
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise CodexRuntimeError("CODEX_TIMEOUT")
        try:
            written = done.get(timeout=remaining)
        except queue.Empty:
            raise CodexRuntimeError("CODEX_TIMEOUT") from None
        if not written:
            raise CodexRuntimeError("CODEX_PROCESS_IO_ERROR")

    def receive(self) -> dict:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise CodexRuntimeError("CODEX_TIMEOUT")
        try:
            line = self.incoming.get(timeout=remaining)
        except queue.Empty:
            raise CodexRuntimeError("CODEX_TIMEOUT") from None
        if isinstance(line, str):
            raise CodexRuntimeError(line)
        self.wire_bytes += len(line)
        self.messages += 1
        if self.wire_bytes > MAX_WIRE_BYTES or self.messages > MAX_MESSAGES:
            raise CodexRuntimeError("CODEX_OUTPUT_LIMIT")
        try:
            message = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object)
        except (UnicodeError, json.JSONDecodeError, RecursionError):
            raise CodexRuntimeError("CODEX_INVALID_JSON_ENVELOPE") from None
        if not isinstance(message, dict):
            raise CodexRuntimeError("CODEX_INVALID_JSON_ENVELOPE")
        if "method" in message and "id" in message:
            # Deny every server-side tool, approval, elicitation and future
            # request. Never return an approving response to an unknown method.
            self.send({"id": message["id"], "error": {
                "code": -32601, "message": "JANUS_TEXT_ONLY_REQUEST_DENIED"}})
            raise CodexRuntimeError("CODEX_TOOL_REQUEST_DENIED")
        if "method" in message:
            self._check_notification(message)
        return message

    def _check_notification(self, message: dict):
        method = message["method"]
        if not isinstance(method, str):
            raise CodexRuntimeError("CODEX_INVALID_JSON_ENVELOPE")
        params = message.get("params", {})
        if not isinstance(params, dict):
            raise CodexRuntimeError("CODEX_INVALID_JSON_ENVELOPE")
        if "threadId" in params and self.thread_id and params["threadId"] != self.thread_id:
            raise CodexRuntimeError("CODEX_THREAD_MISMATCH")
        # Resuming a thread replays its latest usage snapshot, whose turnId
        # belongs to a previous turn. It is metadata, never output/completion.
        if (method != "thread/tokenUsage/updated" and "turnId" in params
                and self.turn_id and params["turnId"] != self.turn_id):
            raise CodexRuntimeError("CODEX_TURN_MISMATCH")
        if method.startswith("item/") and method not in {
            "item/started", "item/completed", "item/agentMessage/delta",
            "item/reasoning/summaryTextDelta", "item/reasoning/textDelta",
            "item/reasoning/summaryPartAdded",
        }:
            raise CodexRuntimeError("CODEX_TOOL_EVENT_DENIED")
        if method in {"item/started", "item/completed"}:
            item = params.get("item")
            if not isinstance(item, dict) or item.get("type") not in {
                "agentMessage", "userMessage", "reasoning",
            }:
                raise CodexRuntimeError("CODEX_TOOL_EVENT_DENIED")
        if method == "error":
            raise CodexRuntimeError(_server_error(params))

    def request(self, method: str, params: dict) -> dict:
        self.next_id += 1
        request_id = self.next_id
        self.send({"id": request_id, "method": method, "params": params})
        while True:
            message = self.receive()
            if "id" in message:
                if type(message["id"]) is not int or message["id"] != request_id:
                    raise CodexRuntimeError("CODEX_RESPONSE_ID_MISMATCH")
                if "error" in message:
                    raise CodexRuntimeError(_server_error(message, "CODEX_RPC_FAILED"))
                result = message.get("result")
                if not isinstance(result, dict):
                    raise CodexRuntimeError("CODEX_INVALID_RPC_RESULT")
                return result
            # Turn notifications can race the turn/start response. Delay their
            # acceptance until both thread and turn IDs are known.
            if method == "turn/start":
                self.pending.append(message)
                if len(self.pending) > 64:
                    raise CodexRuntimeError("CODEX_OUTPUT_LIMIT")
            elif message.get("method") == "turn/completed":
                raise CodexRuntimeError("CODEX_UNEXPECTED_COMPLETION")


def _effective_config(result: dict, provider_url: str = "https://api.openai.com/v1") -> None:
    config = result.get("config")
    if not isinstance(config, dict):
        raise CodexRuntimeError("CODEX_UNSAFE_EFFECTIVE_CONFIG")
    features = config.get("features")
    if not isinstance(features, dict) or any(features.get(name) is not False for name in _DISABLED_FEATURES):
        raise CodexRuntimeError("CODEX_UNSAFE_EFFECTIVE_CONFIG")
    if (config.get("sandbox_mode") != "read-only"
            or config.get("approval_policy") != "on-request"
            or config.get("web_search") != "disabled"
            or config.get("cli_auth_credentials_store") != "ephemeral"
            or config.get("mcp_servers") not in (None, {})
            or config.get("hooks") not in (None, {})):
        raise CodexRuntimeError("CODEX_UNSAFE_EFFECTIVE_CONFIG")
    providers = config.get("model_providers")
    provider = providers.get("openai_chatgpt_plan") if isinstance(providers, dict) else None
    expected = {"base_url": provider_url, "env_key": "ACCESS_TOKEN", "wire_api": "responses",
                "requires_openai_auth": False, "supports_websockets": False}
    if (config.get("model_provider") != "openai_chatgpt_plan" or not isinstance(provider, dict)
            or any(provider.get(key) != value for key, value in expected.items())
            or any(provider.get(key) not in (None, {}) for key in
                   ("http_headers", "env_http_headers", "query_params", "auth"))
            or provider.get("experimental_bearer_token") is not None
            or provider.get("model_catalog_url") is not None):
        raise CodexRuntimeError("CODEX_UNSAFE_PROVIDER")
    skills = config.get("skills", {})
    if (not isinstance(skills, dict) or not isinstance(skills.get("bundled"), dict)
            or skills["bundled"].get("enabled") is not False
            or skills.get("include_instructions") is not False):
        raise CodexRuntimeError("CODEX_UNSAFE_EFFECTIVE_CONFIG")


def _default_model(rpc: _Rpc) -> str:
    defaults = []
    cursor = None
    for _ in range(10):
        result = rpc.request("model/list", {"limit": 100, "includeHidden": False, "cursor": cursor})
        data = result.get("data")
        if not isinstance(data, list):
            raise CodexRuntimeError("CODEX_MODEL_SELECTION_REQUIRED")
        for entry in data:
            if isinstance(entry, dict) and entry.get("isDefault") is True:
                defaults.append(_identifier(entry.get("model"), "CODEX_MODEL_SELECTION_REQUIRED"))
        cursor = result.get("nextCursor")
        if cursor is None:
            if len(defaults) != 1:
                raise CodexRuntimeError("CODEX_MODEL_SELECTION_REQUIRED")
            return defaults[0]
        if not isinstance(cursor, str) or len(cursor) > 4096:
            break
    raise CodexRuntimeError("CODEX_MODEL_SELECTION_REQUIRED")


def _completion(rpc: _Rpc, access_token: str, model: str) -> dict[str, Any]:
    parts: dict[str, str] = {}
    text_size = 0
    while True:
        message = rpc.pending.pop(0) if rpc.pending else rpc.receive()
        rpc._check_notification(message)
        method, params = message.get("method"), message.get("params", {})
        if "id" in message:
            raise CodexRuntimeError("CODEX_RESPONSE_ID_MISMATCH")
        if method == "item/agentMessage/delta":
            if params.get("threadId") != rpc.thread_id or params.get("turnId") != rpc.turn_id:
                raise CodexRuntimeError("CODEX_TURN_MISMATCH")
            item_id = _identifier(params.get("itemId"), "CODEX_INVALID_MESSAGE")
            delta = params.get("delta")
            if not isinstance(delta, str):
                raise CodexRuntimeError("CODEX_INVALID_MESSAGE")
            text_size += len(delta.encode("utf-8"))
            if text_size > MAX_TEXT_BYTES:
                raise CodexRuntimeError("CODEX_OUTPUT_LIMIT")
            parts[item_id] = parts.get(item_id, "") + delta
        elif method == "turn/started":
            turn = params.get("turn")
            if (params.get("threadId") != rpc.thread_id or not isinstance(turn, dict)
                    or turn.get("id") != rpc.turn_id):
                raise CodexRuntimeError("CODEX_TURN_MISMATCH")
        elif method == "turn/completed":
            turn = params.get("turn")
            if (params.get("threadId") != rpc.thread_id or not isinstance(turn, dict)
                    or turn.get("id") != rpc.turn_id):
                raise CodexRuntimeError("CODEX_TURN_MISMATCH")
            if turn.get("error") is not None:
                raise CodexRuntimeError(_server_error(turn, "CODEX_TURN_FAILED"))
            items = turn.get("items", [])
            if not isinstance(items, list):
                raise CodexRuntimeError("CODEX_INVALID_MESSAGE")
            for item in items:
                if not isinstance(item, dict) or item.get("type") not in {"agentMessage", "userMessage", "reasoning"}:
                    raise CodexRuntimeError("CODEX_TOOL_EVENT_DENIED")
                if item.get("type") == "agentMessage" and isinstance(item.get("text"), str):
                    item_id = _identifier(item.get("id"), "CODEX_INVALID_MESSAGE")
                    parts[item_id] = item["text"]
            try:
                result = validate_turn_completion(message)
            except ValueError:
                raise CodexRuntimeError("CODEX_TURN_NOT_SUCCESSFUL") from None
            text = "\n".join(parts.values())
            if not text or len(text.encode("utf-8")) > MAX_TEXT_BYTES:
                raise CodexRuntimeError("CODEX_EMPTY_OR_EXCESSIVE_RESPONSE")
            result.update({"turn_id": rpc.turn_id, "model": model,
                           "text": text.replace(access_token, "[REDACTED]"),
                           "model_called": True, "execution_tools_enabled": False,
                           "tool_calls_accepted": False})
            return result


def _resolve_binary(codex_binary: Path | None) -> str:
    executable = str(codex_binary) if codex_binary is not None else shutil.which("codex")
    if (not executable or not Path(executable).is_file()
            or (codex_binary is not None and not Path(executable).is_absolute())
            or (os.name == "nt" and Path(executable).suffix.lower() != ".exe")):
        raise CodexRuntimeError("CODEX_BINARY_UNAVAILABLE")
    return os.path.abspath(executable)


def _reduced_catalog(value: dict) -> dict:
    """Reduce trusted bundled tool metadata, preserving exact model identities."""
    models = value.get("models") if isinstance(value, dict) else None
    if not isinstance(models, list) or not 1 <= len(models) <= 512:
        raise CodexRuntimeError("CODEX_BUNDLED_CATALOG_INVALID")
    reduced, seen = [], set()
    for model in models:
        if not isinstance(model, dict):
            raise CodexRuntimeError("CODEX_BUNDLED_CATALOG_INVALID")
        slug = _identifier(model.get("slug"), "CODEX_BUNDLED_CATALOG_INVALID")
        if slug in seen:
            raise CodexRuntimeError("CODEX_BUNDLED_CATALOG_INVALID")
        seen.add(slug)
        reduced.append({**model, "tool_mode": "direct", "use_responses_lite": True,
                        "apply_patch_tool_type": None, "experimental_supported_tools": [],
                        "node_repl_disabled": True, "multi_agent_version": None,
                        "multi_agent_reasoning_effort": None})
    return {"models": reduced}


def _export_catalog(executable: str, home: Path, work: Path, timeout: float) -> dict:
    """Offline native export; no token, refresh, network discovery or user config."""
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    # Bound output on disk while the process runs; a broken binary must not
    # allocate unbounded capture_output memory. The executable is trusted.
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(
            [executable, "debug", "models", "--bundled"], cwd=str(work),
            env=_child_env("janus-synthetic-catalog-export", home, work),
            stdout=output, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
            shell=False, creationflags=flags,
        )
        deadline = time.monotonic() + min(timeout, 15)
        try:
            while process.poll() is None:
                if os.fstat(output.fileno()).st_size > MAX_WIRE_BYTES:
                    raise CodexRuntimeError("CODEX_BUNDLED_CATALOG_INVALID")
                if time.monotonic() >= deadline:
                    raise CodexRuntimeError("CODEX_TIMEOUT")
                time.sleep(0.02)
            if process.returncode != 0 or os.fstat(output.fileno()).st_size > MAX_WIRE_BYTES:
                raise CodexRuntimeError("CODEX_BUNDLED_CATALOG_UNAVAILABLE")
            output.seek(0)
            try:
                value = json.load(output, object_pairs_hook=_unique_object)
            except (ValueError, UnicodeError, RecursionError):
                raise CodexRuntimeError("CODEX_BUNDLED_CATALOG_INVALID") from None
            return _reduced_catalog(value)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def _execute(access_token: str, model: str | None, prompt: str, *,
             timeout: float = 120, thread_id: str | None = None,
             runtime_dir: Path | None = None, codex_binary: Path | None = None,
             _provider_url: str = "https://api.openai.com/v1") -> dict[str, Any]:
    """Run one bounded text turn after the caller validates OAuth consent.

    Persistent state, when requested, must be dedicated to one validated JANUS
    account profile. No ordinary Codex directory or prior Codex session is read.
    Re-entering this function always starts a new child with the caller's fresh
    access token, so an expired child token is never reused after refresh.
    """
    if (not isinstance(access_token, str) or not access_token
            or len(access_token) > 32768 or any(ch.isspace() for ch in access_token)):
        raise CodexRuntimeError("CODEX_ACCESS_TOKEN_REQUIRED")
    if model is not None:
        _identifier(model, "CODEX_INVALID_MODEL")
    if thread_id is not None:
        _identifier(thread_id, "CODEX_INVALID_THREAD")
    if not isinstance(prompt, str) or not prompt.strip() or "\x00" in prompt or len(prompt.encode("utf-8")) > 16384:
        raise CodexRuntimeError("CODEX_INVALID_PROMPT")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise CodexRuntimeError("CODEX_INVALID_TIMEOUT")
    executable = _resolve_binary(codex_binary)
    process, rpc = None, None
    try:
        with _runtime_home(runtime_dir, thread_id is not None) as code_home:
            if thread_id is not None:
                _thread_manifest(runtime_dir, thread_id)
            with tempfile.TemporaryDirectory(prefix="janus-codex-work-") as scratch:
                work = Path(scratch)
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                started_at = time.monotonic()
                catalog = _export_catalog(executable, code_home, work, timeout)
                if model is not None and model not in {entry["slug"] for entry in catalog["models"]}:
                    raise CodexRuntimeError("CODEX_MODEL_NOT_IN_BUNDLED_CATALOG")
                catalog_path = work / "text-models.json"
                catalog_path.write_text(json.dumps(catalog, ensure_ascii=True), encoding="utf-8")
                argv = _argv(executable)
                argv += ["-c", "model_catalog_json=" + json.dumps(str(catalog_path))]
                if _provider_url != "https://api.openai.com/v1":
                    argv += ["-c", 'model_providers.openai_chatgpt_plan.base_url="' + _provider_url + '"',
                             "-c", "features.enable_request_compression=false",
                             "-c", "model_providers.openai_chatgpt_plan.request_max_retries=0",
                             "-c", "model_providers.openai_chatgpt_plan.stream_max_retries=0"]
                process = subprocess.Popen(
                    argv, cwd=str(work), env=_child_env(access_token, code_home, work),
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    shell=False, creationflags=flags,
                )
                rpc = _Rpc(process, started_at + timeout)
                try:
                    rpc.request("initialize", {"clientInfo": {
                        "name": CLIENT_NAME, "title": "JANUS local bridge", "version": "0.2.0"}})
                    rpc.send({"method": "initialized"})
                    _effective_config(rpc.request("config/read", {"includeLayers": False, "cwd": str(work)}), _provider_url)
                    mcp = rpc.request("mcpServerStatus/list", {"limit": 100})
                    if mcp.get("data") != [] or mcp.get("nextCursor") is not None:
                        raise CodexRuntimeError("CODEX_MCP_CONNECTION_REFUSED")
                    selected = model or _default_model(rpc)
                    params = {"model": selected, "modelProvider": "openai_chatgpt_plan",
                              "cwd": str(work), "sandbox": "read-only", "approvalPolicy": "on-request",
                              "baseInstructions": _INSTRUCTIONS, "developerInstructions": _INSTRUCTIONS}
                    if thread_id:
                        params["threadId"] = thread_id
                        response = rpc.request("thread/resume", params)
                    else:
                        response = rpc.request("thread/start", params)
                    thread = response.get("thread")
                    if not isinstance(thread, dict):
                        raise CodexRuntimeError("CODEX_INVALID_THREAD")
                    rpc.thread_id = _identifier(thread.get("id"), "CODEX_INVALID_THREAD")
                    if thread_id and rpc.thread_id != thread_id:
                        raise CodexRuntimeError("CODEX_THREAD_MISMATCH")
                    if response.get("model") != selected or response.get("modelProvider") != "openai_chatgpt_plan":
                        raise CodexRuntimeError("CODEX_MODEL_MISMATCH")
                    sandbox = response.get("sandbox")
                    if (not isinstance(sandbox, dict) or sandbox.get("type") != "readOnly"
                            or response.get("approvalPolicy") != "on-request"):
                        raise CodexRuntimeError("CODEX_UNSAFE_THREAD_CONFIG")
                    if runtime_dir is not None:
                        _thread_manifest(runtime_dir, rpc.thread_id, save=True)
                    started = rpc.request("turn/start", {"threadId": rpc.thread_id,
                        "input": [{"type": "text", "text": prompt}], "model": selected,
                        "approvalPolicy": "on-request", "sandboxPolicy": {"type": "readOnly"}})
                    turn = started.get("turn")
                    if not isinstance(turn, dict):
                        raise CodexRuntimeError("CODEX_INVALID_TURN")
                    rpc.turn_id = _identifier(turn.get("id"), "CODEX_INVALID_TURN")
                    return _completion(rpc, access_token, selected)
                finally:
                    rpc.stop.set()
                    process.kill()
                    process.wait(timeout=5)
                    process.stdin.close()
                    process.stdout.close()
                    rpc.reader.join(timeout=1)
    except CodexRuntimeError:
        raise
    except (OSError, ValueError, subprocess.SubprocessError):
        raise CodexRuntimeError("CODEX_LOCAL_RUNTIME_FAILED") from None


def _metadata_schema(name: str) -> dict:
    def obj(properties, required):
        return {"type": "object", "properties": properties, "required": required,
                "additionalProperties": False}
    string = {"type": "string"}
    schemas = {
        "get_goal": obj({}, []),
        "create_goal": obj({"objective": string, "token_budget": {"type": "integer"}}, ["objective"]),
        "update_goal": obj({"status": {"type": "string", "enum": ["complete", "blocked", "paused"]}}, ["status"]),
        "request_user_input": obj({"questions": {"type": "array", "items": obj({
            "id": string, "header": string, "question": string,
            "options": {"type": "array", "items": obj({"label": string, "description": string},
                                                        ["label", "description"])},
        }, ["id", "header", "question", "options"])}}, ["questions"]),
    }
    return schemas[name]


def _schema_shape(value):
    if isinstance(value, dict):
        return {key: sorted(item) if key in {"required", "enum"} and isinstance(item, list)
                else _schema_shape(item) for key, item in value.items() if key != "description"}
    if isinstance(value, list):
        return [_schema_shape(item) for item in value]
    return value


def _catalog_tools(body: dict) -> list[str]:
    """Allow only fixed native interaction/goal metadata schemas, no execution.

    These four native tools can affect the bridge's disposable local goal/UI
    metadata. A single functions namespace is allowed for the SIWC Responses
    route. Commands, patches, nested/other namespaces, unknown tools and
    expanded same-name schemas fail compatibility.
    """
    names: list[str] = []

    def inspect(tools, *, in_namespace=False):
        if not isinstance(tools, list):
            raise CodexRuntimeError("CODEX_INVALID_TOOL_CATALOG")
        for tool in tools:
            if not isinstance(tool, dict):
                raise CodexRuntimeError("CODEX_INVALID_TOOL_CATALOG")
            name = tool.get("name", tool.get("type", "unknown"))
            if not isinstance(name, str) or len(name) > 200:
                raise CodexRuntimeError("CODEX_INVALID_TOOL_CATALOG")
            if tool.get("type") == "namespace":
                if name != "functions" or in_namespace:
                    names.append("namespace:" + name)
                inspect(tool.get("tools"), in_namespace=True)
            else:
                if (name in _NONEXECUTING_TOOLS
                        and (tool.get("type") != "function" or
                             _schema_shape(tool.get("parameters")) != _schema_shape(_metadata_schema(name)))):
                    names.append("schema_mismatch:" + name)
                else:
                    names.append(name)
            if len(names) > 1000:
                raise CodexRuntimeError("CODEX_INVALID_TOOL_CATALOG")

    inspect(body.get("tools", []))
    inputs = body.get("input")
    if not isinstance(inputs, list):
        raise CodexRuntimeError("CODEX_INVALID_TOOL_CATALOG")
    for item in inputs:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            inspect(item.get("tools"))
    if len(names) != len(set(names)):
        raise CodexRuntimeError("CODEX_DUPLICATE_TOOL_CATALOG")
    return names


def check_runtime(codex_binary: Path | None = None, *, timeout: float = 30,
                  model: str | None = None) -> dict[str, Any]:
    """Probe the installed binary against a loopback-only rejecting provider.

    No OAuth credential is read and no request is forwarded. Even valid-looking
    config can leave native tools in a binary's request; inspect its real wire
    catalog using a synthetic bearer token before permitting a live child.
    """
    executable = _resolve_binary(codex_binary)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise CodexRuntimeError("CODEX_INVALID_TIMEOUT")
    if model is not None:
        _identifier(model, "CODEX_INVALID_MODEL")
    captured: queue.Queue = queue.Queue(maxsize=1)
    capability = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if (self.path != "/v1/responses" or not 0 < length <= MAX_WIRE_BYTES
                        or self.headers.get("Authorization") != "Bearer " + capability
                        or self.headers.get("Content-Encoding") not in (None, "identity")):
                    self.send_error(400)
                    return
                body = json.loads(self.rfile.read(length), object_pairs_hook=_unique_object)
                if not isinstance(body, dict):
                    self.send_error(400)
                    return
                try:
                    captured.put_nowait(body)
                except queue.Full:
                    pass
                response = b'{"error":{"message":"JANUS_OFFLINE_COMPATIBILITY_PROBE"}}'
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            except (ValueError, OSError, CodexRuntimeError):
                return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = min(timeout, 1)
    # A malicious local peer must not hold the one-shot probe HTTP reader open.
    original_get_request = server.get_request

    def get_request():
        connection, address = original_get_request()
        connection.settimeout(min(timeout, 2))
        return connection, address

    server.get_request = get_request
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    worker.start()
    error = None
    try:
        try:
            _execute(capability, model, "Reply exactly JANUS_OFFLINE_PROBE.", timeout=timeout,
                     codex_binary=Path(executable),
                     _provider_url="http://127.0.0.1:" + str(server.server_port) + "/v1")
        except CodexRuntimeError as exc:
            error = str(exc)
        try:
            body = captured.get_nowait()
        except queue.Empty:
            return {"status": "RUNTIME_INCOMPATIBLE", "compatible": False,
                    "error": error or "CODEX_PROBE_REQUEST_NOT_OBSERVED",
                    "model_called": False, "external_request_sent": False}
        names = _catalog_tools(body)
        selected = _identifier(body.get("model"), "CODEX_MODEL_SELECTION_REQUIRED")
        if body.get("store") is not False or body.get("stream") is not True:
            return {"status": "RUNTIME_INCOMPATIBLE", "compatible": False,
                    "error": "CODEX_UNSAFE_INFERENCE_OPTIONS", "model_called": False,
                    "external_request_sent": False}
        unexpected = [name for name in names if name not in _NONEXECUTING_TOOLS]
        return {"status": "UNSAFE_TOOL_CATALOG" if unexpected else "TEXT_ONLY_COMPATIBLE",
                "compatible": not unexpected, "model": selected, "tool_names": names,
                "catalog_nonexecuting_tools": [name for name in names if name in _NONEXECUTING_TOOLS],
                "execution_tools_enabled": bool(unexpected), "tool_calls_accepted": False,
                "model_called": False, "external_request_sent": False}
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def run_turn(access_token: str, model: str | None, prompt: str, *,
             timeout: float = 120, thread_id: str | None = None,
             runtime_dir: Path | None = None, codex_binary: Path | None = None) -> dict[str, Any]:
    """Validate a synthetic wire probe before creating a credentialed child."""
    if model is not None:
        _identifier(model, "CODEX_INVALID_MODEL")
    if thread_id is not None:
        _identifier(thread_id, "CODEX_INVALID_THREAD")
    if (not isinstance(prompt, str) or not prompt.strip() or "\x00" in prompt
            or len(prompt.encode("utf-8")) > 16384):
        raise CodexRuntimeError("CODEX_INVALID_PROMPT")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise CodexRuntimeError("CODEX_INVALID_TIMEOUT")
    started_at = time.monotonic()
    executable = _resolve_binary(codex_binary)
    compatibility = check_runtime(Path(executable), timeout=min(timeout, 30), model=model)
    if compatibility.get("compatible") is not True:
        raise CodexRuntimeError("CODEX_TEXT_ONLY_RUNTIME_INCOMPATIBLE")
    remaining = timeout - (time.monotonic() - started_at)
    if remaining <= 0:
        raise CodexRuntimeError("CODEX_TIMEOUT")
    result = _execute(access_token, compatibility["model"], prompt, timeout=remaining,
                      thread_id=thread_id, runtime_dir=runtime_dir, codex_binary=Path(executable))
    result["catalog_nonexecuting_tools"] = compatibility.get("catalog_nonexecuting_tools", [])
    return result
