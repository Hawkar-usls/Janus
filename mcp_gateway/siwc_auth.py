"""Local, owner-operated Sign in with ChatGPT public client.

Nothing in this module is an MCP tool. Only an explicit local ``login`` opens
the system browser. Credentials never appear in status results or exceptions.
Windows encrypts the complete vault with user-bound DPAPI; Unix uses a private
directory and files. The OS account and other processes running as that account
are part of the trusted boundary.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import ctypes
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
import uuid
import webbrowser

import jwt


ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
DISCOVERY_URL = ISSUER + "/.well-known/openid-configuration"
JWKS_URL = ISSUER + "/.well-known/jwks.json"
REVOCATION_URL = ISSUER + "/api/accounts/oauth/revoke"
RESOURCE = "https://api.openai.com/v1"
MODELS_URL = RESOURCE + "/models"
AGENT_NAME = "janus_local_bridge"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
MAX_RESPONSE = 1024 * 1024
MAX_VAULT = 2 * 1024 * 1024
PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
CLIENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,256}$")


class SiwcError(Exception):
    """A fixed error code, without remote bodies, URLs, or credentials."""


class _InvalidGrant(SiwcError):
    """Internal classification for a bounded fresh authorization attempt."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http(url: str, data: dict[str, str] | None = None, *, empty=False,
          access_token: str | None = None):
    """Internal fixed-endpoint HTTPS transport, without redirects or retries."""
    if url not in {DISCOVERY_URL, JWKS_URL, TOKEN_URL, REVOCATION_URL, MODELS_URL}:
        raise SiwcError("SIWC_ENDPOINT_REJECTED")
    if url == MODELS_URL:
        if data is not None or not _secret(access_token) or empty:
            raise SiwcError("SIWC_ENDPOINT_REJECTED")
    elif access_token is not None:
        raise SiwcError("SIWC_ENDPOINT_REJECTED")
    body = urllib.parse.urlencode(data).encode("ascii") if data is not None else None
    request = urllib.request.Request(url, data=body, headers={
        "Accept": "application/json", "User-Agent": AGENT_NAME + "/1.0",
        **({"Content-Type": "application/x-www-form-urlencoded"} if body else {}),
        **({"Authorization": "Bearer " + access_token} if access_token is not None else {}),
    })
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=15) as response:
            if response.status != 200:
                raise SiwcError("SIWC_HTTP_REJECTED")
            raw = response.read(MAX_RESPONSE + 1)
        if len(raw) > MAX_RESPONSE:
            raise SiwcError("SIWC_RESPONSE_TOO_LARGE")
        if empty:
            return None
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise SiwcError("SIWC_RESPONSE_INVALID")
        return result
    except urllib.error.HTTPError as exc:
        code = "SIWC_HTTP_RETRYABLE" if exc.code == 429 or exc.code >= 500 else "SIWC_HTTP_REJECTED"
        if url == MODELS_URL and exc.code == 401:
            code = "SIWC_REAUTHENTICATION_REQUIRED"
        error = None
        if exc.code in {400, 401} and url == TOKEN_URL:
            # Read only to classify a standard error. Never expose its text.
            try:
                error = json.loads(exc.read(MAX_RESPONSE + 1)).get("error")
                if error in {"invalid_grant", "invalid_client"}:
                    code = "SIWC_REAUTHENTICATION_REQUIRED"
            except (ValueError, AttributeError, OSError):
                pass
        exc.close()
        if error == "invalid_grant":
            raise _InvalidGrant(code) from None
        raise SiwcError(code) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise SiwcError("SIWC_NETWORK_ERROR") from None
    except (ValueError, UnicodeError):
        raise SiwcError("SIWC_RESPONSE_INVALID") from None


def _metadata():
    metadata = _http(DISCOVERY_URL)
    if (metadata.get("issuer") != ISSUER or metadata.get("jwks_uri") != JWKS_URL
            or metadata.get("revocation_endpoint") != REVOCATION_URL
            or "RS256" not in metadata.get("id_token_signing_alg_values_supported", [])):
        raise SiwcError("SIWC_DISCOVERY_REJECTED")
    return metadata


def _validated_identity(token: str, client_id: str, nonce: str | None,
                        *, require_nonce: bool = True):
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if header.get("alg") != "RS256" or not isinstance(kid, str) or not 1 <= len(kid) <= 256:
            raise SiwcError("SIWC_ID_TOKEN_INVALID")
        _metadata()
        jwks = _http(JWKS_URL)
        keys = jwks.get("keys")
        if not isinstance(keys, list) or not 1 <= len(keys) <= 64:
            raise SiwcError("SIWC_ID_TOKEN_INVALID")
        matches = [key for key in keys if isinstance(key, dict) and key.get("kid") == kid
                   and key.get("use", "sig") == "sig" and key.get("alg", "RS256") == "RS256"]
        if len(matches) != 1:
            raise SiwcError("SIWC_ID_TOKEN_INVALID")
        key = jwt.PyJWK.from_dict(matches[0], algorithm="RS256").key
        claims = jwt.decode(token, key, algorithms=["RS256"], issuer=ISSUER,
                            audience=client_id, leeway=15,
                            options={"require": ["exp", "iat", "iss", "aud", "sub"]})
        subject = claims.get("sub")
        if not isinstance(subject, str) or not 1 <= len(subject) <= 512:
            raise SiwcError("SIWC_ID_TOKEN_INVALID")
        audience = claims["aud"]
        if ((isinstance(audience, list) and len(audience) > 1)
                or "azp" in claims) and claims.get("azp") != client_id:
            raise SiwcError("SIWC_ID_TOKEN_INVALID")
        if require_nonce or "nonce" in claims:
            received_nonce = claims.get("nonce")
            if (not isinstance(received_nonce, str) or not isinstance(nonce, str)
                    or not secrets.compare_digest(received_nonce, nonce)):
                raise SiwcError("SIWC_NONCE_MISMATCH")
        return claims
    except SiwcError:
        raise
    except (jwt.PyJWTError, TypeError, ValueError, KeyError):
        raise SiwcError("SIWC_ID_TOKEN_INVALID") from None


def _secret(value):
    return isinstance(value, str) and 1 <= len(value) <= 65536 and not any(ord(c) < 32 for c in value)


def _token_record(response: dict, *, previous: dict | None = None):
    if (not isinstance(response, dict) or not isinstance(response.get("token_type"), str)
            or response["token_type"].lower() != "bearer"
            or not _secret(response.get("access_token"))
            or not _secret(response.get("refresh_token"))):
        raise SiwcError("SIWC_TOKEN_RESPONSE_INVALID")
    expires_in = response.get("expires_in")
    scope = response.get("scope")
    if (type(expires_in) is not int or not 1 <= expires_in <= 86400
            or not isinstance(scope, str) or len(scope) > 4096):
        raise SiwcError("SIWC_TOKEN_RESPONSE_INVALID")
    granted = scope.split()
    if any(not re.fullmatch(r"[\x21\x23-\x5b\x5d-\x7e]+", item) for item in granted):
        raise SiwcError("SIWC_TOKEN_RESPONSE_INVALID")
    id_token = response.get("id_token", (previous or {}).get("id_token"))
    if not _secret(id_token):
        raise SiwcError("SIWC_TOKEN_RESPONSE_INVALID")
    now = time.time()
    return {"access_token": response["access_token"], "refresh_token": response["refresh_token"],
            "id_token": id_token, "token_type": "Bearer", "scopes": sorted(set(granted)),
            "expires_at": now + expires_in, "saved_at": now}


def _dpapi(data: bytes, decrypt=False) -> bytes:
    """DPAPI CurrentUser, UI forbidden, with explicit Win32 prototypes."""
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    backing = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    source = Blob(len(data), backing)
    destination = Blob()
    if decrypt:
        fn = crypt.CryptUnprotectData
        fn.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
        args = (ctypes.byref(source), None, None, None, None, 1, ctypes.byref(destination))
    else:
        fn = crypt.CryptProtectData
        fn.argtypes = [ctypes.POINTER(Blob), wintypes.LPCWSTR, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
        args = (ctypes.byref(source), "JANUS SIWC credential vault", None,
                None, None, 1, ctypes.byref(destination))
    fn.restype = wintypes.BOOL
    if not fn(*args):
        raise SiwcError("SIWC_VAULT_PROTECTION_FAILED")
    try:
        return ctypes.string_at(destination.data, destination.size)
    finally:
        kernel.LocalFree(destination.data)


def _windows_private(path: Path):
    """Apply an explicit, protected DACL for the current user and SYSTEM."""
    from ctypes import wintypes
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    token = wintypes.HANDLE()
    advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                          wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    advapi.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    advapi.SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
        raise SiwcError("SIWC_VAULT_PERMISSIONS_FAILED")
    try:
        size = wintypes.DWORD()
        advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not advapi.GetTokenInformation(token, 1, buffer, size, ctypes.byref(size)):
            raise SiwcError("SIWC_VAULT_PERMISSIONS_FAILED")
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        sid_text = wintypes.LPWSTR()
        if not advapi.ConvertSidToStringSidW(sid, ctypes.byref(sid_text)):
            raise SiwcError("SIWC_VAULT_PERMISSIONS_FAILED")
        try:
            sddl = "D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;" + sid_text.value + ")"
        finally:
            kernel.LocalFree(sid_text)
        descriptor = ctypes.c_void_p()
        if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl, 1, ctypes.byref(descriptor), None):
            raise SiwcError("SIWC_VAULT_PERMISSIONS_FAILED")
        try:
            if not advapi.SetFileSecurityW(str(path), 4 | 0x80000000, descriptor):
                raise SiwcError("SIWC_VAULT_PERMISSIONS_FAILED")
        finally:
            kernel.LocalFree(descriptor)
    finally:
        kernel.CloseHandle(token)


class _Vault:
    def __init__(self, directory: Path):
        self.directory = Path(directory).absolute()
        self.path = self.directory / "credentials.v1"
        self.marker = self.directory / ".janus-siwc-vault"

    def check_path(self):
        # Resolve nothing through a symlink, junction, or Windows reparse point.
        # This also prevents an innocent --state-dir mistake from traversing a
        # redirected parent before changing permissions or making directories.
        for component in [*reversed(self.directory.parents), self.directory]:
            try:
                info = component.lstat()
            except FileNotFoundError:
                continue
            if (stat.S_ISLNK(info.st_mode)
                    or getattr(info, "st_file_attributes", 0) & 0x400):
                raise SiwcError("SIWC_VAULT_PATH_REJECTED")
            if not stat.S_ISDIR(info.st_mode):
                raise SiwcError("SIWC_VAULT_PATH_REJECTED")

    def _marker(self):
        try:
            info = self.marker.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_size > 512
                    or info.st_nlink != 1 or self.marker.is_symlink()
                    or getattr(info, "st_file_attributes", 0) & 0x400):
                raise SiwcError("SIWC_VAULT_PATH_REJECTED")
            value = json.loads(self.marker.read_bytes())
            if value.get("schema") != "janus.siwc.vault.v1":
                raise ValueError()
            host = value["host_id"]
            if not isinstance(host, str) or not host.startswith("urn:uuid:"):
                raise ValueError()
            uuid.UUID(host[9:])
            return host
        except SiwcError:
            raise
        except (ValueError, KeyError, TypeError, AttributeError):
            raise SiwcError("SIWC_VAULT_INVALID") from None

    def _prepare(self):
        self.check_path()
        if self.directory.exists() and any(self.directory.iterdir()) and not self.marker.exists():
            raise SiwcError("SIWC_VAULT_DIRECTORY_NOT_OWNED")
        if self.marker.exists() or self.marker.is_symlink():
            self._marker()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or self.directory.is_symlink():
            raise SiwcError("SIWC_VAULT_PATH_REJECTED")
        if os.name == "nt":
            _windows_private(self.directory)
        elif info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise SiwcError("SIWC_VAULT_PERMISSIONS_REJECTED")
        if not self.marker.exists():
            record = {"schema": "janus.siwc.vault.v1", "host_id": "urn:uuid:" + str(uuid.uuid4())}
            try:
                fd = os.open(self.marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                self._marker()
            else:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(json.dumps(record).encode("ascii"))
                    stream.flush()
                    os.fsync(stream.fileno())

    @contextmanager
    def locked(self):
        try:
            self._prepare()
            lock_path = self.directory / "credentials.lock"
            try:
                lock_info = lock_path.lstat()
            except FileNotFoundError:
                lock_info = None
            if lock_info and (not stat.S_ISREG(lock_info.st_mode) or lock_info.st_nlink != 1
                              or getattr(lock_info, "st_file_attributes", 0) & 0x400):
                raise SiwcError("SIWC_VAULT_PATH_REJECTED")
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_nlink != 1:
                    raise SiwcError("SIWC_VAULT_PATH_REJECTED")
                if os.name == "nt":
                    import msvcrt
                    if os.fstat(fd).st_size == 0:
                        os.write(fd, b"0")
                    lock = lambda: msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    unlock = lambda: msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    lock = lambda: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    unlock = lambda: fcntl.flock(fd, fcntl.LOCK_UN)
                deadline = time.monotonic() + 30
                while True:
                    try:
                        os.lseek(fd, 0, os.SEEK_SET)
                        lock()
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise SiwcError("SIWC_VAULT_BUSY") from None
                        time.sleep(0.05)
                try:
                    yield
                finally:
                    os.lseek(fd, 0, os.SEEK_SET)
                    unlock()
            finally:
                os.close(fd)
        except SiwcError:
            raise
        except OSError:
            raise SiwcError("SIWC_VAULT_IO_FAILED") from None

    def read(self):
        try:
            self.check_path()
            if not self.path.exists():
                if self.path.is_symlink():
                    raise SiwcError("SIWC_VAULT_PATH_REJECTED")
                host = self._marker() if self.marker.exists() else "urn:uuid:" + str(uuid.uuid4())
                return {"version": 1, "host_id": host,
                        "active_profile": None, "profiles": {}}
            info = self.path.lstat()
            if (not stat.S_ISREG(info.st_mode) or self.path.is_symlink()
                    or info.st_nlink != 1 or getattr(info, "st_file_attributes", 0) & 0x400):
                raise SiwcError("SIWC_VAULT_PATH_REJECTED")
            if os.name != "nt" and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
                raise SiwcError("SIWC_VAULT_PERMISSIONS_REJECTED")
            if info.st_size > MAX_VAULT:
                raise SiwcError("SIWC_VAULT_INVALID")
            raw = self.path.read_bytes()
            expected = b"JANUS-DPAPI-1\n" if os.name == "nt" else b"JANUS-PRIVATE-1\n"
            if not raw.startswith(expected):
                raise SiwcError("SIWC_VAULT_INVALID")
            raw = raw[len(expected):]
            value = json.loads(_dpapi(raw, decrypt=True) if os.name == "nt" else raw)
            if (not isinstance(value, dict) or value.get("version") != 1
                    or not isinstance(value.get("profiles"), dict)
                    or len(value["profiles"]) > 64
                    or not isinstance(value.get("host_id"), str)
                    or not value["host_id"].startswith("urn:uuid:")):
                raise SiwcError("SIWC_VAULT_INVALID")
            uuid.UUID(value["host_id"][9:])
            if value["host_id"] != self._marker():
                raise SiwcError("SIWC_VAULT_INVALID")
            if value.get("active_profile") is not None and value["active_profile"] not in value["profiles"]:
                raise SiwcError("SIWC_VAULT_INVALID")
            for profile_id, profile in value["profiles"].items():
                if (not PROFILE_ID.fullmatch(profile_id) or not isinstance(profile, dict)
                        or not isinstance(profile.get("client_id"), str)
                        or not CLIENT_ID.fullmatch(profile["client_id"])
                        or profile["client_id"] == "dynamic_agent_client"
                        or not isinstance(profile.get("subject"), str)
                        or profile.get("issuer") != ISSUER
                        or type(profile.get("revision")) is not int):
                    raise SiwcError("SIWC_VAULT_INVALID")
                tokens = profile.get("tokens")
                if tokens is not None and (not isinstance(tokens, dict)
                        or not all(_secret(tokens.get(key)) for key in ("access_token", "refresh_token", "id_token"))
                        or not isinstance(tokens.get("scopes"), list)
                        or not all(isinstance(s, str) for s in tokens["scopes"])
                        or not isinstance(tokens.get("expires_at"), (int, float))):
                    raise SiwcError("SIWC_VAULT_INVALID")
            return value
        except SiwcError:
            raise
        except (ValueError, TypeError, KeyError, UnicodeError):
            raise SiwcError("SIWC_VAULT_INVALID") from None

    def write(self, value):
        raw = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        if len(raw) > MAX_RESPONSE:
            raise SiwcError("SIWC_VAULT_TOO_LARGE")
        raw = b"JANUS-DPAPI-1\n" + _dpapi(raw) if os.name == "nt" else b"JANUS-PRIVATE-1\n" + raw
        fd, name = tempfile.mkstemp(prefix=".credentials-", dir=self.directory)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                if os.name == "nt":
                    _windows_private(temporary)
                else:
                    os.fchmod(stream.fileno(), 0o600)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if self.path.is_symlink():
                raise SiwcError("SIWC_VAULT_PATH_REJECTED")
            os.replace(temporary, self.path)
            if os.name != "nt":
                directory_fd = os.open(self.directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)


def _callback(query: str, expected_state: str, issued_client: str | None):
    try:
        if len(query) > 8192:
            raise ValueError()
        values = urllib.parse.parse_qs(query, keep_blank_values=True,
                                       strict_parsing=True, max_num_fields=20)
        if any(len(value) != 1 for value in values.values()):
            raise ValueError()
        state_value = values.get("state", [None])[0]
        if not isinstance(state_value, str) or not secrets.compare_digest(
                state_value.encode("utf-8"), expected_state.encode("utf-8")):
            raise SiwcError("SIWC_STATE_MISMATCH")
        if "error" in values:
            raise SiwcError("SIWC_CONSENT_DENIED" if values["error"][0] == "access_denied"
                            else "SIWC_AUTHORIZATION_REJECTED")
        code = values.get("code", [None])[0]
        client = values.get("client_id", [issued_client])[0]
        if (not _secret(code) or len(code) > 4096 or not isinstance(client, str)
                or not CLIENT_ID.fullmatch(client) or client == "dynamic_agent_client"):
            raise SiwcError("SIWC_CALLBACK_INVALID")
        if issued_client is not None and client != issued_client:
            raise SiwcError("SIWC_CLIENT_ID_MISMATCH")
        if "iss" in values and values["iss"][0] != ISSUER:
            raise SiwcError("SIWC_ISSUER_MISMATCH")
        return code, client
    except SiwcError:
        raise
    except (ValueError, UnicodeError):
        raise SiwcError("SIWC_CALLBACK_INVALID") from None


def _browser_authorization(parameters: dict, state_value: str, client_id: str | None,
                           browser_open, timeout):
    completed = threading.Event()
    result = []
    claim = threading.Lock()

    class CallbackServer(ThreadingHTTPServer):
        daemon_threads = True
        block_on_close = False

        def __init__(self, *args):
            self.slots = threading.BoundedSemaphore(4)
            super().__init__(*args)

        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(2)
            return connection, address

        def process_request(self, request, address):
            if not self.slots.acquire(blocking=False):
                self.shutdown_request(request)
                return
            try:
                super().process_request(request, address)
            except BaseException:
                self.slots.release()
                raise

        def process_request_thread(self, request, address):
            try:
                super().process_request_thread(request, address)
            finally:
                self.slots.release()

        def handle_error(self, request, address):
            # Never print a callback request or exception with query contents.
            pass

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            # Static text only: never reflect callback data into the browser.
            try:
                parts = urllib.parse.urlsplit(self.path)
            except ValueError:
                self.send_error(400)
                return
            if parts.path != "/auth/callback" or parts.scheme or parts.netloc:
                self.send_error(404)
                return
            with claim:
                if completed.is_set():
                    self.send_error(400)
                    return
                try:
                    if self.headers.get_all("Host") != [f"127.0.0.1:{self.server.server_port}"]:
                        raise SiwcError("SIWC_CALLBACK_INVALID")
                    result.append(_callback(parts.query, state_value, client_id))
                except SiwcError as error:
                    result.append(error)
                completed.set()
            self.send_response(200 if not isinstance(result[0], SiwcError) else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'")
            self.end_headers()
            try:
                self.wfile.write(b"Authorization received. Return to JANUS.\n")
            except OSError:
                pass

    try:
        server = CallbackServer(("127.0.0.1", 0), Handler)
    except OSError:
        raise SiwcError("SIWC_LOOPBACK_UNAVAILABLE") from None
    server.timeout = 0.2
    redirect = f"http://127.0.0.1:{server.server_port}/auth/callback"
    parameters = {**parameters, "redirect_uri": redirect}
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    worker.start()
    try:
        try:
            opened = browser_open(AUTHORIZE_URL + "?" + urllib.parse.urlencode(parameters))
        except Exception:
            raise SiwcError("SIWC_BROWSER_UNAVAILABLE") from None
        if opened is False:
            raise SiwcError("SIWC_BROWSER_UNAVAILABLE")
        if not completed.wait(timeout):
            raise SiwcError("SIWC_AUTHORIZATION_TIMEOUT")
        if isinstance(result[0], SiwcError):
            raise result[0]
        return *result[0], redirect
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)


class SiwcClient:
    def __init__(self, storage_dir: Path):
        self._vault = _Vault(storage_dir)

    @staticmethod
    def _status(value):
        profiles = []
        for profile_id, profile in value["profiles"].items():
            tokens = profile.get("tokens")
            profiles.append({"id": profile_id, "email": profile.get("email"),
                             "connected": bool(tokens),
                             "plan_usage_enabled": bool(tokens and PLAN_SCOPE in tokens["scopes"]),
                             "expires_at": tokens["expires_at"] if tokens else None})
        active_id = value["active_profile"]
        active = value["profiles"].get(active_id, {})
        tokens = active.get("tokens")
        state = "AUTHENTICATION_NOT_CONNECTED"
        if tokens:
            state = "READY" if PLAN_SCOPE in tokens["scopes"] else "PLAN_USAGE_NOT_GRANTED"
            if tokens["expires_at"] <= time.time():
                state = "TOKEN_REFRESH_REQUIRED"
        return {"status": state, "active_profile_id": active_id, "profiles": profiles}

    def status(self):
        # Writers replace the complete vault atomically. A status snapshot can
        # therefore remain read-only, including on a first launch.
        directory = self._vault.directory
        self._vault.check_path()
        if directory.exists() or directory.is_symlink():
            info = directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or directory.is_symlink():
                raise SiwcError("SIWC_VAULT_PATH_REJECTED")
            if os.name != "nt" and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
                raise SiwcError("SIWC_VAULT_PERMISSIONS_REJECTED")
        try:
            return self._status(self._vault.read())
        except OSError:
            raise SiwcError("SIWC_VAULT_IO_FAILED") from None

    def login(self, profile_id: str | None = None, browser_open=webbrowser.open,
              timeout=180):
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 600:
            raise SiwcError("SIWC_TIMEOUT_INVALID")
        with self._vault.locked():
            value = self._vault.read()
            self._vault.write(value)
            if profile_id is not None and (not isinstance(profile_id, str) or profile_id not in value["profiles"]):
                raise SiwcError("SIWC_PROFILE_NOT_FOUND")
            if profile_id is None and len(value["profiles"]) >= 64:
                raise SiwcError("SIWC_PROFILE_LIMIT")
            previous = value["profiles"].get(profile_id)
            host_id = value["host_id"]
        client_id = previous["client_id"] if previous else None
        deadline = time.monotonic() + timeout
        for attempt in range(2):
            state_value, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
            parameters = {"client_id": client_id or "dynamic_agent_client", "ext_agent_host_id": host_id,
                          "response_type": "code", "scope": SCOPES, "resource": RESOURCE,
                          "state": state_value, "nonce": nonce, "code_challenge_method": "S256",
                          "code_challenge": challenge}
            if previous:
                if previous.get("tokens"):
                    parameters["id_token_hint"] = previous["tokens"]["id_token"]
                if previous.get("email"):
                    parameters["login_hint"] = previous["email"]
            if not client_id:
                parameters["agent_name_hint"] = AGENT_NAME
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SiwcError("SIWC_AUTHORIZATION_TIMEOUT")
            code, issued_client, redirect = _browser_authorization(parameters, state_value, client_id,
                                                                  browser_open, remaining)
            try:
                response = _http(TOKEN_URL, {"grant_type": "authorization_code", "client_id": issued_client,
                                            "code": code, "code_verifier": verifier,
                                            "redirect_uri": redirect, "resource": RESOURCE})
                break
            except _InvalidGrant:
                if attempt:
                    raise
                # The code is single-use. Keep the just-issued registration,
                # then start one fresh state/nonce/PKCE flow within the same
                # timeout. No unverified registration becomes an active profile.
                client_id = issued_client
        tokens = _token_record(response)
        identity = _validated_identity(tokens["id_token"], issued_client, nonce)
        if previous and (identity["sub"] != previous["subject"] or identity["iss"] != previous["issuer"]):
            raise SiwcError("SIWC_IDENTITY_MISMATCH")
        email = identity.get("email")
        if not isinstance(email, str) or len(email) > 320 or any(ord(c) < 32 for c in email):
            email = None
        with self._vault.locked():
            value = self._vault.read()
            if value["host_id"] != host_id:
                raise SiwcError("SIWC_SESSION_CHANGED")
            if previous and value["profiles"].get(profile_id, {}).get("revision") != previous["revision"]:
                raise SiwcError("SIWC_SESSION_CHANGED")
            if profile_id is None:
                # Keep registrations separate even when emails are identical.
                existing = [key for key, item in value["profiles"].items()
                            if item["client_id"] == issued_client]
                if existing or len(value["profiles"]) >= 64:
                    raise SiwcError("SIWC_REGISTRATION_CONFLICT")
                profile_id = "p-" + uuid.uuid4().hex[:16]
            value["profiles"][profile_id] = {
                "issuer": identity["iss"], "subject": identity["sub"], "email": email,
                "client_id": issued_client, "nonce": nonce, "tokens": tokens,
                "revision": previous["revision"] + 1 if previous else 1,
            }
            value["active_profile"] = profile_id
            self._vault.write(value)
            return self._status(value)

    def select_profile(self, profile_id):
        with self._vault.locked():
            value = self._vault.read()
            if not isinstance(profile_id, str) or profile_id not in value["profiles"]:
                raise SiwcError("SIWC_PROFILE_NOT_FOUND")
            if not value["profiles"][profile_id].get("tokens"):
                raise SiwcError("SIWC_REAUTHENTICATION_REQUIRED")
            value["active_profile"] = profile_id
            self._vault.write(value)
            return self._status(value)

    def list_models(self, profile_id: str | None = None):
        """Read this registration's current catalog; never perform inference.

        The app-server's bundled ``model/list`` is not an entitlement check.
        Query the fixed public endpoint and expose only validated picker data.
        """
        chosen = self.status()["active_profile_id"] if profile_id is None else profile_id
        if not isinstance(chosen, str) or not PROFILE_ID.fullmatch(chosen):
            raise SiwcError("SIWC_REAUTHENTICATION_REQUIRED")
        token = self.access_token(profile_id=chosen)
        response = _http(MODELS_URL, access_token=token)
        if not isinstance(response, dict):
            raise SiwcError("SIWC_MODELS_RESPONSE_INVALID")
        models = response.get("models")
        if not isinstance(models, list) or len(models) > 512:
            raise SiwcError("SIWC_MODELS_RESPONSE_INVALID")
        choices, seen = [], set()
        for model in models:
            if not isinstance(model, dict):
                raise SiwcError("SIWC_MODELS_RESPONSE_INVALID")
            if model.get("visibility") != "list":
                continue
            slug, name = model.get("slug"), model.get("display_name")
            if (not isinstance(slug, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", slug)
                    or slug in seen or not isinstance(name, str) or not 1 <= len(name) <= 160
                    or not name.strip() or any(unicodedata.category(c).startswith("C") for c in name)):
                raise SiwcError("SIWC_MODELS_RESPONSE_INVALID")
            choices.append({"slug": slug, "display_name": name})
            seen.add(slug)
        return {"status": "ACCOUNT_MODEL_CATALOG", "active_profile_id": chosen,
                "models": choices, "model_called": False}

    def access_token(self, profile_id: str | None = None):
        # Network renewal remains under the cross-process lock: a rotating
        # refresh token can never be consumed concurrently by two clients.
        with self._vault.locked():
            value = self._vault.read()
            chosen = value["active_profile"] if profile_id is None else profile_id
            if not isinstance(chosen, str):
                raise SiwcError("SIWC_REAUTHENTICATION_REQUIRED")
            profile = value["profiles"].get(chosen, {})
            tokens = profile.get("tokens")
            if not tokens:
                raise SiwcError("SIWC_REAUTHENTICATION_REQUIRED")
            if PLAN_SCOPE not in tokens["scopes"]:
                raise SiwcError("SIWC_PLAN_USAGE_NOT_GRANTED")
            if tokens["expires_at"] <= time.time() + 90:
                try:
                    response = _http(TOKEN_URL, {"grant_type": "refresh_token", "client_id": profile["client_id"],
                                                "refresh_token": tokens["refresh_token"], "resource": RESOURCE})
                except SiwcError as error:
                    if str(error) == "SIWC_REAUTHENTICATION_REQUIRED":
                        profile["tokens"] = None
                        profile["revision"] += 1
                        self._vault.write(value)
                    raise
                fresh = _token_record(response, previous=tokens)
                if "id_token" in response:
                    identity = _validated_identity(fresh["id_token"], profile["client_id"],
                                                   profile.get("nonce"), require_nonce=False)
                    if identity["sub"] != profile["subject"] or identity["iss"] != profile["issuer"]:
                        raise SiwcError("SIWC_IDENTITY_MISMATCH")
                profile["tokens"] = fresh
                profile["revision"] += 1
                self._vault.write(value)
                if PLAN_SCOPE not in fresh["scopes"]:
                    raise SiwcError("SIWC_PLAN_USAGE_NOT_GRANTED")
                tokens = fresh
            return tokens["access_token"]

    def logout(self, profile_id=None):
        with self._vault.locked():
            value = self._vault.read()
            chosen = profile_id if profile_id is not None else value["active_profile"]
            if not isinstance(chosen, str) or chosen not in value["profiles"]:
                raise SiwcError("SIWC_PROFILE_NOT_FOUND")
            profile = value["profiles"][chosen]
            tokens = profile.get("tokens")
            revoked = not bool(tokens)
            if tokens:
                try:
                    _metadata()
                    for attempt in range(3):
                        try:
                            _http(REVOCATION_URL, {"token": tokens["refresh_token"],
                                                  "token_type_hint": "refresh_token",
                                                  "client_id": profile["client_id"]}, empty=True)
                            revoked = True
                            break
                        except SiwcError as error:
                            if str(error) not in {"SIWC_NETWORK_ERROR", "SIWC_HTTP_RETRYABLE"} or attempt == 2:
                                break
                            time.sleep(0.1 * (2 ** attempt))
                except SiwcError:
                    pass
            profile["tokens"] = None
            profile.pop("nonce", None)
            profile["revision"] += 1
            if value["active_profile"] == chosen:
                value["active_profile"] = None
            self._vault.write(value)
            return {**self._status(value), "remote_revocation_confirmed": revoked,
                    "signed_out_profile": chosen}
