"""Offline OAuth controls: local loopback only, signed fixture ID tokens.

No test opens the real browser, contacts OpenAI, or uses a real credential.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import urllib.error
import urllib.parse
import urllib.request

from cryptography.hazmat.primitives.asymmetric import rsa
import jwt

from mcp_gateway import siwc_auth as auth


class SiwcAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(cls.key.public_key()))
        cls.jwk.update(kid="offline-key", alg="RS256", use="sig")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "private-vault"
        self.client = auth.SiwcClient(self.directory)
        self.authorizations = []
        self.calls = []
        self.client_id = "oaiapp_offline_a"
        self.subject = "offline-subject-a"
        self.email = "same-account@example.invalid"
        self.scopes = auth.SCOPES
        self.response_overrides = {}
        self.claim_overrides = {}
        self.exchange_error = None
        self.browser_query = None
        self.refresh_count = 0
        self.revocation_error = None
        self.model_response = {"models": []}
        self.expected_catalog_token = "offline-access-token"
        self.http_patch = patch.object(auth, "_http", side_effect=self.http)
        self.http_patch.start()
        self.addCleanup(self.http_patch.stop)
        self.addCleanup(self.temporary.cleanup)

    def signed(self, *, nonce="offline-nonce", client_id=None, key=None, **overrides):
        claims = {"iss": auth.ISSUER, "sub": self.subject,
                  "aud": client_id or self.client_id,
                  "exp": int(time.time()) + 600, "iat": int(time.time()),
                  "nonce": nonce, "email": self.email}
        claims.update(overrides)
        return jwt.encode(claims, key or self.key, algorithm="RS256", headers={"kid": "offline-key"})

    def http(self, url, data=None, *, empty=False, access_token=None):
        self.calls.append((url, data, empty))
        if url == auth.DISCOVERY_URL:
            return {"issuer": auth.ISSUER, "jwks_uri": auth.JWKS_URL,
                    "revocation_endpoint": auth.REVOCATION_URL,
                    "id_token_signing_alg_values_supported": ["RS256"]}
        if url == auth.JWKS_URL:
            return {"keys": [self.jwk]}
        if url == auth.TOKEN_URL:
            if self.exchange_error:
                raise auth.SiwcError(self.exchange_error)
            if data["grant_type"] == "authorization_code":
                request = self.authorizations[-1]
                self.assertEqual(data["client_id"], self.client_id)
                self.assertEqual(data["resource"], auth.RESOURCE)
                self.assertEqual(data["redirect_uri"], request["redirect_uri"])
                challenge = base64.urlsafe_b64encode(hashlib.sha256(
                    data["code_verifier"].encode("ascii")).digest()).rstrip(b"=").decode("ascii")
                self.assertEqual(challenge, request["code_challenge"])
                response = {"access_token": "offline-access-token", "refresh_token": "offline-refresh-token",
                            "id_token": self.signed(nonce=request["nonce"], **self.claim_overrides),
                            "token_type": "Bearer", "expires_in": 3600, "scope": self.scopes}
            else:
                self.assertEqual(data["grant_type"], "refresh_token")
                self.assertNotIn("scope", data)
                self.assertNotEqual(data["client_id"], "dynamic_agent_client")
                self.refresh_count += 1
                time.sleep(0.03)
                response = {"access_token": "offline-access-renewed", "refresh_token": "offline-refresh-renewed",
                            "token_type": "Bearer", "expires_in": 3600, "scope": self.scopes}
            return {**response, **self.response_overrides}
        if url == auth.REVOCATION_URL:
            if self.revocation_error:
                raise auth.SiwcError(self.revocation_error)
            self.assertTrue(empty)
            self.assertEqual(data["token_type_hint"], "refresh_token")
            return None
        if url == auth.MODELS_URL:
            self.assertIsNone(data)
            self.assertEqual(access_token, self.expected_catalog_token)
            return self.model_response
        self.fail("Unapproved network endpoint")

    def browser(self, url):
        parts = urllib.parse.urlsplit(url)
        self.assertEqual(parts.scheme + "://" + parts.netloc + parts.path, auth.AUTHORIZE_URL)
        values = {key: value[0] for key, value in urllib.parse.parse_qs(parts.query).items()}
        self.authorizations.append(values)
        callback = {"state": values["state"], "code": "offline-code", "client_id": self.client_id}
        if self.browser_query:
            callback.update(self.browser_query)
        try:
            with urllib.request.urlopen(values["redirect_uri"] + "?" + urllib.parse.urlencode(callback), timeout=2) as result:
                self.assertNotIn(b"offline-code", result.read())
        except urllib.error.HTTPError as error:
            self.assertEqual(error.code, 400)
            error.close()
        return True

    def login(self, profile_id=None):
        return self.client.login(profile_id=profile_id, browser_open=self.browser, timeout=1)

    def mutate(self, operation):
        with self.client._vault.locked():
            value = self.client._vault.read()
            operation(value)
            self.client._vault.write(value)

    def test_status_is_read_only_and_sanitized(self):
        status = self.client.status()
        self.assertEqual(status, {"status": "AUTHENTICATION_NOT_CONNECTED",
                                  "active_profile_id": None, "profiles": []})
        self.assertFalse(self.directory.exists())
        self.assertEqual(self.calls, [])

    def test_initial_login_has_real_loopback_and_verified_identity(self):
        status = self.login()
        self.assertEqual(status["status"], "READY")
        self.assertEqual(self.client.access_token(), "offline-access-token")
        request = self.authorizations[-1]
        self.assertEqual(request["client_id"], "dynamic_agent_client")
        self.assertEqual(request["agent_name_hint"], auth.AGENT_NAME)
        self.assertTrue(request["ext_agent_host_id"].startswith("urn:uuid:"))
        self.assertTrue(request["redirect_uri"].startswith("http://127.0.0.1:"))
        self.assertTrue(request["redirect_uri"].endswith("/auth/callback"))
        self.assertEqual(request["code_challenge_method"], "S256")
        self.assertNotIn("client_secret", request)
        self.assertNotIn("offline-access-token", json.dumps(status))
        self.assertNotIn("offline-refresh-token", json.dumps(self.client.status()))
        self.assertNotIn("offline-access-token", repr(self.client))
        raw = self.client._vault.path.read_bytes()
        if os.name == "nt":
            self.assertTrue(raw.startswith(b"JANUS-DPAPI-1\n"))
            self.assertNotIn(b"offline-access-token", raw)
        else:
            self.assertEqual(self.directory.stat().st_mode & 0o777, 0o700)
            self.assertEqual(self.client._vault.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(auth.SiwcClient(self.directory).status(), status)

    def test_returning_login_reuses_client_host_and_fresh_pkce(self):
        first = self.login()
        self.login(first["active_profile_id"])
        a, b = self.authorizations
        self.assertEqual(b["client_id"], self.client_id)
        self.assertEqual(a["ext_agent_host_id"], b["ext_agent_host_id"])
        self.assertNotIn("agent_name_hint", b)
        self.assertIn("id_token_hint", b)
        self.assertEqual(b["login_hint"], self.email)
        for name in ("state", "nonce", "code_challenge"):
            self.assertNotEqual(a[name], b[name])

    def test_consent_denial_state_mismatch_and_callback_client_are_rejected(self):
        for query, code in [({"error": "access_denied"}, "SIWC_CONSENT_DENIED"),
                            ({"state": "wrong"}, "SIWC_STATE_MISMATCH"),
                            ({"client_id": "dynamic_agent_client"}, "SIWC_CALLBACK_INVALID")]:
            with self.subTest(code=code):
                self.browser_query = query
                self.calls.clear()
                with self.assertRaisesRegex(auth.SiwcError, "^" + code + "$"):
                    self.login()
                self.assertEqual(self.calls, [])
                self.assertEqual(self.client.status()["profiles"], [])

    def test_callback_parser_rejects_duplicate_missing_and_wrong_issuer(self):
        for query in ("state=x&state=x&code=c&client_id=oaiapp_a",
                      "state=x&code=c&code=d&client_id=oaiapp_a",
                      "code=c&client_id=oaiapp_a",
                      "state=x&code=&client_id=oaiapp_a",
                      "state=x&code=c&client_id=oaiapp_a&iss=https%3A%2F%2Fevil.invalid",
                      "state=x&code=c&client_id=oaiapp_b", "state=%D1%8F&code=c&client_id=oaiapp_a"):
            with self.subTest(query=query), self.assertRaises(auth.SiwcError):
                auth._callback(query, "x", "oaiapp_a")
        self.assertEqual(auth._callback("state=x&code=c", "x", "oaiapp_a"), ("c", "oaiapp_a"))
        with self.assertRaises(auth.SiwcError):
            auth._callback("state=x&code=c", "x", None)

    def test_invalid_grant_restarts_once_with_issued_client_and_fresh_pkce(self):
        original = self.http
        exchanges = []
        def invalid_once(url, data=None, *, empty=False):
            if url == auth.TOKEN_URL and data["grant_type"] == "authorization_code":
                exchanges.append(data)
                if len(exchanges) == 1:
                    raise auth._InvalidGrant("SIWC_REAUTHENTICATION_REQUIRED")
            return original(url, data, empty=empty)
        with patch.object(auth, "_http", side_effect=invalid_once):
            result = self.login()
        self.assertEqual(result["status"], "READY")
        self.assertEqual(len(exchanges), 2)
        first, second = self.authorizations
        self.assertEqual(first["client_id"], "dynamic_agent_client")
        self.assertEqual(second["client_id"], self.client_id)
        self.assertNotIn("agent_name_hint", second)
        self.assertEqual(first["ext_agent_host_id"], second["ext_agent_host_id"])
        for key in ("state", "nonce", "code_challenge"):
            self.assertNotEqual(first[key], second[key])

    def test_signed_token_gates(self):
        cases = [({"iss": "https://evil.invalid"}, None),
                 ({"aud": "oaiapp_wrong"}, None), ({"exp": int(time.time()) - 100}, None),
                 ({"nonce": "wrong"}, None), ({"sub": ""}, None),
                 ({"aud": [self.client_id, "other"]}, None), ({}, self.other_key)]
        for overrides, key in cases:
            with self.subTest(fields=list(overrides)), self.assertRaises(auth.SiwcError):
                auth._validated_identity(self.signed(key=key, **overrides), self.client_id, "offline-nonce")
        self.assertEqual(auth._validated_identity(self.signed(), self.client_id, "offline-nonce")["sub"], self.subject)
        unsigned = jwt.encode({"sub": "untrusted"}, key="", algorithm="none")
        with self.assertRaises(auth.SiwcError):
            auth._validated_identity(unsigned, self.client_id, "offline-nonce")

    def test_identity_mismatch_preserves_selected_credentials(self):
        before = self.login()
        profile = before["active_profile_id"]
        self.subject = "different-subject"
        with self.assertRaisesRegex(auth.SiwcError, "SIWC_IDENTITY_MISMATCH"):
            self.login(profile)
        self.assertEqual(self.client.status(), before)
        self.assertEqual(self.client.access_token(), "offline-access-token")

    def test_callback_client_mismatch_does_not_exchange_or_replace(self):
        first = self.login()
        self.calls.clear()
        self.browser_query = {"client_id": "oaiapp_different"}
        with self.assertRaisesRegex(auth.SiwcError, "SIWC_CLIENT_ID_MISMATCH"):
            self.login(first["active_profile_id"])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.client.status(), first)

    def test_scope_comes_from_token_response_not_callback(self):
        self.scopes = "openid profile email offline_access"
        self.browser_query = {"scope": auth.SCOPES}
        status = self.login()
        self.assertEqual(status["status"], "PLAN_USAGE_NOT_GRANTED")
        with self.assertRaisesRegex(auth.SiwcError, "SIWC_PLAN_USAGE_NOT_GRANTED"):
            self.client.access_token()
        self.assertEqual(self.refresh_count, 0)

    def test_invalid_token_response_never_saves_credentials(self):
        for override in ({"expires_in": True}, {"expires_in": 0}, {"token_type": 1},
                         {"refresh_token": ""}, {"scope": None}, {"id_token": "invalid"}):
            with self.subTest(keys=list(override)):
                self.response_overrides = override
                with self.assertRaises(auth.SiwcError):
                    self.login()
                self.assertEqual(self.client.status()["profiles"], [])

    def test_multiple_registrations_same_email_and_exact_profile_access(self):
        first = self.login()["active_profile_id"]
        self.client_id = "oaiapp_offline_b"
        self.subject = "offline-subject-b"
        self.response_overrides = {"access_token": "second-access"}
        second = self.login()["active_profile_id"]
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.client.status()["profiles"]), 2)
        self.assertEqual(self.client.access_token(first), "offline-access-token")
        self.assertEqual(self.client.status()["active_profile_id"], second)
        self.assertEqual(self.client.access_token(second), "second-access")
        self.client.select_profile(first)
        self.assertEqual(self.client.status()["active_profile_id"], first)

    def test_rotating_refresh_is_serialized_and_replaces_whole_record(self):
        profile_id = self.login()["active_profile_id"]
        self.mutate(lambda value: value["profiles"][profile_id]["tokens"].update(expires_at=0))
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(auth.SiwcClient(self.directory).access_token, profile_id) for _ in range(2)]
            self.assertEqual([future.result(timeout=5) for future in futures], ["offline-access-renewed"] * 2)
        self.assertEqual(self.refresh_count, 1)
        with self.client._vault.locked():
            tokens = self.client._vault.read()["profiles"][profile_id]["tokens"]
        self.assertEqual(tokens["refresh_token"], "offline-refresh-renewed")
        self.assertGreater(tokens["expires_at"], time.time() + 3000)

    def test_vault_lock_serializes_independent_processes(self):
        self.login()
        code = (
            "import sys; from pathlib import Path; from mcp_gateway.siwc_auth import _Vault; "
            "vault=_Vault(Path(sys.argv[1])); print('waiting',flush=True); "
            "\nwith vault.locked():\n print('acquired',flush=True)\n"
        )
        process = None
        try:
            with self.client._vault.locked():
                process = subprocess.Popen([sys.executable, "-c", code, str(self.directory)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.assertEqual(process.stdout.readline().strip(), "waiting")
                time.sleep(0.15)
                self.assertIsNone(process.poll())
            output, error = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, error)
            self.assertEqual(output.strip(), "acquired")
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate(timeout=5)

    def test_refresh_revoked_scope_holds_and_invalid_grant_clears_session(self):
        profile_id = self.login()["active_profile_id"]
        self.mutate(lambda value: value["profiles"][profile_id]["tokens"].update(expires_at=0))
        self.scopes = "openid"
        with self.assertRaisesRegex(auth.SiwcError, "SIWC_PLAN_USAGE_NOT_GRANTED"):
            self.client.access_token(profile_id)
        self.assertFalse(self.client.status()["profiles"][0]["plan_usage_enabled"])
        self.scopes = auth.SCOPES
        self.login(profile_id)
        self.mutate(lambda value: value["profiles"][profile_id]["tokens"].update(expires_at=0))
        self.exchange_error = "SIWC_REAUTHENTICATION_REQUIRED"
        with self.assertRaisesRegex(auth.SiwcError, "SIWC_REAUTHENTICATION_REQUIRED"):
            self.client.access_token(profile_id)
        self.assertFalse(self.client.status()["profiles"][0]["connected"])

    def test_logout_preserves_registration_then_reauthorizes_without_hints(self):
        profile_id = self.login()["active_profile_id"]
        result = self.client.logout()
        self.assertTrue(result["remote_revocation_confirmed"])
        self.assertIsNone(result["active_profile_id"])
        self.assertFalse(result["profiles"][0]["connected"])
        self.assertEqual(len(result["profiles"]), 1)
        with self.assertRaises(auth.SiwcError):
            self.client.select_profile(profile_id)
        self.login(profile_id)
        request = self.authorizations[-1]
        self.assertEqual(request["client_id"], self.client_id)
        self.assertNotIn("id_token_hint", request)
        self.assertEqual(request["ext_agent_host_id"], self.authorizations[0]["ext_agent_host_id"])

    def test_failed_revocation_clears_local_tokens_and_reports_unconfirmed(self):
        self.login()
        self.revocation_error = "SIWC_HTTP_RETRYABLE"
        with patch.object(auth.time, "sleep"):
            result = self.client.logout()
        self.assertFalse(result["remote_revocation_confirmed"])
        self.assertFalse(result["profiles"][0]["connected"])
        self.assertEqual(sum(url == auth.REVOCATION_URL for url, _, _ in self.calls), 3)

    def test_failed_first_attempt_still_preserves_stable_host(self):
        with self.assertRaisesRegex(auth.SiwcError, "SIWC_BROWSER_UNAVAILABLE"):
            self.client.login(browser_open=lambda url: self.authorizations.append(
                {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).items()}) or False)
        self.login()
        self.assertEqual(self.authorizations[0]["ext_agent_host_id"], self.authorizations[1]["ext_agent_host_id"])

    def test_partial_http_headers_cannot_hold_login_shutdown(self):
        sockets = []
        def partial(url):
            params = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            target = urllib.parse.urlsplit(params["redirect_uri"][0])
            connection = socket.create_connection((target.hostname, target.port), timeout=1)
            sockets.append(connection)
            connection.sendall(b"GET /auth/callback HTTP/1.1\r\nHost: ")
            return True
        start = time.monotonic()
        try:
            with self.assertRaisesRegex(auth.SiwcError, "SIWC_AUTHORIZATION_TIMEOUT"):
                self.client.login(browser_open=partial, timeout=0.1)
            self.assertLess(time.monotonic() - start, 1.5)
        finally:
            for connection in sockets:
                connection.close()

    def test_unrelated_nonempty_directory_is_never_modified(self):
        self.directory.mkdir()
        innocent = self.directory / "unrelated.txt"
        innocent.write_text("keep me", encoding="utf-8")
        before = self.directory.stat()
        with self.assertRaisesRegex(auth.SiwcError, "SIWC_VAULT_DIRECTORY_NOT_OWNED"):
            self.client.login(browser_open=lambda url: False)
        self.assertEqual(list(self.directory.iterdir()), [innocent])
        self.assertEqual(innocent.read_text(encoding="utf-8"), "keep me")
        self.assertEqual(self.directory.stat().st_mode, before.st_mode)

    def test_corrupted_vault_fails_closed_without_recreation(self):
        self.login()
        path = self.client._vault.path
        path.write_bytes(b"damaged")
        with self.assertRaisesRegex(auth.SiwcError, "SIWC_VAULT_INVALID"):
            self.client.status()
        with self.assertRaisesRegex(auth.SiwcError, "SIWC_VAULT_INVALID"):
            self.client.login(browser_open=lambda url: False)
        self.assertEqual(path.read_bytes(), b"damaged")

    def test_symlink_or_reparse_ancestor_rejected_before_creating_state(self):
        original = Path(self.temporary.name) / "original"
        original.mkdir()
        link = Path(self.temporary.name) / "link"
        try:
            link.symlink_to(original, target_is_directory=True)
        except OSError:
            # Ordinary Windows accounts may lack symlink creation privileges.
            # Exercise the junction/reparse attribute gate without changing OS
            # privileges or creating an actual redirected directory.
            real_lstat = Path.lstat
            def reparse_lstat(path, *args, **kwargs):
                if path == link:
                    return SimpleNamespace(st_mode=0o40700, st_file_attributes=0x400)
                return real_lstat(path, *args, **kwargs)
            with patch.object(Path, "lstat", reparse_lstat):
                redirected = auth.SiwcClient(link / "credentials")
                with self.assertRaisesRegex(auth.SiwcError, "SIWC_VAULT_PATH_REJECTED"):
                    redirected.status()
                with self.assertRaisesRegex(auth.SiwcError, "SIWC_VAULT_PATH_REJECTED"):
                    redirected.login(browser_open=lambda url: False)
            self.assertEqual(list(original.iterdir()), [])
            return
        redirected = auth.SiwcClient(link / "credentials")
        with self.assertRaisesRegex(auth.SiwcError, "SIWC_VAULT_PATH_REJECTED"):
            redirected.status()
        with self.assertRaisesRegex(auth.SiwcError, "SIWC_VAULT_PATH_REJECTED"):
            redirected.login(browser_open=lambda url: False)
        self.assertEqual(list(original.iterdir()), [])

    def test_metadata_endpoint_substitution_rejected(self):
        with patch.object(auth, "_http", return_value={"issuer": auth.ISSUER,
                "jwks_uri": "https://evil.invalid/keys", "revocation_endpoint": auth.REVOCATION_URL,
                "id_token_signing_alg_values_supported": ["RS256"]}):
            with self.assertRaisesRegex(auth.SiwcError, "SIWC_DISCOVERY_REJECTED"):
                auth._metadata()

    def test_no_redirect_and_endpoint_pin(self):
        self.assertIsNone(auth._NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.invalid"))
        self.http_patch.stop()
        with self.assertRaisesRegex(auth.SiwcError, "SIWC_ENDPOINT_REJECTED"):
            auth._http("https://evil.invalid/token", {"token": "offline"})
        self.http_patch.start()

    def test_account_model_catalog_filters_visibility_and_preserves_order(self):
        profile_id = self.login()["active_profile_id"]
        self.model_response = {"models": [
            {"slug": "unknown-safe-model-2", "display_name": "Future Model 2", "visibility": "list",
             "access_token": "untrusted-extra-secret", "description": "untrusted text"},
            {"slug": "not shown", "display_name": "hidden", "visibility": "hidden"},
            {"slug": "unknown-safe-model-1", "display_name": "Future Model 1", "visibility": "list"},
            {"slug": "not shown either", "visibility": "unknown"},
        ], "extra": "drop me"}
        result = self.client.list_models()
        self.assertEqual(result, {"status": "ACCOUNT_MODEL_CATALOG", "active_profile_id": profile_id,
            "models": [{"slug": "unknown-safe-model-2", "display_name": "Future Model 2"},
                       {"slug": "unknown-safe-model-1", "display_name": "Future Model 1"}],
            "model_called": False})
        self.assertNotIn("untrusted-extra-secret", json.dumps(result))
        self.assertEqual(self.refresh_count, 0)

    def test_model_catalog_binds_exact_profile_while_active_account_changes(self):
        first = self.login()["active_profile_id"]
        self.client_id = "oaiapp_offline_b"
        self.subject = "offline-subject-b"
        self.response_overrides = {"access_token": "second-access"}
        second = self.login()["active_profile_id"]
        self.assertEqual(self.client.list_models(first)["active_profile_id"], first)
        self.assertEqual(self.client.status()["active_profile_id"], second)
        self.expected_catalog_token = "second-access"
        self.assertEqual(self.client.list_models()["active_profile_id"], second)

    def test_model_catalog_rejects_malformed_visible_fields_and_duplicates(self):
        self.login()
        valid = {"slug": "safe-model", "display_name": "Safe Model", "visibility": "list"}
        bad = [None, {}, {"models": None}, {"models": ["invalid"]},
               {"models": [{**valid, "slug": "../../escape"}]},
               {"models": [{**valid, "slug": "-option"}]},
               {"models": [{**valid, "slug": "model with spaces"}]},
               {"models": [{**valid, "slug": 1}]},
               {"models": [{**valid, "display_name": "bad\nname"}]},
               {"models": [{**valid, "display_name": "bad\u202ename"}]},
               {"models": [{**valid, "display_name": " "}]},
               {"models": [{**valid, "display_name": "n" * 161}]},
               {"models": [valid, valid]}, {"models": [valid] * 513}]
        for response in bad:
            with self.subTest(kind=type(response).__name__):
                self.model_response = response
                with self.assertRaisesRegex(auth.SiwcError, "^SIWC_MODELS_RESPONSE_INVALID$"):
                    self.client.list_models()

    def test_model_catalog_transport_pins_bearer_get_and_safe_http_errors(self):
        self.http_patch.stop()
        transport = auth._http
        self.http_patch.start()
        response = MagicMock()
        response.status = 200
        response.read.return_value = b'{"models":[]}'
        response.__enter__.return_value = response
        opener = MagicMock()
        opener.open.return_value = response
        with patch.object(auth.urllib.request, "build_opener", return_value=opener):
            self.assertEqual(transport(auth.MODELS_URL, access_token="offline-bearer"), {"models": []})
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/models")
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.get_header("Authorization"), "Bearer offline-bearer")
        self.assertIsNone(request.data)
        for status, expected in [(401, "SIWC_REAUTHENTICATION_REQUIRED"),
                                 (403, "SIWC_HTTP_REJECTED"), (500, "SIWC_HTTP_RETRYABLE")]:
            failure = urllib.error.HTTPError(auth.MODELS_URL, status, "remote secret must not escape", {},
                                            io.BytesIO(b"remote-secret-body"))
            opener.open.side_effect = failure
            with patch.object(auth.urllib.request, "build_opener", return_value=opener):
                with self.assertRaisesRegex(auth.SiwcError, "^" + expected + "$"):
                    transport(auth.MODELS_URL, access_token="offline-bearer")
        with self.assertRaisesRegex(auth.SiwcError, "SIWC_ENDPOINT_REJECTED"):
            transport(auth.JWKS_URL, access_token="offline-bearer")
        with self.assertRaisesRegex(auth.SiwcError, "SIWC_ENDPOINT_REJECTED"):
            transport(auth.MODELS_URL, {"not": "GET"}, access_token="offline-bearer")


if __name__ == "__main__":
    unittest.main()
