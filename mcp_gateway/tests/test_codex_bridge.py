"""Human CLI boundaries; no network, account or model is used."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mcp_gateway import codex_bridge as bridge


class BridgeCliTests(unittest.TestCase):
    def invoke(self, argv):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            code = bridge.main(argv)
        return code, json.loads(output.getvalue())

    def test_model_request_requires_local_authorization_before_vault_access(self):
        with patch.object(bridge, "SiwcClient") as client, patch.object(bridge, "run_turn") as runtime:
            code, result = self.invoke(["smoke"])
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "SIWC_EXPLICIT_PLAN_USAGE_REQUIRED")
        client.assert_not_called()
        runtime.assert_not_called()

    def test_unsafe_prompt_rejected_before_vault_access(self):
        with patch.object(bridge, "SiwcClient") as client:
            code, _ = self.invoke(["ask", "--allow-plan-usage", "--prompt", "secret ghp_example"])
        self.assertEqual(code, 2)
        client.assert_not_called()

    def test_doctor_is_model_specific_and_never_opens_vault(self):
        with patch.object(bridge, "SiwcClient") as client, patch.object(bridge, "check_runtime") as doctor:
            doctor.return_value = {"compatible": False, "model_called": False}
            code, result = self.invoke(["doctor", "--model", "gpt-5.4"])
        client.assert_not_called()
        doctor.assert_called_once_with(codex_binary=None, model="gpt-5.4")
        self.assertEqual(code, 2)
        self.assertFalse(result["model_called"])

    def test_status_does_not_invoke_runtime_or_read_tokens(self):
        with patch.object(bridge, "SiwcClient") as client, patch.object(bridge, "run_turn") as runtime:
            client.return_value.status.return_value = {"status": "LOGIN_REQUIRED", "profiles": []}
            code, result = self.invoke(["status"])
        self.assertEqual(code, 0)
        self.assertFalse(result["model_called"])
        client.return_value.access_token.assert_not_called()
        runtime.assert_not_called()

    def test_smoke_uses_fixed_prompt_and_account_bound_token(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(bridge, "SiwcClient") as client, \
                patch.object(bridge, "run_turn") as runtime:
            client.return_value.status.return_value = {"active_profile_id": "profile-a"}
            client.return_value.access_token.return_value = "fake-test-token"
            runtime.return_value = {"text": "JANUS_BRIDGE_OK", "model_called": True}
            code, result = self.invoke(["--state-dir", directory, "smoke", "--allow-plan-usage"])
            client.return_value.access_token.assert_called_once_with(profile_id="profile-a")
            call = runtime.call_args
            self.assertEqual(call.args, ("fake-test-token", None, bridge.SMOKE_PROMPT))
            self.assertEqual(call.kwargs["runtime_dir"].parent, Path(directory) / "runtime")
        self.assertEqual(code, 0)
        self.assertTrue(result["smoke_response_matches"])
        self.assertFalse(result["auto_merge"])

    def test_failed_smoke_is_nonzero(self):
        with patch.object(bridge, "SiwcClient") as client, patch.object(bridge, "run_turn") as runtime:
            client.return_value.status.return_value = {"active_profile_id": "a"}
            runtime.return_value = {"text": "wrong", "model_called": True}
            code, result = self.invoke(["smoke", "--allow-plan-usage"])
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "CODEX_SMOKE_RESPONSE_MISMATCH")

    def test_unexpected_exception_never_prints_provider_body_or_credentials(self):
        with patch.object(bridge, "SiwcClient", side_effect=RuntimeError("Bearer secret-value")):
            code, result = self.invoke(["status"])
        self.assertEqual(code, 3)
        self.assertEqual(result, {"status": "BRIDGE_INTERNAL_ERROR", "ok": False})

    def test_login_and_logout_are_only_explicit_commands(self):
        with patch.object(bridge, "SiwcClient") as client:
            client.return_value.login.return_value = {"status": "CONNECTED"}
            self.invoke(["login", "--profile", "profile-a", "--timeout", "60"])
            client.return_value.login.assert_called_once_with(profile_id="profile-a", timeout=60)
            client.return_value.logout.assert_not_called()
            client.return_value.logout.return_value = {"status": "SIGNED_OUT"}
            self.invoke(["logout", "--profile", "profile-a"])
            client.return_value.logout.assert_called_once_with(profile_id="profile-a")


if __name__ == "__main__":
    unittest.main()
