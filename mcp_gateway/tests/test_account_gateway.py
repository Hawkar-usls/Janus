"""Offline provider routing and Google CLI isolation; synthetic fixtures only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from mcp_gateway import account_gateway as pool, google_gateway as google
from mcp_gateway.codex_runtime import CodexRuntimeError


class AccountGatewayTests(unittest.TestCase):
    def args(self, *values):
        return pool.parser().parse_args(list(values))

    def test_missing_usage_consent_stops_before_either_provider(self):
        with patch.object(pool, "_chatgpt") as chatgpt, patch.object(google, "run_turn") as gemini:
            with self.assertRaisesRegex(pool.SiwcError, "EXPLICIT_USAGE_REQUIRED"):
                pool.execute(self.args("ask", "--provider", "auto", "--prompt", "hello"))
        chatgpt.assert_not_called()
        gemini.assert_not_called()

    def test_auto_fallback_only_on_quota_and_records_selected_provider(self):
        args = self.args("ask", "--provider", "auto", "--prompt", "hello", "--allow-plan-usage")
        with patch.object(pool, "_chatgpt", side_effect=CodexRuntimeError("CODEX_PROVIDER_USAGE_LIMIT")), \
                patch.object(google, "run_turn", return_value={"provider": "google", "text": "hello"}) as gemini:
            result = pool.execute(args)
        self.assertEqual(result["fallback_reason"], "CODEX_PROVIDER_USAGE_LIMIT")
        gemini.assert_called_once()
        with patch.object(pool, "_chatgpt", side_effect=CodexRuntimeError("CODEX_TEXT_ONLY_RUNTIME_INCOMPATIBLE")), \
                patch.object(google, "run_turn") as gemini:
            with self.assertRaisesRegex(CodexRuntimeError, "INCOMPATIBLE"):
                pool.execute(args)
        gemini.assert_not_called()

    def test_explicit_model_or_thread_is_never_reinterpreted_by_another_provider(self):
        for flag in ("--model", "--thread"):
            with self.assertRaisesRegex(pool.SiwcError, "PROVIDER_NEUTRAL"):
                pool.execute(self.args("ask", "--provider", "auto", flag, "exact-id", "--prompt", "hello", "--allow-plan-usage"))

    def test_deepseek_does_not_start_an_unsupported_session(self):
        with patch.object(google, "run_turn") as gemini:
            result = pool.execute(self.args("smoke", "--provider", "deepseek", "--allow-plan-usage"))
        self.assertFalse(result["model_called"])
        self.assertFalse(result["ok"])
        gemini.assert_not_called()

    def test_chatgpt_smoke_mismatch_is_an_unsuccessful_result(self):
        with patch.object(pool, "_chatgpt", return_value={"status": "CODEX_SMOKE_RESPONSE_MISMATCH"}):
            result = pool.execute(self.args("smoke", "--allow-plan-usage"))
        self.assertFalse(result["ok"])

    def test_native_google_session_has_no_ambient_api_keys_or_github_token(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "synthetic", "GEMINI_API_KEY": "synthetic",
                                   "OPENAI_API_KEY": "synthetic", "GOOGLE_APPLICATION_CREDENTIALS": "synthetic"}):
            env = google._environment()
        for key in ("GITHUB_TOKEN", "GEMINI_API_KEY", "OPENAI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"):
            self.assertNotIn(key, env)

    def test_real_child_gets_deny_policy_and_disposable_workspace(self):
        original = subprocess.Popen

        def spawn(argv, **options):
            self.assertFalse(options["shell"])
            settings = json.loads(Path(argv[2]).read_text())
            self.assertEqual(settings["permissions"]["deny"], [name + "(*)" for name in google.DENIED_ACTIONS])
            self.assertNotIn("GITHUB_TOKEN", options["env"])
            return original([sys.executable, "-c", "print('synthetic output')"], **options)

        with patch.object(google.subprocess, "Popen", side_effect=spawn):
            code, output, errors = google._run(sys.executable, ["models"], 5)
        self.assertEqual(code, 0)
        self.assertEqual(output.strip(), "synthetic output")

    def test_google_timeout_and_output_limit_reap_the_real_child(self):
        original = subprocess.Popen
        for script, limit, expected in (
            ("import time; time.sleep(60)", google.MAX_BYTES, "GOOGLE_TIMEOUT"),
            ("import time; print('x'*8192, flush=True); time.sleep(60)", 1024, "GOOGLE_OUTPUT_LIMIT"),
        ):
            children = []

            def spawn(argv, **options):
                child = original([sys.executable, "-c", script], **options)
                children.append(child)
                return child

            with patch.object(google.subprocess, "Popen", side_effect=spawn), \
                    patch.object(google, "MAX_BYTES", limit), \
                    self.assertRaisesRegex(google.GoogleGatewayError, "^" + expected + "$"):
                google._run(sys.executable, [], 1)
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].poll())

    def test_only_success_envelope_is_a_google_completion(self):
        fixture = {"status": "SUCCESS", "response": "JANUS_BRIDGE_OK", "conversation_id": "00000000-0000-4000-8000-000000000001"}
        with patch.object(google, "_binary", return_value=sys.executable), \
                patch.object(google, "_run", return_value=(0, json.dumps(fixture), "")) as run:
            result = google.run_turn("hello", model="exact-model")
            args = run.call_args.args[1]
            self.assertIn("--disable-slash-commands", args)
            self.assertIn("plan", args)
            self.assertNotIn("--dangerously-skip-permissions", args)
            self.assertEqual(result["model"], "exact-model")
        for state in ("ERROR", "CANCELED", "INTERRUPTED", "WAITING", "RUNNING"):
            fixture["status"] = state
            with patch.object(google, "_binary", return_value=sys.executable), \
                    patch.object(google, "_run", return_value=(0, json.dumps(fixture), "")), \
                    self.assertRaises(google.GoogleGatewayError):
                google.run_turn("hello")
        fixture["status"] = "SUCCESS"
        fixture["conversation_id"] = "-" * 36
        with patch.object(google, "_binary", return_value=sys.executable), \
                patch.object(google, "_run", return_value=(0, json.dumps(fixture), "")), \
                self.assertRaisesRegex(google.GoogleGatewayError, "RESPONSE_INVALID"):
            google.run_turn("hello")

    def test_google_catalog_and_errors_exclude_untrusted_native_output(self):
        with patch.object(google, "_binary", return_value=sys.executable), \
                patch.object(google, "_run", return_value=(0, "Fetching available models...\ngemini-test\tGemini Test\n", "")):
            self.assertEqual(google.list_models()["models"][0]["slug"], "gemini-test")
        with patch.object(google, "_binary", return_value=sys.executable), \
                patch.object(google, "_run", return_value=(1, "", "Please sign in synthetic-secret")), \
                self.assertRaisesRegex(google.GoogleGatewayError, "^GOOGLE_ACCOUNT_LOGIN_REQUIRED$"):
            google.list_models()


if __name__ == "__main__":
    unittest.main()
