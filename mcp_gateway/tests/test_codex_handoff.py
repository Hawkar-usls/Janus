"""Offline negative controls for Codex Cloud and ChatGPT-plan gateway drafts."""
import unittest

from mcp_gateway.codex_handoff import (
    CodexHandoffError, prepare_codex_cloud_task,
)
from mcp_gateway.codex_app_server import (
    CodexProtocolError, app_server_argv, app_server_readiness,
    validate_turn_completion,
)


class CodexGatewayTests(unittest.TestCase):
    def test_draft_is_deterministic_and_non_executing(self):
        a = prepare_codex_cloud_task("Repair one failing test",
                                     ["mcp_gateway", "docs"])
        b = prepare_codex_cloud_task("Repair one failing test",
                                     ["mcp_gateway", "docs"])
        self.assertEqual(a, b)
        self.assertEqual(a["repository"], "Hawkar-usls/Janus")
        self.assertEqual(a["status"], "DRAFT_USER_AUTHORIZATION_REQUIRED")
        self.assertTrue(a["requires_user_chatgpt_sign_in"])
        self.assertFalse(a["model_called"])
        self.assertFalse(a["oauth_token_accessed"])
        self.assertFalse(a["github_write_performed"])
        self.assertEqual(a["production_authority_delta"], 0)
        self.assertIn("Run relevant offline tests", a["codex_cloud_prompt"])
        self.assertIn("Never auto-merge", a["codex_cloud_prompt"])

    def test_allowlisted_repository_only(self):
        with self.assertRaisesRegex(CodexHandoffError, "NOT_ALLOWLISTED"):
            prepare_codex_cloud_task("Edit code", repository="Hawkar-usls/SkinGPT")

    def test_path_traversal_controls(self):
        for bad in ("../secrets", "x/../../tmp", "/root", "a//b",
                    "a/./b", "a\\x", "a?b", "a\nfoo"):
            with self.subTest(bad=bad), self.assertRaises(CodexHandoffError):
                prepare_codex_cloud_task("Safe task", [bad])

    def test_reject_sensitive_or_excessive_task(self):
        for bad in ("", " ", "secret ghp_abcdef", "x" * 9000, "x\x00y"):
            with self.subTest(bad=bad), self.assertRaises(CodexHandoffError):
                prepare_codex_cloud_task(bad)
        with self.assertRaises(CodexHandoffError):
            prepare_codex_cloud_task("Task", paths=[])

    def test_app_server_command_uses_approved_token_transport(self):
        command = app_server_argv()
        joined = " ".join(command)
        self.assertEqual(command[:4], ["codex", "app-server", "--listen", "stdio://"])
        self.assertIn('env_key="ACCESS_TOKEN"', joined)
        self.assertIn('wire_api="responses"', joined)
        self.assertNotIn("sk-", joined)
        self.assertNotIn("Bearer ", joined)
        readiness = app_server_readiness()
        self.assertEqual(readiness["status"], "AUTHENTICATION_NOT_CONNECTED")
        self.assertFalse(readiness["codex_app_server_started"])
        self.assertFalse(readiness["chat_history_access"])

    def test_turn_done_is_not_patch_acceptance(self):
        completed = validate_turn_completion({
            "method": "turn/completed",
            "params": {"threadId": "t-1", "turn": {"status": "completed"}},
        })
        self.assertEqual(completed["status"], "MODEL_TURN_COMPLETED_ONLY")
        self.assertFalse(completed["github_pr_created"])
        self.assertFalse(completed["github_patch_verified"])
        for event in (
            {"method": "turn/completed", "params": {"threadId": "t-1",
                                                     "turn": {"status": "failed"}}},
            {"method": "turn/completed", "params": {"threadId": "t-1",
                                                     "turn": {"status": "interrupted"}}},
            {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
            {"method": "turn/start", "params": {"threadId": "t-1"}},
            {},
        ):
            with self.subTest(event=event), self.assertRaises(CodexProtocolError):
                validate_turn_completion(event)


if __name__ == "__main__":
    unittest.main()
