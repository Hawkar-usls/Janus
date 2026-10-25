"""No-network controls for the scheduled zero-credit Rowan GitHub gate."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("github_rowan_readonly", ROOT / "github_rowan_readonly.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeError(Exception):
    def __init__(self, status_code):
        self.response = FakeResponse(status_code)


class FakeFolder:
    def __init__(self, fails=False):
        self.fails = fails

    def contents(self, size):
        assert size == 10
        if self.fails:
            raise FakeError(403)
        return ["DO_NOT_DISCLOSE_PRIVATE_ITEM"]


class FakeSDK:
    def __init__(self, *, auth_fails=False, folder_fails=False):
        self.calls = []
        self.auth_fails = auth_fails
        self.folder_fails = folder_fails

    def whoami(self):
        self.calls.append("whoami")
        if self.auth_fails:
            raise FakeError(403)
        return {"email": "MUST_NOT_BE_EXPOSED", "credits": 900}

    def retrieve_folder(self, uuid):
        self.calls.append("retrieve_folder")
        assert uuid == worker.FOLDER_UUID
        return FakeFolder(fails=self.folder_fails)


class ReadOnlyTests(unittest.TestCase):
    def test_missing_secret_fails_closed_without_any_remote_calls(self):
        api = FakeSDK()
        status = worker.inspect_readonly(api, key_present=False)
        self.assertEqual(status["secret"], "MISSING")
        self.assertEqual(status["auth"], "NOT_CHECKED")
        self.assertEqual(api.calls, [])

    def test_success_is_boolean_and_no_private_metadata(self):
        api = FakeSDK()
        status = worker.inspect_readonly(api, key_present=True)
        self.assertEqual(status["auth"], "VERIFIED")
        self.assertEqual(status["folder"], "VERIFIED_READ_ONLY")
        self.assertEqual(api.calls, ["whoami", "retrieve_folder"])
        self.assertNotIn("email", repr(status))
        self.assertNotIn("credits", repr(status).replace("credit_spending", ""))

    def test_http_403_does_not_try_folder(self):
        api = FakeSDK(auth_fails=True)
        status = worker.inspect_readonly(api, key_present=True)
        self.assertEqual(status["auth"], "HTTP_403")
        self.assertEqual(status["folder"], "NOT_CHECKED")
        self.assertEqual(api.calls, ["whoami"])

    def test_folder_403_is_classified_without_private_data(self):
        api = FakeSDK(folder_fails=True)
        status = worker.inspect_readonly(api, key_present=True)
        self.assertEqual(status["auth"], "VERIFIED")
        self.assertEqual(status["folder"], "HTTP_403")

    def test_non_http_network_error_is_sanitized(self):
        self.assertEqual(worker.safe_error(RuntimeError("sensitive external response")), "PROVIDER_REQUEST_FAILED")

    def test_readonly_does_not_enable_paid_computation(self):
        status = worker.inspect_readonly(FakeSDK(), key_present=True)
        self.assertEqual(status["provider_computation"], "NOT_ATTEMPTED")
        self.assertEqual(status["credit_spending"], "NOT_ATTEMPTED")


if __name__ == "__main__":
    unittest.main()
