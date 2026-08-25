import unittest

from nas_sync.validate_checkpoint import validate


class CheckpointValidationTests(unittest.TestCase):
    def valid(self):
        return {
            "schema": "janus.nas.checkpoint.v1",
            "timestamp_utc": "2026-08-25T18:00:00Z",
            "host_id": "janus-nas-01",
            "canonical_janus_sha": "a" * 40,
            "sync_status": "PASS",
            "summary": {
                "repos_in_sync": 1,
                "repos_dirty": 0,
                "repos_diverged": 0,
                "repos_blocked": 0,
            },
            "repos": [{
                "repo": "Hawkar-usls/Janus",
                "branch": "main",
                "local_head": "a" * 40,
                "remote_head": "a" * 40,
                "status": "IN_SYNC",
                "tests": "PASS",
            }],
        }

    def test_valid_checkpoint(self):
        validate(self.valid())

    def test_reject_secret_key(self):
        payload = self.valid()
        payload["token"] = "redacted"
        with self.assertRaises(ValueError):
            validate(payload)

    def test_reject_absolute_path(self):
        payload = self.valid()
        payload["repos"][0]["branch"] = "/share/CACHEDEV1_DATA/Janus"
        with self.assertRaises(ValueError):
            validate(payload)

    def test_reject_non_hawkar_repo(self):
        payload = self.valid()
        payload["repos"][0]["repo"] = "other/repo"
        with self.assertRaises(ValueError):
            validate(payload)

    def test_dirty_is_receipt_state_not_failure(self):
        payload = self.valid()
        payload["sync_status"] = "PARTIAL"
        payload["repos"][0]["status"] = "DIRTY_LOCAL"
        validate(payload)


if __name__ == "__main__":
    unittest.main()
