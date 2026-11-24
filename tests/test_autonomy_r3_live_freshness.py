"""R3 metadata freshness witness: never follow attacker URLs or grant authority."""
from __future__ import annotations
from datetime import datetime,timezone
import unittest

from autonomy.registry import live_freshness as f
from autonomy.registry import navigator as n

NOW=datetime(2026,11,24,tzinfo=timezone.utc)


class LiveFreshnessTests(unittest.TestCase):
    def mock(self,head=n.MAIN_COMMIT,blob=n.INDEX_BLOB):
        history=[]
        def fetch(url,token):
            history.append(url)
            if url==f.URL_MAIN:
                return {"commit":{"sha":head}}
            if url==f.URL_INDEX_PREFIX+head:
                return {"type":"file","path":n.INDEX,"sha":blob}
            raise AssertionError("unreviewed URL "+url)
        return fetch,history

    def test_current_pin_matches_without_global_health_claim(self):
        fetch,history=self.mock()
        x=f.check(fetch=fetch,at=NOW)
        self.assertEqual(x["status"],
            "SOURCE_HEAD_AND_INDEX_MATCH_FROZEN_PIN__OTHER_SERVICES_UNCHECKED")
        self.assertEqual(len(history),2)
        self.assertFalse(x["full_organism_current_state_validated"])
        self.assertFalse(x["external_review_completed"])
        self.assertFalse(x["p03_admission_granted"])

    def test_changed_head_is_hold_even_if_index_bytes_unchanged(self):
        fetch,_=self.mock(head="a"*40)
        x=f.check(fetch=fetch,at=NOW)
        self.assertEqual(x["status"],
            "HOLD_MAIN_OR_BOOT_INDEX_DRIFT__REVALIDATION_REQUIRED")
        self.assertEqual(x["observed_index_blob_sha1"],n.INDEX_BLOB)

    def test_changed_index_bytes_is_hold(self):
        fetch,_=self.mock(blob="f"*40)
        x=f.check(fetch=fetch,at=NOW)
        self.assertEqual(x["status"],
            "HOLD_MAIN_OR_BOOT_INDEX_DRIFT__REVALIDATION_REQUIRED")

    def test_api_failure_stays_unknown_not_pass(self):
        x=f.check(fetch=lambda *_: (_ for _ in ()).throw(OSError("offline")),at=NOW)
        self.assertEqual(x["status"],"HOLD_LIVE_METADATA_UNAVAILABLE")
        self.assertEqual(x["api_get_calls_completed"],0)
        self.assertFalse(x["main_index_may_be_automatically_rewritten"])

    def test_invalid_response_identity_is_hold(self):
        def wrong(url,token):
            if url==f.URL_MAIN:return {"commit":{"sha":n.MAIN_COMMIT}}
            return {"type":"file","path":"secrets.json","sha":"a"*40}
        x=f.check(fetch=wrong,at=NOW)
        self.assertEqual(x["status"],"HOLD_LIVE_METADATA_UNAVAILABLE")
        self.assertEqual(x["api_get_calls_completed"],2)
        self.assertFalse(x["p03_admission_granted"])

    def test_invalid_sha_and_arbitrary_endpoint_denied(self):
        x=f.check(fetch=lambda *_:{"commit":{"sha":"notsha"}},at=NOW)
        self.assertEqual(x["status"],"HOLD_LIVE_METADATA_UNAVAILABLE")
        for url in ("https://example.org/secret",
                    f.URL_INDEX_PREFIX+"../../admin",
                    "https://api.github.com/repos/Hawkar-usls/SkinGPT/branches/main"):
            with self.subTest(url=url),self.assertRaises(n.RegistryHold):
                f.get_json(url)

    def test_receipt_has_no_source_content_or_token(self):
        fetch,_=self.mock()
        x=f.check(fetch=fetch,token="A_FAKE_TEST_SECRET",at=NOW)
        self.assertNotIn("A_FAKE_TEST_SECRET",str(x))
        self.assertFalse(x["new_foreign_code_executed"])
        self.assertFalse(x["model_weights_updated"])
        self.assertEqual(x["receipt_sha256"],n.sha256({k:v for k,v in x.items()
                                                     if k!="receipt_sha256"}))

if __name__=="__main__":
    unittest.main()
