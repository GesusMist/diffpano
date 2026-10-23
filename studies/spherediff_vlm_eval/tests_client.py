"""CPU-only client tests; all paid transport calls are mocked."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from .client import APIError, cache_identity, cost_estimate, evaluate_views, parse_scores


VALID = """- Image Quality: (Clear and detailed.) → 5 (Excellent)
- Aesthetic Appeal: (Balanced composition.) → 4 (Good)
- Distortion Level: (Minor stretching.) → 3 (Fair)
- Connectivity: (The center joins smoothly.) → 5 (Excellent)"""


def response(text=VALID):
    return {"id": "chatcmpl-test", "model": "gpt-4o-2024-08-06", "created": 1,
            "system_fingerprint": "fp-test", "usage": {"prompt_tokens": 530, "completion_tokens": 100},
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}


class ParsingTests(unittest.TestCase):
    def test_exact_published_format(self):
        scores, explanations = parse_scores(VALID)
        self.assertEqual(scores, {"image_quality": 5, "aesthetic_appeal": 4, "distortion": 3, "connectivity": 5})
        self.assertEqual(len(explanations), 4)
        self.assertTrue(all(explanations.values()))

    def test_invalid_values_and_format(self):
        for value in ("0", "6", "4.5", "true", "4/5", "05", "+5", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_scores(VALID.replace("→ 5 (Excellent)", "→ " + value, 1))
        for text in (VALID + "\nSummary: great", VALID.rsplit("\n", 1)[0],
                     VALID.replace("Connectivity:", "Image Quality:"),
                     VALID.replace("(Clear and detailed.)", "()")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_scores(text)

    def test_markdown_bullets_do_not_change_rating(self):
        decorated = VALID.replace("- Image Quality:", "• **Image Quality:**")
        self.assertEqual(parse_scores(decorated)[0]["image_quality"], 5)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        prompt = self.root / "prompt.txt"
        prompt.write_text("Published instructions.\n{image}\nPlease evaluate.", encoding="utf-8")
        self.config = {"prompt_path": str(prompt), "protocol_hash": "test-protocol", "max_retries": 2}
        self.output = self.root / "evaluation"
        image = self.root / "method-secret-generation-prompt.png"
        Image.new("RGB", (512, 512), "red").save(image)
        self.record = {"image_sha256": "a" * 64, "view_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                       "view_path": str(image), "panorama_id": "pano1", "method_id": "method-secret",
                       "prompt_id": "secret-generation-prompt", "prompt_name": "secret", "view_id": "yaw0",
                       "elevation_group": "equator", "is_seam": False}
        self.original_bytes = image.read_bytes()

    def run_paid(self, transport, records=None, **kwargs):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-secret-do-not-log"}):
            return evaluate_views(records or [self.record], self.output, self.config, dry_run=False,
                                  max_requests=kwargs.pop("max_requests", 20), budget_usd=kwargs.pop("budget_usd", 10),
                                  transport=transport, sleep_fn=lambda _: None, **kwargs)

    def test_dry_run_without_key_or_network(self):
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            result = evaluate_views([self.record], self.output, self.config)
        self.assertEqual(result["requests_attempted"], 0)
        self.assertEqual(result["pending_unique_requests"], 1)
        self.assertFalse(result["credential_available"])
        self.assertEqual(result["records"][0]["status"], "pending")
        self.assertEqual(result["records"][0]["scores"], {})
        self.assertFalse(list((self.output / "api_attempts").glob("*.json")))

    def test_single_image_blinding_dedup_and_resume(self):
        calls = []
        def transport(payload, key, timeout):
            calls.append(payload)
            self.assertEqual(len(payload["messages"]), 1)
            content = payload["messages"][0]["content"]
            self.assertEqual(sum(p["type"] == "image_url" for p in content), 1)
            text = "".join(p["text"] for p in content if p["type"] == "text")
            self.assertEqual(text, "Published instructions.\n\nPlease evaluate.")
            self.assertNotIn("method-secret", json.dumps(payload))
            return response()
        duplicate = {**self.record, "panorama_id": "pano2", "method_id": "other"}
        first = self.run_paid(transport, [self.record, duplicate])
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["pre_execution_pending_unique_requests"], 1)
        self.assertEqual(first["pending_unique_requests"], 0)
        self.assertEqual(first["estimated_cost_usd"], 0)
        self.assertEqual(first["cached_unique_requests"], 0)
        self.assertEqual(first["successful_records"], 2)
        self.assertEqual(first["records"][1]["method_id"], "other")
        second = self.run_paid(lambda *_: self.fail("Repeated a successful request"), [self.record, duplicate])
        self.assertEqual(second["requests_attempted"], 0)
        self.assertEqual(second["pending_unique_requests"], 0)
        self.assertEqual(second["cached_unique_requests"], 1)
        self.assertEqual(second["records"][0]["resolved_model"], "gpt-4o-2024-08-06")
        self.assertEqual(Path(self.record["view_path"]).read_bytes(), self.original_bytes)
        self.assertNotIn("sk-test-secret-do-not-log", "".join(p.read_text() for p in self.output.rglob("*.json")))

    def test_cache_invalidation_all_required_inputs(self):
        base = cache_identity(self.record, self.config)[0]
        changes = [({**self.record, "image_sha256": "b" * 64}, self.config),
                   ({**self.record, "view_sha256": "b" * 64}, self.config),
                   (self.record, {**self.config, "protocol_hash": "new"}),
                   (self.record, {**self.config, "api_parameters": {"seed": 42}}),
                   (self.record, {**self.config, "image_detail": "low"})]
        for record, config in changes:
            self.assertNotEqual(base, cache_identity(record, config)[0])
        Path(self.config["prompt_path"]).write_text("New instructions {image}")
        self.assertNotEqual(base, cache_identity(self.record, self.config)[0])

    def test_transient_retry_counts_attempts_and_cost(self):
        calls = []
        def transport(*_):
            calls.append(1)
            if len(calls) < 3:
                raise APIError(429, code="rate_limit_exceeded")
            return response()
        result = self.run_paid(transport)
        self.assertEqual(result["requests_attempted"], 3)
        self.assertEqual(result["successful_records"], 1)
        attempts = list((self.output / "api_attempts").glob("*.json"))
        self.assertEqual(len(attempts), 3)
        self.assertGreater(result["cost_reserved_usd"], result["records"][0]["accounted_cost_usd"])

    def test_authentication_failure_never_retries(self):
        calls = []
        def transport(*_):
            calls.append(1)
            raise APIError(401, code="invalid_api_key")
        result = self.run_paid(transport)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["records"][0]["status"], "failed")
        self.assertEqual(result["records"][0]["scores"], {})
        self.assertEqual(result["pending_unique_requests"], 0)
        self.assertEqual(result["failed_unique_requests"], 1)
        self.run_paid(lambda *_: self.fail("Terminal error silently retried"))

    def test_insufficient_quota_does_not_retry(self):
        calls = []
        def transport(*_):
            calls.append(1)
            raise APIError(429, code="insufficient_quota")
        self.run_paid(transport)
        self.assertEqual(len(calls), 1)

    def test_parse_failure_saved_without_retry(self):
        result = self.run_paid(lambda *_: response(VALID.replace("→ 5", "→ 6", 1)))
        self.assertEqual(result["requests_attempted"], 1)
        self.assertEqual(result["records"][0]["status"], "failed")
        saved = json.loads(next((self.output / "responses").glob("*.json")).read_text())
        self.assertIn("→ 6", saved["raw_response"])
        self.assertEqual(saved["scores"], {})
        self.run_paid(lambda *_: self.fail("Invalid response silently retried"))

    def test_hard_request_guard_includes_retries(self):
        def transport(*_):
            raise APIError(code="network_error")
        result = self.run_paid(transport, max_requests=2)
        self.assertEqual(result["requests_attempted"], 2)
        self.assertEqual(result["records"][0]["status"], "blocked")
        self.assertEqual(result["pending_unique_requests"], 0)
        self.assertEqual(result["remaining_unscored_unique_requests"], 1)
        result = self.run_paid(lambda *_: self.fail("Ambiguous request repeated"), max_requests=10)
        self.assertEqual(result["requests_attempted"], 0)
        self.assertEqual(result["records"][0]["error_code"], "ambiguous_previous_attempt")

    def test_budget_guard_includes_ambiguous_request_reservations(self):
        reserve = cost_estimate(self.config, 1)["reserved_cost_per_attempt_usd"]
        def transport(*_):
            raise APIError(code="network_error")
        result = self.run_paid(transport, budget_usd=reserve * 1.5)
        self.assertEqual(result["requests_attempted"], 1)
        self.assertEqual(result["records"][0]["error_code"], "budget_reached")
        self.assertAlmostEqual(result["cost_reserved_usd"], reserve)

    def test_preflight_blocks_insufficient_budget_or_request_count(self):
        for count, budget in ((0, 10), (20, 0)):
            result = self.run_paid(lambda *_: self.fail("Request passed preflight guard"), max_requests=count, budget_usd=budget)
            self.assertEqual(result["requests_attempted"], 0)
            self.assertEqual(result["records"][0]["status"], "blocked")
            self.assertEqual(result["pending_unique_requests"], 1)

    def test_unattempted_view_remains_pending_when_retries_exhaust_guard(self):
        calls = []
        def transport(*_):
            calls.append(1)
            if len(calls) == 1:
                raise APIError(429, code="rate_limit_exceeded")
            return response()
        second = {**self.record, "image_sha256": "b" * 64, "panorama_id": "second"}
        result = self.run_paid(transport, [self.record, second], max_requests=2)
        self.assertEqual(result["requests_attempted"], 2)
        self.assertEqual(result["pre_execution_pending_unique_requests"], 2)
        self.assertEqual(result["pending_unique_requests"], 1)
        self.assertEqual(result["blocked_unique_requests"], 1)
        self.assertEqual(result["remaining_unscored_unique_requests"], 1)

    def test_recover_received_response_without_calling_again(self):
        self.run_paid(lambda *_: response())
        next((self.output / "responses").glob("*.json")).unlink()
        result = self.run_paid(lambda *_: self.fail("Received response was repeated"))
        self.assertEqual(result["successful_records"], 1)
        self.assertEqual(result["records"][0]["cache_status"], "recovered")

    def test_missing_and_modified_views_have_no_false_scores(self):
        missing = {**self.record, "view_path": str(self.root / "missing.png")}
        changed = {**self.record, "view_sha256": "0" * 64}
        result = evaluate_views([missing, changed], self.output, self.config)
        self.assertEqual(result["failed_records"], 2)
        self.assertTrue(all(not row["scores"] for row in result["records"]))

    def test_prompt_integrity_and_no_fallback(self):
        for config in ({**self.config, "prompt_sha256": "wrong"}, {**self.config, "model": "gpt-4o-mini"},
                       {**self.config, "api_key": "never-allowed"}):
            with self.assertRaises(ValueError):
                evaluate_views([self.record], self.output, config)

    def test_returned_model_switch_is_rejected(self):
        wrong = response()
        wrong["model"] = "gpt-4o-mini"
        result = self.run_paid(lambda *_: wrong)
        self.assertEqual(result["records"][0]["status"], "failed")
        self.assertEqual(result["records"][0]["scores"], {})

    def test_unverified_requested_snapshots_rejected_before_cost_or_network(self):
        for model in ("gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20"):
            self.config["model"] = model
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "snapshot-specific pricing"):
                self.run_paid(lambda *_: self.fail("Unpriced model reached the API"))
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "snapshot-specific pricing"):
                cost_estimate(self.config, 1)
        self.assertFalse(self.output.exists())

    def test_empty_key_is_reported_unavailable(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            result = evaluate_views([self.record], self.output, self.config)
        self.assertFalse(result["credential_available"])

    def test_default_verified_prompt_dry_run(self):
        from .protocol import PROMPT_SHA256, rendering_protocol_sha256
        result = evaluate_views([self.record], self.output, {"protocol_hash": rendering_protocol_sha256()})
        self.assertEqual(result["prompt_sha256"], PROMPT_SHA256)
        self.assertEqual(result["requested_model"], "gpt-4o")
        self.assertEqual(result["image_tokens_per_request"], 255)


if __name__ == "__main__":
    unittest.main()
