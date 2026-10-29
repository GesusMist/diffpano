"""Offline protocol, crash recovery, shared accounting, and concurrency checks."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
import struct
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import vlm, vlm_protocol as protocol
from .api_ledger import BudgetBlocked, Ledger, PendingBlock


def discovery():
    models = {}
    for model in protocol.MODELS:
        supported = ["seed", "max_tokens", "response_format", "structured_outputs"]
        supported += ["temperature", "top_p"] if model == protocol.MODELS[0] else ["reasoning"]
        endpoint = {"provider_name": "OpenAI", "model_id": model, "tag": "openai", "service_tier": None,
                    "supported_parameters": supported, "pricing": {"prompt": "0.000002", "completion": "0.00001"}}
        models[model] = {"catalog_entry": {"architecture": {"input_modalities": ["text", "image"]}},
                         "canonical_slug": model, "selected_endpoint": endpoint,
                         "endpoints_metadata": {"data": {"endpoints": [endpoint]}}}
    return {"models": models, "blockers": []}


def scores(score=3):
    return {key: {"score": score, "justification": "Visible evidence."} for key in protocol.CRITERIA}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.frozen = protocol.protocol(discovery(), {"protocol_id": "common_v2"})

    def test_model_specific_parameters_and_shared_schema(self):
        first, second = [self.frozen["judges"][model] for model in protocol.MODELS]
        self.assertEqual(first["parameters"]["max_tokens"], 700)
        self.assertEqual(second["parameters"]["max_tokens"], 4096)
        self.assertNotIn("temperature", second["parameters"])
        self.assertNotIn("top_p", second["parameters"])
        self.assertEqual(second["parameters"]["reasoning"]["effort"], "medium")
        self.assertEqual(first["parameters"]["response_format"], second["parameters"]["response_format"])
        self.assertFalse(first["provider"]["allow_fallbacks"])
        self.assertTrue(first["provider"]["require_parameters"])

    def test_payload_has_one_png_and_no_method_context(self):
        image = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 512, 512)
        body = protocol.payload(image, self.frozen["judges"][protocol.MODELS[0]], self.frozen)
        content = body["messages"][0]["content"]
        self.assertEqual(sum(part["type"] == "image_url" for part in content), 1)
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertNotIn("tools", body)
        self.assertNotIn("SphereDiff", json.dumps(body))

    def test_parser_rejects_boolean_extra_duplicate_and_out_of_range(self):
        self.assertEqual(protocol.parse_scores(json.dumps(scores(1)))[0]["image_quality"], 1)
        for invalid in (True, 0, 6, 3.5, "3"):
            value = scores()
            value["image_quality"]["score"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                protocol.parse_scores(json.dumps(value))
        with self.assertRaises(ValueError):
            protocol.parse_scores('{"image_quality":1,"image_quality":2}')
        value = scores()
        value["extra"] = {}
        with self.assertRaises(ValueError):
            protocol.parse_scores(json.dumps(value))

    def test_cache_separates_judge_rubric_renderer_and_repeat(self):
        view = {"image_sha256": "pixels", "view_id": "front", "renderer_protocol_sha256": "render"}
        first = self.frozen["judges"][protocol.MODELS[0]]
        original = protocol.cache_identity(view, first, self.frozen)[0]
        changed = copy.deepcopy(self.frozen)
        changed["protocol_sha256"] = "other"
        values = [protocol.cache_identity(view, self.frozen["judges"][protocol.MODELS[1]], self.frozen)[0],
                  protocol.cache_identity(view, first, self.frozen, "repeat-1")[0],
                  protocol.cache_identity(dict(view, renderer_protocol_sha256="other"), first, self.frozen)[0],
                  protocol.cache_identity(view, first, changed)[0]]
        self.assertNotIn(original, values)
        self.assertEqual(len(set(values)), 4)

    def test_missing_model_or_image_capability_is_a_localized_blocker(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, path = protocol.discover(temporary, fetch=lambda url: {"data": []})
            self.assertEqual(len(result["blockers"]), 2)
            with self.assertRaises(ValueError):
                protocol.judge_configs(result)

    def test_matched_seed_cohorts_reuse_overlap_and_keep_confirmation_separate(self):
        views = []
        for prompt, methods, seeds in (("dev", vlm.PRIMARY_METHODS, (0, 1)), ("fresh", ("sphere20", "select", "spherope"), (0,))):
            for method in methods:
                for seed in seeds:
                    for index in range(14):
                        views.append({"prompt_id": prompt, "method_id": method, "seed": seed,
                                      "case_id": "%s-%s-%d" % (prompt, method, seed), "view_id": str(index),
                                      "image_sha256": "%s-%s-%d-%d" % (prompt, method, seed, index)})
        cohorts = {"cohorts": {"development15": {"prompt_ids": ["dev"]}, "development24": {"prompt_ids": ["dev"]},
                                "confirmation24": {"prompt_ids": ["fresh"]}}}
        selected, manifest = vlm.select_cohort_views(views, cohorts)
        self.assertEqual(len(selected), (4*2+3)*14)
        self.assertEqual(len(manifest[0]["case_ids"]), 8)
        self.assertEqual(len(manifest[1]["case_ids"]), 6)
        self.assertEqual(manifest[2]["prompt_ids"], ["fresh"])


class LedgerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "ledger.sqlite3"

    def test_whole_prompt_is_reserved_before_first_request(self):
        ledger = Ledger(self.path, 1, 10)
        with self.assertRaises(BudgetBlocked):
            ledger.reserve_block("prompt", [{"request_id": str(i), "max_cost_usd": ".6"} for i in range(2)], "production")
        self.assertEqual(ledger.snapshot()["blocks"], {})
        self.assertEqual(ledger.snapshot()["totals"]["attempts"], 0)

    def test_provider_managed_mode_has_no_artificial_local_caps(self):
        ledger = Ledger(self.path, mode="provider_managed")
        ledger.reserve_block("prompt", [{"request_id": "one", "max_cost_usd": "10000"}], "production")
        self.assertIsNone(ledger.snapshot()["configured_budget_usd"])
        self.assertIsNone(ledger.snapshot()["configured_max_requests"])

    def test_provider_reported_remaining_is_respected(self):
        ledger = Ledger(self.path, mode="provider_managed")
        with self.assertRaises(BudgetBlocked):
            ledger.reserve_block("prompt", [{"request_id": "one", "max_cost_usd": "1"}], "pilot", provider_remaining_usd=".5")

    def test_all_stages_share_one_global_request_cap(self):
        ledger = Ledger(self.path, 10, 1)
        ledger.reserve_block("caption", [{"request_id": "one", "max_cost_usd": "1"}], "caption")
        attempt = ledger.start_attempt("caption", "one", {})
        ledger.finish(attempt, {"state": "valid", "content": "Caption"}, actual_cost=".1")
        with self.assertRaises(BudgetBlocked):
            ledger.reserve_block("pilot", [{"request_id": "two", "max_cost_usd": "1"}], "pilot")

    def test_concurrent_reservations_never_exceed_two_inflight(self):
        ledger = Ledger(self.path, 10, 10)
        ledger.reserve_block("prompt", [{"request_id": str(i), "max_cost_usd": "1"} for i in range(3)], "production")
        def start(index):
            try:
                return ledger.start_attempt("prompt", str(index), {})
            except PendingBlock:
                return None
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(start, range(3)))
        self.assertEqual(sum(value is not None for value in results), 2)
        self.assertEqual(ledger.snapshot()["totals"]["committed_requests"], 3)

    def test_ambiguous_attempt_retains_cost_and_blocks_new_prompt(self):
        ledger = Ledger(self.path, 10, 10)
        ledger.reserve_block("prompt", [{"request_id": "one", "max_cost_usd": "1"}], "production")
        attempt = ledger.start_attempt("prompt", "one", {})
        ledger.finish(attempt, {"state": "uncertain"})
        self.assertEqual(ledger.snapshot()["totals"]["accounted_attempt_cost_usd"], "1")
        with self.assertRaises(PendingBlock):
            ledger.reserve_block("next", [{"request_id": "two", "max_cost_usd": "1"}], "production")

    def test_actual_cost_releases_unused_reservation_and_cache_deducts(self):
        ledger = Ledger(self.path, 2, 2)
        ledger.reserve_block("prompt", [{"request_id": "one", "max_cost_usd": "1"}], "production")
        attempt = ledger.start_attempt("prompt", "one", {})
        ledger.finish(attempt, {"state": "valid", "scores": {"test": 1}}, actual_cost=".2")
        ledger.reserve_block("reused", [{"request_id": "one", "max_cost_usd": "1"}], "production")
        self.assertEqual(ledger.snapshot()["totals"]["committed_cost_usd"], "0.2")
        self.assertEqual(ledger.snapshot()["totals"]["committed_requests"], 1)


class ClientTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.frozen = protocol.protocol(discovery(), {"protocol_id": "common_v2"})
        self.judge = self.frozen["judges"][protocol.MODELS[0]]
        self.request = {"request_id": "observation", "identity": {"judge": self.judge}, "max_cost_usd": "1",
                        "payload_factory": lambda: {"model": self.judge["model"]}}
        self.response = {"id": "gen-synthetic", "provider": "OpenAI", "model": self.judge["model"],
                         "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(scores(1))}}],
                         "usage": {"cost": ".01"}}
        patcher = patch.dict(os.environ, {"OPENROUTER_API_KEY": "synthetic-offline-test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def fetch(self, path, key):
        if path == "key":
            return {"data": {"limit": None, "limit_remaining": None, "usage": 0, "label": "must_not_persist"}}
        return {"data": {"id": "gen-synthetic", "total_cost": ".01", "provider_name": "OpenAI", "model": self.judge["model"]}}

    def test_valid_low_score_is_cached_without_retry_or_rebilling(self):
        calls = []
        def send(payload, key):
            calls.append(payload)
            return self.response
        first = vlm.execute_requests(self.run, "block", [self.request], "pilot", send=send, fetch=self.fetch)
        second = vlm.execute_requests(self.run, "block", [self.request], "pilot", send=send, fetch=self.fetch)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first[0]["scores"]["image_quality"], 1)
        self.assertEqual(second[0]["state"], "valid")
        self.assertNotIn("must_not_persist", (self.run / "api/usage_summary.json").read_text())

    def test_crash_after_response_is_recovered_without_sending_again(self):
        ledger = vlm.ledger_for(self.run, "provider_managed")
        ledger.reserve_block("block", [{"request_id": "observation", "max_cost_usd": "1"}], "pilot")
        attempt = ledger.start_attempt("block", "observation", {})
        ledger.save_response(attempt, self.response)
        result = vlm.execute_requests(self.run, "block", [self.request], "pilot",
                    send=lambda *_args: self.fail("Rebilled saved response"), fetch=self.fetch)
        self.assertEqual(result[0]["state"], "valid")
        self.assertEqual(ledger.snapshot()["totals"]["attempts"], 1)

    def test_network_uncertainty_never_automatically_retries(self):
        calls = []
        def send(*args):
            calls.append(1)
            raise vlm.RequestFailure("network_outcome_unknown")
        result = vlm.execute_requests(self.run, "block", [self.request], "pilot", send=send, fetch=self.fetch)
        self.assertEqual(result[0]["state"], "uncertain")
        vlm.execute_requests(self.run, "block", [self.request], "pilot", send=send, fetch=self.fetch)
        self.assertEqual(len(calls), 1)

    def test_http200_errors_and_provider_drift_are_not_scores(self):
        self.assertEqual(vlm.parse_response({"error": {"code": 402}}, self.request)["state"], "api_error")
        self.assertEqual(vlm.parse_response(dict(self.response, provider="Azure"), self.request)["state"], "provider_drift")
        self.assertEqual(vlm.parse_response(dict(self.response, model="other-model"), self.request)["state"], "model_drift")

    def test_truncation_refusal_and_invalid_scores_remain_explicit_failures(self):
        for choice, state in (({"finish_reason": "length", "message": {"content": "partial"}}, "truncated"),
                              ({"finish_reason": "stop", "message": {"refusal": "refused"}}, "refusal"),
                              ({"finish_reason": "stop", "message": {"content": json.dumps(scores(0))}}, "invalid_scores")):
            with self.subTest(state=state):
                self.assertEqual(vlm.parse_response(dict(self.response, choices=[choice]), self.request)["state"], state)


if __name__ == "__main__":
    unittest.main()
