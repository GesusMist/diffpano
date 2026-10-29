"""Bounded, resumable OpenRouter execution for the frozen dual-judge study."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
import email.utils
import fcntl
import io
import json
import os
from pathlib import Path
import re
import socket
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request

from studies.spherediff_vlm_eval.client import atomic_json, canonical_json, _safe_response
from studies.spherediff_vlm_eval.io_utils import write_csv
from . import vlm_protocol as vp
from .api_ledger import BudgetBlocked, Ledger, PendingBlock, configured_limits, money

PRIMARY_METHODS = ("sphere20", "rgb", "select", "meanmax050")
METHOD_SETS = {
    "development15": {"methods": PRIMARY_METHODS, "membership": "development15", "scope": "primary"},
    "development24_three_method_secondary": {"methods": ("sphere20", "rgb", "select"), "membership": "development24", "scope": "secondary"},
    "confirmation24": {"methods": ("sphere20", "select", "spherope"), "membership": "confirmation24", "scope": "primary"},
}


def read(path):
    return json.loads(Path(path).read_text())


class RequestFailure(Exception):
    def __init__(self, code, status=None, retry_after=0, response=None):
        self.code = code if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", str(code)) else "api_error"
        self.status = status
        self.retry_after = retry_after
        self.response = response
        self.uncertain = status is None
        self.pause = status in (401, 402, 403)
        self.retryable = status in (408, 409, 429, 500, 502, 503, 504)
        super().__init__(self.code)


def retry_after_seconds(value):
    try:
        return max(0., float(value))
    except (ValueError, TypeError):
        try:
            return max(0., (email.utils.parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, AttributeError):
            return 0.


def transport(payload, key, timeout=180):
    request = urllib.request.Request(vp.ENDPOINT, data=canonical_json(payload).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            headers = {name: response.headers.get(name) for name in ("x-request-id", "x-generation-id")}
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise RequestFailure("unreadable_response", response={"_http_metadata": headers}) from None
        if not isinstance(result, dict):
            raise RequestFailure("nonobject_response")
        result["_http_metadata"] = headers
        return _safe_response(result, key)
    except urllib.error.HTTPError as error:
        try:
            body = json.loads(error.read())
        except (ValueError, UnicodeDecodeError):
            body = {"error": {"code": "http_error"}}
        body = _safe_response(body, key)
        if not isinstance(body, dict):
            body = {"error": {"code": "nonobject_http_error"}}
        if not isinstance(body.get("error"), dict):
            body["error"] = {"code": "unstructured_http_error"}
        code = body.get("error", {}).get("code", "http_error")
        raise RequestFailure(str(code), error.code, retry_after_seconds(error.headers.get("Retry-After")), body) from None
    except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError):
        raise RequestFailure("network_outcome_unknown") from None


def authenticated_get(path, key):
    request = urllib.request.Request("https://openrouter.ai/api/v1/" + path,
                                     headers={"Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return _safe_response(json.load(response), key)
    except urllib.error.HTTPError as error:
        raise RequestFailure("metadata_http_error", error.code) from None
    except (urllib.error.URLError, TimeoutError, socket.timeout, ValueError):
        raise RequestFailure("metadata_unavailable") from None


def provider_balance(key, fetch=authenticated_get):
    """Do not expose key labels, key hashes, or the full key metadata object."""
    data = fetch("key", key).get("data", {})
    return {name: data.get(name) for name in ("limit", "limit_remaining", "usage", "is_free_tier")}


def ledger_for(run, budget_mode):
    if budget_mode == "provider_managed":
        return Ledger(Path(run) / "api/ledger.sqlite3", mode="provider_managed")
    budget, maximum = configured_limits()
    return Ledger(Path(run) / "api/ledger.sqlite3", budget, maximum)


def parse_response(response, request):
    record = {"request_id": request["request_id"], "identity": request["identity"],
              "generation_id": response.get("id"), "model": response.get("model"),
              "provider": response.get("provider"), "usage": response.get("usage"),
              "state": "invalid_response"}
    if response.get("error"):
        error = response["error"]
        record.update(state="api_error", error_code=str(error.get("code", "api_error")) if isinstance(error, dict) else "unstructured_api_error")
        return record
    judge = request["identity"].get("judge", {})
    permitted_models = {judge.get("model"), judge.get("canonical_slug")}
    if response.get("model") not in permitted_models:
        record.update(state="model_drift", expected_models=sorted(item for item in permitted_models if item))
        return record
    if response.get("provider") != "OpenAI":
        record["state"] = "provider_drift" if response.get("provider") else "provider_unverified"
        return record
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return record
    choice = choices[0]
    if not isinstance(choice, dict):
        return record
    record["finish_reason"] = choice.get("finish_reason")
    message = choice.get("message", {})
    if not isinstance(message, dict):
        return record
    if message.get("refusal") or choice.get("finish_reason") == "content_filter":
        record["state"] = "refusal"
        return record
    if choice.get("finish_reason") != "stop":
        record["state"] = "truncated" if choice.get("finish_reason") == "length" else "nonfinal_response"
        return record
    try:
        if request.get("response_kind", "scores") == "scores":
            scores, explanations = vp.parse_scores(message.get("content"))
            record.update(state="valid", scores=scores, explanations=explanations)
        else:
            # Caption schema validation belongs to the frozen caller protocol.
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty caption response")
            record.update(state="valid", content=content)
    except (ValueError, TypeError, KeyError):
        record["state"] = "invalid_scores" if request.get("response_kind", "scores") == "scores" else "invalid_caption"
    return record


def actual_cost(response):
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        return None
    if usage.get("cost") is None:
        return None
    try:
        return str(money(usage["cost"]))
    except (ValueError, ArithmeticError):
        return None


def validated_record(response, request, key, fetch):
    generation = None
    if response.get("id"):
        try:
            generation = fetch("generation?id=" + urllib.parse.quote(response["id"]), key).get("data", {})
        except RequestFailure:
            pass
    enriched = dict(response)
    if generation and not enriched.get("provider"):
        enriched["provider"] = generation.get("provider_name")
    record = parse_response(enriched, request)
    if generation:
        record["generation_metadata"] = generation
        if generation.get("provider_name") and generation["provider_name"] != "OpenAI":
            record["state"] = "provider_drift"
        judge = request["identity"].get("judge", {})
        if generation.get("model") and generation["model"] not in {judge.get("model"), judge.get("canonical_slug")}:
            record["state"] = "model_drift"
    cost = actual_cost(response)
    if cost is None and generation and generation.get("total_cost") is not None:
        cost = str(money(generation["total_cost"]))
    return record, cost, generation


def execute_requests(run, block_id, requests, stage, *, budget_mode="provider_managed",
                     send=transport, fetch=authenticated_get, sleep=time.sleep):
    """Shared caption/pilot/production entrypoint; never stores payload images/keys.

    Each request supplies request_id, identity, payload_factory(), max_cost_usd,
    and optional response_kind='caption'. Factories are evaluated only after the
    complete block has been reserved. Hold the global worker lock in the caller.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise BudgetBlocked("OPENROUTER_API_KEY is absent on this execution host")
    ledger = ledger_for(run, budget_mode)
    balance = provider_balance(key, fetch)
    remaining = balance.get("limit_remaining")
    ledger.reserve_block(block_id, [{"request_id": request["request_id"], "max_cost_usd": request["max_cost_usd"]}
                                    for request in requests], stage, provider_remaining_usd=remaining)
    by_id = {request["request_id"]: request for request in requests}

    def one(request):
        cached = ledger.observation(request["request_id"])
        if cached:
            return cached
        while True:
            state = ledger.snapshot()
            item = state["blocks"][block_id]["items"][request["request_id"]]
            if item["state"] in ("inflight", "uncertain"):
                attempt_id = item["attempt_ids"][-1]
                saved = ledger.response(attempt_id)
                if saved is not None:
                    record, cost, generation = validated_record(saved, request, key, fetch)
                    ledger.finish(attempt_id, record, cost)
                    if generation:
                        ledger.reconcile(attempt_id, generation)
                    return record
                # No repeat request for ambiguous network/crash outcomes.
                return {"request_id": request["request_id"], "state": "uncertain", "attempt_id": attempt_id}
            if item["state"] != "pending":
                return state["attempts"][item["attempt_ids"][-1]]["result"] if item["attempt_ids"] else cached
            payload = request["payload_factory"]()
            attempt_id = ledger.start_attempt(block_id, request["request_id"], {
                "identity": request["identity"], "payload_sha256": vp.digest(payload),
                "response_kind": request.get("response_kind", "scores")})
            retry, delay = False, 0
            try:
                response = send(payload, key)
                ledger.save_response(attempt_id, response)
                record, cost, generation = validated_record(response, request, key, fetch)
                if record["state"] == "api_error":
                    error_body = response.get("error")
                    code = error_body.get("code") if isinstance(error_body, dict) else None
                    numeric = int(code) if str(code).isdigit() else None
                    retry = numeric in (408, 409, 429, 500, 502, 503, 504)
                    record["pause"] = numeric in (401, 402, 403)
                ledger.finish(attempt_id, record, cost, retry=retry)
                if generation:
                    ledger.reconcile(attempt_id, generation)
            except RequestFailure as error:
                if error.response is not None:
                    ledger.save_response(attempt_id, error.response)
                record = {"request_id": request["request_id"], "identity": request["identity"],
                          "state": "uncertain" if error.uncertain else "api_error", "error_code": error.code,
                          "http_status": error.status, "pause": error.pause,
                          "generation_id": (error.response or {}).get("id")}
                retry, delay = error.retryable, error.retry_after
                ledger.finish(attempt_id, record, actual_cost(error.response or {}), retry=retry)
            record["attempt_id"] = attempt_id
            if not retry:
                return record
            item = ledger.snapshot()["blocks"][block_id]["items"][request["request_id"]]
            if item["state"] != "pending":
                return record
            # Honor Retry-After without an uninterruptible long sleep.
            until = time.monotonic() + max(delay, 2 ** len(item["attempt_ids"]))
            while time.monotonic() < until:
                sleep(min(30, until - time.monotonic()))

    results = []
    # Two adjacent judge requests form a pair; no next image after a hard pause.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for offset in range(0, len(requests), 2):
            pair = list(pool.map(one, requests[offset:offset + 2]))
            results.extend(pair)
            if any(result and (result.get("pause") or result.get("state") in
                    ("uncertain", "model_drift", "provider_drift", "provider_unverified")) for result in pair):
                break
    atomic_json(Path(run) / "api/usage_summary.json", {"ledger_path": str(ledger.path),
                "budget_mode": budget_mode, "provider_limits": balance, "totals": ledger.snapshot()["totals"],
                "by_stage_and_judge": ledger.usage_by_stage_and_judge()})
    return results


def worker_lock(run):
    directory = Path(run) / "api"
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / "worker.lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


def request_records(views, frozen, replicate_id="primary"):
    from .rendering import read_view_bytes
    requests = []
    for view in sorted(views, key=lambda item: (item["view_id"], PRIMARY_METHODS.index(item["method_id"])
                                                if item.get("method_id") in PRIMARY_METHODS else 99, item["case_id"])):
        for model in vp.MODELS:
            judge = frozen["judges"][model]
            request_id, identity = vp.cache_identity(view, judge, frozen, replicate_id)
            requests.append({"request_id": request_id, "identity": identity, "view": view,
                "max_cost_usd": vp.max_request_cost(judge, frozen),
                "payload_factory": lambda view=view, judge=judge: vp.payload(read_view_bytes(view), judge, frozen)})
    # Shared pixel identities are evaluated once while all view mappings survive.
    return list({request["request_id"]: request for request in requests}.values())


def select_cohort_views(views, cohorts):
    """Require fourteen views per panorama and matched seeds, not fourteen per method."""
    from collections import defaultdict
    panoramas = defaultdict(list)
    for view in views:
        panoramas[(view["prompt_id"], view["method_id"], view.get("seed", 0), view["case_id"])].append(view)
    preferred = {(item["prompt_id"], item["method_id"], item.get("seed", 0)): item["case_id"]
                 for item in cohorts.get("selected_cells", [])}
    cells = defaultdict(list)
    for (prompt, method, seed, case), items in panoramas.items():
        if len(items) == 14 and len({item["view_id"] for item in items}) == 14:
            if (prompt, method, seed) not in preferred or case == preferred[(prompt, method, seed)]:
                cells[(prompt, method, seed)].append(items)
    resolved = {}
    for key, candidates in cells.items():
        hashes = {vp.digest(sorted((item["view_id"], item["image_sha256"]) for item in items)) for items in candidates}
        if len(hashes) == 1:
            resolved[key] = sorted(candidates, key=lambda items: items[0]["case_id"])[0]
    manifest_cohorts, output = [], {}
    for cohort_id, specification in METHOD_SETS.items():
        membership = cohorts.get("cohorts", {}).get(specification["membership"], {})
        allowed = set(membership.get("prompt_ids", []))
        cases, prompts = [], []
        for prompt in sorted(allowed):
            common_seeds = None
            for method in specification["methods"]:
                present = {seed for candidate, variant, seed in resolved if candidate == prompt and variant == method}
                common_seeds = present if common_seeds is None else common_seeds & present
            if common_seeds:
                prompts.append(prompt)
            for seed in sorted(common_seeds or []):
                for method in specification["methods"]:
                    items = resolved[(prompt, method, seed)]
                    cases.append(items[0]["case_id"])
                    for view in items:
                        output[(view["case_id"], view["view_id"])] = view
        manifest_cohorts.append({"cohort_id": cohort_id, "scope": specification["scope"],
                                "method_ids": list(specification["methods"]), "prompt_ids": prompts,
                                "case_ids": cases, "missing_prompt_ids": sorted(allowed - set(prompts))})
    return list(output.values()), manifest_cohorts


def prepare(run, views_path=None):
    run = Path(run)
    directory = run / "vlm"
    directory.mkdir(exist_ok=True)
    discovered = sorted((directory / "discovery").glob("discovery-*.json"))
    if not discovered:
        discovery, discovery_path = vp.discover(directory / "discovery")
    else:
        discovery_path = max(discovered, key=lambda path: read(path)["observed_utc"])
        discovery = read(discovery_path)
    from .rendering import protocol_definitions
    registry_path = run / "protocol_registry.json"
    if not registry_path.exists():
        raise ValueError("Freeze the shared protocol registry before VLM preparation")
    registry = read(registry_path)
    renderer = protocol_definitions()["common_v2"]
    frozen = vp.protocol(discovery, renderer)
    protocol_path = directory / "pilot_protocol.json"
    if protocol_path.exists() and read(protocol_path) != frozen:
        raise ValueError("Existing pilot protocol differs; use a new explicit protocol version")
    atomic_json(protocol_path, frozen)
    if views_path:
        value = read(views_path)
        views = value["views"] if isinstance(value, dict) else value
    else:
        views = []
    cohorts = read(run / "cohorts.json") if (run / "cohorts.json").exists() else {"cohorts": {}}
    views, selected_cohorts = select_cohort_views(views, cohorts)
    paired_prompts = sorted({view["prompt_id"] for view in views})
    repeats = [view for view in views if int(vp.digest({"image": view["image_sha256"], "seed": 20261026})[:8], 16) < int(.05 * 2**32)]
    atomic_json(directory / "production_views.json", {"views": views, "cohort_prompt_ids": paired_prompts,
                "cohorts": selected_cohorts,
                "repeat_view_identities": [{"image_sha256": view["image_sha256"], "view_id": view["view_id"]} for view in repeats],
                "selection_before_scores": True, "registry_sha256": vp.digest(registry)})
    costs = {model: vp.max_request_cost(judge, frozen) for model, judge in frozen["judges"].items()}
    count = len(request_records(views, frozen))
    maximum = sum((money(request["max_cost_usd"]) for request in request_records(views, frozen)), Decimal(0))
    result = {"state": "pilot_pending", "paired_prompts": len(paired_prompts), "primary_unique_requests": count,
              "cohorts": [{key: value for key, value in item.items() if key != "case_ids"} for item in selected_cohorts],
              "expected_15_scene_requests": 1680, "expected_24_scene_requests": 2688,
              "per_request_max_usd": costs, "production_max_usd_before_repeats_retries": str(maximum),
              "preselected_repeat_images": len(repeats), "protocol_sha256": frozen["protocol_sha256"],
              "discovery_path": str(discovery_path), "paid_requests": 0}
    atomic_json(directory / "preparation.json", result)
    return result


def calibration(run, source, source_id):
    """Deterministic real-view perturbations; never choose examples by judge scores."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from .rendering import render_common_views
    run, source = Path(run), Path(source)
    source_hash = vp.sha256_bytes(source.read_bytes())
    provenance_path = source.with_suffix(".json")
    provenance = read(provenance_path) if provenance_path.exists() else None
    if provenance is None or provenance.get("sha256") != source_hash or provenance.get("excluded_from_confirmation") is not True:
        raise ValueError("Calibration needs verified real-source provenance excluded from confirmation")
    with Image.open(source) as image:
        base = render_common_views(np.asarray(image.convert("RGB")), [(0, 0)])[0]
    variants = []
    y, x = np.indices(base.shape[:2], dtype=np.float32)
    for level, amount in enumerate((0, 2, 6, 16, 32)):
        displaced = base.copy()
        displaced[:, 256:] = np.roll(base[:, 256:], amount, axis=0)
        variants.append(("seam-%d" % level, displaced, {"kind": "seam_displacement", "level": level, "right_half_vertical_roll_px": amount}))
        warped = cv2.remap(base, x + amount * np.sin(2 * np.pi * y / 512), y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        variants.append(("warp-%d" % level, warped, {"kind": "geometric_warp", "level": level, "sinusoidal_horizontal_amplitude_px": amount}))
    for sigma in (2, 5):
        variants.append(("blur-%d" % sigma, cv2.GaussianBlur(base, (0, 0), sigma), {"kind": "blur", "sigma": sigma}))
    for sigma in (5, 15):
        noise = np.random.default_rng(20261026).normal(0, sigma, base.shape)
        variants.append(("noise-%d" % sigma, np.clip(np.rint(base + noise), 0, 255).astype(np.uint8), {"kind": "noise", "sigma": sigma, "seed": 20261026}))
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    variants.append(("grayscale-style", np.repeat(gray[..., None], 3, axis=2), {"kind": "geometry_preserving_style", "operation": "RGB_to_grayscale"}))
    database = (run / "vlm/calibration.sqlite3").resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with sqlite3.connect(str(database)) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS views (key TEXT PRIMARY KEY, sha256 TEXT NOT NULL, png BLOB NOT NULL)")
        for name, array, settings in variants:
            stream = io.BytesIO()
            Image.fromarray(array).save(stream, format="PNG", compress_level=6, optimize=False)
            raw = stream.getvalue()
            identity = {"source_sha256": source_hash, "source_id": source_id, "settings": settings,
                        "base_view": {"yaw": 0, "pitch": 0, "fov": 90, "size": 512}, "version": "real-calibration-v1"}
            key = vp.digest(identity)
            checksum = vp.sha256_bytes(raw)
            old = connection.execute("SELECT sha256 FROM views WHERE key=?", (key,)).fetchone()
            if old and old[0] != checksum:
                raise ValueError("Calibration byte identity changed")
            connection.execute("INSERT OR IGNORE INTO views VALUES(?,?,?)", (key, checksum, raw))
            rows.append({"prompt_id": "calibration/" + source_id, "case_id": name, "method_id": "real_calibration",
                         "view_id": name, "image_path": "sqlite://" + str(database) + "#" + key,
                         "image_sha256": checksum, "renderer_protocol_sha256": vp.digest(identity),
                         "is_polar": False, "is_seam": settings["kind"] == "seam_displacement", "seed": 0,
                         "calibration": identity})
    result = {"source_path": str(source), "source_id": source_id, "source_sha256": source_hash,
              "source_provenance": provenance, "separate_from_confirmation": True, "views": rows}
    path = run / "vlm/calibration.json"
    if path.exists() and read(path) != result:
        raise ValueError("Frozen calibration source or perturbations changed")
    atomic_json(path, result)
    sheet = Image.new("RGB", (5 * 256, 3 * 280), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, array, settings) in enumerate(variants):
        column, row = index % 5, index // 5
        sheet.paste(Image.fromarray(array).resize((256, 256)), (column * 256, row * 280 + 24))
        draw.text((column * 256 + 4, row * 280 + 4), name, fill="black")
    sheet.save(run / "vlm/calibration/contact_sheet.png")
    return {"calibration_views": len(rows), "planned_dual_judge_requests": 2 * len(rows), "manifest": str(path)}


def evaluate(run, stage, budget_mode="provider_managed"):
    run = Path(run)
    with worker_lock(run):
        frozen = read(run / "vlm/pilot_protocol.json")
        for name, expected in frozen["implementation_sha256"].items():
            if vp.sha256_bytes((Path(__file__).parent / name).read_bytes()) != expected:
                raise ValueError("VLM implementation changed after protocol freeze: " + name)
        if stage == "pilot":
            views = read(run / "vlm/calibration.json")["views"]
        else:
            pilot = read(run / "vlm/pilot_result.json")
            if not pilot.get("production_ready"):
                raise ValueError("A valid calibrated pilot is required before production")
            production_protocol = run / "vlm/production_protocol.json"
            if production_protocol.exists() and read(production_protocol) != frozen:
                raise ValueError("Production protocol drift")
            atomic_json(production_protocol, frozen)
            production_manifest = read(run / "vlm/production_views.json")
            views = production_manifest["views"]
        repeat_keys = set() if stage == "pilot" else {
            (item["image_sha256"], item["view_id"]) for item in production_manifest["repeat_view_identities"]}
        grouped = {}
        for view in views:
            grouped.setdefault(view["prompt_id"], []).append(view)
        results = []
        blocked = None
        priority = {} if stage == "pilot" else {prompt: min(index for index, cohort in enumerate(production_manifest["cohorts"])
              if prompt in cohort["prompt_ids"]) for prompt in grouped}
        for prompt in sorted(grouped, key=lambda prompt: (priority.get(prompt, 0), prompt)):
            requests = request_records(grouped[prompt], frozen)
            repeat_views = [view for view in grouped[prompt] if (view["image_sha256"], view["view_id"]) in repeat_keys]
            requests += request_records(repeat_views, frozen, replicate_id="repeat-1")
            block_id = vp.digest({"stage": stage, "prompt_id": prompt, "protocol": frozen["protocol_sha256"]})
            try:
                block_results = execute_requests(run, block_id, requests, stage, budget_mode=budget_mode)
            except (BudgetBlocked, PendingBlock, RequestFailure) as error:
                blocked = type(error).__name__ + ": " + str(error)
                break
            results.extend(block_results)
            if any(row and row["state"] in ("uncertain", "model_drift", "provider_drift", "provider_unverified") or row and row.get("pause") for row in block_results):
                blocked = "unfinished_or_drifted_block"
                break
        valid = [row for row in results if row and row.get("state") == "valid"]
        planned = len(request_records(views, frozen)) + len(request_records(
            [view for view in views if (view["image_sha256"], view["view_id"]) in repeat_keys], frozen, "repeat-1"))
        result = {"stage": stage, "state": "blocked" if blocked else "complete", "blocker": blocked,
                  "planned_requests": planned, "valid_requests": len(valid),
                  "recorded_results": len(results), "results": results,
                  "protocol_sha256": frozen["protocol_sha256"]}
        if stage == "pilot":
            result["production_ready"] = not blocked and len(valid) == result["planned_requests"]
            result["calibration_note"] = "All calibration observations retained; sensitivity and style-confounding diagnostics are exported separately. No judge selection or rubric tuning by method rank."
            result["calibration_diagnostics"] = calibration_diagnostics(views, frozen, ledger_for(run, budget_mode))
        atomic_json(run / ("vlm/" + stage + "_result.json"), result)
        if stage == "production":
            aggregate(run, budget_mode)
        return {key: value for key, value in result.items() if key != "results"}


def calibration_diagnostics(views, frozen, ledger):
    result = {}
    for model in vp.MODELS:
        rows = []
        for view in views:
            request_id, _ = vp.cache_identity(view, frozen["judges"][model], frozen)
            observation = ledger.observation(request_id)
            if observation:
                rows.append({"view_id": view["view_id"], **view["calibration"]["settings"], "scores": observation["scores"]})
        diagnostics = {"observations": rows, "valid_count": len(rows), "expected_count": len(views), "flags": []}
        for family, criterion in (("seam_displacement", "local_continuity"), ("geometric_warp", "geometric_distortion")):
            levels = sorted([row for row in rows if row["kind"] == family], key=lambda row: row["level"])
            if len(levels) == 5:
                delta = levels[-1]["scores"][criterion] - levels[0]["scores"][criterion]
                diagnostics[family + "_severe_minus_clean"] = delta
                if delta >= 0:
                    diagnostics["flags"].append(family + "_no_score_decline_in_this_control")
        clean = next((row for row in rows if row["view_id"] == "warp-0"), None)
        style = next((row for row in rows if row["kind"] == "geometry_preserving_style"), None)
        if clean and style:
            diagnostics["style_geometry_minus_clean"] = style["scores"]["geometric_distortion"] - clean["scores"]["geometric_distortion"]
        result[model] = diagnostics
    return result


def aggregate(run, budget_mode="provider_managed"):
    """Separate judges, complete view subsets, and equal-weight prompt clusters."""
    from collections import defaultdict
    import numpy as np
    from .statistics import summarize
    run = Path(run)
    frozen = read(run / "vlm/pilot_protocol.json")
    manifest = read(run / "vlm/production_views.json")
    ledger = ledger_for(run, budget_mode)
    views = manifest["views"]
    view_rows, panorama_rows, group_rows, paired_rows = [], [], [], []
    grouped = defaultdict(list)
    for view in views:
        for model in vp.MODELS:
            request_id, _ = vp.cache_identity(view, frozen["judges"][model], frozen)
            observation = ledger.observation(request_id)
            row = {key: view.get(key) for key in ("prompt_id", "case_id", "method_id", "seed", "view_id", "is_polar", "is_seam", "image_sha256")}
            row.update(judge=model, request_id=request_id, state="valid" if observation else "missing_or_failed",
                       scores=observation["scores"] if observation else None)
            view_rows.append(row)
            grouped[(model, view["case_id"])].append(row)
    for (model, case_id), rows in sorted(grouped.items()):
        metrics, coverage = {}, {}
        for metric, criterion, subset, required in (
                ("quality", "image_quality", lambda row: True, 14),
                ("aesthetics", "aesthetic_appeal", lambda row: True, 14),
                ("distortion", "geometric_distortion", lambda row: True, 14),
                ("polar_distortion", "geometric_distortion", lambda row: row["is_polar"], 2),
                ("wrap_continuity", "local_continuity", lambda row: row["is_seam"], 3),
                ("all_view_continuity", "local_continuity", lambda row: True, 14)):
            chosen = [row for row in rows if subset(row)]
            valid = [row for row in chosen if row["scores"] is not None]
            coverage[metric] = {"valid_views": len(valid), "required_views": required}
            metrics[metric] = float(np.mean([row["scores"][criterion] for row in valid])) if len(chosen) == len(valid) == required else None
        panorama_rows.append({"judge": model, "case_id": case_id, "prompt_id": rows[0]["prompt_id"],
                              "method_id": rows[0]["method_id"], "seed": rows[0]["seed"], "metrics": metrics, "coverage": coverage})
    agreement = []
    for cohort in manifest["cohorts"]:
        selected_panorama_rows = [row for row in panorama_rows if row["case_id"] in cohort["case_ids"]]
        comparisons = [(baseline, comparison) for baseline, comparison in
                       (("sphere20", "select"), ("rgb", "select"), ("select", "meanmax050"), ("spherope", "select"))
                       if baseline in cohort["method_ids"] and comparison in cohort["method_ids"]]
        for model in vp.MODELS:
            selected = [row for row in selected_panorama_rows if row["judge"] == model]
            for metric in ("quality", "aesthetics", "distortion", "polar_distortion", "wrap_continuity", "all_view_continuity"):
                groups, pairs = summarize(selected, metric, cohort["cohort_id"], cohort["prompt_ids"], comparisons=comparisons)
                for row in groups + pairs:
                    row.update(judge=model, scale="1_to_5_higher_is_better", protocol_sha256=frozen["protocol_sha256"])
                    if "scope" in row:
                        row["scope"] = "secondary_diagnostic" if metric == "all_view_continuity" else cohort["scope"]
                group_rows.extend(groups)
                paired_rows.extend(pairs)
        agreement.extend(judge_agreement(selected_panorama_rows, cohort["cohort_id"], comparisons))
    output = {"per_view": view_rows, "per_panorama": panorama_rows, "group_metrics": group_rows,
              "paired_comparisons": paired_rows, "judge_agreement": agreement, "ledger_totals": ledger.snapshot()["totals"]}
    for name, rows in output.items():
        atomic_json(run / ("vlm/" + name + ".json"), rows)
        if isinstance(rows, list) and rows:
            write_csv(run / ("vlm/" + name + ".csv"), rows, sorted({key for row in rows for key in row}))
    return {"panoramas": len(panorama_rows), "views": len(view_rows), "group_rows": len(group_rows)}


def judge_agreement(panorama_rows, cohort_id, comparisons):
    import numpy as np
    agreement = []
    by_judge = {model: {(row["method_id"], row["prompt_id"], row["seed"]): row for row in panorama_rows if row["judge"] == model} for model in vp.MODELS}
    for metric in ("quality", "aesthetics", "distortion", "polar_distortion", "wrap_continuity"):
        common = sorted(set(by_judge[vp.MODELS[0]]) & set(by_judge[vp.MODELS[1]]))
        common = [key for key in common if all(by_judge[model][key]["metrics"][metric] is not None for model in vp.MODELS)]
        a = np.array([by_judge[vp.MODELS[0]][key]["metrics"][metric] for key in common])
        b = np.array([by_judge[vp.MODELS[1]][key]["metrics"][metric] for key in common])
        def ranks(values):
            return np.array([np.mean(np.where(np.sort(values) == value)[0]) for value in values])
        rank_correlation = None
        if len(common) > 1 and np.std(a) > 0 and np.std(b) > 0:
            rank_correlation = float(np.corrcoef(ranks(a), ranks(b))[0, 1])
        disagreements = sorted([{"method_id": key[0], "prompt_id": key[1], "seed": key[2],
                                 "gpt4o": float(first), "gpt6sol": float(second), "difference": float(second-first)}
                                for key, first, second in zip(common, a, b)], key=lambda row: (-abs(row["difference"]), row["prompt_id"]))[:5]
        ordering = []
        for prompt in sorted({key[1] for key in common}):
            for baseline, comparison in comparisons:
                differences = []
                for model in vp.MODELS:
                    left = [row["metrics"][metric] for key, row in by_judge[model].items() if key[0] == baseline and key[1] == prompt and key in common]
                    right = [row["metrics"][metric] for key, row in by_judge[model].items() if key[0] == comparison and key[1] == prompt and key in common]
                    differences.append(float(np.mean(right) - np.mean(left)) if left and right else None)
                if all(value is not None for value in differences):
                    ordering.append({"prompt_id": prompt, "baseline": baseline, "comparison": comparison,
                                     "differences": dict(zip(vp.MODELS, differences)), "same_order": bool(np.sign(differences[0]) == np.sign(differences[1]))})
        agreement.append({"cohort_id": cohort_id, "metric": metric, "common_panorama_count": len(common), "common_prompt_ids": sorted({key[1] for key in common}),
                          "spearman_panorama": rank_correlation, "mean_absolute_score_difference": float(np.mean(np.abs(b-a))) if len(common) else None,
                          "paired_method_ordering": ordering, "largest_disagreements": disagreements})
    return agreement


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("discover", "prepare", "calibration", "pilot", "production", "aggregate"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--views")
    parser.add_argument("--source")
    parser.add_argument("--source-id")
    parser.add_argument("--budget-mode", choices=("provider_managed", "local_caps"), default="provider_managed")
    args = parser.parse_args(argv)
    if args.command == "discover":
        data, path = vp.discover(Path(args.run) / "vlm/discovery")
        result = {"path": str(path), "blockers": data["blockers"], "paid_requests": 0}
    elif args.command == "prepare":
        result = prepare(args.run, args.views)
    elif args.command == "calibration":
        if not args.source or not args.source_id:
            parser.error("calibration requires --source and --source-id")
        result = calibration(args.run, args.source, args.source_id)
    elif args.command == "aggregate":
        result = aggregate(args.run, args.budget_mode)
    else:
        result = evaluate(args.run, args.command, args.budget_mode)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
