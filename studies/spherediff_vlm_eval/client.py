"""Image-only GPT-4o judge with durable request and spending guards.

The standard-library transport is intentionally independent of generation code.
Nothing contacts the network unless evaluate_views(..., dry_run=False) is used.
Budgets and request counts apply cumulatively to an output directory. An attempt
is journaled *before* sending; interrupted attempts retain their cost reservation
and are not automatically repeated. This cannot guarantee exactly-once behavior
at the remote service, but does prevent a resume from repeating known successes.
"""
from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import tempfile
import time
from typing import Any, Callable
import urllib.error
import urllib.request


CRITERIA = ("image_quality", "aesthetic_appeal", "distortion", "connectivity")
LABELS = ("Image Quality", "Aesthetic Appeal", "Distortion Level", "Connectivity")
DEFAULT_API_PARAMETERS = {"temperature": 0, "top_p": 1, "seed": 0, "max_tokens": 700}
ENDPOINT = "https://api.openai.com/v1/chat/completions"
PRICING = {
    "as_of": "2026-10-16",
    "model_family": "gpt-4o",
    "input_usd_per_million": 2.5,
    "output_usd_per_million": 10.0,
    "image_tokens_512_high": 255,
    "image_tokens_512_low": 85,
    "source": "https://developers.openai.com/api/docs/models/gpt-4o",
    "vision_source": "https://developers.openai.com/api/docs/guides/images-vision",
    "note": "Published standard rates; no caching discount assumed. Account access is unverified.",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Replace a complete JSON file, never exposing a partially written record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_scores(raw: str) -> tuple[dict[str, int], dict[str, str]]:
    """Parse Table 5's four ``- Criterion: reason → Score`` lines strictly.

    Markdown bullets/bold labels and parenthesized rating names are harmless
    formatting variations. Floats, fractions, booleans, missing/duplicate scores,
    out-of-range values, extra prose, and empty explanations are rejected.
    """
    if not isinstance(raw, str):
        raise ValueError("Judge response must be text")
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if len(lines) != 4:
        raise ValueError("Expected exactly four criterion lines")
    scores: dict[str, int] = {}
    explanations: dict[str, str] = {}
    labels = dict(zip(LABELS, CRITERIA))
    for line in lines:
        line = re.sub(r"^(?:[-*•]\s*)", "", line)
        line = line.replace("**", "")
        match = re.fullmatch(
            r"(Image Quality|Aesthetic Appeal|Distortion Level|Connectivity)\s*:\s*"
            r"(.+?)\s*(?:→|->)\s*([+-]?\d+)(?:\s*\([^\r\n()]+\))?\s*", line
        )
        if not match:
            raise ValueError("Response does not match the published criterion format")
        label, reason, value = match.groups()
        key = labels[label]
        if key in scores:
            raise ValueError("Duplicate criterion")
        if not re.fullmatch("[1-5]", value):
            raise ValueError("Scores must be integer values from 1 to 5")
        reason = reason.strip()
        if reason.startswith("(") and reason.endswith(")"):
            reason = reason[1:-1].strip()
        if not reason or "→" in reason or "->" in reason:
            raise ValueError("Each criterion requires one brief explanation")
        scores[key] = int(value)
        explanations[key] = reason
    if set(scores) != set(CRITERIA):
        raise ValueError("Missing criterion")
    return scores, explanations


class APIError(Exception):
    """Transport error containing safe metadata, never a raw HTTP error body."""

    def __init__(self, status: int | None = None, *, code: str = "transport_error", retry_after: float = 0):
        self.status = status
        self.code = code if re.fullmatch(r"[a-zA-Z0-9_]{1,64}", code) else "api_error"
        self.retry_after = max(0, min(float(retry_after), 60))
        self.retryable = (status in (408, 409, 429) or status is None or (status is not None and status >= 500))
        if self.code in {"insufficient_quota", "billing_hard_limit_reached", "invalid_api_key", "model_not_found"}:
            self.retryable = False
        super().__init__(self.code)


def http_transport(payload: dict, api_key: str, timeout: float) -> dict:
    """Send one API request; retries are owned by the durable outer loop."""
    request = urllib.request.Request(
        ENDPOINT, data=canonical_json(payload).encode("utf-8"), method="POST",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            headers = {name: response.headers.get(name) for name in ("x-request-id", "openai-processing-ms")}
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise APIError(502, code="invalid_json_response") from None
        if not isinstance(result, dict):
            raise APIError(502, code="invalid_json_response")
        result["_http_metadata"] = headers
        return result
    except urllib.error.HTTPError as error:
        code = "http_error"
        try:
            code = str(json.loads(error.read()).get("error", {}).get("code") or code)
        except (ValueError, UnicodeDecodeError, AttributeError):
            pass
        try:
            retry_after = float(error.headers.get("Retry-After", "0"))
        except (ValueError, TypeError):
            retry_after = 0
        raise APIError(error.code, code=code, retry_after=retry_after) from None
    except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError):
        raise APIError(code="network_error") from None


def _configuration(config: dict) -> dict:
    if any("key" in str(key).lower() or "secret" in str(key).lower() for key in config):
        raise ValueError("API credentials must come only from OPENAI_API_KEY")
    model = config.get("model", "gpt-4o")
    if model != "gpt-4o":
        raise ValueError("This protocol requests exactly gpt-4o; snapshot-specific pricing is not verified and no model fallback is allowed")
    parameters = dict(DEFAULT_API_PARAMETERS)
    supplied = config.get("api_parameters", {})
    if set(supplied) - set(DEFAULT_API_PARAMETERS):
        raise ValueError("Unsupported API parameters; evaluator must remain image-only")
    parameters.update(supplied)
    if type(parameters["max_tokens"]) is not int or not 1 <= parameters["max_tokens"] <= 16384:
        raise ValueError("max_tokens must be an integer from 1 to 16384")
    if type(parameters["seed"]) is not int:
        raise ValueError("seed must be an integer")
    for key, maximum in (("temperature", 2), ("top_p", 1)):
        value = parameters[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError("Invalid sampling parameter")
    detail = config.get("image_detail", "high")
    if detail not in ("high", "low", "auto"):
        raise ValueError("image_detail must be high, low, or auto")
    prompt_path = config.get("prompt_path")
    expected_hash = config.get("prompt_sha256")
    if prompt_path is None:
        from .protocol import PROMPT_SHA256, prompt_text
        prompt = prompt_text()
        prompt_bytes = prompt.encode("utf-8")
        expected_hash = PROMPT_SHA256
    else:
        prompt_bytes = Path(prompt_path).read_bytes()
        prompt = prompt_bytes.decode("utf-8")
    prompt_hash = sha256_bytes(prompt_bytes)
    if expected_hash and prompt_hash != expected_hash:
        raise ValueError("Evaluation prompt checksum does not match the verified asset")
    if prompt.count("{image}") != 1:
        raise ValueError("Published prompt must contain exactly one {image} placeholder")
    protocol_hash = config.get("protocol_hash") or config.get("rendering_protocol_sha256")
    if not protocol_hash:
        raise ValueError("An explicit evaluation protocol_hash is required")
    retries = config.get("max_retries", 3)
    if type(retries) is not int or not 0 <= retries <= 10:
        raise ValueError("max_retries must be an integer from 0 to 10")
    timeout = float(config.get("timeout_seconds", 90))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be positive")
    return {"model": model, "api_parameters": parameters, "image_detail": detail, "prompt": prompt,
            "prompt_sha256": prompt_hash, "protocol_hash": str(protocol_hash), "max_retries": retries,
            "timeout_seconds": timeout, "endpoint": ENDPOINT}


def cache_identity(view_record: dict, config: dict) -> tuple[str, dict]:
    """Content identity omits method, prompt, and path to allow safe deduplication."""
    c = config if "prompt" in config and "endpoint" in config else _configuration(config)
    identity = {"version": 1, "image_sha256": view_record["image_sha256"],
                "view_sha256": view_record["view_sha256"], "judge_model": c["model"],
                "evaluation_prompt_sha256": c["prompt_sha256"], "protocol_hash": c["protocol_hash"],
                "rendering_protocol_sha256": view_record.get("rendering_protocol_sha256", c["protocol_hash"]),
                "api_parameters": c["api_parameters"], "image_detail": c["image_detail"], "endpoint": ENDPOINT}
    return sha256_bytes(canonical_json(identity).encode("utf-8")), identity


def cost_estimate(config: dict, pending: int) -> dict:
    """Expected output length is an estimate; budgeting uses a conservative cap.

    A byte-based bound safely exceeds text-token count without adding a tokenizer
    dependency. The image reserve uses the legacy 765-token high-detail bound,
    although current documentation specifies 255 for a 512-square image. This
    avoids exhausting a user budget if service-side preprocessing changes.
    """
    c = config if "prompt" in config and "endpoint" in config else _configuration(config)
    text = c["prompt"].replace("{image}", "")
    estimated_text = math.ceil(len(text) / 4) + 32
    max_text = len(text.encode("utf-8")) + 256
    image_tokens = 85 if c["image_detail"] == "low" else 255
    max_image_tokens = 85 if c["image_detail"] == "low" else 765
    estimated_output = min(280, c["api_parameters"]["max_tokens"])
    expected = ((estimated_text + image_tokens) * 2.5 + estimated_output * 10.0) / 1e6
    reserve = ((max_text + max_image_tokens) * 2.5 + c["api_parameters"]["max_tokens"] * 10.0) / 1e6
    return {"pending_unique_requests": pending, "estimated_cost_usd": round(expected * pending, 8),
            "estimated_cost_per_request_usd": expected, "conservative_max_cost_usd": round(reserve * pending, 8),
            "reserved_cost_per_attempt_usd": reserve, "estimated_text_tokens": estimated_text,
            "image_tokens_per_request": image_tokens, "estimated_output_tokens": estimated_output,
            "max_output_tokens": c["api_parameters"]["max_tokens"], "pricing": dict(PRICING),
            "estimate_note": "Approximate text/output tokens; guards reserve a conservative cost before every attempt, including retries."}


def _payload(image_data: bytes, config: dict) -> dict:
    before, after = config["prompt"].split("{image}")
    parts = []
    if before:
        parts.append({"type": "text", "text": before})
    parts.append({"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(image_data).decode("ascii"),
        "detail": config["image_detail"],
    }})
    if after:
        parts.append({"type": "text", "text": after})
    return {"model": config["model"], "messages": [{"role": "user", "content": parts}],
            **config["api_parameters"]}


def _safe_response(response: dict, key: str) -> dict:
    """Preserve API metadata/raw judge output while redacting any credential echo."""
    serialized = json.dumps(response, ensure_ascii=False)
    if key:
        serialized = serialized.replace(key, "[REDACTED]")
    serialized = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[REDACTED]", serialized)
    return json.loads(serialized)


def _response_record(response: dict, identity: dict, cache_key: str) -> dict:
    result = {"cache_key": cache_key, "identity": identity, "requested_model": identity["judge_model"],
              "resolved_model": response.get("model"), "api_response_id": response.get("id"),
              "api_created": response.get("created"), "system_fingerprint": response.get("system_fingerprint"),
              "usage": response.get("usage"), "raw_api_response": response}
    try:
        if not result["resolved_model"]:
            raise ValueError("Missing resolved model identifier")
        choice = response["choices"][0]
        raw = choice["message"]["content"]
        result["raw_response"] = raw
        result["finish_reason"] = choice.get("finish_reason")
        if choice.get("finish_reason") != "stop":
            raise ValueError("Judge response did not finish normally")
        resolved = str(result["resolved_model"])
        if not re.fullmatch(r"gpt-4o(?:-\d{4}-\d{2}-\d{2})?", resolved):
            raise ValueError("Returned model is outside the requested GPT-4o family")
        if identity["judge_model"] != "gpt-4o" and resolved != identity["judge_model"]:
            raise ValueError("Returned model does not match the explicitly requested snapshot")
        scores, explanations = parse_scores(raw)
        result.update(status="success", scores=scores, explanations=explanations)
    except (ValueError, KeyError, IndexError, TypeError) as error:
        result.update(status="failed", error_code="invalid_judge_response", error_detail=str(error), scores={}, explanations={})
    return result


def _usage_cost(response: dict, reserve: float) -> tuple[float, bool]:
    usage = response.get("usage") or {}
    i, o = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if type(i) is int and type(o) is int and i >= 0 and o >= 0:
        return (i * 2.5 + o * 10.0) / 1e6, True
    return reserve, False


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError("Invalid evaluation journal record")
    return result


@contextlib.contextmanager
def _output_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".api.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another evaluator holds this output directory lock") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def evaluate_views(view_records: list[dict], output: Path, config: dict, dry_run: bool = True,
                   max_requests: int = 0, budget_usd: float = 0, *,
                   transport: Callable | None = None, sleep_fn: Callable = time.sleep) -> dict:
    """Evaluate one image/request; safe defaults perform an offline dry run.

    Optional ``transport(payload, api_key, timeout)`` and ``sleep_fn(seconds)``
    make API behavior fully mockable. API keys are never accepted in config.
    Terminal errors are retained and require explicit human inspection; changing
    output directories is not a substitute for resolving an ambiguous request.
    """
    if type(max_requests) is not int or max_requests < 0:
        raise ValueError("max_requests must be a nonnegative integer")
    if not math.isfinite(budget_usd) or budget_usd < 0:
        raise ValueError("budget_usd must be nonnegative and finite")
    c = _configuration(config)
    output = Path(output)
    with _output_lock(output):
        return _evaluate_locked(view_records, output, c, dry_run, max_requests, budget_usd, transport, sleep_fn)


def _evaluate_locked(view_records, output, c, dry_run, max_requests, budget_usd, transport, sleep_fn):
    cache_dir, attempts_dir = output / "responses", output / "api_attempts"
    cache_dir.mkdir(exist_ok=True)
    attempts_dir.mkdir(exist_ok=True)
    attempts = [_load(path) for path in sorted(attempts_dir.glob("*.json"))]
    by_key: dict[str, list[dict]] = {}
    for attempt in attempts:
        by_key.setdefault(attempt["cache_key"], []).append(attempt)
    rows, unique = [], {}
    for original in view_records:
        row = dict(original)
        row.update(scores={}, explanations={}, requested_model=c["model"], resolved_model=None)
        try:
            cache_key, identity = cache_identity(row, c)
            image_path = Path(row["view_path"])
            data = image_path.read_bytes()
            if sha256_bytes(data) != row["view_sha256"]:
                raise ValueError("Rendered view checksum mismatch")
            if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", data[16:24]) != (512, 512):
                raise ValueError("Expected a 512 x 512 PNG perspective view")
            row["cache_key"] = cache_key
            row["status"] = "pending"
            unique.setdefault(cache_key, {"identity": identity, "view_path": str(image_path)})
        except (OSError, KeyError, ValueError) as error:
            row.update(status="failed", error_code="invalid_view", error_detail=str(error))
        rows.append(row)
    results = {}
    for cache_key, entry in unique.items():
        path = cache_dir / (cache_key + ".json")
        if path.exists():
            result = _load(path)
            if result.get("identity") != entry["identity"]:
                raise ValueError("Cached response identity does not match its filename")
            if result.get("status") == "success":
                scores, explanations = parse_scores(result["raw_response"])
                if result.get("scores") != scores or result.get("explanations") != explanations:
                    raise ValueError("Cached response score validation failed")
            results[cache_key] = {**result, "cache_status": "cached"}
            continue
        historical = by_key.get(cache_key, [])
        received = [a for a in historical if "raw_api_response" in a]
        if received:
            result = _response_record(received[-1]["raw_api_response"], entry["identity"], cache_key)
            atomic_json(path, result)
            results[cache_key] = {**result, "cache_status": "recovered"}
        elif historical:
            # An interruption may have happened after the service processed the
            # image. Never make a potentially duplicate call automatically.
            results[cache_key] = {"status": "blocked", "error_code": "ambiguous_previous_attempt",
                                  "scores": {}, "explanations": {}, "cache_status": "journal"}
    pending_keys = [key for key in unique if key not in results]
    estimate = cost_estimate(c, len(pending_keys))
    cached_before_execution = sum(result.get("status") == "success" for result in results.values())
    attempted_keys = set(by_key)
    reserve = estimate["reserved_cost_per_attempt_usd"]
    cumulative_cost = sum(float(a["accounted_cost_usd"]) for a in attempts)
    prior_attempt_count = len(attempts)
    count = prior_attempt_count
    # Report only a boolean; dry runs never initialize an authenticated client.
    credential_available = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    key = ""
    guard_reason = None
    if not dry_run and pending_keys:
        if max_requests < count + len(pending_keys):
            guard_reason = "request_limit_insufficient_for_pending"
        elif budget_usd + 1e-12 < cumulative_cost + reserve * len(pending_keys):
            guard_reason = "budget_insufficient_for_pending"
        else:
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not key:
                guard_reason = "missing_api_key"
    for cache_key in pending_keys:
        entry = unique[cache_key]
        if dry_run:
            results[cache_key] = {"status": "pending", "cache_status": "not_requested", "scores": {}, "explanations": {}}
            continue
        if guard_reason:
            results[cache_key] = {"status": "blocked", "error_code": guard_reason, "scores": {}, "explanations": {}}
            continue
        result = None
        for retry in range(c["max_retries"] + 1):
            if count >= max_requests or cumulative_cost + reserve > budget_usd + 1e-12:
                guard_reason = "max_requests_reached" if count >= max_requests else "budget_reached"
                result = {"status": "blocked", "error_code": guard_reason, "scores": {}, "explanations": {}}
                break
            count += 1
            attempted_keys.add(cache_key)
            attempt_path = attempts_dir / (f"{count:08d}.json")
            if attempt_path.exists():
                raise ValueError("Request journal numbering conflict")
            attempt = {"attempt_number": count, "cache_key": cache_key, "identity": entry["identity"],
                       "status": "in_flight", "retry_index": retry, "started_unix": time.time(),
                       "accounted_cost_usd": reserve, "cost_is_actual": False}
            atomic_json(attempt_path, attempt)
            cumulative_cost += reserve
            try:
                image_data = Path(entry["view_path"]).read_bytes()
                if sha256_bytes(image_data) != entry["identity"]["view_sha256"]:
                    raise ValueError("Rendered view changed after preflight")
                response = (transport or http_transport)(_payload(image_data, c), key, c["timeout_seconds"])
                response = _safe_response(response, key)
                cost, known = _usage_cost(response, reserve)
                cumulative_cost += cost - reserve
                attempt.update(status="response_received", raw_api_response=response,
                               accounted_cost_usd=cost, cost_is_actual=known, completed_unix=time.time())
                atomic_json(attempt_path, attempt)
                result = _response_record(response, entry["identity"], cache_key)
                result["attempt_number"] = count
                result["accounted_cost_usd"] = cost
                atomic_json(cache_dir / (cache_key + ".json"), result)
                break
            except APIError as error:
                attempt.update(status="request_failed", error_code=error.code, http_status=error.status,
                               retryable=error.retryable, completed_unix=time.time())
                atomic_json(attempt_path, attempt)
                if error.retryable and retry < c["max_retries"]:
                    sleep_fn(max(error.retry_after, min(2 ** retry, 60)))
                    continue
                result = {"cache_key": cache_key, "identity": entry["identity"], "status": "failed",
                          "error_code": error.code, "http_status": error.status, "retryable": error.retryable,
                          "requested_model": c["model"], "scores": {}, "explanations": {}}
                atomic_json(cache_dir / (cache_key + ".json"), result)
                if error.status in (401, 403, 404) or error.code in {"insufficient_quota", "billing_hard_limit_reached", "model_not_found"}:
                    guard_reason = "fatal_api_error_" + error.code
                break
        results[cache_key] = {**result, "cache_status": "new"}
    for row in rows:
        if row.get("cache_key") in results:
            result = results[row["cache_key"]]
            row.update({key: result[key] for key in ("status", "scores", "explanations", "requested_model", "resolved_model",
                        "cache_status", "error_code", "error_detail", "usage", "api_response_id", "accounted_cost_usd") if key in result})
            row["response_path"] = str(cache_dir / (row["cache_key"] + ".json")) if (cache_dir / (row["cache_key"] + ".json")).exists() else ""
    # Pending means safe to submit on a later invocation. A received invalid
    # response or an ambiguous attempt needs inspection, not an automatic retry.
    remaining_pending = sum(result["status"] == "pending" or
                            (result["status"] == "blocked" and cache_key not in attempted_keys)
                            for cache_key, result in results.items())
    remaining_estimate = cost_estimate(c, remaining_pending)
    summary = {**remaining_estimate,
               "pre_execution_pending_unique_requests": estimate["pending_unique_requests"],
               "pre_execution_estimated_cost_usd": estimate["estimated_cost_usd"],
               "pre_execution_conservative_max_cost_usd": estimate["conservative_max_cost_usd"],
               "remaining_unscored_unique_requests": sum(r["status"] != "success" for r in results.values()),
               "blocked_unique_requests": sum(r["status"] == "blocked" for r in results.values()),
               "failed_unique_requests": sum(r["status"] == "failed" for r in results.values()),
               "pending_definition": "Unattempted requests safe to submit later; terminal failures and ambiguous attempts are separately blocked or failed.",
               "dry_run": dry_run, "records": rows, "view_records": len(rows),
               "unique_view_requests": len(unique), "successful_records": sum(r["status"] == "success" for r in rows),
               "failed_records": sum(r["status"] == "failed" for r in rows),
               "blocked_records": sum(r["status"] == "blocked" for r in rows),
               "cached_unique_requests": cached_before_execution,
               "requests_attempted": count - prior_attempt_count, "cumulative_requests_attempted": count,
               "cost_reserved_usd": round(cumulative_cost, 8), "max_requests": max_requests, "budget_usd": budget_usd,
               "credential_available": credential_available, "network_access": "not_checked" if dry_run or count == prior_attempt_count else "attempted",
               "requested_model": c["model"], "api_parameters": c["api_parameters"], "image_detail": c["image_detail"],
               "prompt_sha256": c["prompt_sha256"], "protocol_hash": c["protocol_hash"],
               "budget_scope": "Cumulative over every actual request attempt in this output directory, including retries and ambiguous failures"}
    atomic_json(output / ("api_dry_run.json" if dry_run else "api_run_summary.json"), summary)
    return summary
