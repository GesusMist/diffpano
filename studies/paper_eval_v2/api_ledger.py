"""One SQLite budget ledger for captions, calibration, repeats, and both judges.

Transactions reserve complete prompt blocks before any billable request. A crash
after send retains its full attempt reservation; no unknown attempt is rebilled.
Raw responses are SQLite blobs to avoid a file per API request on quota-limited
filesystems. All public records are credential-free.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import sqlite3
import uuid

from studies.spherediff_vlm_eval.client import canonical_json


class BudgetBlocked(RuntimeError):
    pass


class PendingBlock(RuntimeError):
    pass


def money(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("Expected finite nonnegative monetary value")
    return result


def configured_limits(environ=None):
    environ = os.environ if environ is None else environ
    try:
        budget = money(environ.get("OPENROUTER_EVAL_BUDGET_USD", ""))
        count = int(environ.get("OPENROUTER_EVAL_MAX_REQUESTS", ""))
    except (ValueError, InvalidOperation):
        raise BudgetBlocked("Positive cumulative API budget and request limits are required") from None
    if budget <= 0 or count <= 0:
        raise BudgetBlocked("Positive cumulative API budget and request limits are required")
    return budget, count


def now():
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, path, budget_usd=None, max_requests=None, mode="local_caps"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if mode not in ("local_caps", "provider_managed"):
            raise ValueError("Unknown budget mode")
        self.mode = mode
        self.budget = money(budget_usd) if budget_usd is not None else None
        self.max_requests = int(max_requests) if max_requests is not None else None
        if mode == "local_caps" and (self.budget is None or self.max_requests is None or self.budget <= 0 or self.max_requests <= 0):
            raise BudgetBlocked("Positive cumulative limits are required")
        with self.connection() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS ledger (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS responses (attempt_id TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS observations (request_id TEXT PRIMARY KEY, value TEXT NOT NULL)")
            if not connection.execute("SELECT 1 FROM ledger WHERE id=1").fetchone():
                self._save(connection, {"version": 1, "created_utc": now(), "blocks": {}, "attempts": {}, "limit_history": []})
        with self.transaction() as (connection, state):
            limits = {"budget_usd": str(self.budget) if self.budget is not None else None,
                      "max_requests": self.max_requests, "mode": mode}
            if not state["limit_history"] or any(state["limit_history"][-1][key] != value for key, value in limits.items()):
                state["limit_history"].append(dict(limits, observed_utc=now()))

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(str(self.path), timeout=60, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _save(connection, state):
        connection.execute("INSERT OR REPLACE INTO ledger(id,value) VALUES(1,?)", (canonical_json(state),))

    @contextmanager
    def transaction(self):
        with self.connection() as connection:
            state = json.loads(connection.execute("SELECT value FROM ledger WHERE id=1").fetchone()[0])
            yield connection, state
            self._save(connection, state)

    @staticmethod
    def totals(state):
        attempts = state["attempts"].values()
        charged = sum((money(attempt.get("accounted_cost_usd", attempt["max_cost_usd"])) for attempt in attempts), Decimal(0))
        pending = [item for block in state["blocks"].values() for item in block["items"].values()
                   if item["state"] == "pending"]
        reservation = sum((money(item["max_cost_usd"]) for item in pending), Decimal(0))
        return {"accounted_attempt_cost_usd": charged, "pending_block_reservation_usd": reservation,
                "committed_cost_usd": charged + reservation, "attempts": len(state["attempts"]),
                "reserved_requests": len(pending), "committed_requests": len(state["attempts"]) + len(pending)}

    def _check(self, state):
        totals = self.totals(state)
        if state.get("execution_pause"):
            raise BudgetBlocked(state["execution_pause"])
        if ((self.budget is not None and totals["committed_cost_usd"] > self.budget)
                or (self.max_requests is not None and totals["committed_requests"] > self.max_requests)):
            raise BudgetBlocked("Remaining cumulative budget cannot reserve the complete paired block")

    def reserve_block(self, block_id, requests, stage, provider_remaining_usd=None):
        """Requests contain request_id/max_cost_usd only; valid cache is deducted."""
        if len({item["request_id"] for item in requests}) != len(requests):
            raise ValueError("A block must contain unique request identities")
        identities = {item["request_id"]: str(money(item["max_cost_usd"])) for item in requests}
        with self.transaction() as (connection, state):
            if block_id in state["blocks"]:
                block = state["blocks"][block_id]
                if block["stage"] != stage or {key: item["max_cost_usd"] for key, item in block["items"].items()} != identities:
                    raise ValueError("An existing block cannot change identities or reservation settings")
                self._check(state)
                return block
            active = [key for key, block in state["blocks"].items()
                      if any(item["state"] in ("pending", "inflight", "uncertain") for item in block["items"].values())]
            if active:
                raise PendingBlock("Resume the existing unfinished block before starting another: " + active[0])
            items = {}
            for request_id, maximum in identities.items():
                cached = connection.execute("SELECT value FROM observations WHERE request_id=?", (request_id,)).fetchone()
                items[request_id] = {"max_cost_usd": maximum, "state": "cached" if cached else "pending", "attempt_ids": []}
            block = {"stage": stage, "created_utc": now(), "items": items}
            if provider_remaining_usd is not None:
                available = money(provider_remaining_usd)
                needed = sum((money(item["max_cost_usd"]) for item in items.values() if item["state"] == "pending"), Decimal(0))
                if needed > available:
                    raise BudgetBlocked("Provider-reported remaining balance cannot reserve the complete paired block")
                block["provider_remaining_usd_at_preflight"] = str(available)
            state["blocks"][block_id] = block
            self._check(state)
            return block

    def start_attempt(self, block_id, request_id, metadata):
        with self.transaction() as (connection, state):
            item = state["blocks"][block_id]["items"][request_id]
            if item["state"] != "pending":
                raise PendingBlock("Observation is cached, terminal, or awaiting reconciliation")
            if sum(attempt["state"] == "inflight" for attempt in state["attempts"].values()) >= 2:
                raise PendingBlock("Global API concurrency limit reached")
            self._check(state)
            attempt_id = uuid.uuid4().hex
            item["state"] = "inflight"
            item["attempt_ids"].append(attempt_id)
            state["attempts"][attempt_id] = {"block_id": block_id, "request_id": request_id,
                    "stage": state["blocks"][block_id]["stage"], "metadata": metadata,
                    "max_cost_usd": item["max_cost_usd"], "state": "inflight", "started_utc": now()}
            return attempt_id

    def save_response(self, attempt_id, response):
        """Persist before parsing/reconciliation so crash recovery can reuse it."""
        with self.connection() as connection:
            existing = connection.execute("SELECT value FROM responses WHERE attempt_id=?", (attempt_id,)).fetchone()
            value = canonical_json(response)
            if existing and existing[0] != value:
                raise ValueError("Raw final response is immutable")
            connection.execute("INSERT OR IGNORE INTO responses VALUES(?,?)", (attempt_id, value))

    def response(self, attempt_id):
        with self.connection() as connection:
            row = connection.execute("SELECT value FROM responses WHERE attempt_id=?", (attempt_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def observation(self, request_id):
        with self.connection() as connection:
            row = connection.execute("SELECT value FROM observations WHERE request_id=?", (request_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def finish(self, attempt_id, record, actual_cost=None, retry=False, max_attempts=2):
        with self.transaction() as (connection, state):
            attempt = state["attempts"][attempt_id]
            item = state["blocks"][attempt["block_id"]]["items"][attempt["request_id"]]
            if attempt["state"] not in ("inflight", "uncertain"):
                return attempt
            charged = money(actual_cost) if actual_cost is not None else money(attempt["max_cost_usd"])
            attempt.update(state=record["state"], finished_utc=now(), accounted_cost_usd=str(charged),
                           cost_status="actual" if actual_cost is not None else "reserved_uncertain",
                           generation_id=record.get("generation_id"), result=record)
            item["state"] = "uncertain" if record["state"] == "uncertain" else "terminal"
            if record["state"] == "valid":
                existing = connection.execute("SELECT value FROM observations WHERE request_id=?", (attempt["request_id"],)).fetchone()
                if existing and json.loads(existing[0]) != record:
                    raise ValueError("Never overwrite a valid observation")
                connection.execute("INSERT OR IGNORE INTO observations VALUES(?,?)", (attempt["request_id"], canonical_json(record)))
            if retry and item["state"] != "uncertain" and len(item["attempt_ids"]) < max_attempts:
                item["state"] = "pending"
                totals = self.totals(state)
                if ((self.budget is not None and totals["committed_cost_usd"] > self.budget)
                        or (self.max_requests is not None and totals["committed_requests"] > self.max_requests)):
                    item["state"] = "terminal"
                    attempt["retry_blocker"] = "cumulative_budget_exhausted"
            if charged > money(attempt["max_cost_usd"]):
                state["execution_pause"] = "actual_cost_exceeded_conservative_reservation"
            return attempt

    def reconcile(self, attempt_id, generation):
        """Free metadata lookup updates cost, never initiates another completion."""
        with self.transaction() as (connection, state):
            attempt = state["attempts"][attempt_id]
            if generation.get("id") != attempt.get("generation_id"):
                raise ValueError("Generation reconciliation ID mismatch")
            cost = generation.get("total_cost")
            if cost is not None:
                attempt["accounted_cost_usd"] = str(money(cost))
                attempt["cost_status"] = "actual_generation_metadata"
            attempt["generation_metadata"] = generation
            attempt["reconciled_utc"] = now()

    def snapshot(self):
        with self.transaction() as (connection, state):
            totals = {key: str(value) if isinstance(value, Decimal) else value for key, value in self.totals(state).items()}
            return dict(state, totals=totals, budget_mode=self.mode,
                        configured_budget_usd=str(self.budget) if self.budget is not None else None,
                        configured_max_requests=self.max_requests)

    def usage_by_stage_and_judge(self):
        grouped = {}
        for attempt in self.snapshot()["attempts"].values():
            model = attempt.get("metadata", {}).get("identity", {}).get("judge", {}).get("model", "unspecified")
            key = (attempt["stage"], model)
            row = grouped.setdefault(key, {"stage": key[0], "judge": key[1], "attempts": 0,
                       "actual_cost_usd": Decimal(0), "uncertain_reserved_cost_usd": Decimal(0), "states": {}})
            row["attempts"] += 1
            row["states"][attempt["state"]] = row["states"].get(attempt["state"], 0) + 1
            field = "actual_cost_usd" if str(attempt.get("cost_status", "")).startswith("actual") else "uncertain_reserved_cost_usd"
            row[field] += money(attempt.get("accounted_cost_usd", attempt["max_cost_usd"]))
        return [{key: str(value) if isinstance(value, Decimal) else value for key, value in row.items()}
                for _, row in sorted(grouped.items())]
