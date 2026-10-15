"""Offline QA for prospective experiment reservations (no provider calls).

QA: process/thread concurrency cannot oversubscribe; different experiments and
historic usage logs do not share spend; Gemini versions share one cap, Opus a
separate cap; each retry reserves independently; missing/failed usage never
releases an uncertain charge; settlement releases unused capacity; unsupported
rates/bounds block before transport; >272K Codex calls execute and use real
full-request pricing. Snapshot/configuration alone must create no state.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import date
import json
import multiprocessing
import sqlite3
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from substage import budget, gemini, llm
from substage.config import Settings, Workspace
from substage.gemini import GeminiClient, GeminiError
from substage.llm import AnthropicClient, CodexClient, HttpResponse


class Answer(BaseModel):
    text: str


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(gemini, "_backoff", lambda attempt: None)


@pytest.fixture
def settings():
    return Settings(experiment_id="approved-dpaf", service_tier=None)


@pytest.fixture
def ws(tmp_path):
    return Workspace("video", tmp_path / "data", tmp_path / "cache", tmp_path / "logs")


def _reserve_one(_: int) -> bool:
    settings = Settings(experiment_id="concurrent", gemini_budget_usd=5.0)
    try:
        budget.reserve(settings, "gemini-3.8-flash", "s", 1.0)
    except budget.BudgetError:
        return False
    return True


@pytest.mark.parametrize("executor", ["threads", "processes"])
def test_concurrent_requests_cannot_oversubscribe(executor):
    if executor == "threads":
        pool = ThreadPoolExecutor(max_workers=8)
    else:
        pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
    with pool:
        accepted = list(pool.map(_reserve_one, range(20)))
    assert sum(accepted) == 5
    ledger = budget.snapshot(Settings(experiment_id="concurrent", gemini_budget_usd=5.0))
    assert ledger["groups"]["gemini"]["outstanding_usd"] == 5.0
    assert ledger["groups"]["gemini"]["spent_usd"] == 0.0


@pytest.mark.parametrize("experiment_id", ["", "../escape", "a/b", ".", "한글", "a" * 65, 1])
def test_experiment_id_is_validated(experiment_id):
    with pytest.raises(ValueError, match="experiment_id"):
        Settings(experiment_id=experiment_id)


@pytest.mark.parametrize("field, value", [("gemini_budget_usd", 501), ("opus_budget_usd", 301), ("gemini_budget_usd", float("inf")), ("opus_budget_usd", -1)])
def test_budget_cannot_expand_approved_total(field, value):
    with pytest.raises(ValueError, match=field):
        Settings(**{field: value})


def test_snapshot_and_settings_create_no_ledger(settings):
    path = budget.ledger_path(settings)
    assert budget.snapshot(settings)["reservations"] == []
    assert not path.parent.exists()


def test_rejected_first_call_leaves_durable_readable_schema(settings):
    settings = replace(settings, gemini_budget_usd=1.0)
    with pytest.raises(budget.BudgetError, match="exceeds approved"):
        budget.reserve(settings, "gemini-3.8-flash", "rejected", 1.1)
    path = budget.ledger_path(settings)
    with closing(sqlite3.connect(path)) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")}
        assert tables == {"limits", "reservations"}
        assert connection.execute("SELECT COUNT(*) FROM limits").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0
    before = path.read_bytes()
    ledger = budget.snapshot(settings)
    assert ledger["reservations"] == []
    assert ledger["groups"]["gemini"]["spent_usd"] == 0
    assert ledger["groups"]["gemini"]["outstanding_usd"] == 0
    assert path.read_bytes() == before  # The recovery read must remain read-only.
    budget.reserve(settings, "gemini-3.8-flash", "accepted", 0.5)
    assert budget.snapshot(settings)["groups"]["gemini"]["outstanding_usd"] == 0.5


@pytest.mark.parametrize("schema_started", [False, True])
def test_snapshot_is_valid_during_uncommitted_initialization(settings, schema_started):
    path = budget.ledger_path(settings)
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path, isolation_level=None)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        if schema_started:
            writer.execute("CREATE TABLE limits (budget_group TEXT PRIMARY KEY, units INTEGER NOT NULL)")
        with ThreadPoolExecutor(max_workers=1) as pool:
            ledger = pool.submit(budget.snapshot, settings).result(timeout=5)
        assert writer.in_transaction  # The reader neither commits nor mutates it.
        assert ledger["reservations"] == []
        assert ledger["groups"]["gemini"]["spent_usd"] == 0
        assert ledger["groups"]["gemini"]["outstanding_usd"] == 0
        writer.rollback()


def test_snapshot_and_authorization_reject_committed_partial_schema(settings):
    path = budget.ledger_path(settings)
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE limits (budget_group TEXT PRIMARY KEY, units INTEGER NOT NULL)")
        connection.commit()
    with pytest.raises(budget.BudgetError, match="schema is incomplete"):
        budget.snapshot(settings)
    with pytest.raises(budget.BudgetError, match="schema is incomplete"):
        budget.reserve(settings, "gemini-3.8-flash", "invalid-schema", 1.0)


def test_old_experiments_and_historic_usage_do_not_count(settings, ws):
    old = replace(settings, experiment_id="previous", gemini_budget_usd=1.0)
    budget.reserve(old, "gemini-4-argon", "old", 1.0).settle(1.0)
    ws.log_dir.mkdir()
    usage = ws.log_dir / "usage.jsonl"
    historic = '{"stage":"old","model":"gemini-4-argon","cost_usd":99999}\n'
    usage.write_text(historic)
    new = budget.reserve(settings, "gemini-3.8-flash", "new", 1.0)
    assert new.experiment_id == settings.experiment_id
    assert budget.snapshot(settings)["groups"]["gemini"]["spent_usd"] == 0
    assert usage.read_text() == historic


def test_gemini_versions_share_cap_and_opus_is_separate(settings):
    settings = replace(settings, gemini_budget_usd=1.0, opus_budget_usd=1.0)
    budget.reserve(settings, "gemini-3.8-flash", "one", 0.6)
    with pytest.raises(budget.BudgetError, match="exceeds approved"):
        budget.reserve(settings, "gemini-4-argon", "two", 0.5)
    opus = budget.reserve(settings, "anthropic/claude-opus-5-5", "judge", 1.0)
    opus.settle(0.25)
    groups = budget.snapshot(settings)["groups"]
    assert groups["gemini"]["outstanding_usd"] == pytest.approx(0.6)
    assert groups["opus"]["spent_usd"] == 0.25


def test_settlement_releases_only_unused_reservation(settings):
    settings = replace(settings, gemini_budget_usd=1.0)
    first = budget.reserve(settings, "gemini-3.8-flash", "one", 1.0)
    first.settle(0.2)
    first.settle(0.2)  # accounting idempotence
    with pytest.raises(budget.BudgetError, match="different spend"):
        first.settle(0.1)
    budget.reserve(settings, "gemini-4-argon", "two", 0.8)
    with pytest.raises(budget.BudgetError):
        budget.reserve(settings, "gemini-4-argon", "three", 0.000000001)


def test_pinned_limits_cannot_change_between_workspaces(settings):
    budget.reserve(settings, "gemini-3.8-flash", "one", 1.0)
    with pytest.raises(budget.BudgetError, match="pinned limits"):
        budget.reserve(replace(settings, gemini_budget_usd=499.0), "gemini-4-argon", "two", 1.0)


def test_upper_bound_breach_is_recorded_and_blocks_new_calls(settings):
    first = budget.reserve(settings, "gemini-3.8-flash", "one", 1.0)
    with pytest.raises(budget.BudgetError, match="exceeded"):
        first.settle(1.1)
    ledger = budget.snapshot(settings)
    assert ledger["groups"]["gemini"]["spent_usd"] == 1.1
    assert ledger["reservations"][0]["status"] == "over_bound"
    with pytest.raises(budget.BudgetError, match="breached"):
        budget.reserve(settings, "gemini-4-argon", "two", 1.0)


@pytest.mark.parametrize("bound", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_bounds_fail_closed(settings, bound):
    with pytest.raises(budget.BudgetError):
        budget.reserve(settings, "gemini-3.8-flash", "one", bound)
    assert not budget.ledger_path(settings).exists()


def _native_response(*, usage=True, text='{"text":"ok"}'):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=10,
                                       thoughts_token_count=5, cached_content_token_count=0) if usage else None,
    )


class NativeModels:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return outcome()
        return outcome


def _native(ws, settings, outcomes):
    models = NativeModels(outcomes)
    return GeminiClient(ws, settings, client=SimpleNamespace(models=models)), models


def test_unconfigured_budgeted_call_is_blocked_before_network(ws):
    client, models = _native(ws, Settings(), [_native_response()])
    with pytest.raises(GeminiError, match="experiment_id"):
        client.generate(stage="s", prompt="p", schema=Answer)
    assert models.calls == []


@pytest.mark.parametrize("model,tier", [("unknown", None), ("gemini-3.8-flash", "unsupported")])
def test_unsupported_native_bound_or_rate_never_dispatches(ws, settings, model, tier):
    client, models = _native(ws, replace(settings, service_tier=tier), [_native_response()])
    with pytest.raises(GeminiError, match="no validated request bound"):
        client.generate(stage="s", prompt="p", schema=Answer, model=model)
    assert models.calls == []
    assert not budget.ledger_path(settings).exists()


def test_native_request_has_finite_output_and_settles_actual_cost(ws, settings):
    client, models = _native(ws, settings, [_native_response()])
    assert client.generate(stage="s", prompt="p", schema=Answer).text == "ok"
    config = models.calls[0]["config"]
    assert config.max_output_tokens == 65_536
    ledger = budget.snapshot(settings)
    assert ledger["groups"]["gemini"]["outstanding_usd"] == 0
    assert ledger["groups"]["gemini"]["spent_usd"] == pytest.approx((100 * 0.75 + 15 * 3.75) / 1e6)


def test_native_no_usage_keeps_reservation(ws, settings):
    client, _ = _native(ws, settings, [_native_response(usage=False)])
    client.generate(stage="s", prompt="p", schema=Answer)
    ledger = budget.snapshot(settings)
    assert ledger["groups"]["gemini"]["outstanding_usd"] == gemini._request_bound(
        "gemini-3.8-flash",
        settings,
        "s",
        prompt="p",
        schema=Answer,
    )[1]
    assert ledger["reservations"][0]["status"] == "reserved"


def test_native_bound_is_independent_of_prompt_media_and_schema(settings):
    short = gemini._request_bound(
        "gemini-3.8-flash", settings, "s", prompt="short", schema=Answer,
        clips=[gemini.Clip(SimpleNamespace(uri="u", mime_type="video/mp4"), 0, 1)],
    )
    long = gemini._request_bound(
        "gemini-3.8-flash", replace(settings, media_resolution="high"), "s",
        prompt="x" * 4_000_000, system="s" * 100_000, schema=Answer,
        clips=[gemini.Clip(SimpleNamespace(uri="u", mime_type="video/mp4"), 0, 1000, fps=8)],
        images=[SimpleNamespace()], audio=[SimpleNamespace()],
    )
    assert short == long
    assert short[0] == 65_536
    assert short[1] == pytest.approx((1_048_576 * 1.5 + 65_536 * 7.5) / 1e6)


def test_native_priority_and_flex_reservations_cover_standard_fallback(settings):
    standard = gemini._request_bound("gemini-3.8-flash", settings, "s")[1]
    priority = gemini._request_bound(
        "gemini-3.8-flash", replace(settings, service_tier="priority"), "s",
    )[1]
    flex = gemini._request_bound(
        "gemini-3.8-flash", replace(settings, service_tier="flex"), "s",
    )[1]
    assert priority == pytest.approx(standard * 1.8)
    assert flex == standard


def test_native_timeout_keeps_reservation_and_retries_reserve_new(ws, settings, monkeypatch):
    monkeypatch.setattr(gemini, "_retryable", lambda exc: isinstance(exc, TimeoutError))
    client, models = _native(ws, settings, [TimeoutError("uncertain"), _native_response()])
    client.generate(stage="s", prompt="p", schema=Answer)
    ledger = budget.snapshot(settings)
    assert len(models.calls) == 2
    assert [row["status"] for row in ledger["reservations"]] == ["reserved", "settled"]
    assert ledger["groups"]["gemini"]["outstanding_usd"] > 0


def test_invalid_json_bills_attempt_before_retry(ws, settings):
    client, _ = _native(ws, settings, [_native_response(text="not json"), _native_response()])
    client.generate(stage="s", prompt="p", schema=Answer)
    ledger = budget.snapshot(settings)
    assert [row["status"] for row in ledger["reservations"]] == ["settled", "settled"]
    assert ledger["groups"]["gemini"]["outstanding_usd"] == 0


def test_native_transport_does_not_hold_ledger_lock(ws, settings):
    def response():
        with ThreadPoolExecutor(max_workers=1) as pool:
            nested = pool.submit(budget.reserve, settings, "gemini-4-argon", "other", 1.0).result(timeout=5)
        nested.settle(0.0)
        return _native_response()
    client, _ = _native(ws, settings, [response])
    client.generate(stage="s", prompt="p", schema=Answer)
    assert len(budget.snapshot(settings)["reservations"]) == 2


@pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-6-luna"])
@pytest.mark.parametrize("tokens,multipliers", [(272_000, (1.0, 1.0)), (272_001, (2.0, 1.5)), (300_000, (2.0, 1.5))])
def test_codex_large_context_executes_without_budget_and_prices_full_request(ws, model, tokens, multipliers):
    calls = []
    def transport(url, headers, body, timeout):
        calls.append(json.loads(body))
        return HttpResponse(200, json.dumps({
            "status": "completed",
            "usage": {"input_tokens": tokens, "output_tokens": 1000, "input_tokens_details": {"cached_tokens": 100_000}},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"text":"ok"}'}]}],
        }).encode())
    settings = Settings(gemini_budget_usd=0, opus_budget_usd=0)
    client = CodexClient(ws, settings, f"codex/{model}", transport=transport, api_key="test", base_url="https://example.invalid")
    assert client.generate(stage="s", prompt="p", schema=Answer).text == "ok"
    assert len(calls) == 1
    pin, pout, pread, _ = llm._LIST_PRICE[model]
    m_in, m_out = multipliers
    usage = json.loads((ws.log_dir / "usage.jsonl").read_text())
    assert usage["cost_usd"] == pytest.approx(((tokens - 100_000) * pin * m_in + 100_000 * pread * m_in + 1000 * pout * m_out) / 1e6)


def test_opus_timeout_retry_preserves_uncertain_reservation(ws, settings):
    attempts = []
    def transport(url, headers, body, timeout):
        attempts.append(json.loads(body))
        if len(attempts) == 1:
            raise TimeoutError("response lost after paid work")
        return HttpResponse(200, json.dumps({
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "content": [{"type": "tool_use", "input": {"text": "ok"}}],
        }).encode())
    client = AnthropicClient(ws, settings, "anthropic/claude-opus-5-5", transport=transport, api_key="test", base_url="https://example.invalid")
    assert client.generate(stage="judge", prompt="p", schema=Answer).text == "ok"
    ledger = budget.snapshot(settings)
    assert len(attempts) == 2
    assert [row["status"] for row in ledger["reservations"]] == ["reserved", "settled"]
    assert ledger["groups"]["opus"]["spent_usd"] == pytest.approx(0.0008)
    assert ledger["groups"]["gemini"]["spent_usd"] == 0


def test_gemini_actual_priority_and_future_rates():
    assert gemini.price_for("gemini-3.8-flash", date(2026, 10, 14), tier="priority") == pytest.approx((1.35, 6.75))
    assert gemini.price_for("gemini-4-argon", date(2026, 10, 14), tier="priority") == pytest.approx((3.6, 18.0))
    assert gemini.price_for("gemini-4-argon", date(2026, 12, 1)) == (4.0, 20.0)
    assert gemini.price_for("gemini-3.8-flash", date(2027, 1, 1)) == (1.5, 7.5)


@pytest.mark.parametrize("family", ["gemini", "opus"])
def test_insufficient_headroom_blocks_current_call_not_just_next(ws, settings, family):
    settings = replace(settings, gemini_budget_usd=0.01, opus_budget_usd=0.01)
    calls = []
    if family == "gemini":
        client, models = _native(ws, settings, [_native_response()])
        calls = models.calls
    else:
        def transport(url, headers, body, timeout):
            calls.append(body)
            raise AssertionError("must not dispatch an unreserved request")
        client = AnthropicClient(ws, settings, "anthropic/claude-opus-5-5",
                                 transport=transport, api_key="test", base_url="https://example.invalid")
    with pytest.raises(GeminiError, match="exceeds approved"):
        client.generate(stage="s", prompt="p", schema=Answer)
    assert calls == []


def test_retry_needs_new_headroom_for_uncertain_attempt(ws, settings, monkeypatch):
    _, bound = gemini._request_bound(
        "gemini-3.8-flash", settings, "s", prompt="p", schema=Answer
    )
    settings = replace(settings, gemini_budget_usd=bound)
    monkeypatch.setattr(gemini, "_retryable", lambda exc: True)
    client, models = _native(ws, settings, [TimeoutError("paid outcome unknown"), _native_response()])
    with pytest.raises(GeminiError, match="exceeds approved"):
        client.generate(stage="s", prompt="p", schema=Answer)
    assert len(models.calls) == 1
    assert budget.snapshot(settings)["reservations"][0]["status"] == "reserved"


def test_hard_failure_keeps_uncertain_reservation(ws, settings):
    client, models = _native(ws, settings, [TimeoutError("response lost")])
    with pytest.raises(GeminiError, match="API error"):
        client.generate(stage="s", prompt="p", schema=Answer)
    assert len(models.calls) == 1
    assert budget.snapshot(settings)["groups"]["gemini"]["outstanding_usd"] > 0


def test_incomplete_usage_keeps_uncertain_reservation(ws, settings):
    response = _native_response()
    response.usage_metadata.candidates_token_count = None
    client, _ = _native(ws, settings, [response])
    client.generate(stage="s", prompt="p", schema=Answer)
    assert budget.snapshot(settings)["reservations"][0]["status"] == "reserved"


def test_provider_endpoint_dominates_model_prefix():
    assert budget.group_for("gemini-4-argon", "openai-codex") is None
    assert budget.group_for("codex/claude-opus-5-5", "anthropic") == "opus"
    assert budget.group_for("codex/gpt-6-luna", "gemini") == "gemini"


def test_codex_bare_override_uses_endpoint_context_pricing():
    assert llm._price_for("gpt-6.1-sol", 300_000, provider="openai-codex") == (4.0, 15.0, 0.2, 5.0)
    assert llm._price_for("gpt-6-luna", 300_000, provider="openai-codex") == (0.2, 0.75, 0.02, 0.25)


def test_opus_bound_uses_full_context_and_requested_output(ws, settings):
    client = AnthropicClient(
        ws, settings, "anthropic/claude-opus-5-5",
        transport=lambda *args: None, api_key="test", base_url="https://example.invalid",
    )
    payload = {"max_tokens": 16_384}
    short = client._request_bound("s", "anthropic/claude-opus-5-5", payload, prompt="x", schema=Answer)
    long = client._request_bound(
        "s", "anthropic/claude-opus-5-5", payload, prompt="x" * (4 * 272_001),
        system="s" * 100_000, schema=Answer, images=[SimpleNamespace()],
    )
    assert long == short == pytest.approx((1_000_000 * 5.0 + 16_384 * 20.0) / 1e6)
    thinking = client._payload("s", "p", Answer, [], client.model, "high", None)
    assert thinking["max_tokens"] == 16_384 + 16_000
    assert client._request_bound("s", client.model, thinking) == pytest.approx(
        (1_000_000 * 5.0 + thinking["max_tokens"] * 20.0) / 1e6
    )


@pytest.mark.parametrize("with_total", [False, True])
@pytest.mark.parametrize("finish_reason", ["STOP", "MAX_TOKENS"])
def test_thinking_only_terminal_response_settles_evidenced_output_without_holding_headroom(
    ws, settings, with_total, finish_reason
):
    response = SimpleNamespace(
        text="",
        candidates=[SimpleNamespace(finish_reason=finish_reason, content=None)],
        usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=None,
            thoughts_token_count=65_536, cached_content_token_count=0,
            total_token_count=65_636 if with_total else None,
        ),
    )
    client, models = _native(ws, settings, [response] * gemini._MAX_ATTEMPTS)
    with pytest.raises(GeminiError, match="failed after 4 attempts"):
        client.generate(stage="thinking-only", prompt="p", schema=Answer)
    assert len(models.calls) == gemini._MAX_ATTEMPTS
    ledger = budget.snapshot(settings)
    assert all(row["status"] == "settled" for row in ledger["reservations"])
    assert ledger["groups"]["gemini"]["outstanding_usd"] == 0
    expected = (100 * 0.75 + 65_536 * 3.75) / 1e6
    assert ledger["groups"]["gemini"]["spent_usd"] == pytest.approx(expected * gemini._MAX_ATTEMPTS)
    records = [json.loads(line) for line in (ws.log_dir / "usage.jsonl").read_text().splitlines()]
    assert all(row["output_tokens"] == 65_536 and row["usage_complete"] for row in records)


def test_thinking_only_with_unknown_prompt_usage_keeps_reservation(ws, settings):
    response = SimpleNamespace(
        text="", candidates=[SimpleNamespace(finish_reason="MAX_TOKENS", content=None)],
        usage_metadata=SimpleNamespace(prompt_token_count=None, candidates_token_count=None,
                                       thoughts_token_count=100, cached_content_token_count=0),
    )
    client, _ = _native(ws, settings, [response] * gemini._MAX_ATTEMPTS)
    with pytest.raises(GeminiError, match="failed after"):
        client.generate(stage="unknown-input", prompt="p", schema=Answer)
    assert all(row["status"] == "reserved" for row in budget.snapshot(settings)["reservations"])


def test_total_usage_can_evidence_omitted_candidate_count(ws, settings):
    response = _native_response()
    response.usage_metadata.candidates_token_count = None
    response.usage_metadata.total_token_count = 115
    client, _ = _native(ws, settings, [response])
    client.generate(stage="total-evidence", prompt="p", schema=Answer)
    row = budget.snapshot(settings)["reservations"][0]
    assert row["status"] == "settled"
    assert row["cost_usd"] == pytest.approx((100 * 0.75 + 15 * 3.75) / 1e6)


@pytest.mark.parametrize("field,value", [
    ("total_token_count", 1),
    ("prompt_token_count", -1),
    ("candidates_token_count", True),
    ("thoughts_token_count", 1.5),
])
def test_native_malformed_usage_preserves_call_stage_and_uncertain_reservation(
    ws, settings, field, value
):
    response = _native_response()
    setattr(response.usage_metadata, field, value)
    client, _ = _native(ws, settings, [response])
    with pytest.raises(GeminiError, match="invalid provider usage") as error:
        client.generate(stage="per-call-stage", prompt="p", schema=Answer)
    assert error.value.stage == "per-call-stage"
    assert budget.snapshot(settings)["reservations"][0]["status"] == "reserved"


@pytest.mark.parametrize("confirmed_tier,multiplier", [("standard", 1.0), ("priority", 1.8), ("flex", 0.5)])
def test_native_cost_uses_response_confirmed_tier(ws, settings, confirmed_tier, multiplier):
    settings = replace(settings, service_tier="priority")
    response = _native_response()
    response.sdk_http_response = SimpleNamespace(headers={"X-Gemini-Service-Tier": confirmed_tier})
    client, _ = _native(ws, settings, [response])
    client.generate(stage="tier", prompt="p", schema=Answer)
    record = json.loads((ws.log_dir / "usage.jsonl").read_text())
    assert record["cost_usd"] == pytest.approx((100 * 0.75 + 15 * 3.75) * multiplier / 1e6)
    assert record["requested_service_tier"] == "priority"
    assert record["confirmed_service_tier"] == record["priced_service_tier"] == confirmed_tier
    assert record["cost_basis"] == "confirmed_tier_list_price"
    assert budget.snapshot(settings)["groups"]["gemini"]["spent_usd"] == pytest.approx(record["cost_usd"])


def test_native_missing_tier_header_is_labeled_conservative_estimate(ws, settings):
    settings = replace(settings, service_tier="priority")
    client, _ = _native(ws, settings, [_native_response()])
    client.generate(stage="tier-unconfirmed", prompt="p", schema=Answer)
    record = json.loads((ws.log_dir / "usage.jsonl").read_text())
    assert record["confirmed_service_tier"] is None
    assert record["priced_service_tier"] == "priority"
    assert record["cost_basis"] == "conservative_tier_estimate"
    assert record["cost_usd"] == pytest.approx((100 * 0.75 + 15 * 3.75) * 1.8 / 1e6)


def test_native_unknown_confirmed_tier_fails_with_call_stage(ws, settings):
    response = _native_response()
    response.sdk_http_response = SimpleNamespace(headers={"x-gemini-service-tier": "unknown"})
    client, _ = _native(ws, settings, [response])
    with pytest.raises(GeminiError, match="unknown confirmed") as error:
        client.generate(stage="tier-unknown", prompt="p", schema=Answer)
    assert error.value.stage == "tier-unknown"
    assert budget.snapshot(settings)["reservations"][0]["status"] == "reserved"


@pytest.mark.parametrize("max_output", [True, 0, 128_001, "16384"])
def test_opus_invalid_output_bound_blocks_before_transport(ws, settings, max_output):
    client = AnthropicClient(
        ws, settings, "anthropic/claude-opus-5-5",
        transport=lambda *args: None, api_key="test", base_url="https://example.invalid",
    )
    with pytest.raises(GeminiError, match="no validated request bound") as error:
        client._request_bound("opus-bound", client.model, {"max_tokens": max_output})
    assert error.value.stage == "opus-bound"
    assert budget.snapshot(settings)["reservations"] == []


@pytest.mark.parametrize("provider,model", [
    ("anthropic", "anthropic/claude-unknown"),
    ("anthropic", "anthropic/gpt-6.1-sol"),
    ("unknown", "anthropic/claude-opus-5-5"),
])
def test_unknown_non_openai_provider_or_model_is_not_accidentally_uncapped(
    ws, settings, provider, model
):
    calls = []
    client = AnthropicClient(
        ws, settings, model, transport=lambda *args: calls.append(args),
        api_key="test", base_url="https://example.invalid",
    )
    client._OMP_PROVIDER = provider
    with pytest.raises(GeminiError, match="no validated request bound") as error:
        client.generate(stage="unsupported-proxy", prompt="p", schema=Answer)
    assert error.value.stage == "unsupported-proxy"
    assert calls == []
    assert budget.snapshot(settings)["reservations"] == []


@pytest.mark.parametrize("confirmed_tier,multiplier", [(None, 1.0), ("default", 1.0), ("priority", 2.0), ("future-tier", 1.0)])
def test_codex_confirmed_tier_and_cache_write_pricing_are_truthful_without_cutoff(
    ws, confirmed_tier, multiplier
):
    calls = []
    def transport(url, headers, body, timeout):
        calls.append(json.loads(body))
        data = {
            "status": "completed",
            "usage": {
                "input_tokens": 300_000, "output_tokens": 1000,
                "input_tokens_details": {"cached_tokens": 100_000, "cache_write_tokens": 50_000},
            },
            "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"text":"ok"}'}]}],
        }
        if confirmed_tier is not None:
            data["service_tier"] = confirmed_tier
        return HttpResponse(200, json.dumps(data).encode())
    settings = Settings(service_tier="priority", gemini_budget_usd=0, opus_budget_usd=0)
    client = CodexClient(
        ws, settings, "codex/gpt-6.1-sol", transport=transport,
        api_key="test", base_url="https://example.invalid",
    )
    assert client.generate(stage="codex-tier", prompt="p", schema=Answer).text == "ok"
    assert len(calls) == 1
    assert "max_output_tokens" not in calls[0]
    record = json.loads((ws.log_dir / "usage.jsonl").read_text())
    assert record["input_tokens"] == 300_000
    assert record["cache_write_tokens"] == 50_000
    assert record["cost_usd"] == pytest.approx(
        (150_000 * 4.0 + 100_000 * 0.2 + 50_000 * 5.0 + 1000 * 15.0) * multiplier / 1e6
    )
    assert record["confirmed_service_tier"] == confirmed_tier
    assert record["cost_basis"] == (
        "standard_list_price_estimate" if confirmed_tier is None else
        "unvalidated_tier_list_price_estimate" if confirmed_tier == "future-tier" else
        "confirmed_tier_list_price"
    )


def test_opus_malformed_usage_preserves_stage_and_outstanding_reservation(ws, settings):
    def transport(url, headers, body, timeout):
        return HttpResponse(200, json.dumps({
            "usage": {"input_tokens": "not-a-token-count", "output_tokens": 10},
            "content": [{"type": "tool_use", "input": {"text": "ok"}}],
        }).encode())
    client = AnthropicClient(
        ws, settings, "anthropic/claude-opus-5-5", transport=transport,
        api_key="test", base_url="https://example.invalid",
    )
    with pytest.raises(GeminiError, match="invalid provider usage") as error:
        client.generate(stage="opus-usage", prompt="p", schema=Answer)
    assert error.value.stage == "opus-usage"
    assert budget.snapshot(settings)["reservations"][0]["status"] == "reserved"


def test_opus_full_cache_write_usage_fits_model_wide_reservation(ws, settings):
    def transport(url, headers, body, timeout):
        ledger = budget.snapshot(settings)
        assert ledger["reservations"][0]["status"] == "reserved"
        assert ledger["reservations"][0]["upper_usd"] == pytest.approx(
            (1_000_000 * 5.0 + json.loads(body)["max_tokens"] * 20.0) / 1e6
        )
        return HttpResponse(200, json.dumps({
            "usage": {
                "input_tokens": 0, "output_tokens": 1000,
                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 1_000_000,
            },
            "content": [{"type": "tool_use", "input": {"text": "ok"}}],
        }).encode())
    client = AnthropicClient(
        ws, settings, "anthropic/claude-opus-5-5", transport=transport,
        api_key="test", base_url="https://example.invalid",
    )
    assert client.generate(stage="opus-cache-write", prompt="p", schema=Answer).text == "ok"
    ledger = budget.snapshot(settings)
    assert ledger["reservations"][0]["status"] == "settled"
    assert ledger["groups"]["opus"]["outstanding_usd"] == 0
    assert ledger["groups"]["opus"]["spent_usd"] == pytest.approx(5.02)
