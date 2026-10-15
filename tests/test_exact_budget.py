"""Offline QA for native Gemini model-wide budget authorization.

QA: reserve full input/output maxima before generation; never depend on an
incomplete countTokens shape; retain the provider's full thinking/output
allowance; cover high-resolution and schema overhead; block the current call
without sufficient worst-case headroom; keep injected SDK clients equally safe.
No test here invokes a real provider or writes the experiment's live ledger.
"""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from substage import budget, gemini
from substage.config import Settings, Workspace
from substage.gemini import Clip, GeminiClient, GeminiError, UploadedFile
from substage.models import ChunkObservations, PerceptionLedger, SceneTranslation


class Answer(BaseModel):
    text: str


class _Models:
    def __init__(self, *, response=None, count_result=31, count_error=None):
        self.response = response or SimpleNamespace(text='{"text":"ok"}', usage_metadata=None)
        self.count_result = count_result
        self.count_error = count_error
        self.count_calls: list[dict] = []
        self.generate_calls: list[dict] = []

    def count_tokens(self, **kwargs):
        self.count_calls.append(kwargs)
        if self.count_error is not None:
            raise self.count_error
        return SimpleNamespace(total_tokens=self.count_result)

    def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        return self.response


class _MissingCountModels:
    def __init__(self):
        self.generate_calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        return SimpleNamespace(text='{"text":"ok"}', usage_metadata=None)


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(gemini, "_backoff", lambda attempt: None)


@pytest.fixture
def settings() -> Settings:
    return Settings(experiment_id="native-maxima-budget", service_tier=None)


@pytest.fixture
def ws(tmp_path) -> Workspace:
    return Workspace("video", tmp_path / "data", tmp_path / "cache", tmp_path / "logs")


def _live(monkeypatch, ws, settings, models):
    # Exercise client=None without constructing a network-enabled SDK client.
    monkeypatch.setattr(gemini.genai, "Client", lambda: SimpleNamespace(models=models))
    return GeminiClient(ws, settings)


def test_live_native_reserves_maxima_and_preserves_complete_generation_body(
    monkeypatch, tmp_path, ws, settings
):
    models = _Models()
    image = tmp_path / "frame.png"
    image.write_bytes(b"offline image payload")
    audio = tmp_path / "dialogue.wav"
    audio.write_bytes(b"offline audio payload")
    clip = Clip(UploadedFile("files/scene", "https://files.example/scene", "video/mp4"), 5, 9, fps=2)
    client = _live(monkeypatch, ws, settings, models)

    assert client.generate(
        stage="native", prompt="a" * 400, schema=Answer, clips=[clip],
        images=[image], audio=[audio], system="Translate into Korean.",
    ).text == "ok"

    assert models.count_calls == []
    generation = models.generate_calls[0]
    assert generation["model"] == settings.model_main
    assert [content.role for content in generation["contents"]] == ["user"]
    parts = generation["contents"][0].parts
    assert len(parts) == 4
    assert parts[0].file_data.file_uri == clip.file.uri
    assert parts[0].video_metadata.start_offset == "5s"
    assert parts[0].video_metadata.end_offset == "9s"
    assert parts[0].video_metadata.fps == 2
    assert parts[1].inline_data.data == image.read_bytes()
    assert parts[2].inline_data.data == audio.read_bytes()
    assert parts[3].text == "a" * 400
    assert generation["config"].system_instruction == "Translate into Korean."
    assert generation["config"].max_output_tokens == 65_536
    assert generation["config"].candidate_count == 1
    ledger = budget.snapshot(settings)
    # The announced maximum rates cover requests that span a pricing change.
    expected = (1_048_576 * 1.50 + 65_536 * 7.50) / 1e6
    assert ledger["groups"]["gemini"]["outstanding_usd"] == pytest.approx(expected)
    assert ledger["reservations"][0]["status"] == "reserved"


@pytest.mark.parametrize("model", ["gemini-3.8-flash", "gemini-4-argon"])
@pytest.mark.parametrize("schema", [Answer, SceneTranslation, PerceptionLedger, ChunkObservations])
def test_every_schema_keeps_provider_output_maximum(settings, model, schema):
    max_output, bound = gemini._request_bound(model, settings, "native", schema=schema)
    assert max_output == 65_536
    pin, pout = (1.5, 7.5) if model == "gemini-3.8-flash" else (4.0, 20.0)
    assert bound == pytest.approx((1_048_576 * pin + 65_536 * pout) / 1e6)


@pytest.mark.parametrize("count_error", [
    RuntimeError("count endpoint unavailable"),
    ValueError("system_instruction parameter is only supported in Gemini Enterprise Agent Platform mode, not in Gemini Developer API mode."),
])
def test_unsupported_count_shape_never_blocks_or_invents_system_content(
    monkeypatch, ws, settings, count_error
):
    models = _Models(count_error=count_error)
    client = _live(monkeypatch, ws, settings, models)
    assert client.generate(stage="native", prompt="Dialogue", schema=Answer, system="Translate.").text == "ok"
    assert models.count_calls == []
    assert len(models.generate_calls) == 1
    assert all(content.role == "user" for content in models.generate_calls[0]["contents"])
    assert models.generate_calls[0]["config"].system_instruction == "Translate."


def test_missing_count_endpoint_does_not_block_proven_maximum_bound(monkeypatch, ws, settings):
    models = _MissingCountModels()
    client = _live(monkeypatch, ws, settings, models)
    assert client.generate(stage="native", prompt="p", schema=Answer).text == "ok"
    assert len(models.generate_calls) == 1
    assert budget.snapshot(settings)["reservations"][0]["upper_usd"] == pytest.approx(
        gemini._request_bound(settings.model_main, settings, "native")[1]
    )


def test_injected_sdk_client_cannot_bypass_live_maxima(ws, settings):
    models = _Models(count_result=1)
    client = GeminiClient(ws, settings, client=SimpleNamespace(models=models))
    assert client.generate(stage="injected", prompt="p", schema=Answer).text == "ok"
    assert models.count_calls == []
    assert budget.snapshot(settings)["reservations"][0]["upper_usd"] == pytest.approx(
        gemini._request_bound(settings.model_main, settings, "injected")[1]
    )


def test_undercounted_high_resolution_request_cannot_exceed_current_reservation(
    monkeypatch, ws, settings
):
    settings = replace(settings, media_resolution="high")
    response = SimpleNamespace(
        text='{"text":"ok"}',
        usage_metadata=SimpleNamespace(prompt_token_count=400_000, candidates_token_count=8_000,
                                       thoughts_token_count=0, cached_content_token_count=0),
    )
    models = _Models(count_result=100_000, response=response)
    client = _live(monkeypatch, ws, settings, models)
    clip = Clip(UploadedFile("files/scene", "https://files.example/scene", "video/mp4"), 0, 60, fps=8)
    assert client.generate(stage="high-res", prompt="p", schema=Answer, clips=[clip], system="s").text == "ok"
    assert models.count_calls == []
    assert str(models.generate_calls[0]["config"].media_resolution).endswith("HIGH")
    ledger = budget.snapshot(settings)
    row = ledger["reservations"][0]
    assert row["status"] == "settled"
    assert row["upper_usd"] > row["cost_usd"]
    assert row["cost_usd"] == pytest.approx((400_000 * 0.75 + 8_000 * 3.75) / 1e6)
    assert ledger["groups"]["gemini"]["outstanding_usd"] == 0
    assert json.loads((ws.log_dir / "usage.jsonl").read_text())["usage_complete"] is True


def test_full_bound_is_reserved_before_transport(monkeypatch, ws, settings):
    models = _Models()
    original = models.generate_content

    def generation(**kwargs):
        ledger = budget.snapshot(settings)
        assert len(ledger["reservations"]) == 1
        assert ledger["reservations"][0]["status"] == "reserved"
        assert ledger["groups"]["gemini"]["outstanding_usd"] == pytest.approx(
            gemini._request_bound(settings.model_main, settings, "before")[1]
        )
        return original(**kwargs)

    models.generate_content = generation
    client = _live(monkeypatch, ws, settings, models)
    client.generate(stage="before", prompt="p", schema=Answer)


def test_counted_small_input_cannot_authorize_insufficient_worst_case_headroom(
    monkeypatch, ws, settings
):
    settings = replace(settings, gemini_budget_usd=0.5, media_resolution="high")
    models = _Models(count_result=1)
    client = _live(monkeypatch, ws, settings, models)
    with pytest.raises(GeminiError, match="exceeds approved") as error:
        client.generate(stage="high-res", prompt="p", schema=Answer, system="s")
    assert error.value.stage == "high-res"
    assert models.count_calls == []
    assert models.generate_calls == []
    assert budget.snapshot(settings)["reservations"] == []


def test_unknown_live_model_fails_before_any_provider_operation(monkeypatch, ws, settings):
    models = _Models()
    client = _live(monkeypatch, ws, settings, models)
    with pytest.raises(GeminiError, match="no validated request bound") as error:
        client.generate(stage="unsupported", prompt="p", schema=Answer, model="gemini-unknown")
    assert error.value.stage == "unsupported"
    assert models.count_calls == models.generate_calls == []
    assert budget.snapshot(settings)["reservations"] == []
