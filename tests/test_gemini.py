"""Offline tests for gemini.py using an injected fake client (no network).

The fake exposes the two surfaces GeminiClient touches:
  client.models.generate_content(model=, contents=, config=)
  client.files.upload(file=, config=) / files.get(name=)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from google.genai import _transformers, errors, models as genai_models, types

from substage.config import Settings, Workspace
from substage import gemini
from substage.gemini import (
    Clip,
    GeminiBlocked,
    GeminiClient,
    GeminiError,
    UploadedFile,
    price_for,
)
from substage.models import SceneTranslation, SemanticDraft, SemanticSource, UsageRecord
from substage.translate import _SemanticAttributionDraft


class KoLine(BaseModel):
    cue_id: str
    ko: str


# ------------------------------------------------------------------ fakes


class _Usage:
    def __init__(self, prompt=100, candidates=10, thoughts=5, cached=0):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.thoughts_token_count = thoughts
        self.cached_content_token_count = cached


class _Response:
    def __init__(self, text, usage=None, prompt_feedback=None, candidates=None):
        self.text = text
        self.usage_metadata = usage if usage is not None else _Usage()
        self.prompt_feedback = prompt_feedback
        self.candidates = candidates if candidates is not None else []

class _FakeModels:
    """Serves queued outcomes: _Response returns, Exception raises, callable → response."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeFile:
    def __init__(self, name="files/abc123", uri="https://files.example/abc123",
                 mime_type="video/mp4", state="ACTIVE", expiration_time=None,
                 sha256_hash=None):
        self.name = name
        self.uri = uri
        self.mime_type = mime_type
        self.state = state
        self.expiration_time = expiration_time
        self.sha256_hash = sha256_hash


class _FakeFiles:
    def __init__(self, file: _FakeFile):
        self._file = file
        self.uploads: list[Path] = []
        self.gets: int = 0

    def upload(self, *, file, config=None):
        self.uploads.append(Path(file))
        return self._file

    def get(self, *, name, config=None):
        self.gets += 1
        return self._file


class _FakeClient:
    def __init__(self, outcomes, file=None):
        self.models = _FakeModels(outcomes)
        self.files = _FakeFiles(file or _FakeFile())


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(
        video_id="test-video",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    import substage.gemini as g

    monkeypatch.setattr(g, "_backoff", lambda attempt: None)
    monkeypatch.setattr(g.time, "sleep", lambda s: None)


def _ok(text='{"cue_id": "c0001", "ko": "안녕"}', **usage_kw):
    return _Response(text, _Usage(**usage_kw))


def _read_usage(ws: Workspace) -> list[UsageRecord]:
    path = ws.log_dir / "usage.jsonl"
    if not path.exists():
        return []
    return [UsageRecord.model_validate_json(l) for l in path.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------ pricing


def test_price_for_date_aware():
    assert price_for("gemini-3.8-flash", date(2026, 6, 1)) == (0.75, 3.75)
    assert price_for("gemini-3.8-flash", date(2026, 12, 31)) == (0.75, 3.75)
    assert price_for("gemini-3.8-flash", date(2027, 1, 1)) == (1.50, 7.50)
    assert price_for("gemini-3.8-flash", date(2028, 3, 15)) == (1.50, 7.50)
    assert price_for("gemini-3.5-flash-lite", date(2027, 6, 1)) == (0.30, 2.50)
    assert price_for("gemini-4-argon", date(2026, 10, 14)) == (2.00, 10.00)


def test_argon_usage_is_logged_at_catalog_price(ws):
    fake = _FakeClient([_ok(prompt=1000, candidates=20, thoughts=30, cached=7)])
    gc = GeminiClient(ws, Settings(service_tier=None), client=fake)
    gc.generate(stage="comparison", prompt="x", schema=KoLine, model="gemini-4-argon")

    (rec,) = _read_usage(ws)
    assert rec.model == "gemini-4-argon"
    assert (rec.input_tokens, rec.output_tokens, rec.cached_tokens) == (1000, 50, 7)
    expected = 993 * 2.0 / 1e6 + 7 * 0.1 / 1e6 + 50 * 10.0 / 1e6
    assert rec.cost_usd == pytest.approx(expected)


# ------------------------------------------------------------------ generate


def test_generate_returns_validated_object(ws):
    fake = _FakeClient([_ok()])
    gc = GeminiClient(ws, Settings(), client=fake)
    out = gc.generate(stage="s3", prompt="translate", schema=KoLine)
    assert out == KoLine(cue_id="c0001", ko="안녕")
    assert len(fake.models.calls) == 1


def _schema_nodes(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _schema_nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _schema_nodes(value)


def _assert_schema_fields_preserved(original, wire, definitions):
    if "$ref" in original:
        original = definitions[original["$ref"].rsplit("/", 1)[-1]] | {
            key: value for key, value in original.items() if key != "$ref"
        }
    alternatives = original.get("anyOf", [])
    if any(option.get("type") == "null" for option in alternatives):
        assert wire["nullable"] is True
        alternatives = [option for option in alternatives if option.get("type") != "null"]
        if len(alternatives) == 1:
            _assert_schema_fields_preserved(alternatives[0], wire, definitions)
            return
    if "properties" in original:
        assert set(wire["properties"]) == set(original["properties"])
        assert wire.get("required", []) == original.get("required", [])
        for name, field in original["properties"].items():
            _assert_schema_fields_preserved(field, wire["properties"][name], definitions)
    if "items" in original:
        _assert_schema_fields_preserved(original["items"], wire["items"], definitions)
    if "enum" in original:
        assert wire["enum"] == original["enum"]
    if alternatives:
        assert len(wire["any_of"]) == len(alternatives)
        for expected, actual in zip(alternatives, wire["any_of"], strict=True):
            _assert_schema_fields_preserved(expected, actual, definitions)


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (SemanticSource, {"scene_id": "s1", "acts": []}),
        (SemanticDraft, {"scene_id": "s1", "cues": []}),
        (_SemanticAttributionDraft, {"fragments": []}),
    ],
)
def test_generate_closed_schema_reaches_developer_wire_and_keeps_strict_validation(ws, schema, payload):
    original = schema.model_json_schema()
    extra = payload | {"unexpected": "must be rejected locally"}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        schema.model_validate_json(json.dumps(extra))
    fake = _FakeClient([_ok(json.dumps(extra)), _ok(json.dumps(payload))])
    fake._api_client = SimpleNamespace(vertexai=False)
    gc = GeminiClient(ws, Settings(), client=fake)

    assert gc.generate(stage="s3", prompt="translate", schema=schema) == schema.model_validate(payload)
    assert len(fake.models.calls) == 2  # The original closed model rejects the first response.
    cfg = fake.models.calls[0]["config"]
    assert isinstance(cfg.response_schema, dict)
    assert "$defs" in cfg.response_schema  # Preflight conversion must not mutate the config.
    assert schema.model_json_schema() == original
    for node in _schema_nodes(cfg.response_schema):
        assert "additionalProperties" not in node
        assert "additional_properties" not in node

    converted = _transformers.t_schema(fake._api_client, deepcopy(cfg.response_schema))
    parameters = genai_models._GenerateContentParameters_to_mldev(
        fake._api_client, {"config": cfg}
    )
    wire = parameters["generationConfig"]["responseSchema"].model_dump(exclude_none=True)
    assert wire == converted.model_dump(exclude_none=True)
    for node in _schema_nodes(wire):
        assert "additional_properties" not in node
        assert "additionalProperties" not in node
        assert "$defs" not in node
        assert "$ref" not in node
    _assert_schema_fields_preserved(original, wire, original.get("$defs", {}))


@pytest.mark.parametrize("schema", [SemanticSource, SemanticDraft, _SemanticAttributionDraft, SceneTranslation])
def test_wire_schema_vertex_preserves_original_model(schema):
    assert gemini._wire_schema(SimpleNamespace(vertexai=True), schema) is schema


def test_wire_schema_open_scene_translation_preserves_original_model(ws):
    assert gemini._wire_schema(_FakeClient([]), SceneTranslation) is SceneTranslation
    fake = _FakeClient([_ok('{"scene_id":"s1","mode":"av","cues":[]}')])
    gc = GeminiClient(ws, Settings(), client=fake)
    gc.generate(stage="s3", prompt="translate", schema=SceneTranslation)
    assert fake.models.calls[0]["config"].response_schema is SceneTranslation


def test_wire_schema_preserves_property_names_and_literal_data():
    class KeywordFields(BaseModel):
        model_config = ConfigDict(extra="forbid")
        additionalProperties: bool
        additional_properties: bool
        data: dict = {"additionalProperties": False, "additional_properties": False}

    original = KeywordFields.model_json_schema()
    wire = gemini._wire_schema(_FakeClient([]), KeywordFields)
    assert isinstance(wire, dict)
    assert "additionalProperties" not in wire
    assert set(wire["properties"]) == {"additionalProperties", "additional_properties", "data"}
    assert wire["properties"]["additionalProperties"] == original["properties"]["additionalProperties"]
    assert wire["properties"]["additional_properties"] == original["properties"]["additional_properties"]
    assert wire["properties"]["data"]["default"] == {
        "additionalProperties": False, "additional_properties": False
    }


@pytest.mark.parametrize("keyword", ["additionalProperties", "additional_properties"])
def test_wire_schema_strips_only_false_schema_keywords(keyword):
    class SnakeSchema(BaseModel):
        @classmethod
        def model_json_schema(cls):
            return {
                "type": "object",
                keyword: False,
                "properties": {
                    keyword: {"type": "boolean"},
                    "nested": {"type": "array", "items": {"type": "object", keyword: False}},
                },
                "anyOf": [{"type": "object", keyword: False}, {"type": "null"}],
            }

    wire = gemini._wire_schema(_FakeClient([]), SnakeSchema)
    assert keyword not in wire
    assert wire["properties"][keyword] == {"type": "boolean"}
    assert keyword not in wire["properties"]["nested"]["items"]
    assert keyword not in wire["anyOf"][0]


@pytest.mark.parametrize("keyword", ["additionalProperties", "additional_properties"])
@pytest.mark.parametrize("additional", [True, {"type": "string"}])
@pytest.mark.parametrize("closed_child", [False, True])
def test_wire_schema_preserves_unsupported_open_maps(keyword, additional, closed_child):
    class MapSchema(BaseModel):
        @classmethod
        def model_json_schema(cls):
            child = {"type": "object", "additionalProperties": False} if closed_child else {"type": "string"}
            return {"type": "object", keyword: deepcopy(additional), "properties": {"child": child}}

    wire = gemini._wire_schema(_FakeClient([]), MapSchema)
    if closed_child:
        assert isinstance(wire, dict)
        assert wire[keyword] == additional
        assert "additionalProperties" not in wire["properties"]["child"]
    else:
        assert wire is MapSchema
    with pytest.raises(ValueError, match="additionalProperties"):
        _transformers.t_schema(SimpleNamespace(vertexai=False), wire)


def test_generate_sdk_schema_error_precedes_budget_reservation(ws, monkeypatch):
    class MapAnswer(BaseModel):
        values: dict[str, str]

    fake = _FakeClient([])
    fake._api_client = SimpleNamespace(vertexai=False)
    reservations = []
    monkeypatch.setattr(gemini, "_reserve", lambda *args, **kwargs: reservations.append(args))
    gc = GeminiClient(ws, Settings(), client=fake)
    with pytest.raises(GeminiError, match=r"\[s3\].*schema rejected before request"):
        gc.generate(stage="s3", prompt="translate", schema=MapAnswer)
    assert reservations == []
    assert fake.models.calls == []
    assert not (ws.log_dir / "usage.jsonl").exists()


def test_generate_nested_extra_fields_remain_forbidden(ws):
    valid = {"fragments": [{"text": "안녕"}]}
    invalid = {"fragments": [{"text": "안녕", "unexpected": "extra"}]}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        _SemanticAttributionDraft.model_validate_json(json.dumps(invalid))
    fake = _FakeClient([_ok(json.dumps(invalid)), _ok(json.dumps(valid))])
    gc = GeminiClient(ws, Settings(), client=fake)
    result = gc.generate(stage="s3", prompt="attribute", schema=_SemanticAttributionDraft)
    assert result == _SemanticAttributionDraft.model_validate(valid)
    assert len(fake.models.calls) == 2


def test_generate_logs_usage_thinking_billed_as_output(ws):
    fake = _FakeClient([_ok(prompt=1000, candidates=20, thoughts=30, cached=7)])
    gc = GeminiClient(ws, Settings(service_tier=None), client=fake)
    gc.generate(stage="s3", prompt="x", schema=KoLine)

    recs = _read_usage(ws)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.stage == "s3" and rec.model == "gemini-3.8-flash"
    assert rec.input_tokens == 1000
    assert rec.output_tokens == 50  # 20 candidates + 30 thinking, billed as output
    assert rec.cached_tokens == 7
    pin, pout = price_for("gemini-3.8-flash")
    expected = 993 * pin / 1e6 + 7 * pin * 0.1 / 1e6 + 50 * pout / 1e6
    assert rec.cost_usd == pytest.approx(expected)


def test_gemini_shared_budget_stops_only_after_exhaustion(ws):
    usage = dict(prompt=100, candidates=10, thoughts=5)
    spent = (usage["prompt"] * 2.0 + (usage["candidates"] + usage["thoughts"]) * 10.0) / 1e6
    bound = gemini._request_bound("gemini-4-argon", Settings(service_tier=None), "comparison")[1]
    settings = Settings(service_tier=None, gemini_budget_usd=bound + spent - 0.000001)
    first = _FakeClient([_ok(**usage)])
    second = _FakeClient([_ok(**usage)])
    other_ws = Workspace(
        video_id="other-video",
        data_dir=ws.data_dir.parent / "other-data",
        cache_dir=ws.cache_dir.parent / "other-cache",
        log_dir=ws.log_dir.parent / "other-logs",
    )
    first_gc = GeminiClient(ws, settings, client=first)
    second_gc = GeminiClient(other_ws, settings, client=second)

    # The first Argon call is allowed at zero spend; its catalog-priced usage
    # exhausts the shared Gemini (Flash + Argon) cap across workspaces.
    first_gc.generate(stage="comparison", prompt="x", schema=KoLine, model="gemini-4-argon")
    with pytest.raises(GeminiError, match="budget.*exhausted"):
        second_gc.generate(stage="comparison", prompt="x", schema=KoLine, model="gemini-4-argon")
    assert len(first.models.calls) == 1
    assert second.models.calls == []


def test_generate_retryable_errors_then_success(ws):
    outcomes = [
        errors.ServerError(503, {"error": {"message": "unavailable"}}),
        errors.ClientError(429, {"error": {"message": "rate limited"}}),
        _ok(),
    ]
    fake = _FakeClient(outcomes)
    gc = GeminiClient(ws, Settings(), client=fake)
    out = gc.generate(stage="s4", prompt="x", schema=KoLine)
    assert out.ko == "안녕"
    assert len(fake.models.calls) == 3


def test_generate_exhausts_retries_raises_gemini_error(ws):
    outcomes = [errors.ServerError(500, {"error": {"message": "boom"}})] * 10
    fake = _FakeClient(outcomes)
    gc = GeminiClient(ws, Settings(), client=fake)
    with pytest.raises(GeminiError) as ei:
        gc.generate(stage="s3", prompt="x", schema=KoLine)
    assert ei.value.stage == "s3"
    assert len(fake.models.calls) == 4  # 1 initial + 3 retries


def test_generate_invalid_json_retries_then_raises(ws):
    fake = _FakeClient([_Response("{not json"), _Response('{"wrong": 1}')] + [_ok()])
    gc = GeminiClient(ws, Settings(), client=fake)
    out = gc.generate(stage="s3", prompt="x", schema=KoLine)
    assert out.ko == "안녕"
    assert len(fake.models.calls) == 3
    # both failed attempts still billed (the API charged for them)
    assert len(_read_usage(ws)) == 3


def test_generate_invalid_json_all_attempts_fails(ws):
    fake = _FakeClient([_Response("garbage")] * 10)
    gc = GeminiClient(ws, Settings(), client=fake)
    with pytest.raises(GeminiError):
        gc.generate(stage="s3", prompt="x", schema=KoLine)
    assert len(fake.models.calls) == 4


def test_generate_non_retryable_error_fails_fast(ws):
    fake = _FakeClient([errors.ClientError(400, {"error": {"message": "bad request"}})])
    gc = GeminiClient(ws, Settings(), client=fake)
    with pytest.raises(GeminiError) as ei:
        gc.generate(stage="s2a", prompt="x", schema=KoLine)
    assert ei.value.stage == "s2a"
    assert len(fake.models.calls) == 1  # 4xx other than 429: no retry


def test_generate_prompt_blocked_raises_blocked_no_retry(ws):
    resp = _Response(
        None,
        prompt_feedback=SimpleNamespace(block_reason="PROHIBITED_CONTENT"),
    )
    fake = _FakeClient([resp])
    gc = GeminiClient(ws, Settings(), client=fake)
    with pytest.raises(GeminiBlocked) as ei:
        gc.generate(stage="s2b", prompt="x", schema=KoLine)
    assert ei.value.stage == "s2b"
    assert "PROHIBITED_CONTENT" in ei.value.reason
    assert len(fake.models.calls) == 1  # an identical retry would be refused again


def test_generate_safety_finish_reason_raises_blocked_no_retry(ws):
    resp = _Response(
        None,
        candidates=[SimpleNamespace(finish_reason="SAFETY")],
    )
    fake = _FakeClient([resp])
    gc = GeminiClient(ws, Settings(), client=fake)
    with pytest.raises(GeminiBlocked):
        gc.generate(stage="s2v:v01", prompt="x", schema=KoLine)
    assert len(fake.models.calls) == 1


def test_generate_empty_non_blocked_response_still_retries(ws):
    resp = _Response(None, candidates=[SimpleNamespace(finish_reason="OTHER")])
    fake = _FakeClient([resp, _ok()])
    gc = GeminiClient(ws, Settings(), client=fake)
    out = gc.generate(stage="s3", prompt="x", schema=KoLine)
    assert out.ko == "안녕"
    assert len(fake.models.calls) == 2


def test_generate_config_and_contents(ws, tmp_path):
    fake = _FakeClient([_ok()])
    gc = GeminiClient(ws, Settings(thinking_level="medium"), client=fake)

    uploaded = UploadedFile(name="files/xyz", uri="https://u/xyz", mime_type="video/mp4")
    clip = Clip(file=uploaded, start_s=12.5, end_s=80.0, fps=2.0)
    import wave
    wav = tmp_path / "c0001.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 8000)
    png = tmp_path / "frame.png"
    png.write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (768).to_bytes(4, "big") + (768).to_bytes(4, "big")
    )

    gc.generate(
        stage="s3",
        prompt="hello",
        schema=KoLine,
        clips=[clip],
        images=[png],
        audio=[wav],
        model="gemini-3.8-flash",
        system="you are a subtitler",
    )
    call = fake.models.calls[0]
    cfg = call["config"]
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_schema is KoLine
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert cfg.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_LOW
    assert cfg.system_instruction == "you are a subtitler"
    assert {s.threshold for s in cfg.safety_settings} == {types.HarmBlockThreshold.BLOCK_NONE}
    assert len(cfg.safety_settings) >= 5

    parts = call["contents"][0].parts
    clip_part = parts[0]
    assert clip_part.file_data.file_uri == "https://u/xyz"
    assert clip_part.video_metadata.start_offset == "12.5s"
    assert clip_part.video_metadata.end_offset == "80s"
    assert clip_part.video_metadata.fps == 2.0
    assert parts[1].inline_data.mime_type == "image/png"
    assert parts[2].inline_data.mime_type == "audio/wav"
    assert parts[3].text == "hello"


def test_generate_audio_suffix_mimes(ws, tmp_path):
    fake = _FakeClient([_ok(), _ok()])
    gc = GeminiClient(ws, Settings(), client=fake)
    for suffix, mime in [(".m4a", "audio/mp4"), (".mp3", "audio/mpeg")]:
        p = tmp_path / f"a{suffix}"
        p.write_bytes(b"fake")
        gc.generate(stage="x", prompt="p", schema=KoLine, audio=[p])
        part = fake.models.calls[-1]["contents"][0].parts[0]
        assert part.inline_data.mime_type == mime


# ------------------------------------------------------------------- upload


def test_upload_caches_and_reuses(ws, tmp_path):
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"video-bytes")
    fake = _FakeClient([], _FakeFile())
    gc = GeminiClient(ws, Settings(), client=fake)

    up1 = gc.upload(proxy, "video/mp4")
    assert up1.name == "files/abc123" and up1.uri == "https://files.example/abc123"
    assert up1.mime_type == "video/mp4"
    assert len(fake.files.uploads) == 1

    up2 = gc.upload(proxy, "video/mp4")  # cache hit — no second upload
    assert up2.name == up1.name and up2.uri == up1.uri
    assert len(fake.files.uploads) == 1
    entries = json.loads((ws.cache_dir / "gemini_files.json").read_text())
    assert next(iter(entries.values()))["sha256"] == hashlib.sha256(proxy.read_bytes()).hexdigest()


def test_upload_invalidates_same_size_and_mtime_bytes_in_generation(ws, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    proxy = tmp_path / "proxy.mp4"
    first_bytes, second_bytes = b"video-bytes-v1", b"video-bytes-v2"
    proxy.write_bytes(first_bytes)
    original = proxy.stat()
    fake = _FakeClient([_ok()], _FakeFile(name="files/v1", uri="https://files.example/v1"))
    gc = GeminiClient(ws, Settings(experiment_id="upload-byte-identity", service_tier=None), client=fake)
    first = gc.upload(proxy, "video/mp4")
    proxy.write_bytes(second_bytes)
    os.utime(proxy, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert proxy.stat().st_size == original.st_size
    assert proxy.stat().st_mtime_ns == original.st_mtime_ns
    fake.files._file = _FakeFile(name="files/v2", uri="https://files.example/v2")
    second = gc.upload(proxy, "video/mp4")
    assert len(fake.files.uploads) == 2
    assert first.uri != second.uri
    assert second.uri == "https://files.example/v2"
    assert gc.upload(proxy, "video/mp4") == second
    assert len(fake.files.uploads) == 2
    gc.generate(
        stage="fresh-upload", prompt="p", schema=KoLine,
        clips=[Clip(second, 0, 1)],
    )
    assert fake.models.calls[0]["contents"][0].parts[0].file_data.file_uri == second.uri
    cache = json.loads((ws.cache_dir / "gemini_files.json").read_text())
    assert {entry["sha256"] for entry in cache.values()} == {
        hashlib.sha256(first_bytes).hexdigest(), hashlib.sha256(second_bytes).hexdigest(),
    }


def test_upload_unchanged_bytes_reuse_despite_metadata_change(ws, tmp_path):
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"same-video-bytes")
    fake = _FakeClient([], _FakeFile())
    gc = GeminiClient(ws, Settings(), client=fake)
    first = gc.upload(proxy, "video/mp4")
    original = proxy.stat()
    os.utime(proxy, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000))
    assert gc.upload(proxy, "video/mp4") == first
    assert len(fake.files.uploads) == 1


def test_upload_preserves_legacy_undigested_cache_without_trusting_it(ws, tmp_path):
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"video-bytes")
    stat = proxy.stat()
    legacy_key = hashlib.sha256(
        f"{proxy.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    ).hexdigest()
    legacy = {
        "name": "files/legacy", "uri": "https://files.example/legacy",
        "expiry": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
    }
    path = ws.cache_dir / "gemini_files.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({legacy_key: legacy}))
    fake = _FakeClient([], _FakeFile(name="files/digested", uri="https://files.example/digested"))
    gc = GeminiClient(ws, Settings(), client=fake)
    assert gc.upload(proxy, "video/mp4").name == "files/digested"
    assert len(fake.files.uploads) == 1
    entries = json.loads(path.read_text())
    assert entries[legacy_key] == legacy
    assert len(entries) == 2
    current = next(entry for key, entry in entries.items() if key != legacy_key)
    assert current["sha256"] == hashlib.sha256(proxy.read_bytes()).hexdigest()
    gc.upload(proxy, "video/mp4")
    assert len(fake.files.uploads) == 1


@pytest.mark.parametrize("encoding", ["raw32", "ascii-hex64", "ascii-hex64-uppercase"])
def test_upload_records_matching_provider_byte_digest(ws, tmp_path, encoding):
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"video-bytes")
    digest = hashlib.sha256(proxy.read_bytes()).digest()
    encoded_digest = digest if encoding == "raw32" else digest.hex().encode("ascii")
    if encoding == "ascii-hex64-uppercase":
        encoded_digest = encoded_digest.upper()
    fake = _FakeClient([], _FakeFile(sha256_hash=base64.b64encode(encoded_digest).decode("ascii")))
    gc = GeminiClient(ws, Settings(), client=fake)
    gc.upload(proxy, "video/mp4")
    entries = json.loads((ws.cache_dir / "gemini_files.json").read_text())
    entry = next(iter(entries.values()))
    assert entry["sha256"] == entry["provider_sha256"] == digest.hex()
    assert gc.upload(proxy, "video/mp4").name == fake.files._file.name
    assert len(fake.files.uploads) == 1


def test_upload_ascii_hex_provider_digest_reaches_generation_consumer(ws, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"observed-protocol-shape")
    digest = hashlib.sha256(proxy.read_bytes()).hexdigest()
    sdk_file = types.File(
        name="files/canonical-digest", uri="https://files.example/canonical-digest",
        mime_type="video/mp4", state=types.FileState.ACTIVE,
        sha256_hash=base64.b64encode(digest.encode("ascii")).decode("ascii"),
    )
    fake = _FakeClient([_ok()], sdk_file)
    gc = GeminiClient(ws, Settings(experiment_id="canonical-upload-hash", service_tier=None), client=fake)
    uploaded = gc.upload(proxy, "video/mp4")
    gc.generate(stage="digest-consumer", prompt="p", schema=KoLine, clips=[Clip(uploaded, 0, 1)])
    assert len(fake.files.uploads) == len(fake.models.calls) == 1
    assert fake.models.calls[0]["contents"][0].parts[0].file_data.file_uri == sdk_file.uri
    entry = next(iter(json.loads((ws.cache_dir / "gemini_files.json").read_text()).values()))
    assert entry["provider_sha256"] == entry["sha256"] == digest


@pytest.mark.parametrize("provider_hash", [
    "invalid-base64",
    base64.b64encode(hashlib.sha256(b"different-provider-bytes").digest()).decode("ascii"),
    base64.b64encode(hashlib.sha256(b"different-provider-bytes").hexdigest().encode("ascii")).decode("ascii"),
    base64.b64encode(b"z" * 64).decode("ascii"),
    base64.b64encode(b"\xff" * 64).decode("ascii"),
    base64.b64encode(b"0" * 62 + b"  ").decode("ascii"),
    base64.b64encode(b"x" * 31).decode("ascii"),
    base64.b64encode(b"x" * 33).decode("ascii"),
    "0" * 64,  # Bare hexadecimal is not the documented base64 envelope.
    base64.b64encode(b"0" * 63).decode("ascii"),
    base64.b64encode(b"0" * 65).decode("ascii"),
    base64.b64encode(b"\xfb" * 32).decode("ascii").rstrip("="),
    base64.urlsafe_b64encode(b"\xfb" * 32).decode("ascii"),
    base64.b64encode(hashlib.sha256(b"video-bytes").digest()),
])
def test_upload_rejects_invalid_or_mismatched_provider_digest_without_caching(ws, tmp_path, provider_hash):
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"video-bytes")
    fake = _FakeClient([], _FakeFile(sha256_hash=provider_hash))
    gc = GeminiClient(ws, Settings(), client=fake)
    with pytest.raises(GeminiError, match="SHA-256") as error:
        gc.upload(proxy, "video/mp4")
    assert error.value.stage == "upload"
    assert len(fake.files.uploads) == 1
    assert not (ws.cache_dir / "gemini_files.json").exists()


def test_upload_reuploads_after_expiry(ws, tmp_path):
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"video-bytes")
    fake = _FakeClient([], _FakeFile(name="files/v2"))
    gc = GeminiClient(ws, Settings(), client=fake)

    gc.upload(proxy, "video/mp4")
    # age the cached entry past its expiry
    cache_path = ws.cache_dir / "gemini_files.json"
    cache = json.loads(cache_path.read_text())
    for entry in cache.values():
        entry["expiry"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    cache_path.write_text(json.dumps(cache))

    up = gc.upload(proxy, "video/mp4")
    assert len(fake.files.uploads) == 2
    assert up.name == "files/v2"


def test_upload_reuploads_when_content_changes(ws, tmp_path):
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"video-bytes-v1")
    fake = _FakeClient([], _FakeFile())
    gc = GeminiClient(ws, Settings(), client=fake)
    gc.upload(proxy, "video/mp4")

    proxy.write_bytes(b"different-bytes-v2")  # size+content change -> new key
    gc.upload(proxy, "video/mp4")
    assert len(fake.files.uploads) == 2


def test_upload_waits_for_active_state(ws, tmp_path):
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"v")

    processing = _FakeFile(state="PROCESSING")
    active = _FakeFile(state="ACTIVE")

    class _PollingFiles(_FakeFiles):
        def upload(self, *, file, config=None):
            self.uploads.append(Path(file))
            return processing  # not yet active

        def get(self, *, name, config=None):
            self.gets += 1
            return active  # first poll sees it active

    fake = _FakeClient([], processing)
    fake.files = _PollingFiles(processing)
    gc = GeminiClient(ws, Settings(), client=fake)
    up = gc.upload(proxy, "video/mp4")
    assert fake.files.gets == 1
    assert up.uri == active.uri


# ------------------------------------------------------------------ summary


def test_usage_summary_aggregates(ws):
    fake = _FakeClient([
        _ok(prompt=100, candidates=10, thoughts=0),
        _ok(prompt=200, candidates=20, thoughts=10),
    ])
    gc = GeminiClient(ws, Settings(service_tier=None), client=fake)
    gc.generate(stage="s3", prompt="a", schema=KoLine)
    gc.generate(stage="s4", prompt="b", schema=KoLine)

    summary = gc.usage_summary()
    assert summary["s3"]["input_tokens"] == 100
    assert summary["s4"]["output_tokens"] == 30
    assert summary["total"]["input_tokens"] == 300
    assert summary["total"]["output_tokens"] == 40
    pin, pout = price_for("gemini-3.8-flash")
    assert summary["total"]["cost_usd"] == pytest.approx(
        300 * pin / 1e6 + 40 * pout / 1e6
    )


def test_flash_uses_vertex_and_argon_uses_developer_api(ws, tmp_path, monkeypatch):
    monkeypatch.setattr(gemini, "load_secrets", lambda: None)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "runbear")
    monkeypatch.setenv("GOOGLE_VERTEX_LOCATION", "global")
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")

    vertex_fake = _FakeClient([_ok()])
    vertex_fake.vertexai = True
    mldev_fake = _FakeClient([_ok()])
    mldev_fake.vertexai = False
    created: list[dict] = []

    class _StubSDKClient:
        def __new__(cls, **kwargs):
            created.append(kwargs)
            return vertex_fake if kwargs.get("vertexai") else mldev_fake

    monkeypatch.setattr(gemini.genai, "Client", _StubSDKClient)
    monkeypatch.setattr(gemini, "_REAL_GENAI_CLIENT", _StubSDKClient)
    sliced = tmp_path / "sliced.mp4"
    sliced.write_bytes(b"sliced-mp4-bytes")
    monkeypatch.setattr(gemini.media, "slice_proxy_clip", lambda *_a, **_kw: sliced)
    proxy = ws.cache("proxy.mp4")
    proxy.parent.mkdir(parents=True, exist_ok=True)
    proxy.write_bytes(b"proxy-mp4-bytes")

    gc = GeminiClient(ws, Settings(model_main="gemini-3.8-flash", service_tier="priority"))
    clip = Clip(UploadedFile("files/p", "https://files.example/p", "video/mp4"), 10.0, 20.0, fps=1.0)

    gc.generate(stage="s3", prompt="flash", schema=KoLine, clips=[clip], model="gemini-3.8-flash")
    gc.generate(stage="s3", prompt="argon", schema=KoLine, clips=[clip], model="gemini-4-argon")

    assert {"vertexai": True, "project": "runbear", "location": "global"} in created
    assert {"api_key": "test-api-key", "vertexai": False} in created
    assert len(vertex_fake.models.calls) == 1
    assert vertex_fake.models.calls[0]["config"].service_tier is None
    assert vertex_fake.models.calls[0]["contents"][0].parts[0].inline_data.data == b"sliced-mp4-bytes"
    assert len(mldev_fake.models.calls) == 1
    assert mldev_fake.models.calls[0]["config"].service_tier == types.ServiceTier.PRIORITY
    assert mldev_fake.models.calls[0]["contents"][0].parts[0].file_data.file_uri == "https://files.example/p"
