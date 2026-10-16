"""Offline tests for llm.py: AnthropicClient and CodexClient over a fake
transport, credential resolution, the make_client factory, and S3 text-mode
routing through settings.model_translate."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from substage import config, llm, translate
from substage.config import Settings, Workspace
from substage.gemini import GeminiBlocked, GeminiClient, GeminiError
from substage.llm import AnthropicClient, CodexClient, HttpResponse, make_client, strict_schema
from substage.models import Cue, Scene, SceneTranslation, Snapshot, TranslatedCue, UsageRecord


class KoLine(BaseModel):
    cue_id: str
    ko: str
    speaker: str | None = None


KO = {"cue_id": "c1", "ko": "안녕", "speaker": None}


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(video_id="t", data_dir=tmp_path / "d", cache_dir=tmp_path / "c", log_dir=tmp_path / "l")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    import substage.gemini as g

    monkeypatch.setattr(g, "_backoff", lambda attempt: None)


class FakeTransport:
    """Scripted responses; records every request url, payload and headers."""

    def __init__(self, *responses: HttpResponse | Exception):
        self.responses = list(responses)
        self.urls: list[str] = []
        self.requests: list[dict] = []
        self.headers: list[dict] = []

    def __call__(self, url, headers, body, timeout):
        self.urls.append(url)
        self.requests.append(json.loads(body))
        self.headers.append(headers)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _resp(body: dict, status: int = 200) -> HttpResponse:
    return HttpResponse(status, json.dumps(body).encode())


# Anthropic Messages API responses

def a_ok(tool_input=KO, usage=None, stop="tool_use", content=None):
    usage = {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 0} if usage is None else usage
    if content is None:
        content = [{"type": "tool_use", "id": "t1", "name": "KoLine", "input": tool_input}]
    return _resp({"type": "message", "role": "assistant", "content": content, "stop_reason": stop, "usage": usage})


def a_err(status, message="boom", etype="api_error"):
    return _resp({"type": "error", "error": {"type": etype, "message": message}}, status)


# OpenAI Responses API responses

def c_ok(text=json.dumps(KO), usage=None, status="completed", content=None):
    usage = {"input_tokens": 100, "output_tokens": 20,
             "input_tokens_details": {"cached_tokens": 0}} if usage is None else usage
    if content is None:
        content = [{"type": "output_text", "text": text}]
    return _resp({"object": "response", "status": status, "error": None,
                  "output": [{"type": "reasoning", "summary": []},
                             {"type": "message", "role": "assistant", "content": content}],
                  "usage": usage})


def c_err(status, message="boom"):
    return _resp({"error": {"message": message, "type": "server_error"}}, status)


def anthropic(ws, transport, model="anthropic/claude-sonnet-5-5", **settings):
    settings.setdefault("experiment_id", "test-anthropic")
    return AnthropicClient(ws, Settings(**settings), model, transport=transport,
                           api_key="test-key", base_url="https://ant.example/")


def codex(ws, transport, model="codex/gpt-6.1-sol", **settings):
    return CodexClient(ws, Settings(**settings), model, transport=transport,
                       api_key="test-key", base_url="https://codex.example/v1")


# per-client fixtures for the shared retry/usage contract
CLIENTS = {
    "anthropic": (anthropic, a_ok, a_err),
    "codex": (codex, c_ok, c_err),
}


def usage_records(ws) -> list[UsageRecord]:
    p = ws.log_dir / "usage.jsonl"
    return [UsageRecord.model_validate_json(l) for l in p.read_text().splitlines()] if p.exists() else []


# ------------------------------------------------------------------ schema


def test_strict_schema_requires_all_props_and_closes_objects():
    s = strict_schema(SceneTranslation.model_json_schema())

    def walk(node):
        if isinstance(node, dict):
            assert "default" not in node and "title" not in node
            if "$ref" in node:
                assert node == {"$ref": node["$ref"]}
            if "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
                for sub in node["properties"].values():
                    walk(sub)
            for k, v in node.items():
                if k != "properties":
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(s)
    cue = s["$defs"]["TranslatedCue"]
    assert set(cue["required"]) >= {"cue_id", "ko", "speaker", "speech_level", "needs_media"}
    # optional fields stay nullable, so the strict output still validates
    assert {"type": "null"} in cue["properties"]["speaker"]["anyOf"]


def test_strict_schema_keeps_property_named_title():
    class Doc(BaseModel):
        title: str
        default: int = 0

    s = strict_schema(Doc.model_json_schema())
    assert set(s["properties"]) == {"title", "default"} and s["required"] == ["title", "default"]


# ------------------------------------------------------------------ anthropic request


def test_anthropic_request_shape(ws, tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG")
    t = FakeTransport(a_ok())
    out = anthropic(ws, t).generate(stage="s3", prompt="translate", schema=KoLine, images=[img], system="sys")
    assert out == KoLine(cue_id="c1", ko="안녕")
    assert t.urls == ["https://ant.example/v1/messages"]
    h = t.headers[0]
    assert h["x-api-key"] == "test-key" and h["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in h
    req = t.requests[0]
    assert req["model"] == "claude-sonnet-5-5"
    # the shared system prefix (style guide) is a prompt-cache breakpoint
    assert req["system"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    (tool,) = req["tools"]
    assert tool["name"] == "KoLine" and tool["input_schema"]["additionalProperties"] is False
    assert req["tool_choice"] == {"type": "auto"}
    parts = req["messages"][0]["content"]
    assert parts[0]["type"] == "image" and parts[0]["source"]["media_type"] == "image/png"
    assert parts[-1] == {"type": "text", "text": "translate"}
    assert "thinking" not in req  # default thinking_level=low → off


def test_anthropic_caches_tools_when_no_system(ws):
    t = FakeTransport(a_ok())
    anthropic(ws, t).generate(stage="s", prompt="p", schema=KoLine)
    assert t.requests[0]["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_codex_prompt_cache_key_shared_by_same_prefix(ws):
    t = FakeTransport(*[c_ok() for _ in range(3)])
    c = codex(ws, t)
    c.generate(stage="s", prompt="one", schema=KoLine, system="guide")
    c.generate(stage="s", prompt="two", schema=KoLine, system="guide")
    c.generate(stage="s", prompt="one", schema=KoLine, system="other guide")
    keys = [r["prompt_cache_key"] for r in t.requests]
    assert keys[0] == keys[1] != keys[2]


@pytest.mark.parametrize("level,budget", [("minimal", None), ("low", None), ("medium", 4000), ("high", 16000)])
def test_anthropic_thinking_budget(ws, level, budget):
    t = FakeTransport(a_ok())
    anthropic(ws, t).generate(stage="s", prompt="p", schema=KoLine, thinking_level=level)
    req = t.requests[0]
    if budget is None:
        assert "thinking" not in req
    else:
        assert req["thinking"] == {"type": "enabled", "budget_tokens": budget}
        assert req["max_tokens"] > budget


def test_anthropic_model_override(ws):
    t = FakeTransport(a_ok())
    anthropic(ws, t).generate(stage="s", prompt="p", schema=KoLine, model="anthropic/claude-opus-5-5")
    assert t.requests[0]["model"] == "claude-opus-5-5"
    assert usage_records(ws)[0].model == "anthropic/claude-opus-5-5"


def test_anthropic_skips_thinking_blocks(ws):
    content = [{"type": "thinking", "thinking": "hmm", "signature": "x"},
               {"type": "tool_use", "id": "t", "name": "KoLine", "input": KO}]
    t = FakeTransport(a_ok(content=content))
    assert anthropic(ws, t).generate(stage="s", prompt="p", schema=KoLine).ko == "안녕"


def test_anthropic_no_tool_use_retried(ws):
    t = FakeTransport(a_ok(content=[{"type": "text", "text": "sure"}], stop="end_turn"), a_ok())
    assert anthropic(ws, t).generate(stage="s", prompt="p", schema=KoLine).ko == "안녕"
    assert len(t.requests) == 2


class KoBatch(BaseModel):
    lines: list[KoLine]
    note: str = ""


def test_anthropic_decodes_stringified_nested_array(ws):
    """Claude may send a nested array as a JSON string inside the tool input;
    it is decoded (no retry), while ordinary strings that merely start with
    '[' stay strings."""
    stringified = {"lines": json.dumps([KO], ensure_ascii=False), "note": "[sic] kept"}
    t = FakeTransport(a_ok(tool_input=stringified))
    out = anthropic(ws, t).generate(stage="s", prompt="p", schema=KoBatch)
    assert out.lines == [KoLine(**KO)] and out.note == "[sic] kept"
    assert len(t.requests) == 1


def test_anthropic_decodes_stringified_array_with_leaked_closers(ws):
    """Observed with claude-opus: the stringified array ends with the outer
    object's '}' ('[...]}'); trailing garbage that isn't a closer stays invalid."""
    leaked = {"lines": json.dumps([KO], ensure_ascii=False) + "}"}
    out = anthropic(ws, FakeTransport(a_ok(tool_input=leaked))).generate(stage="s", prompt="p", schema=KoBatch)
    assert out.lines == [KoLine(**KO)]
    bad = {"lines": json.dumps([KO], ensure_ascii=False) + " junk"}
    t = FakeTransport(*[a_ok(tool_input=bad)] * 4)
    with pytest.raises(GeminiError):
        anthropic(ws, t).generate(stage="s", prompt="p", schema=KoBatch)


# ------------------------------------------------------------------ codex request


def test_codex_request_shape(ws, tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG")
    t = FakeTransport(c_ok())
    out = codex(ws, t, thinking_level="medium").generate(
        stage="s3", prompt="translate", schema=KoLine, images=[img], system="sys")
    assert out == KoLine(cue_id="c1", ko="안녕")
    assert t.urls == ["https://codex.example/v1/responses"]
    assert t.headers[0]["Authorization"] == "Bearer test-key"
    req = t.requests[0]
    assert req["model"] == "gpt-6.1-sol"
    assert req["instructions"] == "sys"
    assert req["store"] is False
    fmt = req["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True and fmt["name"] == "KoLine"
    assert fmt["schema"]["additionalProperties"] is False
    assert req["reasoning"] == {"effort": "medium"}
    parts = req["input"][0]["content"]
    assert parts[0]["type"] == "input_image" and parts[0]["image_url"].startswith("data:image/png;base64,")
    assert parts[-1] == {"type": "input_text", "text": "translate"}


def test_codex_thinking_and_model_override(ws):
    t = FakeTransport(c_ok())
    codex(ws, t).generate(stage="s", prompt="p", schema=KoLine, thinking_level="high",
                          model="codex/gemini-3.8-flash")
    assert t.requests[0]["reasoning"] == {"effort": "high"}
    assert t.requests[0]["model"] == "gemini-3.8-flash"


def test_codex_failed_status_retried(ws):
    failed = _resp({"status": "failed", "error": {"message": "upstream overloaded"}, "output": []})
    t = FakeTransport(failed, c_ok())
    assert codex(ws, t).generate(stage="s", prompt="p", schema=KoLine).ko == "안녕"


# ------------------------------------------------------------------ shared contract


@pytest.mark.parametrize("kind", CLIENTS)
def test_clips_or_audio_rejected(ws, kind):
    make, _, _ = CLIENTS[kind]
    t = FakeTransport()
    with pytest.raises(GeminiError, match="text \\+ images only"):
        make(ws, t).generate(stage="s", prompt="p", schema=KoLine, clips=[object()])
    with pytest.raises(GeminiError, match="text \\+ images only"):
        make(ws, t).generate(stage="s", prompt="p", schema=KoLine, audio=[Path("a.wav")])
    assert t.requests == []


@pytest.mark.parametrize("kind", CLIENTS)
def test_retries_429_5xx_network_and_invalid_json(ws, kind):
    make, ok, err = CLIENTS[kind]
    bad = ok(tool_input={"cue_id": "c1"}) if kind == "anthropic" else ok(text="{not json")
    t = FakeTransport(err(429), err(503), TimeoutError("slow"), bad)
    with pytest.raises(GeminiError, match="failed after 4 attempts"):
        make(ws, t).generate(stage="s", prompt="p", schema=KoLine)
    # only the attempt that returned usage is logged
    assert len(usage_records(ws)) == 1

    t = FakeTransport(err(500), bad, ok())
    assert make(ws, t).generate(stage="s", prompt="p", schema=KoLine).ko == "안녕"
    assert len(t.requests) == 3


@pytest.mark.parametrize("kind", CLIENTS)
def test_4xx_not_retried(ws, kind):
    make, _, err = CLIENTS[kind]
    t = FakeTransport(err(400, "bad schema"))
    with pytest.raises(GeminiError, match="HTTP 400: bad schema") as ei:
        make(ws, t).generate(stage="s", prompt="p", schema=KoLine)
    assert not isinstance(ei.value, GeminiBlocked) and len(t.requests) == 1


@pytest.mark.parametrize("kind,resp", [
    ("anthropic", a_ok(content=[], stop="refusal")),
    ("anthropic", a_err(400, "Output blocked by content filtering policy: safety", "invalid_request_error")),
    ("codex", c_ok(content=[{"type": "refusal", "refusal": "I can't help with that"}])),
    ("codex", c_err(400, "Request flagged by moderation")),
])
def test_refusals_raise_blocked_without_retry(ws, kind, resp):
    make, ok, _ = CLIENTS[kind]
    t = FakeTransport(resp, ok())
    with pytest.raises(GeminiBlocked):
        make(ws, t).generate(stage="s", prompt="p", schema=KoLine)
    assert len(t.requests) == 1


@pytest.mark.parametrize("kind,usage,expected,cost", [
    # Anthropic: input_tokens excludes cache reads/writes; all four are billed separately
    ("anthropic", {"input_tokens": 1000, "output_tokens": 300, "cache_read_input_tokens": 200,
                   "cache_creation_input_tokens": 500}, (1700, 300, 200, 500),
     (1000 * 2.0 + 300 * 10.0 + 200 * 0.2 + 500 * 2.5) / 1e6),
    # Responses API: input_tokens includes the cached part; no write charge
    ("codex", {"input_tokens": 1000, "output_tokens": 300, "input_tokens_details": {"cached_tokens": 200},
               "output_tokens_details": {"reasoning_tokens": 250}}, (1000, 300, 200, 0),
     (800 * 2.0 + 300 * 10.0 + 200 * 0.1) / 1e6),
    # Codex provider pricing doubles input/cache rates and multiplies output
    # for requests over 272k; this is pricing, not a local execution cutoff.
    ("codex", {"input_tokens": 300_000, "output_tokens": 1000, "input_tokens_details": {"cached_tokens": 100_000}},
     (300_000, 1000, 100_000, 0), (200_000 * 4.0 + 1000 * 15.0 + 100_000 * 0.2) / 1e6),
])
def test_usage_tokens_logged_at_list_price(ws, kind, usage, expected, cost):
    make, ok, _ = CLIENTS[kind]
    t = FakeTransport(ok(usage=usage))
    c = make(ws, t)
    c.generate(stage="s3:x", prompt="p", schema=KoLine)
    (rec,) = usage_records(ws)
    assert (rec.input_tokens, rec.output_tokens, rec.cached_tokens, rec.cache_write_tokens) == expected
    assert rec.cost_usd == pytest.approx(cost)
    assert rec.model.startswith(f"{kind}/")
    assert len(t.requests) == 1


def test_opus_budget_stops_only_after_exhaustion(ws):
    # A bounded request reserves its full finite upper bound before dispatch.
    from dataclasses import replace

    settings = {"experiment_id": "test-opus-budget", "opus_budget_usd": 300.0}
    first_t = FakeTransport(a_ok())
    first = anthropic(ws, first_t, model="anthropic/claude-opus-5-5", **settings)
    payload = first._payload(
        "comparison", "p", KoLine, [], "anthropic/claude-opus-5-5", "low", None
    )
    bound = first._request_bound(
        "comparison", "anthropic/claude-opus-5-5", payload, prompt="p", schema=KoLine
    )
    cap = bound + 0.0008 - 0.000001
    first.settings = replace(first.settings, opus_budget_usd=cap)
    second_t = FakeTransport(a_ok())
    other_ws = Workspace(
        video_id="other-video",
        data_dir=ws.data_dir.parent / "other-data",
        cache_dir=ws.cache_dir.parent / "other-cache",
        log_dir=ws.log_dir.parent / "other-logs",
    )
    second = anthropic(
        other_ws, second_t, model="anthropic/claude-opus-5-5",
        experiment_id="test-opus-budget", opus_budget_usd=cap,
    )

    first.generate(stage="comparison", prompt="p", schema=KoLine)
    with pytest.raises(GeminiError, match="budget.*exhausted"):
        second.generate(stage="comparison", prompt="p", schema=KoLine)
    assert len(first_t.requests) == 1
    assert second_t.requests == []


# ------------------------------------------------------------------ credentials / factory


def _isolate_creds(monkeypatch, omp=(None, None)):
    for var in ("SUBSTAGE_ANTHROPIC_BASE_URL", "SUBSTAGE_ANTHROPIC_API_KEY", "SUBSTAGE_CODEX_BASE_URL", "SUBSTAGE_CODEX_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "load_secrets", lambda: None)
    monkeypatch.setattr(llm, "_omp_model_creds", lambda provider: omp)


def test_make_client_dispatch(ws, monkeypatch):
    _isolate_creds(monkeypatch)
    monkeypatch.setenv("SUBSTAGE_ANTHROPIC_BASE_URL", "https://a")
    monkeypatch.setenv("SUBSTAGE_ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("SUBSTAGE_CODEX_BASE_URL", "https://c/v1")
    monkeypatch.setenv("SUBSTAGE_CODEX_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    a = make_client(ws, Settings(), "anthropic/claude-sonnet-5-5")
    assert isinstance(a, AnthropicClient) and a.model == "anthropic/claude-sonnet-5-5"
    assert a._url == "https://a/v1/messages"
    c = make_client(ws, Settings(), "codex/gemini-3.8-flash")
    assert isinstance(c, CodexClient) and c._url == "https://c/v1/responses"
    g = make_client(ws, Settings(), "gemini-3.5-flash-lite")
    assert isinstance(g, GeminiClient) and g.settings.model_main == "gemini-3.5-flash-lite"


def test_credentials_fall_back_to_omp_models_yml(ws, monkeypatch):
    seen = []
    _isolate_creds(monkeypatch)
    monkeypatch.setattr(llm, "_omp_model_creds", lambda provider: seen.append(provider) or ("https://omp", "ok"))
    a = AnthropicClient(ws, Settings(), "anthropic/claude-sonnet-5-5")
    c = CodexClient(ws, Settings(), "codex/gpt-6.1-sol")
    assert seen == ["anthropic", "openai-codex"]
    assert a._url == "https://omp/v1/messages" and c._url == "https://omp/responses"


def test_omp_models_yml_default_none():
    assert llm._omp_model_creds("anthropic") == (None, None)
    assert llm._omp_model_creds("openai-codex") == (None, None)

@pytest.mark.parametrize("cls,var", [(AnthropicClient, "SUBSTAGE_ANTHROPIC_API_KEY"), (CodexClient, "SUBSTAGE_CODEX_API_KEY")])
def test_missing_credentials(ws, monkeypatch, cls, var):
    _isolate_creds(monkeypatch)
    with pytest.raises(GeminiError, match=var):
        cls(ws, Settings(), "x/y")


# ------------------------------------------------------------------ S3 routing


class Recorder:
    _CUE_LINE = re.compile(r"^  ([A-Za-z0-9_-]+) \| \d{2}:", re.M)

    def __init__(self, settings, blocked=False):
        self.settings = settings
        self.calls: list[dict] = []
        self.blocked = blocked

    def generate(self, *, stage, prompt, schema, clips=(), images=(), audio=(),
                 model=None, thinking_level=None, system=None):
        self.calls.append({"stage": stage, "clips": list(clips)})
        if self.blocked:
            raise GeminiBlocked(stage, "nope")
        sid = re.search(r"Scene (s\d+)", prompt).group(1)
        return SceneTranslation(scene_id=sid, mode="x",
                                cues=[TranslatedCue(cue_id=i, ko="KO") for i in self._CUE_LINE.findall(prompt)])


def _scene_inputs():
    sc = Scene(id="s001", start_ms=1000, end_ms=5000, cue_ids=["c0001"], act=1)
    cues = {"c0001": Cue(id="c0001", start_ms=1000, end_ms=4000, text="hi", raw="hi")}
    return sc, cues


PROXY = SimpleNamespace(name="proxy", uri="gs://proxy", mime_type="video/mp4")


def test_s3_text_calls_use_model_translate_av_stays_gemini(ws, monkeypatch):
    settings = Settings(model_translate="codex/gpt-6.1-sol")
    gc = Recorder(settings)
    alt = Recorder(settings)
    made = []
    monkeypatch.setattr(translate, "make_client", lambda w, s, m: made.append(m) or alt)
    sc, cues = _scene_inputs()
    translate.translate_scene(ws, gc, sc, cues, Snapshot(scene_id="s001"), [], [], mode="text", proxy=None)
    assert len(alt.calls) == 1 and gc.calls == [] and made == ["codex/gpt-6.1-sol"]
    translate.translate_scene(ws, gc, sc, cues, Snapshot(scene_id="s001"), [], [], mode="av", proxy=PROXY)
    assert len(gc.calls) == 1 and gc.calls[0]["clips"] and len(alt.calls) == 1


def test_s3_blocked_av_degrades_through_text_client(ws, monkeypatch):
    settings = Settings(model_translate="anthropic/claude-sonnet-5-5")
    gc = Recorder(settings, blocked=True)
    alt = Recorder(settings)
    monkeypatch.setattr(translate, "make_client", lambda w, s, m: alt)
    sc, cues = _scene_inputs()
    st = translate.translate_scene(ws, gc, sc, cues, Snapshot(scene_id="s001"), [], [], mode="av", proxy=PROXY)
    assert len(gc.calls) == 1 and alt.calls[0]["clips"] == [] and st.cues[0].ko == "KO"


def test_s3_checkpoint_hash_includes_text_model(ws, monkeypatch):
    sc, cues = _scene_inputs()
    video = SimpleNamespace(video_id="t")

    def run(model_translate):
        settings = Settings(model_translate=model_translate)
        rec = Recorder(settings)
        monkeypatch.setattr(translate, "make_client", lambda w, s, m: rec)
        gc = Recorder(settings)
        translate.translate_video(ws, gc, video, [sc], list(cues.values()), {}, [], None, mode="text")
        return len(rec.calls) + len(gc.calls)

    assert run(None) == 1
    assert run(None) == 0  # checkpoint hit
    assert run("codex/gpt-6.1-sol") == 1  # different text model → re-call
    assert run("codex/gpt-6.1-sol") == 0


# ------------------------------------------------------------ text-stage routing


def test_production_defaults_route_text_to_opus():
    s = Settings()
    assert s.model_translate == "anthropic/claude-opus-5-5" and s.translate_thinking == "medium"
    assert s.model_text is None and config.text_model(s) == "anthropic/claude-opus-5-5"
    assert config.text_model(Settings(model_text="codex/gpt-6.1-sol")) == "codex/gpt-6.1-sol"
    assert config.text_model(Settings(model_translate=None)) == s.model_main
    assert config.text_model(SimpleNamespace(model_main="m")) == "m"  # duck-typed settings


def test_text_client_stays_on_gemini_without_override(ws):
    gc = Recorder(Settings(model_translate=None))
    assert llm.text_client(ws, gc) is gc
    gc = Recorder(Settings(model_translate=None, model_text=Settings.model_main))
    assert llm.text_client(ws, gc) is gc


def test_text_client_names_text_model_as_main_and_light(ws, monkeypatch):
    _isolate_creds(monkeypatch)
    monkeypatch.setenv("SUBSTAGE_ANTHROPIC_BASE_URL", "https://a")
    monkeypatch.setenv("SUBSTAGE_ANTHROPIC_API_KEY", "k")
    gc = Recorder(Settings())
    tc = llm.text_client(ws, gc)
    assert isinstance(tc, AnthropicClient) and tc.model == "anthropic/claude-opus-5-5"
    # stage code keys checkpoints on, and requests, model_main/model_light
    assert tc.settings.model_main == tc.settings.model_light == "anthropic/claude-opus-5-5"
    assert gc.settings.model_main == Settings.model_main  # video stages keep Gemini


def test_text_stages_send_text_model_ids(ws, monkeypatch):
    """S5 compress and S5c condense request the client's model_light: through
    text_client that is the text model, never a Gemini id on the proxy."""
    from substage import layout, validate

    sent = []

    def transport(url, headers, body, timeout):
        payload = json.loads(body)
        sent.append(payload["model"])
        schema = payload["tools"][0]["name"]
        data = {"edits": []} if schema == "_CompressBatch" else {"items": []}
        return HttpResponse(200, json.dumps({
            "content": [{"type": "tool_use", "name": schema, "input": data}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 1, "output_tokens": 1},
        }).encode())

    _isolate_creds(monkeypatch)
    monkeypatch.setenv("SUBSTAGE_ANTHROPIC_BASE_URL", "https://a")
    monkeypatch.setenv("SUBSTAGE_ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(llm, "urllib_transport", transport)
    settings = Settings()
    tc = llm.text_client(ws, Recorder(settings))
    cue = Cue(id="c0001", start_ms=0, end_ms=1000, text="hi", raw="hi")
    validate.run_s5(ws, tc, [cue], {"c0001": "가" * 15}, {}, {}, settings)
    layout.layout([cue], {"c0001": "가나다 " * 6}, {"c0001": "s001"}, {}, settings, tc)
    assert sent == ["claude-opus-5-5", "claude-opus-5-5"]


def test_clef_decisions_client_and_decomposed_diagnosis(ws, monkeypatch):
    from substage import review
    from substage.llm import ClefDecisionsClient

    _isolate_creds(monkeypatch)
    monkeypatch.setenv("SUBSTAGE_CLOUDFLARE_ACCOUNT_ID", "acct-123")
    monkeypatch.setenv("SUBSTAGE_CLOUDFLARE_API_TOKEN", "tok-456")
    captured: list[dict] = []

    def fake_transport(url, headers, body, timeout):
        payload = json.loads(body)
        captured.append({"url": url, "headers": headers, "payload": payload})
        answers = {}
        for qid in payload["questions"]:
            # Flag c0001 on omission and translationese; keep c0002 clean
            if qid in ("c0001_omission", "c0001_translat"):
                answers[qid] = {"type": "noul", "noul": 0.91, "confidence": 0.82}
            else:
                answers[qid] = {"type": "noul", "noul": 0.12, "confidence": 0.76}
        return HttpResponse(
            200,
            json.dumps({
                "success": True,
                "result": {
                    "model": "clef-omni",
                    "answers": answers,
                    "usage": {"input_tokens": 1200, "output_tokens": 0},
                },
            }).encode("utf-8"),
        )

    client = ClefDecisionsClient(ws, Settings(), "clef-omni", transport=fake_transport)
    assert client.model == "clef/clef-omni"
    assert client._url.endswith("/accounts/acct-123/ai/run/@cf/cloudflare/clef-omni")
    scenes = [Scene(id="s001", start_ms=1000, end_ms=5000, cue_ids=["c0001", "c0002"], act=1)]
    cues_by_id = {
        "c0001": Cue(id="c0001", start_ms=1000, end_ms=2500, text="So you gotta hold onto love...", raw=""),
        "c0002": Cue(id="c0002", start_ms=2600, end_ms=4500, text="Excuse me. Whoo!", raw=""),
    }
    ko = {"c0001": "그러니까 사랑을", "c0002": "실례"}
    flags = review.diagnose_with_decisions(
        ws, client, scenes, cues_by_id, ko, levels={"c0001": ("alex>sam", "해"), "c0002": ("alex>crowd", "해")},
    )
    assert len(captured) == 1
    assert len(captured[0]["payload"]["questions"]) == 12  # 2 cues * 6 atomic questions
    assert "PRO_SUBTITLE_LAWS" in captured[0]["payload"]["state"]
    assert [f.cue_id for f in flags] == ["c0001"]
    assert "omission/truncation" in flags[0].problem and "translationese_calque" in flags[0].problem
    # Second call hits checkpoint without invoking transport again
    flags2 = review.diagnose_with_decisions(
        ws, client, scenes, cues_by_id, ko, levels={"c0001": ("alex>sam", "해"), "c0002": ("alex>crowd", "해")},
    )
    assert len(captured) == 1 and flags2 == flags

    assert review.resolve_decisions_model("auto", has_images=True) == "clef-omni"
    assert review.resolve_decisions_model("auto", has_images=False) == "clef"
    assert review.resolve_decisions_model("clef-flash", has_images=True) == "clef-flash"

    from substage.models import CueEdit

    def gate_transport(url, headers, body, timeout):
        payload = json.loads(body)
        ans = {}
        for qid in payload["questions"]:
            if qid.startswith("c0001_curr_"):
                ans[qid] = {"type": "noul", "noul": 0.75 if "mistrans" in qid else 0.20}
            elif qid.startswith("c0001_cand_"):
                ans[qid] = {"type": "noul", "noul": 0.10 if "mistrans" in qid else (0.85 if "spoken" in qid else 0.15)}
            elif qid.startswith("c0002_curr_"):
                ans[qid] = {"type": "noul", "noul": 0.05}
            elif qid.startswith("c0002_cand_"):
                ans[qid] = {"type": "noul", "noul": 0.65 if "verbose" in qid else 0.05}
        return HttpResponse(
            200,
            json.dumps({
                "success": True,
                "result": {
                    "model": "clef",
                    "answers": ans,
                    "usage": {"input_tokens": 900, "output_tokens": 0},
                },
            }).encode("utf-8"),
        )

    gate_client = ClefDecisionsClient(ws, Settings(), "clef", transport=gate_transport)
    raw_edits = [
        CueEdit(cue_id="c0001", old_ko="그러니까 사랑을", new_ko="그러니까 사랑을\n꼭 붙잡아", reason="fix truncation", kind="omission"),
        CueEdit(cue_id="c0002", old_ko="실례", new_ko="실례합니다 여러분 모두 비켜주세요", reason="over-expand", kind="naturalness"),
        CueEdit(cue_id="c0001", old_ko="그러니까 사랑을", new_ko="줄 1\n줄 2\n줄 3", reason="3-line illegal", kind="omission"),
    ]
    accepted = review.filter_edits_with_decisions(ws, gate_client, raw_edits, cues_by_id, ko)
    assert [e.cue_id for e in accepted] == ["c0001"]
    assert accepted[0].new_ko == "그러니까 사랑을\n꼭 붙잡아"
