"""Model-agnostic LLM client factory plus clients for the two OMP subscription
proxies (Anthropic Messages API and OpenAI Responses API compatible).

``make_client(ws, settings, model)`` returns an object with the same
``generate(...)`` signature and usage-logging semantics as ``GeminiClient``:

- plain ``gemini-*`` ids → ``GeminiClient`` whose default model is ``model``;
- ``anthropic/<model>`` → ``AnthropicClient`` (Messages API, structured output
  via a forced single tool whose input_schema is the pydantic schema);
- ``codex/<model>`` → ``CodexClient`` (Responses API, strict ``json_schema``
  via ``text.format``). Any model id the proxy serves works, not only gpt-*.

Both proxies are subscription endpoints: token usage and its actual provider
list-price equivalent are logged to ``<log_dir>/usage.jsonl`` on every billed
attempt. Calls retry on 429/5xx/network/invalid JSON like ``GeminiClient`` and
map provider refusals to ``GeminiBlocked`` so degradation logic applies.

Credentials come from ``SUBSTAGE_ANTHROPIC_BASE_URL``/``SUBSTAGE_ANTHROPIC_API_KEY``
and ``SUBSTAGE_CODEX_BASE_URL``/``SUBSTAGE_CODEX_API_KEY`` (env or
``~/.config/substage/secrets.env`` via ``load_secrets``).
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from . import gemini
from .config import Settings, Workspace, load_secrets, text_model
from .gemini import _IMAGE_MIME, Clip, GeminiBlocked, GeminiClient, GeminiError
from .models import UsageRecord

T = TypeVar("T", bound=BaseModel)

ANTHROPIC_PREFIX = "anthropic/"
CODEX_PREFIX = "codex/"
CLEF_PREFIX = "clef/"
# xhigh reasoning on a 40-cue scene can run well past 10 minutes
_TIMEOUT_S = 1800.0

_ANTHROPIC_VERSION = "2023-06-01"

# settings/stage thinking levels → Responses API reasoning.effort
_EFFORT = {"minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh"}

# thinking levels → Anthropic thinking budget_tokens; absent = thinking off
_THINK_BUDGET = {"medium": 4_000, "high": 16_000, "xhigh": 32_000}
# max_tokens when thinking is off (thinking adds its budget on top)
_MAX_OUTPUT_TOKENS = 16_384

# Opus's documented context/input ceiling. Reserve the entire ceiling rather
# than estimating tool, system, image or tokenizer overhead from the payload.
# https://platform.claude.com/docs/en/about-claude/models/overview
_OPUS_MAX_INPUT_TOKENS = 1_000_000
_OPUS_MAX_OUTPUT_TOKENS = 128_000

# Public list prices, USD per 1M tokens: (input, output, cache read, cache
# write) — from the provider catalog OMP ships (`omp models --json`).
_LIST_PRICE: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.0, 20.0, 0.2, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0, 0.2, 2.5),
    "claude-haiku-5-5": (0.1, 0.5, 0.01, 0.125),
    "gpt-6.1-sol": (2.0, 10.0, 0.1, 2.5),
    "gpt-6-luna": (0.1, 0.5, 0.01, 0.125),
    "clef-omni": (0.15, 0.0, 0.0, 0.0),
    "clef": (0.24, 0.0, 0.0, 0.0),
    "clef-flash": (0.038, 0.0, 0.0, 0.0),
}


def _price_for(model: str, input_tokens: int, *, provider: str | None = None) -> tuple[float, float, float, float] | None:
    """Full-request context pricing; a pricing threshold is never a cutoff.

    https://developers.openai.com/api/docs/models/gpt-6.1-sol.md
    https://developers.openai.com/api/docs/models/gpt-6-luna.md
    """
    wire = model.split("/", 1)[-1]
    price = _LIST_PRICE.get(wire)
    codex = provider in {"codex", "openai-codex"} or (provider is None and model.startswith(CODEX_PREFIX))
    if price and codex and wire in {"gpt-6.1-sol", "gpt-6-luna"} and input_tokens > 272_000:
        pin, pout, pread, pwrite = price
        return pin * 2, pout * 1.5, pread * 2, pwrite * 2
    return price

# stop reasons / keywords that mean "refused", not "truncated/failed"
_REFUSAL_STOP = frozenset({"refusal"})
_REFUSAL_KEYWORDS = ("moderation", "content_filter", "content policy", "safety")


class LLMClient(Protocol):
    settings: Settings

    def generate(
        self, *, stage: str, prompt: str, schema: type[T], clips=(), images=(), audio=(),
        model: str | None = None, thinking_level: str | None = None, system: str | None = None,
    ) -> T: ...


def make_client(ws: Workspace, settings: Settings, model: str) -> LLMClient:
    """Client for `model`: ``anthropic/<m>``, ``codex/<m>`` or a plain Gemini id."""
    if model.startswith(ANTHROPIC_PREFIX):
        return AnthropicClient(ws, settings, model)
    if model.startswith(CODEX_PREFIX):
        return CodexClient(ws, settings, model)
    return GeminiClient(ws, replace(settings, model_main=model))


def text_client(ws: Workspace, gc: GeminiClient) -> LLMClient:
    """Client for the text-only stages (video map, timeline, review, compress,
    condense): ``config.text_model`` when it differs from gc's model, else gc.

    Its settings name the text model as both ``model_main`` and ``model_light``,
    so stage code that requests either model, or keys its checkpoints on
    either, addresses the text model."""
    model = text_model(gc.settings)
    if model == gc.settings.model_main:
        return gc
    return make_client(ws, replace(gc.settings, model_main=model, model_light=model), model)


# ---------------------------------------------------------------- transport


@dataclass
class HttpResponse:
    status: int
    body: bytes


Transport = Callable[[str, dict[str, str], bytes, float], HttpResponse]


def urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> HttpResponse:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return HttpResponse(resp.status, resp.read())
    except urllib.error.HTTPError as exc:
        return HttpResponse(exc.code, exc.read() or b"")


# ---------------------------------------------------------------- schema


_DROP_KEYS = frozenset({"title", "default"})
_SUBSCHEMA_LISTS = ("anyOf", "allOf", "oneOf", "prefixItems")


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a pydantic ``model_json_schema()`` acceptable to strict structured output.

    Every object gets ``additionalProperties: false`` and lists all of its
    properties as required; ``title``/``default`` are dropped and ``$ref`` nodes
    keep no sibling keywords. Optional fields stay nullable via their
    ``anyOf [..., null]`` so pydantic still validates the result.
    """
    return _strict_node(copy.deepcopy(schema))


def _strict_node(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        return {"$ref": node["$ref"]}
    out = {k: v for k, v in node.items() if k not in _DROP_KEYS}
    if "properties" in out:
        out["properties"] = {name: _strict_node(sub) for name, sub in out["properties"].items()}
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    elif out.get("type") == "object":
        out["additionalProperties"] = False
    if "$defs" in out:
        out["$defs"] = {name: _strict_node(sub) for name, sub in out["$defs"].items()}
    if "items" in out:
        out["items"] = _strict_node(out["items"])
    for key in _SUBSCHEMA_LISTS:
        if key in out:
            out[key] = [_strict_node(sub) for sub in out[key]]
    return out


# ---------------------------------------------------------------- credentials


def _omp_model_creds(provider: str) -> tuple[str | None, str | None]:
    """Optional fallback hook for external credential providers (returns Nones by default)."""
    return None, None

class _Retryable(Exception):
    pass


class _ProxyClient:
    """Shared generate() contract for the two subscription-proxy clients.

    Subclasses define ``_url``, ``_headers``, ``_payload`` and ``_extract``;
    everything else — secrets, images, retries, refusals, usage logging — is
    identical.
    """

    _ENV_URL: str = ""
    _ENV_KEY: str = ""
    _OMP_PROVIDER: str = ""

    def __init__(
        self,
        ws: Workspace,
        settings: Settings,
        model: str,
        *,
        transport: Transport | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.ws = ws
        self.settings = settings
        self.model = model
        if not (base_url and api_key):
            load_secrets()
            base_url = base_url or os.environ.get(self._ENV_URL)
            api_key = api_key or os.environ.get(self._ENV_KEY)
        if not (base_url and api_key):
            omp_url, omp_key = _omp_model_creds(self._OMP_PROVIDER)
            base_url = base_url or omp_url
            api_key = api_key or omp_key
        self._base_url = (base_url or "").rstrip("/")
        self._api_key = api_key or ""
        if not self._base_url or not self._api_key:
            raise GeminiError(
                "init",
                f"{self._ENV_URL}/{self._ENV_KEY} not set (env or ~/.config/substage/secrets.env)",
            )
        self._transport = transport or urllib_transport

    # -- subclass hooks -----------------------------------------------------

    @property
    def _url(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _payload(self, stage: str, prompt: str, schema: type[BaseModel],
                 images: list[Path], model: str, level: str, system: str | None) -> dict:
        raise NotImplementedError

    def _extract(self, data: dict, stage: str) -> str:
        """JSON text to feed ``schema.model_validate_json``, or raise."""
        raise NotImplementedError

    # -- shared pieces ------------------------------------------------------

    def _image_parts(self, stage: str, images: list[Path]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for img in images:
            p = Path(img)
            mime = _IMAGE_MIME.get(p.suffix.lower())
            if mime is None:
                raise GeminiError(stage, f"unsupported image suffix {p.suffix!r} ({p})")
            out.append((mime, base64.b64encode(p.read_bytes()).decode("ascii")))
        return out

    def _log_usage(self, stage: str, model: str, uncached: int, output_tokens: int,
                   cache_read: int, cache_write: int, *, confirmed_tier: str | None = None,
                   usage_complete: bool = True) -> float:
        """Log usage-priced spend, identifying unconfirmed billing-tier estimates."""
        if min(uncached, output_tokens, cache_read, cache_write) < 0:
            raise GeminiError(stage, "invalid provider usage; reservation remains outstanding")
        price = _price_for(model, uncached + cache_read + cache_write, provider=self._OMP_PROVIDER)
        codex = self._OMP_PROVIDER in {"codex", "openai-codex"}
        tiers = {"default": 1.0, "standard": 1.0, "priority": 2.0, "fast": 2.0}
        known_tier = codex and confirmed_tier in tiers
        # An unpriced OpenAI tier does not impose a new execution cutoff.
        # Keep the public standard equivalent, explicitly labeled as an
        # estimate, rather than inventing a multiplier or discarding output.
        multiplier = tiers[confirmed_tier] if known_tier else 1.0
        cost = 0.0
        if price:
            p_in, p_out, p_read, p_write = price
            cost = (uncached * p_in + output_tokens * p_out + cache_read * p_read + cache_write * p_write) * multiplier / 1e6
        record = UsageRecord(
            stage=stage,
            model=model,
            input_tokens=uncached + cache_read + cache_write,
            output_tokens=output_tokens,
            cached_tokens=cache_read,
            cache_write_tokens=cache_write,
            cost_usd=cost,
        )
        logged = record.model_dump()
        logged.update(
            requested_service_tier=self.settings.service_tier if codex else None,
            confirmed_service_tier=confirmed_tier if codex else None,
            priced_service_tier=(confirmed_tier if known_tier else "standard") if codex else None,
            cost_basis=(
                "unpriced_model" if price is None else
                "unvalidated_tier_list_price_estimate" if codex and confirmed_tier is not None and not known_tier else
                "standard_list_price_estimate" if codex and confirmed_tier is None else
                "confirmed_tier_list_price" if codex else "provider_list_price"
            ),
            usage_complete=usage_complete,
        )
        self.ws.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.ws.log_dir / "usage.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(logged, ensure_ascii=False) + "\n")
        return record.cost_usd

    def _request_bound(
        self,
        stage: str,
        model: str,
        payload: dict,
        *,
        prompt: str = "",
        system: str | None = None,
        schema: type[BaseModel] | None = None,
        images: list[Path] = (),
    ) -> float:
        if self._OMP_PROVIDER in {"codex", "openai-codex"}:
            return 0.0  # No OpenAI budget or local context cutoff.
        wire = model.split("/", 1)[-1]
        if self._OMP_PROVIDER != "anthropic" or wire not in _LIST_PRICE or not wire.startswith("claude-"):
            raise GeminiError(stage, f"no validated request bound/pricing for {model!r} provider {self._OMP_PROVIDER!r}")
        if wire != "claude-opus-5-5":
            return 0.0  # Known Sonnet/Haiku are outside the approved Opus cap.
        max_output = payload.get("max_tokens")
        if isinstance(max_output, bool) or not isinstance(max_output, int) or not 0 < max_output <= _OPUS_MAX_OUTPUT_TOKENS:
            raise GeminiError(stage, f"no validated request bound/pricing for {model!r}")
        price = _price_for(model, _OPUS_MAX_INPUT_TOKENS, provider=self._OMP_PROVIDER)
        if price is None:
            raise GeminiError(stage, f"no validated request pricing for {model!r}")
        p_in, p_out, p_read, p_write = price
        return (_OPUS_MAX_INPUT_TOKENS * max(p_in, p_read, p_write) + max_output * p_out) / 1e6

    def _call(self, stage: str, body: bytes) -> dict:
        """One HTTP attempt → parsed response dict. Raises _Retryable,
        GeminiBlocked, or GeminiError (non-retryable)."""
        try:
            resp = self._transport(self._url, self._headers(), body, _TIMEOUT_S)
        except (OSError, TimeoutError) as exc:  # network failure: retry
            raise _Retryable(f"{type(exc).__name__}: {exc}") from exc
        try:
            data = json.loads(resp.body or b"{}")
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        status = resp.status
        if status != 200:  # 200 bodies are checked for in-band errors by _extract
            msg = (data.get("error") or {}).get("message") if isinstance(data.get("error"), dict) else None
            msg = msg or resp.body[:300].decode("utf-8", "replace")
            if status in (400, 403) and any(k in str(msg).lower() for k in _REFUSAL_KEYWORDS):
                raise GeminiBlocked(stage, f"blocked by provider: {msg}")
            if status in (408, 429) or 500 <= status < 600:  # 408: proxy stream timeout
                raise _Retryable(f"HTTP {status}: {msg}")
            raise GeminiError(stage, f"API error: HTTP {status}: {msg}")
        return data

    def generate(
        self,
        *,
        stage: str,
        prompt: str,
        schema: type[T],
        clips: list[Clip] = (),
        images: list[Path] = (),
        audio: list[Path] = (),
        model: str | None = None,
        thinking_level: str | None = None,
        system: str | None = None,
    ) -> T:
        """One structured-output generation via the proxy (text + images only).

        Retries (3x, exponential backoff) on 429/5xx/network errors and on
        invalid/empty JSON; a refusal raises GeminiBlocked on the first attempt.
        Every attempt that returns usage is billed to usage.jsonl.
        """
        if clips or audio:
            raise GeminiError(stage, f"model {self.model} supports text + images only (got clips/audio)")
        use_model = model or self.model
        level = (thinking_level or self.settings.thinking_level).lower()
        payload = self._payload(stage, prompt, schema, images, use_model, level, system)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        upper_usd = self._request_bound(
            stage,
            use_model,
            payload,
            prompt=prompt,
            system=system,
            schema=schema,
            images=images,
        )
        last_error = "unknown error"
        for attempt in range(gemini._MAX_ATTEMPTS):
            reservation = gemini._reserve(
                self.settings, use_model, stage, upper_usd, provider=self._OMP_PROVIDER,
            )
            try:
                data = self._call(stage, body)
            except _Retryable as exc:
                # A timeout/HTTP error can follow paid work. Preserve this
                # attempt's reservation and reserve independently for retries.
                last_error = str(exc)
                gemini._backoff(attempt)
                continue
            cost = None
            if data.get("usage"):
                usage = data["usage"]
                if not isinstance(usage, dict):
                    raise GeminiError(stage, "invalid provider usage; reservation remains outstanding")
                complete = usage.get("input_tokens") is not None and usage.get("output_tokens") is not None
                try:
                    tokens = self._usage(data)
                except (TypeError, ValueError, AttributeError) as exc:
                    raise GeminiError(stage, "invalid provider usage; reservation remains outstanding") from exc
                cost = self._log_usage(
                    stage, use_model, *tokens,
                    confirmed_tier=data.get("service_tier"),
                    usage_complete=complete,
                )
                if not complete:
                    cost = None
            gemini._settle(reservation, cost, stage)
            try:
                text = self._extract(data, stage)
                try:
                    return schema.model_validate_json(text)
                except ValidationError:
                    # Claude tool calls sometimes send a nested array/object as a
                    # JSON-encoded string; decode those and validate once more.
                    return schema.model_validate(_destringify(json.loads(text)))
            except GeminiError:  # includes GeminiBlocked: never retried
                raise
            except _Retryable as exc:
                last_error = str(exc)
            except Exception as exc:  # noqa: BLE001 - any parse/validation failure retries
                last_error = f"invalid JSON response: {exc}"
            gemini._backoff(attempt)
        raise GeminiError(stage, f"failed after {gemini._MAX_ATTEMPTS} attempts: {last_error}")

    def _usage(self, data: dict) -> tuple[int, int, int, int]:
        """(uncached input, output incl. thinking, cache-read input,
        cache-write input) tokens of a response."""
        raise NotImplementedError


def _usage_tokens(value) -> int:
    """Optional zero usage fields are distinct from malformed token counts."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid usage token count: {value!r}")
    return value


_JSON_DECODER = json.JSONDecoder()


def _destringify(value: Any) -> Any:
    """Recursively replace string values that hold a JSON array/object with
    the decoded value (other strings are kept as-is). The string may carry the
    outer object's leaked closing brackets after the value ('[...]}'); only
    closing brackets and whitespace may trail."""
    if isinstance(value, dict):
        return {k: _destringify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_destringify(v) for v in value]
    if isinstance(value, str) and value.lstrip()[:1] in ("[", "{"):
        text = value.lstrip()
        try:
            decoded, end = _JSON_DECODER.raw_decode(text)
        except ValueError:
            return value
        if text[end:].strip(" \t\r\n]}"):
            return value
        return _destringify(decoded)
    return value


class AnthropicClient(_ProxyClient):
    """Anthropic Messages API client with GeminiClient's generate() contract.

    ``model`` keeps the full ``anthropic/<model>`` id for usage logging; the
    wire id strips the prefix. Structured output is a single tool whose
    input_schema is the strict pydantic schema (``tool_choice: auto`` — the
    proxy does not accept forced tool_choice, but one tool is always called).
    """

    _ENV_URL = "SUBSTAGE_ANTHROPIC_BASE_URL"
    _ENV_KEY = "SUBSTAGE_ANTHROPIC_API_KEY"
    _OMP_PROVIDER = "anthropic"

    @property
    def _url(self) -> str:
        return f"{self._base_url}/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def _wire_model(self, model: str) -> str:
        return model.removeprefix(ANTHROPIC_PREFIX)

    def _payload(self, stage: str, prompt: str, schema: type[BaseModel],
                 images: list[Path], model: str, level: str, system: str | None) -> dict:
        content: list[dict[str, Any]] = [
            {"type": "image",
             "source": {"type": "base64", "media_type": mime, "data": b64}}
            for mime, b64 in self._image_parts(stage, images)
        ]
        content.append({"type": "text", "text": prompt})
        budget = _THINK_BUDGET.get(level)
        payload: dict[str, Any] = {
            "model": self._wire_model(model),
            "max_tokens": _MAX_OUTPUT_TOKENS + (budget or 0),
            "messages": [{"role": "user", "content": content}],
            "tools": [{
                "name": schema.__name__,
                "description": "Emit the response for this request. Always call this tool.",
                "input_schema": strict_schema(schema.model_json_schema()),
            }],
            "tool_choice": {"type": "auto"},
        }
        if system:
            # Prompt caching: one breakpoint on the system block caches the
            # prefix tools + system (the style guide, identical across calls).
            payload["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        else:
            payload["tools"][-1]["cache_control"] = {"type": "ephemeral"}
        if budget:
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
        return payload

    def _usage(self, data: dict) -> tuple[int, int, int, int]:
        u = data.get("usage") or {}
        return (
            # Anthropic's input_tokens already excludes cache reads and writes.
            _usage_tokens(u.get("input_tokens")),
            # output_tokens already includes thinking tokens.
            _usage_tokens(u.get("output_tokens")),
            _usage_tokens(u.get("cache_read_input_tokens")),
            _usage_tokens(u.get("cache_creation_input_tokens")),
        )

    def _extract(self, data: dict, stage: str) -> str:
        if data.get("type") == "error":
            raise _Retryable(f"error in 200 body: {(data.get('error') or {}).get('message', '?')}")
        stop = data.get("stop_reason")
        if stop in _REFUSAL_STOP:
            raise GeminiBlocked(stage, f"blocked: stop_reason={stop}")
        for block in data.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return json.dumps(block.get("input"), ensure_ascii=False)
        raise ValueError(f"response contained no tool_use block (stop_reason={stop})")


class CodexClient(_ProxyClient):
    """OpenAI Responses API client with GeminiClient's generate() contract.

    ``model`` keeps the full ``codex/<model>`` id for usage logging; the wire
    id strips the prefix. Structured output is strict ``json_schema`` via
    ``text.format``; thinking maps to ``reasoning.effort``.
    """

    _ENV_URL = "SUBSTAGE_CODEX_BASE_URL"
    _ENV_KEY = "SUBSTAGE_CODEX_API_KEY"
    _OMP_PROVIDER = "openai-codex"

    @property
    def _url(self) -> str:
        return f"{self._base_url}/responses"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _wire_model(self, model: str) -> str:
        return model.removeprefix(CODEX_PREFIX)

    def _payload(self, stage: str, prompt: str, schema: type[BaseModel],
                 images: list[Path], model: str, level: str, system: str | None) -> dict:
        content: list[dict[str, Any]] = [
            {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"}
            for mime, b64 in self._image_parts(stage, images)
        ]
        content.append({"type": "input_text", "text": prompt})
        payload: dict[str, Any] = {
            "model": self._wire_model(model),
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "strict": True,
                    "schema": strict_schema(schema.model_json_schema()),
                },
            },
            "reasoning": {"effort": _EFFORT.get(level, "low")},
            "store": False,
        }
        if system:
            payload["instructions"] = system
        tier = getattr(self.settings, "service_tier", None)
        if tier:
            payload["service_tier"] = tier
        # Routes calls sharing the same instructions + schema to the same cache.
        payload["prompt_cache_key"] = "substage-" + hashlib.sha256(
            json.dumps([system or "", payload["text"]["format"]], ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()[:32]
        return payload

    def _usage(self, data: dict) -> tuple[int, int, int, int]:
        u = data.get("usage") or {}
        total = _usage_tokens(u.get("input_tokens"))
        details = u.get("input_tokens_details") or {}
        cached = _usage_tokens(details.get("cached_tokens"))
        written = _usage_tokens(details.get("cache_write_tokens"))
        return (
            total - cached - written,
            # output_tokens already includes reasoning tokens.
            _usage_tokens(u.get("output_tokens")),
            cached,
            written,
        )

    def _extract(self, data: dict, stage: str) -> str:
        status = data.get("status")
        if status == "failed" or data.get("error"):
            err = data.get("error") or {}
            msg = err.get("message") or "response failed"
            if any(k in str(msg).lower() for k in _REFUSAL_KEYWORDS):
                raise GeminiBlocked(stage, f"blocked by provider: {msg}")
            raise _Retryable(f"response failed: {msg}")
        for item in data.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for c in item.get("content") or []:
                if isinstance(c, dict) and c.get("type") in ("output_text", "refusal"):
                    if c["type"] == "refusal":
                        raise GeminiBlocked(stage, f"blocked: {str(c.get('refusal'))[:200]}")
                    return c.get("text") or ""
        raise ValueError(f"response contained no output text (status={status}, "
                         f"incomplete={data.get('incomplete_details')})")


_CLEF_MODELS: dict[str, tuple[str, str]] = {
    "clef-omni": ("@cf/cloudflare/clef-omni", "clef-omni"),
    "clef": ("@cf/cloudflare/clef", "clef"),
    "clef-flash": ("@cf/cloudflare/clef-flash", "clef-flash"),
}


class ClefDecisionsClient:
    """Cloudflare Workers AI System One / Decisions API client (`clef-omni`, `clef`, `clef-flash`).

    Evaluates up to 64 independent typed questions (`noul`, `choice`, `score`) in
    a single forward pass (`output_tokens = 0`) over a shared ``state`` and
    optional multimodal ``images`` (up to 4 base64 data URLs).
    """

    _ENV_ACCOUNT = "SUBSTAGE_CLOUDFLARE_ACCOUNT_ID"
    _ENV_TOKEN = "SUBSTAGE_CLOUDFLARE_API_TOKEN"
    _OMP_PROVIDER = "cloudflare-workers-ai"

    def __init__(
        self,
        ws: Workspace,
        settings: Settings,
        model: str = "clef-omni",
        *,
        transport: Transport | None = None,
        account_id: str | None = None,
        api_token: str | None = None,
        base_url: str | None = None,
    ):
        self.ws = ws
        self.settings = settings
        wire = model.removeprefix(CLEF_PREFIX).removeprefix("@cf/cloudflare/")
        if wire not in _CLEF_MODELS:
            raise GeminiError("init", f"unsupported Clef decisions model {model!r}; expected one of {tuple(_CLEF_MODELS)}")
        self.model = f"{CLEF_PREFIX}{wire}"
        self._cf_path, self._selector = _CLEF_MODELS[wire]
        if not (account_id and api_token):
            load_secrets()
            account_id = account_id or os.environ.get(self._ENV_ACCOUNT)
            api_token = api_token or os.environ.get(self._ENV_TOKEN)
        if not (account_id and api_token):
            omp_url, omp_key = _omp_model_creds(self._OMP_PROVIDER)
            base_url = base_url or omp_url
            if omp_key and omp_key.lstrip().startswith("{"):
                try:
                    parsed = json.loads(omp_key)
                except ValueError:
                    parsed = {}
                account_id = account_id or parsed.get("accountId")
                api_token = api_token or parsed.get("token")
            elif omp_key:
                api_token = api_token or omp_key
        self._account_id = str(account_id or "").strip()
        self._api_token = str(api_token or "").strip()
        self._base_url = (base_url or f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}/ai/run").rstrip("/")
        if not self._account_id or not self._api_token:
            raise GeminiError(
                "init",
                f"{self._ENV_ACCOUNT}/{self._ENV_TOKEN} not set (env or ~/.config/substage/secrets.env)",
            )
        self._transport = transport or urllib_transport

    @property
    def _url(self) -> str:
        return f"{self._base_url}/{self._cf_path}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
        }

    def _image_data_urls(self, stage: str, images: Sequence[Path | str]) -> list[str]:
        out: list[str] = []
        for img in list(images)[:4]:
            if isinstance(img, str) and img.startswith("data:"):
                out.append(img)
                continue
            p = Path(img)
            mime = _IMAGE_MIME.get(p.suffix.lower())
            if mime is None:
                raise GeminiError(stage, f"unsupported image suffix {p.suffix!r} ({p})")
            out.append(f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}")
        return out

    def _log_usage(self, stage: str, input_tokens: int, output_tokens: int) -> float:
        price = _price_for(self.model, input_tokens, provider=self._OMP_PROVIDER)
        cost = (input_tokens * price[0] + output_tokens * price[1]) / 1e6 if price else 0.0
        record = UsageRecord(
            stage=stage,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=0,
            cache_write_tokens=0,
            cost_usd=cost,
        )
        logged = record.model_dump()
        logged.update(
            requested_service_tier=None,
            confirmed_service_tier=None,
            priced_service_tier=None,
            cost_basis="provider_list_price" if price else "unpriced_model",
            usage_complete=True,
        )
        self.ws.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.ws.log_dir / "usage.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(logged, ensure_ascii=False) + "\n")
        return record.cost_usd

    def decide(
        self,
        *,
        stage: str,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, dict[str, Any]],
        images: Sequence[Path | str] = (),
    ) -> dict[str, dict[str, Any]]:
        """Evaluate 1..64 typed atomic questions on ``state`` (+ optional ``images``).

        Returns the ``answers`` mapping ``{question_id: answer_dict}``.
        """
        if not questions or len(questions) > 64:
            raise ValueError(f"Clef decisions API requires 1..64 questions per call, got {len(questions)}")
        payload: dict[str, Any] = {
            "model": self._selector,
            "state": state,
            "questions": questions,
        }
        img_urls = self._image_data_urls(stage, images)
        if img_urls:
            payload["images"] = img_urls
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error = "unknown error"
        for attempt in range(gemini._MAX_ATTEMPTS):
            reservation = gemini._reserve(self.settings, self.model, stage, 0.0, provider=self._OMP_PROVIDER)
            try:
                resp = self._transport(self._url, self._headers(), body, _TIMEOUT_S)
            except (OSError, TimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                gemini._backoff(attempt)
                continue
            try:
                data = json.loads(resp.body or b"{}")
            except ValueError:
                data = {}
            if resp.status != 200 or not data.get("success", True):
                msg = resp.body[:300].decode("utf-8", "replace")
                if resp.status in (408, 429) or 500 <= resp.status < 600:
                    last_error = f"HTTP {resp.status}: {msg}"
                    gemini._backoff(attempt)
                    continue
                raise GeminiError(stage, f"Clef API error: HTTP {resp.status}: {msg}")
            res = data.get("result") if isinstance(data.get("result"), dict) else data
            usage = res.get("usage") if isinstance(res, dict) else None
            cost = 0.0
            if isinstance(usage, dict) and usage.get("input_tokens") is not None:
                in_tok = _usage_tokens(usage.get("input_tokens"))
                out_tok = _usage_tokens(usage.get("output_tokens", 0))
                cost = self._log_usage(stage, in_tok, out_tok)
            gemini._settle(reservation, cost, stage)
            answers = res.get("answers") if isinstance(res, dict) else None
            if isinstance(answers, dict):
                return answers
            last_error = "missing answers in Clef response"
            gemini._backoff(attempt)
        raise GeminiError(stage, f"Clef failed after {gemini._MAX_ATTEMPTS} attempts: {last_error}")
