"""Gemini client wrapper: upload cache, structured generation, retries, usage cost.

Every LLM call in the pipeline goes through ``GeminiClient.generate()`` so token
usage and cost are logged per stage to ``<log_dir>/usage.jsonl``.

Pricing is date-aware: ``PRICES`` holds the prices in effect today and
``_PRICE_SCHEDULE`` lists announced future changes; ``price_for()`` resolves the
rate for a given date. Thinking tokens are billed as output tokens.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import _transformers, errors, types
from pydantic import BaseModel

from . import budget, media
from .config import Settings, Workspace, load_secrets
from .models import UsageRecord
from .store import _atomic_write

# USD per 1M tokens: (input, output). gemini-3.8-flash doubles from 2027-01-01
# (announced); the schedule below carries the future rate so historical cost
# logs stay reproducible.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-4-argon": (2.00, 10.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),  # prompts <= 200k tokens
}

_PRICE_SCHEDULE: dict[str, list[tuple[date, tuple[float, float]]]] = {
    "gemini-3.8-flash": [(date(2027, 1, 1), (1.50, 7.50))],
    "gemini-4-argon": [(date(2026, 12, 1), (4.00, 20.00))],
}

# Synchronous generateContent limits (no long-decode API). Argon's unified
# context ceiling is also a conservative input ceiling. Thinking is included
# in the provider's output-token budget; only one candidate is requested.
# https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash
# https://ai.google.dev/gemini-api/docs/models/gemini-4-argon
_REQUEST_LIMITS = {
    "gemini-3.8-flash": (1_048_576, 65_536),
    "gemini-4-argon": (1_048_576, 65_536),
}

# https://ai.google.dev/gemini-api/docs/pricing — synchronous service tiers.
_TIER_MULTIPLIERS = {None: 1.0, "standard": 1.0, "priority": 1.8, "flex": 0.5}
_CACHE_RATIOS = {"gemini-3.8-flash": 0.1, "gemini-4-argon": 0.05}


def price_for(model: str, on: date | None = None, *, tier: str | None = None) -> tuple[float, float]:
    """USD per 1M tokens (input, output) for `model` on date `on` (today if None).

    Raises KeyError for an unknown model — pricing must never silently be 0.
    """
    rates = PRICES[model]
    on = on or date.today()
    for effective, new_rates in _PRICE_SCHEDULE.get(model, ()):
        if on >= effective:
            rates = new_rates
    multiplier = _TIER_MULTIPLIERS[tier]
    return rates[0] * multiplier, rates[1] * multiplier


def _request_bound(
    model: str,
    settings: Settings,
    stage: str,
    *,
    prompt: str = "",
    system: str | None = None,
    schema: type[BaseModel] | None = None,
    clips: list[Clip] = (),
    images: list[Path] = (),
    audio: list[Path] = (),
) -> tuple[int, float]:
    """Reserve model-wide input and the provider's full output allowance.

    The context maximum bounds every accepted input, including media, system
    instructions and schema overhead. Neither a partial countTokens request nor
    a text/media heuristic proves that bound. Input and output maxima are
    reserved independently, even for models with a shared context ceiling.
    Unused headroom is released only after evidenced usage settlement.
    """
    try:
        max_input, max_output = _REQUEST_LIMITS[model]
        # Cover the requested tier and a standard-tier fallback, including
        # announced prices that might take effect while a request is running.
        pin, pout = price_for(model, tier=settings.service_tier)
        multiplier = max(1.0, _TIER_MULTIPLIERS[settings.service_tier])
        standard_in, standard_out = price_for(model)
        pin, pout = max(pin, standard_in), max(pout, standard_out)
        for _, (future_in, future_out) in _PRICE_SCHEDULE.get(model, ()):
            pin, pout = max(pin, future_in * multiplier), max(pout, future_out * multiplier)
    except KeyError as exc:
        raise GeminiError(stage, f"no validated request bound/pricing for {model!r} tier {settings.service_tier!r}") from exc
    return max_output, (max_input * pin + max_output * pout) / 1e6


def _is_vertex_client(client: Any) -> bool:
    return bool(
        getattr(client, "vertexai", False)
        or getattr(getattr(client, "_api_client", None), "vertexai", False)
    )


def _uses_vertex_for_model(model: str) -> bool:
    """gemini-3.8-flash (and 3.x helpers) use OMP Vertex AI; gemini-4-argon uses API key."""
    return not str(model).startswith("gemini-4")


def _wire_schema(client: Any, schema: type[BaseModel]) -> type[BaseModel] | dict[str, Any]:
    """Omit unsupported closed-object keywords only on the Developer API wire."""
    if _is_vertex_client(client):
        return schema
    raw = schema.model_json_schema()
    stripped = False

    def strip(node: Any) -> None:
        nonlocal stripped
        if not isinstance(node, dict):
            return
        for key in ("additionalProperties", "additional_properties"):
            if node.get(key) is False:
                del node[key]
                stripped = True
        # These dictionaries contain names, not schema keywords. In particular,
        # a property named "additionalProperties" must remain a property.
        for key in ("properties", "$defs", "definitions", "patternProperties", "dependentSchemas"):
            mapping = node.get(key)
            if isinstance(mapping, dict):
                for child in mapping.values():
                    strip(child)
        for key in (
            "items", "additionalItems", "contains", "not", "if", "then", "else",
            "propertyNames", "additionalProperties", "additional_properties",
            "unevaluatedProperties", "unevaluatedItems", "contentSchema",
        ):
            child = node.get(key)
            if isinstance(child, list):
                for item in child:
                    strip(item)
            else:
                strip(child)
        for key in ("allOf", "anyOf", "oneOf", "prefixItems", "any_of", "prefix_items"):
            for child in node.get(key, ()):
                strip(child)

    strip(raw)
    return raw if stripped else schema


def _reserve(settings: Settings, model: str, stage: str, upper_usd: float, *, provider: str | None = None):
    try:
        return budget.reserve(settings, model, stage, upper_usd, provider=provider)
    except (budget.BudgetError, OSError) as exc:
        raise GeminiError(stage, str(exc)) from exc


def _settle(reservation: budget.Reservation | None, cost_usd: float | None, stage: str) -> None:
    # Missing/incomplete usage is not proof of a free response.
    if reservation is not None and cost_usd is not None:
        try:
            reservation.settle(cost_usd)
        except (budget.BudgetError, OSError) as exc:
            raise GeminiError(stage, str(exc)) from exc


# Videos contain violence, profanity, sexual references — disable filtering on
# every category that can be adjusted per request (image_* and JAILBREAK are not
# adjustable).
_SAFETY_SETTINGS = [
    types.SafetySetting(category=cat, threshold=types.HarmBlockThreshold.BLOCK_NONE)
    for cat in (
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
    )
]

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
}

_UPLOAD_TTL = timedelta(hours=48)  # Files API keeps uploads ~48h
_MAX_ATTEMPTS = 4  # 1 initial call + 3 retries
_UPLOAD_POLL_S = 2.0
_UPLOAD_TIMEOUT_S = 600.0


class GeminiError(RuntimeError):
    """A pipeline-stage LLM call failed (after retries where applicable)."""

    def __init__(self, stage: str, reason: str):
        self.stage = stage
        self.reason = reason
        super().__init__(f"[{stage}] {reason}")


class GeminiBlocked(GeminiError):
    """The provider refused the prompt/candidate (safety block). Retrying an
    identical request is wasted money — callers degrade (bisect/degrade media)
    instead of retrying."""


@dataclass
class UploadedFile:
    name: str
    uri: str
    mime_type: str


@dataclass
class Clip:
    """A time window of an uploaded video file."""

    file: UploadedFile
    start_s: float
    end_s: float
    fps: float = 1.0


def _seconds(v: float) -> str:
    return f"{v:g}s"


def _retryable(exc: Exception) -> bool:
    """429 (rate limit) and 5xx (server) are retried; other 4xx are not."""
    if isinstance(exc, errors.APIError):
        code = exc.code or 0
        return code == 429 or 500 <= code < 600
    return False


def _backoff(attempt: int) -> None:
    # attempt 0 → 1s, 1 → 2s, 2 → 4s, … capped at 8s
    time.sleep(min(2.0**attempt, 8.0))


_REAL_GENAI_CLIENT = genai.Client


class GeminiClient:
    def __init__(
        self,
        ws: Workspace,
        settings: Settings,
        client: genai.Client | None = None,
    ):
        # Injecting an SDK client must not weaken authorization: it may be live.
        self.ws = ws
        self.settings = settings
        self._injected = client is not None
        self._vertex_client: Any = None
        self._mldev_client: Any = None
        if client is None:
            load_secrets()
            if genai.Client is not _REAL_GENAI_CLIENT:
                self._injected = True
                client = genai.Client()
            else:
                client = self._client_for(settings.model_main)
        self._client = client

    def _client_for(self, model: str) -> Any:
        if self._injected or genai.Client is not _REAL_GENAI_CLIENT:
            return self._client
        if _uses_vertex_for_model(model):
            if self._vertex_client is None:
                project = os.environ.get("GOOGLE_CLOUD_PROJECT")
                location = os.environ.get("GOOGLE_VERTEX_LOCATION", "global")
                if project:
                    self._vertex_client = genai.Client(
                        vertexai=True, project=project, location=location
                    )
                else:
                    self._vertex_client = genai.Client()
            return self._vertex_client
        if self._mldev_client is None:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            self._mldev_client = (
                genai.Client(api_key=api_key, vertexai=False)
                if api_key
                else genai.Client(vertexai=False)
            )
        return self._mldev_client
    # ------------------------------------------------------------------ files

    def _upload_cache_path(self) -> Path:
        return self.ws.cache("gemini_files.json")

    def upload(self, path: Path, mime_type: str) -> UploadedFile:
        """Upload `path` via the Files API, reusing a cache keyed by content identity.

        Cache: ws.cache("gemini_files.json") maps a versioned content key to
        {name, uri, sha256, expiry, ...}. Each lookup streams the current bytes:
        size/mtime cannot prove identity after a proxy is regenerated. Legacy
        undigested entries remain available for diagnostics but are never reused.
        Entries are reused until `expiry` (≤48h after upload).
        """
        path = Path(path)
        with path.open("rb") as source:
            content_digest = hashlib.file_digest(source, "sha256").digest()
        content_sha256 = content_digest.hex()
        st = path.stat()
        key = "sha256-v1:" + hashlib.sha256(
            f"{path.resolve()}|{mime_type}|{content_sha256}".encode()
        ).hexdigest()

        cache_path = self._upload_cache_path()
        cache: dict = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                cache = {}

        now = datetime.now(timezone.utc)
        entry = cache.get(key)
        if entry and entry.get("sha256") == content_sha256:
            try:
                expiry = datetime.fromisoformat(entry["expiry"])
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)
            except (KeyError, ValueError, TypeError):
                expiry = None
            if expiry is not None and now < expiry:
                return UploadedFile(name=entry["name"], uri=entry["uri"], mime_type=mime_type)
        upload_client = self._client
        if not self._injected and _is_vertex_client(upload_client):
            if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
                upload_client = self._client_for("gemini-4-argon")
            else:
                expiry = now + _UPLOAD_TTL
                local_uri = f"file://{path.resolve()}"
                local_name = f"local/{path.name}"
                cache[key] = {
                    "name": local_name, "uri": local_uri, "expiry": expiry.isoformat(),
                    "sha256": content_sha256, "provider_sha256": content_sha256,
                    "path": str(path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                    "mime_type": mime_type,
                }
                _atomic_write(cache_path, json.dumps(cache, indent=2).encode("utf-8"))
                return UploadedFile(name=local_name, uri=local_uri, mime_type=mime_type)

        f = upload_client.files.upload(
            file=path, config=types.UploadFileConfig(mime_type=mime_type)
        )
        # Wait until the file is ACTIVE (PROCESSING → ACTIVE | FAILED).
        deadline = time.monotonic() + _UPLOAD_TIMEOUT_S
        while getattr(f, "state", None) in (types.FileState.PROCESSING, "PROCESSING"):
            if time.monotonic() > deadline:
                raise GeminiError("upload", f"file {f.name!r} still PROCESSING after {_UPLOAD_TIMEOUT_S:.0f}s")
            time.sleep(_UPLOAD_POLL_S)
            f = upload_client.files.get(name=f.name)
        state = getattr(f, "state", None)
        if state in (types.FileState.FAILED, "FAILED"):
            raise GeminiError("upload", f"file {getattr(f, 'name', '?')!r} reached FAILED state")

        api_expiry = getattr(f, "expiration_time", None)
        expiry = now + _UPLOAD_TTL
        if api_expiry is not None:
            if api_expiry.tzinfo is None:
                api_expiry = api_expiry.replace(tzinfo=timezone.utc)
            expiry = min(expiry, api_expiry)

        provider_sha256 = None
        reported_hash = getattr(f, "sha256_hash", None)
        if reported_hash is not None:
            try:
                if not isinstance(reported_hash, str):
                    raise ValueError("uploaded SHA-256 must be a base64 string")
                provider_digest = base64.b64decode(reported_hash, validate=True)
                # The Files API is documented as base64. Its live metadata also
                # encodes the ASCII hexadecimal hash inside that envelope;
                # normalize either representation to the same strict SHA32.
                if len(provider_digest) == 64:
                    provider_digest = bytes.fromhex(provider_digest.decode("ascii"))
                if len(provider_digest) != 32:
                    raise ValueError("uploaded SHA-256 must contain exactly 32 digest bytes")
                provider_sha256 = provider_digest.hex()
            except (ValueError, TypeError) as exc:
                raise GeminiError("upload", f"file {f.name!r} returned an invalid SHA-256 hash") from exc
            if provider_digest != content_digest:
                raise GeminiError("upload", f"file {f.name!r} uploaded bytes do not match the local SHA-256")
        cache[key] = {
            "name": f.name, "uri": f.uri, "expiry": expiry.isoformat(),
            "sha256": content_sha256, "provider_sha256": provider_sha256,
            "path": str(path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "mime_type": mime_type,
        }
        _atomic_write(cache_path, json.dumps(cache, indent=2).encode("utf-8"))
        return UploadedFile(name=f.name, uri=f.uri, mime_type=getattr(f, "mime_type", None) or mime_type)

    # --------------------------------------------------------------- generate

    def _resolve_clip_path(self, uploaded: UploadedFile) -> Path | None:
        if uploaded.uri.startswith("file://"):
            candidate = Path(uploaded.uri.removeprefix("file://"))
            if candidate.is_file():
                return candidate
        cache_path = self._upload_cache_path()
        if cache_path.is_file():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
                for entry in cache.values():
                    if isinstance(entry, dict) and (
                        entry.get("uri") == uploaded.uri or entry.get("name") == uploaded.name
                    ):
                        p = entry.get("path")
                        if p and Path(p).is_file():
                            return Path(p)
            except Exception:
                pass
        fallback = self.ws.cache("proxy.mp4")
        return fallback if fallback.is_file() else None

    def _build_parts(
        self,
        stage: str,
        prompt: str,
        clips: list[Clip],
        images: list[Path],
        audio: list[Path],
        *,
        active_client: Any = None,
    ) -> list[types.Part]:
        parts: list[types.Part] = []
        vertex_mode = _is_vertex_client(active_client if active_client is not None else self._client)
        for c in clips:
            local_video = self._resolve_clip_path(c.file) if vertex_mode else None
            if vertex_mode and local_video is not None:
                sliced = media.slice_proxy_clip(self.ws, local_video, c.start_s, c.end_s)
                parts.append(
                    types.Part(
                        inline_data=types.Blob(data=sliced.read_bytes(), mime_type=c.file.mime_type),
                        video_metadata=types.VideoMetadata(fps=c.fps),
                    )
                )
            else:
                parts.append(
                    types.Part(
                        file_data=types.FileData(file_uri=c.file.uri, mime_type=c.file.mime_type),
                        video_metadata=types.VideoMetadata(
                            start_offset=_seconds(c.start_s),
                            end_offset=_seconds(c.end_s),
                            fps=c.fps,
                        ),
                    )
                )
        for img in images:
            img = Path(img)
            mime = _IMAGE_MIME.get(img.suffix.lower())
            if mime is None:
                raise GeminiError(stage, f"unsupported image suffix {img.suffix!r} ({img})")
            parts.append(types.Part.from_bytes(data=img.read_bytes(), mime_type=mime))
        for a in audio:
            a = Path(a)
            mime = _AUDIO_MIME.get(a.suffix.lower())
            if mime is None:
                raise GeminiError(stage, f"unsupported audio suffix {a.suffix!r} ({a})")
            parts.append(types.Part.from_bytes(data=a.read_bytes(), mime_type=mime))
        parts.append(types.Part.from_text(text=prompt))
        return parts

    def _log_usage(self, stage: str, model: str, usage, *, response=None) -> float | None:
        if usage is None:
            return None
        prompt = getattr(usage, "prompt_token_count", None)
        candidate = getattr(usage, "candidates_token_count", None)
        thoughts = getattr(usage, "thoughts_token_count", None)
        cached = getattr(usage, "cached_content_token_count", None)
        total = getattr(usage, "total_token_count", None)
        tool_prompt = getattr(usage, "tool_use_prompt_token_count", None)
        values = (prompt, candidate, thoughts, cached, total, tool_prompt)
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0) for value in values):
            raise GeminiError(stage, "invalid provider usage; reservation remains outstanding")
        input_tokens, thinking_tokens, cached_tokens = prompt or 0, thoughts or 0, cached or 0
        if cached_tokens > input_tokens:
            raise GeminiError(stage, "invalid provider usage; reservation remains outstanding")
        complete = prompt is not None and not tool_prompt
        if candidate is None:
            if total is not None and complete:
                # The documented total includes prompt, candidates and thinking.
                candidate = total - input_tokens - thinking_tokens
                if candidate < 0:
                    raise GeminiError(stage, "invalid provider usage total; reservation remains outstanding")
            elif thinking_tokens and _thinking_only_response(response):
                # Optional zero-valued candidate counts can be omitted when
                # all generated output was thinking. The terminal response
                # must independently show that no candidate content was emitted.
                candidate = 0
            else:
                complete = False
        output_tokens = (candidate or 0) + thinking_tokens
        if total is not None and complete and total != input_tokens + output_tokens:
            raise GeminiError(stage, "invalid provider usage total; reservation remains outstanding")
        confirmed_tier = _response_tier(response, stage)
        if confirmed_tier is None:
            # Without the billing header this is a conservative list-price
            # estimate, not a claim that the requested priority tier was used.
            requested = self.settings.service_tier
            tier = requested if _TIER_MULTIPLIERS.get(requested, 1.0) >= 1.0 else "standard"
        else:
            tier = confirmed_tier
        try:
            pin, pout = price_for(model, tier=tier)
        except KeyError as exc:
            raise GeminiError(stage, f"no validated usage pricing for {model!r} tier {tier!r}") from exc
        cache_price = pin * _CACHE_RATIOS.get(model, 1.0)
        record = UsageRecord(
            stage=stage,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=((input_tokens - cached_tokens) * pin + cached_tokens * cache_price + output_tokens * pout) / 1e6,
        )
        logged = record.model_dump()
        logged.update(
            requested_service_tier=self.settings.service_tier,
            confirmed_service_tier=confirmed_tier,
            priced_service_tier=tier or "standard",
            cost_basis="confirmed_tier_list_price" if confirmed_tier is not None else "conservative_tier_estimate",
            usage_complete=complete,
        )
        self.ws.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.ws.log_dir / "usage.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(logged, ensure_ascii=False) + "\n")
        return record.cost_usd if complete else None

    def generate[T: BaseModel](
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
        """One structured-output generation; returns a validated `schema` object.

        - contents: clip Parts (file_data + video_metadata), inline image/audio
          Parts, then the text prompt.
        - JSON mode with a provider-compatible wire schema; result validated
          with the original schema.model_validate_json.
        - BLOCK_NONE on all adjustable harm categories.
        - Reserve the documented model input maximum and full 65,536-token
          output allowance before each attempt. countTokens is not required;
          its SDK surface cannot count every generation setting exactly.
        - Retries (3x, exponential backoff) on HTTP 429/5xx and on invalid/empty
          JSON; a safety-blocked prompt/candidate raises GeminiBlocked on the
          first attempt; any other failure or exhausted retries raises
          GeminiError.
        - Every attempt that returns usage_metadata is billed to usage.jsonl.
        """
        use_model = model or self.settings.model_main
        active_client = self._client_for(use_model)
        vertex_mode = _is_vertex_client(active_client)
        parts = self._build_parts(
            stage, prompt, clips, images, audio, active_client=active_client
        )
        contents = [types.Content(role="user", parts=parts)]
        max_output, upper_usd = _request_bound(use_model, self.settings, stage)
        level = (thinking_level or self.settings.thinking_level).upper()
        effective_tier = None if vertex_mode else self.settings.service_tier
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_wire_schema(active_client, schema),
            candidate_count=1,
            max_output_tokens=max_output,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(level)
            ),
            media_resolution=getattr(
                types.MediaResolution,
                f"MEDIA_RESOLUTION_{self.settings.media_resolution.upper()}",
                types.MediaResolution.MEDIA_RESOLUTION_LOW,
            ),
            safety_settings=_SAFETY_SETTINGS,
            system_instruction=system,
            service_tier=types.ServiceTier(effective_tier) if effective_tier else None,
        )
        api_client = getattr(active_client, "_api_client", None)
        if api_client is not None:
            # SDK schema conversion is local and must fail before reserving paid
            # work. It mutates dict schemas, so preserve the config's wire schema.
            try:
                _transformers.t_schema(api_client, deepcopy(config.response_schema))
            except (ValueError, TypeError) as exc:
                raise GeminiError(stage, f"schema rejected before request: {exc}") from exc
        last_error = "unknown error"
        for attempt in range(_MAX_ATTEMPTS):
            reservation = _reserve(self.settings, use_model, stage, upper_usd, provider="gemini")
            try:
                resp = active_client.models.generate_content(
                    model=use_model, contents=contents, config=config
                )
            except Exception as exc:  # noqa: BLE001 — classified below
                # Even a 429/5xx may follow paid work at a proxy. No usage proof
                # means the reservation stays outstanding; retry reserves anew.
                if not _retryable(exc):
                    raise GeminiError(stage, f"API error: {exc}") from exc
                last_error = f"{type(exc).__name__}: {exc}"
                _backoff(attempt)
                continue
            cost = self._log_usage(stage, use_model, getattr(resp, "usage_metadata", None), response=resp)
            _settle(reservation, cost, stage)
            try:
                text = getattr(resp, "text", None)
                if not text:
                    blocked = _blocked_reason(resp)
                    if blocked is not None:
                        raise GeminiBlocked(stage, blocked)
                    raise ValueError(f"response contained no text ({_why_empty(resp)})")
                return schema.model_validate_json(text)
            except GeminiBlocked:
                raise  # identical retry is refused again — degrade, don't retry
            except Exception as exc:  # noqa: BLE001 — any parse/validation failure retries
                last_error = f"invalid JSON response: {exc}"
                _backoff(attempt)
                continue

        raise GeminiError(stage, f"failed after {_MAX_ATTEMPTS} attempts: {last_error}")

    # ---------------------------------------------------------------- summary

    def usage_summary(self) -> dict[str, dict[str, float]]:
        """Per-stage token/cost totals parsed from usage.jsonl, plus "total"."""
        per: dict[str, dict[str, float]] = {}
        path = self.ws.log_dir / "usage.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = UsageRecord.model_validate_json(line)
                d = per.setdefault(
                    rec.stage,
                    {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0},
                )
                d["input_tokens"] += rec.input_tokens
                d["output_tokens"] += rec.output_tokens
                d["cached_tokens"] += rec.cached_tokens
                d["cost_usd"] += rec.cost_usd
        total = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0}
        for d in per.values():
            for k in total:
                total[k] += d[k]
        per["total"] = total
        return per


def _response_tier(response, stage: str) -> str | None:
    """Read the provider-confirmed billing tier, never infer it from the request."""
    headers = getattr(getattr(response, "sdk_http_response", None), "headers", None) or {}
    for key, value in headers.items():
        if str(key).lower() == "x-gemini-service-tier":
            tier = str(value).lower()
            if tier not in _TIER_MULTIPLIERS or tier is None:
                raise GeminiError(stage, f"unknown confirmed Gemini service tier {tier!r}")
            return tier
    return None


def _thinking_only_response(response) -> bool:
    """Evidence for an omitted zero candidate count, not missing usage generally."""
    if response is None or getattr(response, "text", None):
        return False
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return False
    for candidate in candidates:
        reason = getattr(candidate, "finish_reason", None)
        name = getattr(reason, "name", None) or str(reason or "")
        if name.rsplit(".", 1)[-1] not in {"STOP", "MAX_TOKENS"}:
            return False
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        if any(not getattr(part, "thought", False) for part in parts):
            return False
    return True


def _why_empty(resp) -> str:
    """Finish/block reason of an empty response, for actionable errors."""
    feedback = getattr(resp, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        return f"prompt blocked: {feedback.block_reason}"
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return "no candidates"
    return f"finish_reason={getattr(candidates[0], 'finish_reason', None)}"


# candidate finish_reason values that mean "refused", not "truncated/failed"
_BLOCK_FINISH_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"})


def _blocked_reason(resp) -> str | None:
    """Block description when the provider refused the request, else None."""
    feedback = getattr(resp, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        return f"prompt blocked: {feedback.block_reason}"
    for cand in getattr(resp, "candidates", None) or []:
        fr = getattr(cand, "finish_reason", None)
        name = getattr(fr, "name", None) or str(fr or "")
        if name.rsplit(".", 1)[-1] in _BLOCK_FINISH_REASONS:
            return f"finish_reason={name.rsplit('.', 1)[-1]}"
    return None

