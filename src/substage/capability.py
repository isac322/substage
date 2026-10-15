"""Deterministic model-capability and route selection primitives.

This module deliberately contains no provider or network integration.  It is a
closed-world capability gate for the models and route families used by substage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class _StringEnum(str, Enum):
    """String-valued enum with useful serialization semantics."""

    def __str__(self) -> str:
        return self.value


class Architecture(_StringEnum):
    AUTO = "auto"
    DIRECT_AV = "direct-av"
    AV_TWO_PASS = "av-two-pass"
    TEXT_SNAPSHOT = "text-snapshot"
    DPAF = "dpaf"
    CASCADE = "cascade"
    MIXED = "mixed"
    AGENTIC = "agentic"
    SEMANTIC_CLIP = "semantic-clip"
    SEMANTIC_IMAGE = "semantic-image"


class TranslationMode(_StringEnum):
    TEXT = "text"
    AV = "av"
    DPAF = "dpaf"
    CASCADE = "cascade"
    MIXED = "mixed"


class MediaModality(_StringEnum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT_IMAGE = "text+image"
    VIDEO_AUDIO = "video+audio"
    TEXT_VIDEO_AUDIO = "text+video+audio"
    TEXT_IMAGE_VIDEO_AUDIO = "text+image+video+audio"


class Provider(_StringEnum):
    GEMINI = "gemini"
    CODEX = "codex"
    ANTHROPIC = "anthropic"


class CapabilityError(ValueError):
    """Raised when a model/architecture pair cannot be routed safely."""

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        architecture: str | Architecture | None = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.architecture = architecture.value if isinstance(architecture, Enum) else architecture


@dataclass(frozen=True, slots=True)
class ModelCapability:
    """Static, provider-independent capabilities for one model identity."""

    model: str
    provider: Provider
    modalities: frozenset[MediaModality]
    supports_direct_av: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "provider": self.provider.value,
            "modalities": sorted(modality.value for modality in self.modalities),
            "supports_direct_av": self.supports_direct_av,
        }


# These are the only native AV identities accepted by the direct-av gate.
GEMINI_3_8_FLASH = "gemini-3.8-flash"
GEMINI_4_ARGON = "gemini-4-argon"
CODEX_GPT_6_LUNA = "codex/gpt-6-luna"
CODEX_GPT_6_1_SOL = "codex/gpt-6.1-sol"
ANTHROPIC_CLAUDE_OPUS_5_5 = "anthropic/claude-opus-5-5"

_TEXT_IMAGE_MODALITIES = frozenset({MediaModality.TEXT, MediaModality.IMAGE})
_ALL_MODALITIES = frozenset(
    {
        MediaModality.TEXT,
        MediaModality.IMAGE,
        MediaModality.VIDEO,
        MediaModality.AUDIO,
    }
)

_MODEL_CAPABILITIES: dict[str, ModelCapability] = {
    GEMINI_3_8_FLASH: ModelCapability(
        model=GEMINI_3_8_FLASH,
        provider=Provider.GEMINI,
        modalities=_ALL_MODALITIES,
        supports_direct_av=True,
    ),
    GEMINI_4_ARGON: ModelCapability(
        model=GEMINI_4_ARGON,
        provider=Provider.GEMINI,
        modalities=_ALL_MODALITIES,
        supports_direct_av=True,
    ),
    CODEX_GPT_6_LUNA: ModelCapability(
        model=CODEX_GPT_6_LUNA,
        provider=Provider.CODEX,
        modalities=_TEXT_IMAGE_MODALITIES,
        supports_direct_av=False,
    ),
    CODEX_GPT_6_1_SOL: ModelCapability(
        model=CODEX_GPT_6_1_SOL,
        provider=Provider.CODEX,
        modalities=_TEXT_IMAGE_MODALITIES,
        supports_direct_av=False,
    ),
    ANTHROPIC_CLAUDE_OPUS_5_5: ModelCapability(
        model=ANTHROPIC_CLAUDE_OPUS_5_5,
        provider=Provider.ANTHROPIC,
        modalities=_TEXT_IMAGE_MODALITIES,
        supports_direct_av=False,
    ),
}

# The table is immutable so route decisions cannot change after import.
MODEL_CAPABILITIES: Mapping[str, ModelCapability] = MappingProxyType(_MODEL_CAPABILITIES)

_ARCHITECTURE_VALUES = {architecture.value: architecture for architecture in Architecture}
_TRANSLATION_FOR_ARCHITECTURE: Mapping[Architecture, TranslationMode] = MappingProxyType(
    {
        Architecture.DIRECT_AV: TranslationMode.AV,
        Architecture.AV_TWO_PASS: TranslationMode.AV,
        Architecture.TEXT_SNAPSHOT: TranslationMode.TEXT,
        Architecture.DPAF: TranslationMode.DPAF,
        Architecture.CASCADE: TranslationMode.CASCADE,
        Architecture.MIXED: TranslationMode.MIXED,
        Architecture.SEMANTIC_CLIP: TranslationMode.AV,
        Architecture.SEMANTIC_IMAGE: TranslationMode.TEXT,
    }
)
_DEFAULT_MODALITY_FOR_ARCHITECTURE: Mapping[Architecture, MediaModality] = MappingProxyType(
    {
        Architecture.DIRECT_AV: MediaModality.VIDEO_AUDIO,
        Architecture.AV_TWO_PASS: MediaModality.TEXT_VIDEO_AUDIO,
        Architecture.TEXT_SNAPSHOT: MediaModality.TEXT_IMAGE,
        Architecture.DPAF: MediaModality.TEXT_VIDEO_AUDIO,
        Architecture.CASCADE: MediaModality.TEXT_IMAGE,
        Architecture.MIXED: MediaModality.TEXT_IMAGE_VIDEO_AUDIO,
        Architecture.SEMANTIC_CLIP: MediaModality.TEXT_VIDEO_AUDIO,
        Architecture.SEMANTIC_IMAGE: MediaModality.TEXT_IMAGE,
    }
)
_REQUIRED_MODALITIES_FOR_ARCHITECTURE: Mapping[Architecture, frozenset[MediaModality]] = MappingProxyType(
    {
        Architecture.DIRECT_AV: frozenset({MediaModality.VIDEO, MediaModality.AUDIO}),
        Architecture.AV_TWO_PASS: frozenset(
            {MediaModality.TEXT, MediaModality.VIDEO, MediaModality.AUDIO}
        ),
        Architecture.TEXT_SNAPSHOT: frozenset({MediaModality.TEXT, MediaModality.IMAGE}),
        Architecture.DPAF: frozenset(
            {MediaModality.TEXT, MediaModality.VIDEO, MediaModality.AUDIO}
        ),
        Architecture.CASCADE: frozenset({MediaModality.TEXT, MediaModality.IMAGE}),
        Architecture.MIXED: frozenset(
            {
                MediaModality.TEXT,
                MediaModality.IMAGE,
                MediaModality.VIDEO,
                MediaModality.AUDIO,
            }
        ),
        Architecture.SEMANTIC_CLIP: frozenset(
            {MediaModality.TEXT, MediaModality.VIDEO, MediaModality.AUDIO}
        ),
        Architecture.SEMANTIC_IMAGE: frozenset({MediaModality.TEXT, MediaModality.IMAGE}),
    }
)
_MODALITY_COMPONENTS: Mapping[MediaModality, frozenset[MediaModality]] = MappingProxyType(
    {
        MediaModality.TEXT: frozenset({MediaModality.TEXT}),
        MediaModality.IMAGE: frozenset({MediaModality.IMAGE}),
        MediaModality.VIDEO: frozenset({MediaModality.VIDEO}),
        MediaModality.AUDIO: frozenset({MediaModality.AUDIO}),
        MediaModality.TEXT_IMAGE: frozenset({MediaModality.TEXT, MediaModality.IMAGE}),
        MediaModality.VIDEO_AUDIO: frozenset({MediaModality.VIDEO, MediaModality.AUDIO}),
        MediaModality.TEXT_VIDEO_AUDIO: frozenset(
            {MediaModality.TEXT, MediaModality.VIDEO, MediaModality.AUDIO}
        ),
        MediaModality.TEXT_IMAGE_VIDEO_AUDIO: frozenset(
            {
                MediaModality.TEXT,
                MediaModality.IMAGE,
                MediaModality.VIDEO,
                MediaModality.AUDIO,
            }
        ),
    }
)


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """A fully resolved, immutable route decision."""

    primary_model: str
    architecture: Architecture
    translation_mode: TranslationMode
    media_modality: MediaModality
    mixed_team: bool

    @property
    def av_passes(self) -> int:
        """AV2 uses the primary model for both passes; direct AV uses one."""

        if self.architecture in {Architecture.AV_TWO_PASS, Architecture.SEMANTIC_CLIP}:
            return 2
        return 1 if self.architecture is Architecture.DIRECT_AV else 0

    @property
    def semantic_passes(self) -> int:
        return 2 if self.architecture in {
            Architecture.SEMANTIC_CLIP, Architecture.SEMANTIC_IMAGE,
        } else 0

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "primary_model": self.primary_model,
            "architecture": self.architecture.value,
            "translation_mode": self.translation_mode.value,
            "media_modality": self.media_modality.value,
            "mixed_team": self.mixed_team,
            "av_passes": self.av_passes,
        }
        if self.semantic_passes:
            result.update({
                "semantic_passes": self.semantic_passes,
                "extract_stage": "s3_semantic_extract",
                "write_stage": "s3_semantic_write",
                "route_version": "semantic-route-v1",
            })
        return result


def _coerce_architecture(value: str | Architecture) -> Architecture:
    if isinstance(value, Architecture):
        return value
    if not isinstance(value, str):
        raise CapabilityError(f"architecture must be one of {sorted(_ARCHITECTURE_VALUES)}")
    try:
        return _ARCHITECTURE_VALUES[value]
    except KeyError as exc:
        raise CapabilityError(
            f"unsupported architecture {value!r}; choose one of {sorted(_ARCHITECTURE_VALUES)}",
            architecture=value,
        ) from exc


def _coerce_translation_mode(value: str | TranslationMode) -> TranslationMode:
    if isinstance(value, TranslationMode):
        return value
    if not isinstance(value, str):
        raise CapabilityError("translation_mode must be a string or TranslationMode")
    try:
        return TranslationMode(value)
    except ValueError as exc:
        raise CapabilityError(
            f"unsupported translation mode {value!r}; choose text, av, dpaf, cascade, or mixed"
        ) from exc


def _coerce_modality(value: str | MediaModality) -> MediaModality:
    if isinstance(value, MediaModality):
        return value
    if not isinstance(value, str):
        raise CapabilityError("media_modality must be a string or MediaModality")
    try:
        return MediaModality(value)
    except ValueError as exc:
        raise CapabilityError(
            "unsupported media modality "
            f"{value!r}; choose text, image, video, audio, text+image, video+audio, "
            "text+video+audio, or text+image+video+audio"
        ) from exc


def _capability_for_model(model: str) -> ModelCapability:
    if not isinstance(model, str) or not model:
        raise CapabilityError("model must be a non-empty exact model identifier", model=model)
    capability = MODEL_CAPABILITIES.get(model)
    if capability is not None:
        return capability

    # Provider prefixes are part of the route contract for text-only clients.
    # Do not infer capabilities from arbitrary model names or from Gemini-like
    # names: unknown native AV models fail closed.
    if model.startswith("codex/"):
        return ModelCapability(
            model=model,
            provider=Provider.CODEX,
            modalities=_TEXT_IMAGE_MODALITIES,
            supports_direct_av=False,
        )
    if model.startswith("anthropic/"):
        return ModelCapability(
            model=model,
            provider=Provider.ANTHROPIC,
            modalities=_TEXT_IMAGE_MODALITIES,
            supports_direct_av=False,
        )
    raise CapabilityError(
        f"unknown model {model!r}; direct AV accepts only {GEMINI_3_8_FLASH!r} and {GEMINI_4_ARGON!r}",
        model=model,
    )


def required_modalities(architecture: str | Architecture) -> frozenset[MediaModality]:
    """Return the atomic input modalities required by a route family."""

    resolved = _coerce_architecture(architecture)
    if resolved is Architecture.AUTO:
        raise CapabilityError(
            "required modalities are undefined for architecture 'auto'; resolve a concrete route first",
            architecture=resolved,
        )
    if resolved is Architecture.AGENTIC:
        raise CapabilityError(
            "agentic routing is not implemented; use direct-av, av-two-pass, text-snapshot, dpaf, cascade, or mixed",
            architecture=resolved,
        )
    return _REQUIRED_MODALITIES_FOR_ARCHITECTURE[resolved]


def _validate_plan(plan: RoutePlan) -> None:
    architecture = plan.architecture
    if architecture is Architecture.AUTO:
        raise CapabilityError("a resolved RoutePlan cannot retain architecture 'auto'", model=plan.primary_model)
    if architecture is Architecture.AGENTIC:
        raise CapabilityError(
            "agentic routing is not implemented; use direct-av, av-two-pass, text-snapshot, dpaf, cascade, or mixed",
            model=plan.primary_model,
            architecture=architecture,
        )
    capability = _capability_for_model(plan.primary_model)
    if architecture is Architecture.SEMANTIC_IMAGE and plan.primary_model not in {
        CODEX_GPT_6_LUNA, CODEX_GPT_6_1_SOL,
    }:
        raise CapabilityError(
            "semantic-image accepts only the verified Codex image models",
            model=plan.primary_model, architecture=architecture,
        )
    if architecture is Architecture.SEMANTIC_IMAGE and plan.media_modality is not MediaModality.TEXT_IMAGE:
        raise CapabilityError("semantic-image requires exactly text+image", model=plan.primary_model)
    if architecture is Architecture.SEMANTIC_CLIP and plan.media_modality is not MediaModality.TEXT_VIDEO_AUDIO:
        raise CapabilityError("semantic-clip requires exactly text+video+audio", model=plan.primary_model)
    expected_mode = _TRANSLATION_FOR_ARCHITECTURE[architecture]
    if plan.translation_mode is not expected_mode:
        raise CapabilityError(
            f"translation mode {plan.translation_mode.value!r} does not match architecture {architecture.value!r}",
            model=plan.primary_model,
            architecture=architecture,
        )
    if plan.mixed_team is not (architecture is Architecture.MIXED):
        raise CapabilityError(
            "mixed_team must be true only for the mixed architecture",
            model=plan.primary_model,
            architecture=architecture,
        )
    required = required_modalities(architecture)
    available = _MODALITY_COMPONENTS[plan.media_modality]
    if not required.issubset(available):
        raise CapabilityError(
            f"media modality {plan.media_modality.value!r} does not provide required modalities "
            f"{sorted(modality.value for modality in required)}",
            model=plan.primary_model,
            architecture=architecture,
        )
    if architecture in {Architecture.DIRECT_AV, Architecture.AV_TWO_PASS, Architecture.SEMANTIC_CLIP} and (
        capability.provider is not Provider.GEMINI or not capability.supports_direct_av
    ):
        raise CapabilityError(
            f"model {plan.primary_model!r} does not support {architecture.value}; use text-snapshot or an explicit dpaf/cascade/mixed route",
            model=plan.primary_model,
            architecture=architecture,
        )


def resolve_route(
    model: str,
    architecture: str | Architecture = Architecture.AUTO,
    *,
    translation_mode: str | TranslationMode | None = None,
    media_modality: str | MediaModality | None = None,
    mixed_team: bool | None = None,
) -> RoutePlan:
    """Resolve a model and requested architecture without making provider calls."""

    capability = _capability_for_model(model)
    requested = _coerce_architecture(architecture)
    if requested is Architecture.AGENTIC:
        raise CapabilityError(
            "agentic routing is not implemented; interactions/API calls are intentionally unsupported",
            model=model,
            architecture=requested,
        )
    if requested is Architecture.AUTO:
        if capability.supports_direct_av and capability.provider is Provider.GEMINI:
            requested = Architecture.DIRECT_AV
        elif capability.provider in {Provider.CODEX, Provider.ANTHROPIC}:
            requested = Architecture.TEXT_SNAPSHOT
        else:  # Defensive; current capability table has no other provider.
            raise CapabilityError(
                f"no automatic route is defined for model {model!r}",
                model=model,
                architecture=Architecture.AUTO,
            )

    if requested in {Architecture.DIRECT_AV, Architecture.AV_TWO_PASS, Architecture.SEMANTIC_CLIP} and (
        capability.provider is not Provider.GEMINI or not capability.supports_direct_av
    ):
        raise CapabilityError(
            f"model {model!r} cannot use {requested.value}; only plain native Gemini models "
            f"{GEMINI_3_8_FLASH!r} and {GEMINI_4_ARGON!r} are accepted",
            model=model,
            architecture=requested,
        )

    expected_mode = _TRANSLATION_FOR_ARCHITECTURE[requested]
    mode = expected_mode if translation_mode is None else _coerce_translation_mode(translation_mode)
    if mode is not expected_mode:
        raise CapabilityError(
            f"translation mode {mode.value!r} is incompatible with architecture {requested.value!r}",
            model=model,
            architecture=requested,
        )

    modality = (
        _DEFAULT_MODALITY_FOR_ARCHITECTURE[requested]
        if media_modality is None
        else _coerce_modality(media_modality)
    )
    mixed = requested is Architecture.MIXED if mixed_team is None else mixed_team
    if not isinstance(mixed, bool):
        raise CapabilityError("mixed_team must be a boolean", model=model, architecture=requested)

    plan = RoutePlan(
        primary_model=model,
        architecture=requested,
        translation_mode=mode,
        media_modality=modality,
        mixed_team=mixed,
    )
    _validate_plan(plan)
    return plan


__all__ = [
    "Architecture",
    "TranslationMode",
    "MediaModality",
    "Provider",
    "CapabilityError",
    "ModelCapability",
    "RoutePlan",
    "MODEL_CAPABILITIES",
    "GEMINI_3_8_FLASH",
    "GEMINI_4_ARGON",
    "CODEX_GPT_6_LUNA",
    "CODEX_GPT_6_1_SOL",
    "ANTHROPIC_CLAUDE_OPUS_5_5",
    "required_modalities",
    "resolve_route",
]
