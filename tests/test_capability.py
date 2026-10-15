"""Offline tests for deterministic model capability routing."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from substage.capability import (
    Architecture,
    ANTHROPIC_CLAUDE_OPUS_5_5,
    CapabilityError,
    CODEX_GPT_6_1_SOL,
    CODEX_GPT_6_LUNA,
    GEMINI_3_8_FLASH,
    GEMINI_4_ARGON,
    MediaModality,
    MODEL_CAPABILITIES,
    TranslationMode,
    resolve_route,
    required_modalities,
)
from substage.models import SceneTranslation, TranslatedCue


FIXED_MODELS = (
    GEMINI_3_8_FLASH,
    GEMINI_4_ARGON,
    CODEX_GPT_6_LUNA,
    CODEX_GPT_6_1_SOL,
)
GEMINI_MODELS = frozenset({GEMINI_3_8_FLASH, GEMINI_4_ARGON})
CODEX_MODELS = frozenset({CODEX_GPT_6_LUNA, CODEX_GPT_6_1_SOL})


@pytest.mark.parametrize("model", FIXED_MODELS)
def test_auto_route_is_capability_aware_and_preserves_selected_model(model):
    plan = resolve_route(model, Architecture.AUTO)

    expected_architecture = (
        Architecture.DIRECT_AV if model in GEMINI_MODELS else Architecture.TEXT_SNAPSHOT
    )
    expected_mode = (
        TranslationMode.AV if model in GEMINI_MODELS else TranslationMode.TEXT
    )
    expected_modality = (
        MediaModality.VIDEO_AUDIO
        if model in GEMINI_MODELS
        else MediaModality.TEXT_IMAGE
    )

    assert plan.primary_model == model
    assert plan.architecture is expected_architecture
    assert plan.translation_mode is expected_mode
    assert plan.media_modality is expected_modality
    assert plan.mixed_team is False
    assert plan.av_passes == (1 if model in GEMINI_MODELS else 0)


@pytest.mark.parametrize("model", FIXED_MODELS)
def test_explicit_route_families_keep_architecture_and_primary_model(model):
    expected_routes = (
        (Architecture.TEXT_SNAPSHOT, TranslationMode.TEXT, MediaModality.TEXT_IMAGE),
        (Architecture.CASCADE, TranslationMode.CASCADE, MediaModality.TEXT_IMAGE),
        (Architecture.DPAF, TranslationMode.DPAF, MediaModality.TEXT_VIDEO_AUDIO),
        (
            Architecture.MIXED,
            TranslationMode.MIXED,
            MediaModality.TEXT_IMAGE_VIDEO_AUDIO,
        ),
    )

    for architecture, mode, modality in expected_routes:
        plan = resolve_route(model, architecture)
        assert plan.primary_model == model
        assert plan.architecture is architecture
        assert plan.translation_mode is mode
        assert plan.media_modality is modality
        assert plan.mixed_team is (architecture is Architecture.MIXED)
        assert plan.av_passes == 0

    if model in GEMINI_MODELS:
        plan = resolve_route(model, Architecture.DIRECT_AV)
        assert plan.primary_model == model
        assert plan.architecture is Architecture.DIRECT_AV
        assert plan.translation_mode is TranslationMode.AV
        assert plan.media_modality is MediaModality.VIDEO_AUDIO
        assert plan.mixed_team is False
        assert plan.av_passes == 1
    else:
        with pytest.raises(CapabilityError, match="direct-av"):
            resolve_route(model, Architecture.DIRECT_AV)


@pytest.mark.parametrize("model", sorted(CODEX_MODELS))
def test_codex_direct_av_is_rejected_instead_of_silently_falling_back(model):
    with pytest.raises(CapabilityError, match="direct-av") as exc_info:
        resolve_route(model, Architecture.DIRECT_AV)

    assert exc_info.value.model == model
    assert exc_info.value.architecture == Architecture.DIRECT_AV.value


@pytest.mark.parametrize(
    "architecture",
    [Architecture.AUTO, Architecture.DIRECT_AV, Architecture.AV_TWO_PASS],
)
def test_unknown_native_av_model_fails_closed(architecture):
    with pytest.raises(CapabilityError, match="unknown model"):
        resolve_route("gemini-3.8-flash-preview", architecture)


@pytest.mark.parametrize("architecture", [Architecture.AGENTIC, "agentic"])
def test_agentic_route_is_actionably_unsupported(architecture):
    with pytest.raises(CapabilityError, match="agentic.*not implemented") as exc_info:
        resolve_route(GEMINI_3_8_FLASH, architecture)

    assert exc_info.value.model == GEMINI_3_8_FLASH
    assert exc_info.value.architecture == Architecture.AGENTIC.value


def test_route_plan_is_immutable_and_serializable():
    plan = resolve_route(CODEX_GPT_6_1_SOL, Architecture.MIXED)
    expected = {
        "primary_model": CODEX_GPT_6_1_SOL,
        "architecture": "mixed",
        "translation_mode": "mixed",
        "media_modality": "text+image+video+audio",
        "mixed_team": True,
        "av_passes": 0,
    }

    assert plan.to_dict() == expected
    assert json.loads(json.dumps(plan.to_dict())) == expected

    with pytest.raises(FrozenInstanceError):
        plan.primary_model = GEMINI_3_8_FLASH


@pytest.mark.parametrize("model", sorted(GEMINI_MODELS))
def test_av_two_pass_uses_exact_gemini_primary_and_requires_text_av(model):
    plan = resolve_route(model, Architecture.AV_TWO_PASS)

    assert plan.primary_model == model
    assert plan.architecture is Architecture.AV_TWO_PASS
    assert plan.translation_mode is TranslationMode.AV
    assert plan.media_modality is MediaModality.TEXT_VIDEO_AUDIO
    assert plan.mixed_team is False
    assert plan.av_passes == 2
    assert plan.to_dict()["primary_model"] == model
    assert plan.to_dict()["av_passes"] == 2
    assert required_modalities(plan.architecture) == frozenset(
        {MediaModality.TEXT, MediaModality.VIDEO, MediaModality.AUDIO}
    )
    with pytest.raises(FrozenInstanceError):
        plan.primary_model = GEMINI_3_8_FLASH


@pytest.mark.parametrize(
    "model", [*sorted(CODEX_MODELS), ANTHROPIC_CLAUDE_OPUS_5_5]
)
def test_av_two_pass_rejects_non_native_gemini_models(model):
    with pytest.raises(CapabilityError, match="av-two-pass") as exc_info:
        resolve_route(model, Architecture.AV_TWO_PASS)
    assert exc_info.value.model == model
    assert exc_info.value.architecture == Architecture.AV_TWO_PASS.value


@pytest.mark.parametrize(
    "model", ["gemini-3.8-flash-preview", "codex/gemini-4-argon", "gemini/gemini-4-argon"]
)
def test_av_two_pass_does_not_infer_native_av_from_model_names(model):
    with pytest.raises(CapabilityError):
        resolve_route(model, Architecture.AV_TWO_PASS)


@pytest.mark.parametrize(
    "modality", [MediaModality.TEXT_IMAGE, MediaModality.VIDEO_AUDIO]
)
def test_av_two_pass_rejects_inputs_missing_text_or_av(modality):
    with pytest.raises(CapabilityError, match="required modalities"):
        resolve_route(
            GEMINI_4_ARGON, Architecture.AV_TWO_PASS, media_modality=modality
        )


def test_av_two_pass_rejects_a_text_mode_or_mixed_team_override():
    with pytest.raises(CapabilityError, match="incompatible"):
        resolve_route(
            GEMINI_4_ARGON, Architecture.AV_TWO_PASS, translation_mode="text"
        )
    with pytest.raises(CapabilityError, match="mixed_team"):
        resolve_route(GEMINI_4_ARGON, Architecture.AV_TWO_PASS, mixed_team=True)


def test_capability_table_uses_exact_models_not_effort_suffixed_ids():
    assert ANTHROPIC_CLAUDE_OPUS_5_5 == "anthropic/claude-opus-5-5"
    assert set(MODEL_CAPABILITIES) == {*FIXED_MODELS, ANTHROPIC_CLAUDE_OPUS_5_5}
    assert MODEL_CAPABILITIES[ANTHROPIC_CLAUDE_OPUS_5_5].supports_direct_av is False
    assert resolve_route(ANTHROPIC_CLAUDE_OPUS_5_5).primary_model == ANTHROPIC_CLAUDE_OPUS_5_5


def test_legacy_scene_translation_loads_without_provenance_metadata():
    legacy = {
        "scene_id": "S001",
        "mode": "text",
        "cues": [{"cue_id": "C001", "ko": "안녕"}],
        "errata": [],
    }
    result = SceneTranslation.model_validate_json(json.dumps(legacy))

    assert result.model is None
    assert result.degraded is None
    assert result.degradation_reason is None
    assert result.cues[0].model is None
    assert result.cues[0].ko == "안녕"
    assert result.errata == []


def test_scene_provenance_roundtrip_preserves_mixed_cue_origins():
    result = SceneTranslation(
        scene_id="S001",
        mode="av",
        cues=[
            TranslatedCue(cue_id="C001", ko="안녕", model=CODEX_GPT_6_1_SOL),
            TranslatedCue(cue_id="C002", ko="잘 가", model=GEMINI_4_ARGON),
        ],
        model=CODEX_GPT_6_1_SOL,
        degraded=True,
        degradation_reason="The explicit mixed-team AV writer handled C002.",
    )
    serialized = result.model_dump(mode="json")
    restored = SceneTranslation.model_validate_json(result.model_dump_json())

    assert serialized["model"] == CODEX_GPT_6_1_SOL
    assert serialized["degraded"] is True
    assert serialized["degradation_reason"] == result.degradation_reason
    assert [cue["model"] for cue in serialized["cues"]] == [
        CODEX_GPT_6_1_SOL, GEMINI_4_ARGON
    ]
    assert restored == result


def test_non_degraded_scene_records_actual_writer_without_inventing_reason():
    result = SceneTranslation(
        scene_id="S001",
        mode="av",
        cues=[TranslatedCue(cue_id="C001", ko="안녕", model=GEMINI_4_ARGON)],
        model=GEMINI_4_ARGON,
        degraded=False,
    )
    assert result.model_dump()["degraded"] is False
    assert result.model_dump()["degradation_reason"] is None


@pytest.mark.parametrize("model", FIXED_MODELS)
def test_explicit_semantic_route_preserves_primary_and_supported_modality(model):
    architecture = "semantic-clip" if model in GEMINI_MODELS else "semantic-image"
    plan = resolve_route(model, architecture)
    assert plan.primary_model == model
    assert plan.semantic_passes == 2
    assert plan.mixed_team is False
    assert plan.translation_mode == (TranslationMode.AV if model in GEMINI_MODELS else TranslationMode.TEXT)
    assert required_modalities(architecture) == (
        frozenset({MediaModality.TEXT, MediaModality.VIDEO, MediaModality.AUDIO})
        if model in GEMINI_MODELS else frozenset({MediaModality.TEXT, MediaModality.IMAGE})
    )


@pytest.mark.parametrize("model,architecture", [
    (CODEX_GPT_6_LUNA, "semantic-clip"),
    (CODEX_GPT_6_1_SOL, "semantic-clip"),
    (GEMINI_3_8_FLASH, "semantic-image"),
    (GEMINI_4_ARGON, "semantic-image"),
    ("codex/unverified-image-model", "semantic-image"),
    (ANTHROPIC_CLAUDE_OPUS_5_5, "semantic-image"),
])
def test_semantic_routes_fail_closed_without_cross_provider_fallback(model, architecture):
    with pytest.raises(CapabilityError):
        resolve_route(model, architecture)


@pytest.mark.parametrize("model,architecture,modality", [
    (GEMINI_3_8_FLASH, "semantic-clip", MediaModality.TEXT_IMAGE_VIDEO_AUDIO),
    (CODEX_GPT_6_1_SOL, "semantic-image", MediaModality.TEXT_IMAGE_VIDEO_AUDIO),
    (GEMINI_4_ARGON, "semantic-clip", MediaModality.TEXT_IMAGE),
    (CODEX_GPT_6_LUNA, "semantic-image", MediaModality.TEXT),
])
def test_semantic_variants_reject_media_switches_or_missing_inputs(model, architecture, modality):
    with pytest.raises(CapabilityError):
        resolve_route(model, architecture, media_modality=modality)
