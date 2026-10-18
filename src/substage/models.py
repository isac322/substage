"""Shared data model for every pipeline stage.

All artifacts are persisted as JSON/YAML through these models so that stages can be
re-run independently and a human can edit narrative artifacts between stages.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Register(StrEnum):
    """Korean speech level a speaker uses toward an addressee."""

    HASIPSIO = "hasipsio"  # 하십시오체 (formal polite)
    HAEYO = "haeyo"  # 해요체 (informal polite)
    HAE = "hae"  # 해체 / 반말
    HAERA = "haera"  # 해라체 (plain; narration, commands, monologue)
    MIXED = "mixed"  # deliberately alternating / unresolved
    UNKNOWN = "unknown"


class MediaNeed(StrEnum):
    NONE = "none"
    AUDIO = "audio"
    VIDEO = "video"
    BOTH = "both"


# ---------------------------------------------------------------- S0 ingest


class Cue(BaseModel):
    id: str  # "c0001"…, stable for the whole run; on-screen-text cues use "ost-0001"
    start_ms: int
    end_ms: int
    text: str  # clean English dialogue (SDH labels/sound tags removed)
    raw: str  # original subtitle text
    speaker_hint: str | None = None  # SDH label such as "PETER"
    sound_cues: list[str] = Field(default_factory=list)  # "[explosion]", "(laughs)"
    lyric: bool = False  # ♪ line
    forced: bool = False  # came from a forced (on-screen text / foreign dialogue) track
    source: str = "main"  # main | forced | ost
    ost_kind: str | None = None
    ost_addressee: str | None = None
    speech_level: str | None = None

class VideoInfo(BaseModel):
    video_id: str  # slug, e.g. "sample-2016"
    title: str
    year: int | None = None
    video_path: str
    duration_ms: int
    audio_stream: int  # absolute ffprobe stream index of the dialogue audio track
    en_sub_path: str  # English subtitle actually used
    en_sub_origin: str  # "embedded:<idx>" | "downloaded:<provider>" | "user"
    forced_sub_path: str | None = None
    proxy_path: str | None = None  # low-res A/V proxy uploaded to Gemini
    sync_offset_s: float | None = None  # ffsubsync result for downloaded subs
    sync_scale: float | None = None


class Scene(BaseModel):
    id: str  # "s001"
    start_ms: int
    end_ms: int
    cue_ids: list[str]
    act: int | None = None


# ---------------------------------------------------------------- S2v viewing pass


class PerceptionCandidate(BaseModel):
    """One possible speaker or addressee identified during the viewing pass."""

    value: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class VisualDeixis(BaseModel):
    """A picture-dependent reference that may affect how a cue is translated."""

    target: str | None = None
    relation: str | None = None  # points_to | looks_at | indicates | reacts_to | other
    description: str | None = None
    keyframe_ms: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class SubTurnObservation(BaseModel):
    """Perception for one turn within a cue, preserving the cue's identity."""

    text: str | None = None
    speaker_candidates: list[PerceptionCandidate] = Field(default_factory=list)
    addressee_candidates: list[PerceptionCandidate] = Field(default_factory=list)
    delivery: str | None = None
    prosody: str | None = None
    visual_deixis: VisualDeixis | None = None
    ambiguity: str | None = None
    media_need: MediaNeed | None = None
    keyframe_ms: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class CuePerception(BaseModel):
    """Advisory, cue-level multimodal perception keyed by an existing cue id."""

    cue_id: str
    speaker_candidates: list[PerceptionCandidate] = Field(default_factory=list)
    addressee_candidates: list[PerceptionCandidate] = Field(default_factory=list)
    delivery: str | None = None
    prosody: str | None = None
    visual_deixis: VisualDeixis | None = None
    ambiguity: str | None = None
    media_need: MediaNeed | None = None
    keyframe_ms: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    # A cue may contain one ordinary turn or two turns represented by one
    # displayed subtitle.  The list order is the source/display order.
    turns: list[SubTurnObservation] | None = Field(default=None, min_length=1, max_length=2)


class PerceptionLedger(BaseModel):
    """Cue-id keyed advisory perception records; no new cue-id namespace."""

    cues: dict[str, CuePerception] = Field(default_factory=dict)


class CueObservation(BaseModel):
    cue_id: str
    turns: list[SubTurnObservation] | None = Field(default=None, min_length=1, max_length=2)
    speaker: str | None = None  # character name as seen/heard
    addressee: str | None = None  # who the line is said to ("group", "audience", name)
    delivery: str | None = None  # tone/prosody: sarcastic, whispered, shouting, teasing…
    nonverbal: str | None = None  # gesture/expression/action relevant to meaning
    lyric_plot_relevant: bool | None = None  # only for ♪ cues: does the sung line carry story/dialogue/callback?


class StoryEvent(BaseModel):
    t_ms: int
    description: str
    characters: list[str] = Field(default_factory=list)
    knowledge_change: str | None = None  # "Peter learns Liz's father is the Falcon"


class OstBeat(BaseModel):
    beat_id: str = "b1"
    start_ms: int
    end_ms: int
    text: str
    ost_kind: str = "sign_banner"
    sender: str | None = None
    addressee: str | None = None
    speech_level: str | None = None


class OnScreenText(BaseModel):
    start_ms: int
    end_ms: int
    text: str  # English text as shown
    plot_critical: bool = False
    covered_by_cue: str | None = None  # existing cue id if already subtitled
    beats: list[OstBeat] = Field(default_factory=list)

class ChunkObservations(BaseModel):
    chunk_id: str  # "v01"
    start_ms: int
    end_ms: int
    present_characters: list[str] = Field(default_factory=list)
    cues: list[CueObservation] = Field(default_factory=list)
    events: list[StoryEvent] = Field(default_factory=list)
    on_screen_text: list[OnScreenText] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------- S2a video map


class VoiceCard(BaseModel):
    idiolect: str  # how the character talks (≤25 words)
    vocabulary: list[str] = Field(default_factory=list)  # Korean word palette hints
    profanity_ceiling: int = 1  # 0 none … 3 strongest Korean profanity allowed
    signature_ko: str | None = None  # one example line in the target Korean voice


class Character(BaseModel):
    id: str  # stable slug, e.g. "peter"
    en_name: str
    ko_name: str  # transliteration per 외래어 표기법 / established Korean usage
    aliases: list[str] = Field(default_factory=list)
    persona: str  # personality, role, age band, what drives them (≤40 words)
    voice: VoiceCard


# ---------------------------------------------------------------- S2r register table

EdgeLevel = Literal["hae", "haeyo", "hasipsio", "haera"]
ShiftReason = Literal["role_on", "role_off", "identity_reveal", "first_private_talk_equals", "relationship_change"]


class RegisterShift(BaseModel):
    """A beat where one pair's level changes (style guide §3 'Shifts')."""

    at_cue: str  # first cue in the new level
    level: EdgeLevel
    reason: ShiftReason
    until_cue: str | None = None  # first cue back at the edge level; None = to the video's end


def shifted_level(level: str, shifts: list[RegisterShift], cue_id: str, pos: dict[str, int]) -> str:
    """*level* with the shift whose span ``[at_cue, until_cue)`` holds *cue_id*
    applied; *pos* maps cue id -> video-order index. Unknown cues keep *level*."""
    at = pos.get(cue_id)
    if at is None:
        return level
    for sh in shifts:
        start = pos.get(sh.at_cue)
        end = pos.get(sh.until_cue, len(pos)) if sh.until_cue else len(pos)
        if start is not None and start <= at < end:
            return sh.level
    return level


class RegisterEdge(BaseModel):
    """One directed speaker->addressee entry of the register table (S2r).

    The feature fields record why the style guide §3 ladder step applies;
    ``firm`` is computed (:func:`substage.register.edge_firm`), never stored."""

    speaker: str  # character id
    addressee: str  # character id, or "group"
    relation: Literal[
        "own_parent", "own_child", "elder_kin", "surrogate_parent", "mentor", "protege", "boss",
        "subordinate", "crew_buddy", "peer", "sibling", "partner", "professional", "stranger_civil",
        "enemy", "ai_to_user", "adult_to_kid", "kid_to_adult", "group",
    ]
    apparent_generation: Literal["speaker_older", "same", "addressee_older"]
    history: Literal[
        "raised_together", "raised_by_addressee", "raised_addressee", "longtime_crew_or_buddies",
        "met_as_adults", "met_as_child", "none",
    ]
    stance: Literal["civil", "open_enmity", "contempt_from_power"]
    ai_user: bool = False
    speaker_comic_mockery: bool = False
    evidence_cues: list[str] = Field(default_factory=list)
    step: int  # style guide §3 ladder step that decided the level
    level: EdgeLevel
    shifts: list[RegisterShift] = Field(default_factory=list)
    pronoun: Literal["너", "자네", "당신", "none"] = "none"
    vocative: str = "none"  # kin/role/name term for the addressee
    basis: str = ""


class RegisterMap(BaseModel):
    """register.yaml: the video's register table (human-editable)."""

    edges: list[RegisterEdge] = Field(default_factory=list)
    # "spk>adr" pairs S2r had to cover (audit only; edits here change nothing)
    required_pairs: list[str] = Field(default_factory=list)


class SceneRegister(BaseModel):
    """One pair's register as a scene's snapshot delivers it."""

    level: EdgeLevel  # the table level (outside every shift)
    pronoun: str = "none"
    vocative: str = "none"
    shifts: list[RegisterShift] = Field(default_factory=list)  # shifts whose span overlaps the scene

    def level_at(self, cue_id: str, pos: dict[str, int]) -> str:
        return shifted_level(self.level, self.shifts, cue_id, pos)


class EmotionWaypoint(BaseModel):
    cue_id: str
    state: str
    cause: str


class EmotionalArc(BaseModel):
    character: str
    waypoints: list[EmotionWaypoint] = Field(default_factory=list)


class KnownBy(BaseModel):
    character: str
    from_cue: str


class TwistFact(BaseModel):
    fact: str
    audience_reveal_cue: str  # first cue at which the audience can know it
    known_by: list[KnownBy] = Field(default_factory=list)


class Callback(BaseModel):
    motif: str
    setup_cues: list[str]
    payoff_cues: list[str]
    note: str = ""


class GlossaryEntry(BaseModel):
    en: str
    ko: str
    note: str = ""


class AmbiguityFlag(BaseModel):
    cue_id: str
    note: str  # what must stay double-edged and why


class ContextMap(BaseModel):
    synopsis: str
    tone_baseline: str
    rating: str | None = None
    characters: list[Character] = Field(default_factory=list)
    emotional_arcs: list[EmotionalArc] = Field(default_factory=list)
    twist_facts: list[TwistFact] = Field(default_factory=list)
    callbacks: list[Callback] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    ambiguous: list[AmbiguityFlag] = Field(default_factory=list)


# ---------------------------------------------------------------- S2b timeline


class SceneEvent(BaseModel):
    cue_id: str
    description: str
    characters: list[str] = Field(default_factory=list)


class Revelation(BaseModel):
    cue_id: str
    fact: str
    to_whom: list[str]  # character ids, or ["audience"]


class EmotionChange(BaseModel):
    cue_id: str
    character: str
    from_state: str
    to_state: str


class SceneBeat(BaseModel):
    scene_id: str
    beat: str  # ≤20 words
    chars_present: list[str] = Field(default_factory=list)
    events: list[SceneEvent] = Field(default_factory=list)
    revelations: list[Revelation] = Field(default_factory=list)
    emotion_changes: list[EmotionChange] = Field(default_factory=list)


class Timeline(BaseModel):
    scenes: list[SceneBeat] = Field(default_factory=list)


# ---------------------------------------------------------------- S2c snapshot


class CharacterState(BaseModel):
    emotion: str | None = None
    knows: list[str] = Field(default_factory=list)
    recent_events: list[str] = Field(default_factory=list)  # ≤5, most relevant past events


class CueDelta(BaseModel):
    cue_id: str
    emotion_changes: list[EmotionChange] = Field(default_factory=list)
    revelations: list[Revelation] = Field(default_factory=list)


class Snapshot(BaseModel):
    """Causal state for one scene: what holds at scene entry plus cue-indexed deltas."""

    scene_id: str
    # "peter>tony" -> the register table entry for pairs speaking in this scene
    registers: dict[str, SceneRegister] = Field(default_factory=dict)
    characters: dict[str, CharacterState] = Field(default_factory=dict)
    cue_deltas: list[CueDelta] = Field(default_factory=list)
    voice_cards: dict[str, VoiceCard] = Field(default_factory=dict)  # present characters only
    glossary: list[GlossaryEntry] = Field(default_factory=list)  # terms used in this scene
    callbacks_live: list[Callback] = Field(default_factory=list)
    preserve_ambiguity: list[AmbiguityFlag] = Field(default_factory=list)
    character_names: dict[str, str] = Field(default_factory=dict)  # present character id -> ko_name (use exactly)
    tone_baseline: str = ""  # the video's overall dialogue register (video map)
    rating: str | None = None
    personas: dict[str, str] = Field(default_factory=dict)  # present character id -> one-line persona

    def for_prompt(self, narrative: str) -> Snapshot:
        """The snapshot a prompt may see at narrative level ``full`` (as is)
        or ``lite``: story facts plus the register table — character names,
        personas, video tone and rating, the scene's speaker>addressee registers
        (level, address terms, shifts), knowledge, recent events, glossary,
        callbacks, ambiguity flags and in-scene revelations; no emotion deltas,
        voice cards or emotion states."""
        if narrative == "full":
            return self
        if narrative != "lite":
            raise ValueError(f"narrative must be lite|full, got {narrative!r}")
        return self.model_copy(
            update={
                "voice_cards": {},
                "characters": {cid: st.model_copy(update={"emotion": None}) for cid, st in self.characters.items()},
                "cue_deltas": [
                    CueDelta(cue_id=d.cue_id, revelations=list(d.revelations))
                    for d in self.cue_deltas
                    if d.revelations
                ],
            }
        )


# ---------------------------------------------------------------- S3 translation


class TranslatedCue(BaseModel):
    cue_id: str
    speaker: str | None = None
    addressee: str | None = None
    speech_level: Register = Register.UNKNOWN
    ko: str
    needs_media: MediaNeed = MediaNeed.NONE
    media_reason: str | None = None
    model: str | None = None  # model that last wrote this cue's Korean text
    degraded: bool | None = None  # this cue used a provider-blocked route fallback
    degradation_reason: str | None = None


class TimelineErratum(BaseModel):
    cue_ids: list[str]
    claim: str  # what the timeline/video map gets wrong
    evidence: str



class PragmaticCueNote(BaseModel):
    cue_id: str
    category: str = "literal"  # literal | body_or_partitive_metonymy | visual_action_deixis | idiom_or_slang | conversational_implicature | referent_or_ellipsis
    surface_literal_trap: str = ""
    situational_intent: str = ""
    korean_explicitation_anchor: str = ""


class SceneSituationCard(BaseModel):
    scene_id: str
    physical_blocking: str = ""
    cues: list[PragmaticCueNote] = Field(default_factory=list)

class SceneTranslation(BaseModel):
    scene_id: str
    mode: str  # text | av
    cues: list[TranslatedCue]
    errata: list[TimelineErratum] = Field(default_factory=list)
    model: str | None = None  # scene-level writer; mixed cues carry their own model
    degraded: bool | None = None
    degradation_reason: str | None = None


class _SemanticModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceAct(_SemanticModel):
    """An advisory act anchored to an exact, half-open English source span."""

    ref: str
    cue_id: str
    turn_ordinal: int = Field(ge=1)
    act_ordinal: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=1)
    text: str
    speaker: str | None = None
    meaning: str
    essential: bool | None = None  # unknown unless explicitly claimed; never truth
    uncertainty: list[str] = Field(default_factory=list)


class SourceRelation(_SemanticModel):
    kind: Literal["elapsed", "relative_time", "question_answer", "order", "list"]
    refs: list[str] = Field(min_length=1)
    claim: str
    uncertainty: list[str] = Field(default_factory=list)


class SemanticSource(_SemanticModel):
    scene_id: str
    acts: list[SourceAct]
    relations: list[SourceRelation] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)


class RetainedFragment(_SemanticModel):
    """Actual emitted Korean, not a source-turn count or proof of fidelity."""

    text: str
    refs: list[str] = Field(default_factory=list)
    speaker: str | None = None
    attribution: Literal["declared", "unverified"] = "declared"
    prior_refs: list[str] = Field(default_factory=list)


class OmittedAct(_SemanticModel):
    ref: str
    reason: str


class SemanticAttributionProvenance(_SemanticModel):
    stage: str
    model: str
    input_hash: str
    schema_version: str
    policy_hash: str


class SemanticAttributionResult(_SemanticModel):
    fragments: list[RetainedFragment]
    provenance: SemanticAttributionProvenance
    omissions: list[OmittedAct] = Field(default_factory=list)


class SemanticLineage(_SemanticModel):
    source: SemanticSource
    fragments_by_cue: dict[str, list[RetainedFragment]]
    omissions: list[OmittedAct] = Field(default_factory=list)
    structural_valid: bool = False
    semantic_assessment: str = "unassessed"
    attribution_passes: list[SemanticAttributionProvenance] = Field(default_factory=list)


class SemanticDraftCue(_SemanticModel):
    cue_id: str
    fragments: list[RetainedFragment]
    addressee: str | None = None
    speech_level: Register = Register.UNKNOWN
    needs_media: MediaNeed = MediaNeed.NONE
    media_reason: str | None = None


class SemanticDraft(_SemanticModel):
    """New-route writer wire schema; the legacy SceneTranslation is unchanged."""

    scene_id: str
    cues: list[SemanticDraftCue]
    omissions: list[OmittedAct] = Field(default_factory=list)
    errata: list[TimelineErratum] = Field(default_factory=list)
    source_conflicts: list[str] = Field(default_factory=list)


class SemanticProvenance(_SemanticModel):
    route_variant: Literal["semantic-clip", "semantic-image"]
    model: str
    extract_stage: str
    write_stage: str
    extract_hash: str
    write_hash: str
    media_hash: str
    route_version: str
    source_schema_version: str
    target_schema_version: str
    prompt_version: str
    policy_hash: str


class SemanticSceneTranslation(SceneTranslation):
    """Explicit result variant retaining extraction and actual target lineage."""

    model_config = ConfigDict(extra="forbid")
    lineage: SemanticLineage
    provenance: SemanticProvenance


# ---------------------------------------------------------------- S4/S5/C7


class CueEdit(BaseModel):
    cue_id: str
    new_ko: str
    reason: str
    kind: str  # register | omission | consistency | callback | naturalness | condensation | picture


class ReadFlag(BaseModel):
    cue_id: str
    problem: str  # 번역투, unclear referent, unnatural phrasing…


class ValidationIssue(BaseModel):
    cue_id: str
    rule: str  # cpl | lines | cps | punctuation | empty | linebreak
    detail: str


# ---------------------------------------------------------------- S5c layout


class OutEvent(BaseModel):
    """One displayed Korean subtitle: the cues it covers may be merged or split."""

    start_ms: int
    end_ms: int
    ko: str
    cue_ids: list[str] = Field(default_factory=list)
    fragments: list[RetainedFragment] = Field(default_factory=list)
    omissions: list[OmittedAct] = Field(default_factory=list)
    attribution_passes: list[SemanticAttributionProvenance] = Field(default_factory=list)
    channel: Literal["dialogue", "ost"] = "dialogue"
    position: Literal["an2", "an8"] = "an2"

# ---------------------------------------------------------------- accounting


class UsageRecord(BaseModel):
    stage: str
    model: str
    input_tokens: int
    output_tokens: int  # includes thinking tokens (billed as output)
    cached_tokens: int = 0  # cache reads
    cache_write_tokens: int = 0  # cache creation (Anthropic reports it separately)
    cost_usd: float
