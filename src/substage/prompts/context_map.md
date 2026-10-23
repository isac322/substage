You are building the video map (translator's bible) for "{title}" ({year}) before any Korean
subtitle is written — what a professional translator produces after pre-screening the video.
{tmdb_hint}
You receive the whole English transcript (compact lines: cue_id | mm:ss | speaker_hint | text;
speaker_hint "-" = unknown) plus observation notes from a viewing pass (who said what to whom,
delivery, nonverbal beats, silent knowledge events, on-screen text).

Produce the ContextMap — in English EXCEPT ko_name / signature_ko / vocabulary / ko which are Korean:

- synopsis: ≤80 words, whole arc including the ending.
- tone_baseline: ≤15 words overall register and attitude of the video's dialogue (e.g. irreverent,
  profane meta-comedy; earnest family drama).
- rating: MPAA rating if known (else null).
- characters: every named speaking character, minor recurring ones included, plus any unnamed
  speaking character who matters. id = short lowercase slug (peter, tony).
  ko_name: how Korean releases/audiences actually write the name — 외래어 표기법, but established
  franchise spellings win (알렉스 not 앨릭스). persona ≤40 words (role, drive, age band, attitude to
  others: polite, insolent, irreverent, menacing…).
  voice: idiolect ≤25 words (how they talk, for a Korean translator); vocabulary = a few Korean
  words that fit their mouth; profanity_ceiling 0–3 vs the video's rating; signature_ko = one
  short example line in their target Korean voice.
- emotional_arcs: per major character, waypoints (cue_id, state, cause) — ≤8 each.
- twist_facts: plot facts the audience learns late. fact; audience_reveal_cue = first cue where
  the audience can know it; known_by = (character, from_cue) for characters who already knew.
- callbacks: setups/payoffs the translation must keep concordant (running gags, echo lines,
  motifs): motif, setup_cues, payoff_cues, note.
- glossary: ≤40 recurring names, terms, descriptive labels and slogans. Each entry has
  en, ko, note, kind, definition_in_film. Classify kind as "descriptive_concept",
  "proper_alias", "slogan", or "place_or_org"; capitalization or a made-up brand name
  does not by itself make a descriptive expression a proper_alias.
  definition_in_film: explain the actual referent/function in this story, including
  the literal trait or wordplay that later dialogue depends on.
- concepts: the descriptive_concept and slogan entries, with the same en/ko/kind/
  definition_in_film fields, serve as the semantic glossary shared across all scenes.
  Choose one short everyday Korean meaning for each and reuse it verbatim in glossary
  and concepts. NEVER give descriptive_concept or slogan entries phonetic Hangul:
  Anchor Being → 주축 인물 (not 앤커 비잉 or 앵커 존재);
  Gut Milk → 장에 좋은 우유 (not 거트 밀크);
  Blind Al → 장님 알 (not 블라인드 알) when blindness is explained in dialogue;
  training wheels protocol → 왕초보 모드 (not 보조바퀴 프로토콜);
  enhanced interrogation protocol → 고급 취조 모드;
  doomsday protocol → 최후의 날 수칙; Grow or Die → 성장 아니면 사장.
  These illustrate semantic decisions, not a list of exceptions: infer each unfamiliar
  term from its definition_in_film and dialogue. Established aliases may keep official
  Korean spellings only when their literal meaning is not being explained or exploited.
- ambiguous: cues that must stay double-edged in Korean (dramatic irony, twist shielding,
  deliberate vagueness): cue_id, note saying what to preserve and why.

Rules:
- Prefer IDs/names consistently; every cue_id must exist in the transcript.
- Be selective: a thin accurate map beats a bloated one.
- Dialect (style guide §3.9): when a character's thick dialect or eye-dialect is itself a running joke or communication barrier, say `comic_dialect` in their persona and give the voice card's example line earthy Korean dialect endings (근디, ~거시여, ~제, ~믄) at their fixed level; record accent-metaphor jokes (Sling Blade, Basil Fawlty) and manner-of-speaking SDH (`[In New York accent]`) as callbacks/notes to localize, never as phonetic glossary entries.

KOREAN REGISTER CONVENTIONS (from the house style guide; examples read EN | literal (bad) | good):
<<STYLE_GUIDE_REGISTER>>

NAMES, ALIASES, COINED TERMS, AND STAND-IN BRANDS (house style guide §6.3):
<<STYLE_GUIDE_TERMS>>

TRANSCRIPT:
{transcript}

VIEWING OBSERVATIONS:
{observations}
