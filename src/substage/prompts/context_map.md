You are building the video map (translator's bible) for "{title}" ({year}) before any Korean
subtitle is written — what a professional translator produces after pre-screening the video.
{tmdb_hint}
You receive the whole English transcript (compact lines: cue_id | mm:ss | speaker_hint | text;
speaker_hint "-" = unknown) plus observation notes from a viewing pass (who said what to whom,
delivery, nonverbal beats, silent knowledge events, on-screen text).

Produce the ContextMap — in English EXCEPT ko_name / signature_ko / vocabulary which are Korean:

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
- glossary: ≤40 entries — franchise/proper nouns with official Korean renderings when a
  well-known convention exists (established series…), plus coined terms the video relies on. en/ko/note.
- ambiguous: cues that must stay double-edged in Korean (dramatic irony, twist shielding,
  deliberate vagueness): cue_id, note saying what to preserve and why.

Rules:
- Prefer IDs/names consistently; every cue_id must exist in the transcript.
- Be selective: a thin accurate map beats a bloated one.
- Dialect (style guide §3.9): when a character's thick dialect or eye-dialect is itself a running joke or communication barrier, say `comic_dialect` in their persona and give the voice card's example line earthy Korean dialect endings (근디, ~거시여, ~제, ~믄) at their fixed level; record accent-metaphor jokes (Sling Blade, Basil Fawlty) and manner-of-speaking SDH (`[In New York accent]`) as callbacks/notes to localize, never as phonetic glossary entries.

KOREAN REGISTER CONVENTIONS (from the house style guide; examples read EN | literal (bad) | good):
<<STYLE_GUIDE_REGISTER>>

TRANSCRIPT:
{transcript}

VIEWING OBSERVATIONS:
{observations}
