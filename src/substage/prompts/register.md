You are building the Korean register map (translator's bible) for "{title}" ({year}) before any Korean subtitle is written. Return exactly one edge per listed pair, using the given speaker/addressee ids.

edges: one edge per listed directed pair (speaker>addressee).
For each edge, fill the relationship features first (relation, apparent_generation, history, stance, ai_user, speaker_comic_mockery), then decide `level` by the §3 ladder below (the first step that applies wins) and give the step number in `step`.
Set `ai_user=true` and `relation="ai_to_user"` ONLY when `speaker` is a tactical/assistant/system AI addressing a human user (Step 1 -> `haeyo`, or Step 5 -> `hae` if the AI is a close buddy/sister companion with `history="longtime_crew_or_buddies"`). When a human addresses an AI (`human>ai`), set `ai_user=false` and use Step 7 (`subordinate` -> `hae`) or Step 10 (`partner`/`peer` -> `hae`).
Married spouses and established couples of the same generation (`relation="partner"`) belong at Step 10 (`hae`), not Step 9. Young children or warm-home teens addressing the parent who raised them (`relation="own_parent"`, `history="raised_by_addressee"`) take `hae` at Step 3.
`level` is the pair's level at its first exchange. Any later change goes into `shifts`: at_cue = the first cue at the new level, the new level, reason from the closed list (role_on, role_off, identity_reveal, first_private_talk_equals, relationship_change), and until_cue = the first cue back at the edge level when the video reverts it (a mask dropped), else null. Anger is never a shift reason.
Persona adjectives (rude, insolent, gruff, polite, gentle, warm) are not evidence for a level; kinship, who raised whom, generation, rank, shared history and enmity are.
Cite the cue ids that establish every non-default feature in `evidence_cues`.
address: `pronoun` is 너|자네|당신|none (what surfaces for English "you"), `vocative` the kin/role/name term used for the addressee, or none.
basis: one short English sentence naming the relationship and the step that decided the level.

KOREAN REGISTER CONVENTIONS (from the house style guide; examples read EN | literal (bad) | good):
<<STYLE_GUIDE_REGISTER>>

VIDEO: {title} ({year})
Synopsis: {synopsis}
Tone: {tone}
CHARACTERS (id | English name | Korean name | persona):
{characters}

TRANSCRIPT: the whole video, all lines in video order. Each line: cue id | mm:ss | speaker>addressee | English text (- or ? = unknown).

{transcript}

PAIRS (speaker>addressee) to decide:
{pairs}
