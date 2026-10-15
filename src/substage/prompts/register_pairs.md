You are listing who speaks to whom in "{title}" ({year}), so a Korean translator knows which speaker>addressee pairs need a speech level.

Read the whole transcript and return `pairs`: every directed pair (speaker>addressee) that exchanges at least one line in the video, with `cue_ids` = every cue in which the speaker addresses that addressee.
- `speaker` and `addressee` must be ids from CHARACTERS, exactly as written there. Never invent ids; skip lines whose speaker or addressee is not on the list.
- `addressee` = "group" when the speaker addresses several people at once, a crowd, an audience, the press or viewers.
- A line belongs to the person it is said to, not to the people it mentions. Lines to oneself, to an animal, or to nobody in particular belong to no pair.
- The speaker -> addressee names in the transcript come from a viewing pass and an SDH speaker label; they may be missing (?), misspelled, a title or surname only (Dr. Mindy, the President), or wrong. Use the dialogue and its context to decide, and map each name to its CHARACTERS id.
- One entry per directed pair; peter>tony and tony>peter are separate entries.

CHARACTERS (id | English name | Korean name | aliases):
{characters}

TRANSCRIPT: the whole video, all lines in video order. Each line: cue id | mm:ss | speaker -> addressee | English text (? = unknown).

{transcript}
