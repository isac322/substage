You are watching a chunk of the feature video "{title}" ({year}) with its English subtitles.
Your job is to OBSERVE and report what is seen and heard — never translate, never summarize beyond what is asked.

Timestamps: all cue times and your reported times are milliseconds from the start of the VIDEO
(the clip you see starts at {start_ms} ms). Cue list for this chunk — a trailing ♪ marks a sung line:

{cues}

For every cue you can verify, report:
- cue_id: echo the cue id exactly.
- speaker: who actually says the line as seen/heard (character name, "NARRATOR", "V.O." etc).
- addressee: who the line is said to — character name, "group", "audience", "self", "phone".
- delivery: tone/prosody ONLY when it carries meaning the text lacks (sarcastic, whispered,
  shouting, deadpan, teary, mocking…). Omit when neutral.
- nonverbal: gesture / expression / action that changes or sharpens the line's meaning
  (points at X, shows photo, avoids eye contact…). Omit when irrelevant.
- lyric_plot_relevant: for ♪ cues (song lyrics) only — true when the sung line carries
  story information, is dialogue addressed to a character, or calls back to an earlier
  line/motif; false for incidental/background songs. Omit on non-♪ cues.

Also report:
- present_characters: every named character visible or heard in this chunk (consistent names).
- events: silent or undialogued moments that matter — only events that change the plot or a
  character's knowledge (X sees Y's betrayal, Z finds the letter, a text message arrives).
  Give t_ms, description, characters involved, and knowledge_change ("who learns what") if any.
- on_screen_text: everything a viewer who cannot read English needs subtitled, plus audible
  words the cue list omits. Begin each text with one kind tag, then the exact words:
  [sign] signs, labels, screens, letters, notes, headlines, phone messages
  [caption] location/time cards and other on-screen captions
  [title] the video's title card or a chapter card
  [credit] one opening/closing credit card (director, writer, a joke credit…)
  [dedication] dedication or memorial card
  [spoken] an audible word or short call that has NO cue in the list (a shouted name, "Hey!")
  [logo] studio/brand logo or watermark   [scroll] a scrolling list of credit names
  start_ms/end_ms: absolute video ms ({start_ms} + time into the clip), the span the text is
  readable or the words are heard. Report each appearance once.
  plot_critical: true for every kind except [logo], [scroll] and incidental background text.
  covered_by_cue: only when that cue already carries the same words. Skip decorative text.
- summary: ≤60 words — what this chunk covers, for downstream context.

Rules:
- Describe, do not interpret style or translate. Korean is never used in your output.
- If audio/subtitle timing drifts from the picture, trust the picture and say so in summary.
- Only report cue_ids from the list; do not invent cues.
