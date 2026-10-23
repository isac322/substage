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
- lyric_plot_relevant: for ♪ cues (song lyrics) only — true when an on-screen or named
  character sings/chants the line (including rhyme games, parodies, children's songs, hymns,
  or birthday songs), when the line carries story information, is dialogue addressed to a
  character, or calls back to an earlier line/motif; false for unattributed background BGM,
  credit songs, songs playing under foreground speech/signs, or pure vocable noise. Omit on non-♪ cues.

Also report:
- present_characters: every named character visible or heard in this chunk (consistent names).
- events: silent or undialogued moments that matter — only events that change the plot or a
  character's knowledge (X sees Y's betrayal, Z finds the letter, a text message arrives).
  Give t_ms, description, characters involved, and knowledge_change ("who learns what") if any.
- on_screen_text: everything a viewer who cannot read English needs subtitled, plus audible
  words the cue list omits. Begin each text with one kind tag, then the exact words:
  [message] one phone text/chat bubble or SMS (report EACH bubble as its own item with its own start_ms/end_ms; never concatenate sequential bubbles into one string)
  [ui] caller ID, phone status ("Do Not Disturb", "No Service"), HUD, or computer/device prompt
  [note] handwritten note, letter, pamphlet, magazine/comic cover, calendar, or book cover
  [headline] news/magazine article headline or TV chyron
  [callout] highlighted article sentence, giant pop-out phrase, or kinetic subtext/thought overlay shown on screen simultaneously with spoken dialogue
  [sign] physical signs, street blades, banners, placards, close-up prop labels (coffee cup name, pizza box, cylinder/bottle label, arcade prize ticket sign, photo label), license plates, or nametags held in the shot
  [caption] location/time cards ("IXTENCO, MEXICO", "VENICE, ITALY", "SIX YEARS LATER") and character lower-thirds
  [title] the video's title card or a chapter card
  [gag_credit] satirical or comedic opening/closing credit card ("STARRING GOD'S PERFECT IDIOT", "PRODUCED BY ASSHATS")
  [credit] real opening/closing studio/cast/crew credit card (director, writer, actor billing)
  [dedication] dedication or memorial card ("For Leah", "In Memoriam")
  [spoken] an audible word or short call that has NO cue in the list (a shouted name, "Hey!")
  [logo] studio/brand logo or watermark   [scroll] a scrolling list of credit names
  Report [headline] and [callout] even during continuous dialogue; do not omit them because a spoken cue overlaps their on-screen lifetime.
  start_ms/end_ms: absolute video ms ({start_ms} + time into the clip), the span the text is
  readable or the words are heard. When a top-left burned-in video timecode (HH:MM:SS.mmm) is visible on the frame, read start_ms/end_ms directly from that burned-in movie clock (converted to milliseconds: (HH*3600 + MM*60 + SS)*1000 + mmm) instead of estimating clip offsets. Report each bubble, card, or insert appearance separately, even when a close-up prop or street sign is on screen for only 1–2 seconds.
  For rapid montages of headlines, articles, slides, or social-media posts, report every distinct shot/cut in chronological order with its own start_ms/end_ms. Never collapse the montage into only its first and last item.
  When one shot shows multiple headlines, clippings, or tweets at once, combine its key readable headline/callout into one concise item for that shot's [start_ms, end_ms]; do not emit multiple overlapping items with identical timestamps. Omit UI chrome and boilerplate such as "Click to add title", browser toolbars, and author bylines.
  plot_critical: true for every kind except [logo], [scroll] and out-of-focus background clutter. Any close-up shot whose subject is a readable prop, label, cover, or sign is plot_critical: true.
  covered_by_cue: only when that cue already speaks the exact same words aloud verbatim. Skip decorative text.

Rules:
- Describe, do not interpret style or translate. Korean is never used in your output.
- If audio/subtitle timing drifts from the picture, trust the picture and say so in summary.
- Only report cue_ids from the list; do not invent cues.
- Dialect and accent (translation style guide §3.9): when a speaker's thick accent or dialect is itself the butt of a joke or a communication barrier, put `comic_dialect` plus the accent (e.g. "comic_dialect: thick Cajun") in delivery; when a line is a put-on accent or the subtitle carries a manner-of-speaking tag (`(IN BRITISH ACCENT)`, `[In New York accent]`), note it in delivery, and name the accent an accent-metaphor joke refers to (Sling Blade → Southern drawl).
