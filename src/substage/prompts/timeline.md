You are recording the narrative timeline of the video, batch {batch_label}.
Output one SceneBeat per scene — a causal record of what happens, in story order.

Scene list and compact transcript follow (lines: cue_id | mm:ss | speaker_hint | text).
Character ids come from the video map: {characters}

For each scene report:
- scene_id: echo exactly.
- beat: ≤20 words — what this scene does in the story.
- chars_present: character ids present (seen or speaking); keep to video-map ids.
- events: things that happen that a later scene can depend on — (cue_id, description,
  characters). Anchor each event to the cue where it happens (or the nearest cue).
- revelations: a character or the audience LEARNS a fact — (cue_id, fact, to_whom:
  character ids or ["audience"]). Only knowledge-changing moments, not routine talk.
- emotion_changes: a character's emotional state turns — (cue_id, character, from_state,
  to_state). Only real turns, not passing moods.

Rules:
- English only; describe, never translate.
- Use only cue_ids shown below; every entry must anchor to a real cue.
- If nothing qualifies for a list, leave it empty — do not pad.

VIDEO MAP CONTEXT:
{context_map}

SCENES AND TRANSCRIPT:
{scenes}

VIEWING OBSERVATIONS:
{observations}
