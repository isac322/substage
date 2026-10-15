# substage

Scene-aware multimodal English-to-Korean video subtitle translator.

Unlike line-by-line machine translation, `substage` watches the video's audio and video to identify who is speaking to whom, infer character relationships and Korean speech levels (`해체` / `해요체` / `하십시오체`), read on-screen text (signs, title cards, gag credits), and format the final subtitles to professional Korean broadcast standards (`.srt`).

## Key Features

- **Hardware-Accelerated Media Ingestion (`S0–S1`)**: Automatically uses `jellyfin-ffmpeg` (`ffmpeg-rk` / `rkmpp` + `scale_rkrga` on Rockchip Linux, or VideoToolbox on macOS) to build lightweight 360p proxies, extract dialogue audio, and segment subtitle cues into scenes.
- **Whole-Video Viewing & Editable Guidebooks (`S2`)**: Watches the video in ~10-minute chunks (plus non-dialogue opening sequences) to record speakers, addressees, delivery, and on-screen text. Synthesizes human-editable YAML guidebooks (`context_map.yaml`, `timeline.yaml`, `register.yaml`) that lock speech levels and address terms across every scene.
- **Multimodal Scene Translation & Register Gating (`S3–S4`)**: Supports direct audio/video translation (`direct-av`, `semantic-clip`), keyframe snapshot translation (`text-snapshot`, `semantic-image`), and selective A/V escalation (`cascade`, `dpaf`). Every post-translation edit is gated by the character register table so speech levels never drift mid-scene.
- **Korean Subtitle Formatting & Timing Layout (`S5–S6`)**: Enforces two-speaker dialogue dashes (`- Speaker A` / `- Speaker B`) while keeping single-speaker lines dashless, enforces character-per-line (`≤14` chars, max `16`) and reading speed (`≤8` CPS) limits, and emits UTF-8 `.ko.srt` alongside a full provenance sidecar (`.ko.json`).
- **Official-Reference Evaluation (`substage refeval`)**: Evaluates candidate subtitles against official human-translated Korean subtitles (anchored at 100 points), scoring meaning preservation, register agreement, two-speaker dash recall, and chrF++.
- **Web Review Server (`substage review-server`)**: Built-in zero-dependency web UI for side-by-side playback of the video scene with English, official Korean, and `substage` subtitles, cue search, diff filtering, and timestamped review notes.

## Pipeline Overview

| Stage | Output Artifact | Description |
|---|---|---|
| `S0` Ingest | `video.json`, `cues.json`, `proxy.mp4` | Probes media streams, extracts or downloads English subtitles (`subliminal` + `ffsubsync`), and builds the 360p A/V proxy. |
| `S1` Scenes | `scenes.json` | Groups subtitle cues into narrative scenes by timing gaps and length. |
| `S2v` Viewing | `observations.json` | Multimodal pass recording speaker/addressee turns, tone, nonverbal beats, and on-screen text (`OST`). |
| `S2a–S2r` Guidebooks | `context_map.yaml`, `timeline.yaml`, `register.yaml` | Character roster, causal scene timeline, and pairwise Korean speech-level / vocative table (human-editable). |
| `S2c` Snapshots | `snapshots.json` | Scene-entry causal state (spoiler-free up to that point in the video) + active register rules. |
| `S3` Translate | `translations.json` | Scene-by-scene Korean translation with optional BM25 translator-memory style examples. |
| `S4–C7` Review & Check | `edits.json` | Bilingual window review, whole-video consistency audit, blind Korean read, and A/V picture checks. |
| `S5–S6` Layout & Emit | `<video>.ko.srt`, `<video>.ko.json` | Re-segments cues into non-overlapping display events, enforces Korean CPS/CPL norms, and writes the final `.srt`. |

Every stage is checkpointed by input hash; re-running resumes automatically from the first changed stage.

## Installation

Requires **Python 3.15+** and [`uv`](https://docs.astral.sh/uv/), plus `ffmpeg` / `ffprobe` on `PATH` (on Rockchip Linux, `ffmpeg-rk` / `ffprobe-rk` from `jellyfin-ffmpeg` are detected automatically):

```sh
git clone https://github.com/isac322/substage.git
cd substage
uv sync
```

## Configuration

Store API credentials in `$XDG_CONFIG_HOME/substage/secrets.env` (`~/.config/substage/secrets.env`, mode `0600`):

```ini
# Gemini Developer API or Vertex AI
GEMINI_API_KEY=your_gemini_api_key
# Optional: Vertex AI configuration
GOOGLE_CLOUD_PROJECT=your_gcp_project
GOOGLE_VERTEX_LOCATION=global

# Optional: Anthropic / OpenAI-compatible endpoints
SUBSTAGE_ANTHROPIC_BASE_URL=https://api.anthropic.com
SUBSTAGE_ANTHROPIC_API_KEY=your_anthropic_key
SUBSTAGE_CODEX_BASE_URL=https://api.openai.com/v1
SUBSTAGE_CODEX_API_KEY=your_openai_key
```

Optional runtime overrides go in `~/.config/substage/config.toml`:

| Setting | Default | Description |
|---|---|---|
| `model_main` | `"gemini-3.8-flash"` | Primary multimodal model for viewing, A/V translation, and picture checks |
| `model_light` | `"gemini-3.5-flash-lite"` | Lightweight model for blind Korean reads and line compression |
| `model_translate` | `"anthropic/claude-opus-5-5"` | Default text-stage translator (`gemini-*`, `anthropic/*`, or `codex/*`) |
| `cpl_limit` / `cpl_one_line` | `14` / `12` | Target characters per line (`16` hard cap) and single-line retention threshold |
| `cps_target` / `cps_max` | `4.9` / `8.0` | Target and maximum Korean reading speed (characters per second, excluding spaces) |
| `min_duration_ms` / `max_duration_ms` | `1000` / `5700` | Minimum and maximum display duration per subtitle event |
| `merge_max_gap_ms` / `split_min_ms` | `250` / `2400` | Adjacent cue merge gap and minimum duration for splitting long single-speaker cues |
| `ffmpeg_bin` / `ffprobe_bin` | `"ffmpeg-rk"` (Linux) / `"ffmpeg"` | Media binaries used for hardware-accelerated proxy and clip extraction |

All workspaces follow the XDG Base Directory specification:
- Artifacts & YAML guidebooks: `~/.local/share/substage/<video_id>/`
- Proxies & media cache: `~/.cache/substage/<video_id>/`
- Usage logs: `~/.local/state/substage/<video_id>/usage.jsonl`

## Usage

```sh
# Translate a video end-to-end
substage run video.mkv --video-id sample-2016 --title "SampleVideo" --year 2016

# Explicit model and capability route (direct-av, text-snapshot, semantic-clip, semantic-image, cascade, dpaf)
substage run video.mkv --video-id sample-2016 --model gemini-3.8-flash --architecture direct-av

# Stop after building the character/register guidebooks, edit YAML if desired, then resume
substage run video.mkv --video-id sample-2016 --stop-after register
$EDITOR ~/.local/share/substage/sample-2016/register.yaml
substage run video.mkv --video-id sample-2016

# Inspect token usage and estimated cost by stage
substage cost sample-2016

# Evaluate against an official Korean subtitle file
substage refeval sample-2016 --arms substage=sample-2016 --ref official.ko.srt --judge

# Build a 720p review video and launch the side-by-side web review UI (default http://127.0.0.1:8000/)
substage review-video video.mkv --video-id sample-2016
substage review-server --host 127.0.0.1 --port 8000
```

## License

[MIT](LICENSE)
