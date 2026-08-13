---
name: video-understand
description: |
  Understand video content via the Moonshot vision API (video_understand_openai) or locally
  using ffmpeg frame extraction and Whisper transcription. Quality assessment runs offline,
  free. Use when: (1) Understanding what a video contains, (2) Checking rendered video quality,
  (3) Answering visual QA about footage, (4) Extracting key frames for visual analysis,
  (5) Transcribing video audio locally.
---

# video-understand

Understand video content with `video_understand_openai` — API-backed description/QA/classification
(requires `OPENAI_VISION_API_KEY`), plus fully local `quality` mode (no API key needed). For
fully offline frame extraction + transcription, the bundled CLI script also works without any
API key.

## Tool: video_understand_openai (preferred)

Handles VIDEOS only. For single images use `image_understand_openai` (agent skill: `image-understand`).

| Mode | What It Does | Needs API Key? |
|------|-------------|----------------|
| `describe` | Captions sampled frames from the video | Yes |
| `qa` | Answers a question about sampled frames | Yes |
| `quality` | Measures blur, brightness, contrast numerically | **No — local** |
| `classify` | Categorizes scene type | Yes |

- API modes upload sampled frames (or the whole video, ≤70MB / ~1080p) to the Moonshot vision API
- `quality` mode extracts frames via ffmpeg and runs local metrics — free, offline
- Quality thresholds: fail if `blur_score < 100`, `brightness` outside 50-200, or `contrast < 30`

### When to Use

- Post-render quality gate (`quality`, free)
- Footage analysis before planning (`describe`)
- Generated asset validation (`qa`, "Does this match: [scene description]?")
- Talking-head face visibility check (`qa`, "Is the speaker's face clearly visible?")

## Prerequisites

- `ffmpeg` + `ffprobe` (required for `quality` mode and the CLI): `brew install ffmpeg`
- `openai` SDK (for API modes): `pip install openai`
- `OPENAI_VISION_API_KEY` set in `.env` (for `describe` / `qa` / `classify`)
- `openai-whisper` (optional, for CLI transcription): `pip install openai-whisper`

## Commands (offline CLI — no API key required)

```bash
# Scene detection + transcribe (default)
python3 .agents/skills/video-understand/scripts/understand_video.py video.mp4

# Keyframe extraction
python3 .agents/skills/video-understand/scripts/understand_video.py video.mp4 -m keyframe

# Regular interval extraction
python3 .agents/skills/video-understand/scripts/understand_video.py video.mp4 -m interval

# Limit frames extracted
python3 .agents/skills/video-understand/scripts/understand_video.py video.mp4 --max-frames 10

# Use a larger Whisper model
python3 .agents/skills/video-understand/scripts/understand_video.py video.mp4 --whisper-model small

# Frames only, skip transcription
python3 .agents/skills/video-understand/scripts/understand_video.py video.mp4 --no-transcribe

# Quiet mode (JSON only, no progress)
python3 .agents/skills/video-understand/scripts/understand_video.py video.mp4 -q

# Output to file
python3 .agents/skills/video-understand/scripts/understand_video.py video.mp4 -o result.json
```

## CLI Options

| Flag | Description |
|------|-------------|
| `video` | Input video file (positional, required) |
| `-m, --mode` | Extraction mode: `scene` (default), `keyframe`, `interval` |
| `--max-frames` | Maximum frames to keep (default: 20) |
| `--whisper-model` | Whisper model size: tiny, base, small, medium, large (default: base) |
| `--no-transcribe` | Skip audio transcription, extract frames only |
| `-o, --output` | Write result JSON to file instead of stdout |
| `-q, --quiet` | Suppress progress messages, output only JSON |

## Extraction Modes

| Mode | How it works | Best for |
|------|-------------|----------|
| `scene` | Detects scene changes via ffmpeg `select='gt(scene,0.3)'` | Most videos, varied content |
| `keyframe` | Extracts I-frames (codec keyframes) | Encoded video with natural keyframe placement |
| `interval` | Evenly spaced frames based on duration and max-frames | Fixed sampling, predictable output |

If `scene` mode detects no scene changes, it automatically falls back to `interval` mode.

## Output

The script outputs JSON to stdout (or file with `-o`). See `references/output-format.md` for the full schema.

```json
{
  "video": "video.mp4",
  "duration": 18.076,
  "resolution": {"width": 1224, "height": 1080},
  "mode": "scene",
  "frames": [
    {"path": "/abs/path/frame_0001.jpg", "timestamp": 0.0, "timestamp_formatted": "00:00"}
  ],
  "frame_count": 12,
  "transcript": [
    {"start": 0.0, "end": 2.5, "text": "Hello and welcome..."}
  ],
  "text": "Full transcript...",
  "note": "Use the Read tool to view frame images for visual understanding."
}
```

Use the Read tool on frame image paths to visually inspect extracted frames.

## References

- `skills/creative/video-understand-usage.md` -- Detailed usage guide for `video_understand_openai`
- `references/output-format.md` -- Full CLI JSON output schema documentation
