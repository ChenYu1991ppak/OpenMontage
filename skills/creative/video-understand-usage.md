# Video Understanding Usage for OpenMontage

> Sources: OpenMontage video_understand_openai tool implementation, Moonshot vision API,
> OpenCV image quality metrics

## Quick Reference Card

```
DEFAULT MODE:     describe — generates captions for sampled frames
FOR REVIEW:       quality — assesses blur, brightness, contrast (FREE, local)
FOR Q&A:          qa mode with a query — "Is the speaker visible?" "Is the text readable?"
DEFAULT MODEL:    Moonshot vision model (API); quality mode is fully offline
MAX FRAMES:       5 default for video — sample strategically, not exhaustively
```

## Which Tool to Use

| Need | Tool | Notes |
|------|------|-------|
| **Video content understanding** | `video_understand_openai` | API modes need `OPENAI_VISION_API_KEY`; `quality` mode is local & free |
| **Single image understanding** | `image_understand_openai` | See `skills/creative/image-understand-usage.md` |
| **Local frame extraction / probe** | `visual_qa` | ffmpeg-based, no model — pairs with `video_understand_openai` |

> **Note:** The legacy `video_understand` tool (local CLIP/BLIP2/LLaVA via transformers) is
> **unavailable** unless `transformers` + `torch` are installed. Use `video_understand_openai`
> instead — it is API-backed (Moonshot), and its `quality` mode is fully local.

## When to Use video_understand_openai

- **Visual QA during review** — check rendered output quality before delivering
- **Footage analysis** — understand what's in user-provided footage before planning
- **Highlight extraction** — identify the most visually interesting frames
- **Quality gating** — programmatic check for blur, exposure, scene coherence
- **Scene classification** — categorize footage by content type
- **Asset validation** — verify generated images match the intended scene description

## Mode Selection

| Mode | What It Does | When to Use |
|------|-------------|-------------|
| `describe` | Generates a text description of sampled frames | Understanding footage content, logging |
| `qa` | Answers a specific question about sampled frames | Targeted checks ("Is text readable?", "Is face visible?") |
| `quality` | Measures blur, brightness, contrast numerically | Automated quality gating, comparing takes |
| `classify` | Categorizes the scene type | Sorting footage, pipeline routing |

### Quality Mode Metrics

| Metric | What It Measures | Bad | Good |
|--------|-----------------|-----|------|
| `blur_score` | Laplacian variance | Below 100 = blurry | Above 500 = sharp |
| `brightness` | Mean pixel value (0-255) | Below 50 = too dark, above 200 = overexposed | 50-200 |
| `contrast` | Pixel standard deviation | Below 30 = flat/washed out | Above 80 = good contrast |

> `quality` mode runs fully offline (ffmpeg frame extraction + local metrics), no API key needed.
> `describe` / `qa` / `classify` upload sampled frames to the Moonshot vision API
> (requires `OPENAI_VISION_API_KEY`; videos ≤70MB / ~1080p for whole-video upload).

## Frame Selection for Video

- Default samples `max_frames` (5) evenly across the video
- Use `frame_indices` to target specific frames (e.g., check quality at specific timestamps)
- For quality review, sample the first frame, middle frame, and last frame minimum

## Common Workflows

### 1. Pre-Edit Footage Review

```
video_understand_openai (describe, 10 frames) → inform scene_plan
```

Analyze user-provided footage before planning cuts or edits. Use rich descriptions that inform the scene plan.

### 2. Post-Render Quality Gate

```
video_understand_openai (quality) → pass/fail → re-render if needed
```

Run after composing the final video. Fail if any frame has blur_score < 100, brightness outside 50-200, or contrast < 30.

### 3. Highlight Selection

```
video_understand_openai (describe, 20 frames) → rank by visual interest → select clips
```

Sample many frames, describe each, then select the most visually compelling segments for a montage or trailer.

### 4. Asset Validation

```
video_understand_openai (qa, "Does this match: [scene description]?") → confirm or regenerate
```

After generating an image or video clip, verify it matches the intended scene description before proceeding. For stills, `image_understand_openai` is the right tool.

### 5. Talking-Head Analysis

```
video_understand_openai (qa, "Is the speaker's face clearly visible?") → face_enhance if needed
```

Check face visibility and framing before applying lip-sync or face restoration tools.

## Quality Checklist

- Descriptions accurately match what's in the frames
- Quality scores correlate with visual inspection (manually spot-check)
- QA answers are consistent across similar frames
- Classification categories are stable across adjacent frames
- No false positives in quality gating (good frames passing, bad frames failing)

## Applying to OpenMontage

When using the `video_understand_openai` tool:

1. **Use `quality` mode as a post-render gate in the compose stage** — reject outputs below quality thresholds (free, local)
2. **Use `describe` mode to analyze user-provided footage** at the start of the talking-head pipeline
3. **Use API modes (`describe`/`qa`/`classify`) when `OPENAI_VISION_API_KEY` is set** — richer understanding than local metrics
4. **Sample at least 3 frames for quality assessment** — beginning, middle, end
5. **Quality thresholds for passing:** blur_score > 100, brightness 50-200, contrast > 30
6. **Use `qa` mode to validate generated assets:** "Does this image show [expected content]?"
7. **In the review stage**, combine video_understand_openai quality data with the reviewer skill's rubric
8. **Do NOT run video_understand_openai on every frame of a long video** — sample strategically
9. **For single images, use `image_understand_openai`** (agent skill: `image-understand`)
