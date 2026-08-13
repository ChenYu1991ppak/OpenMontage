---
name: image-understand
description: |
  Understand image content using the Moonshot vision API or local quality metrics.
  Use when: (1) Describing what an image shows, (2) Answering visual QA questions about an image,
  (3) Checking image quality (blur/brightness/contrast), (4) Classifying image scene type,
  (5) Validating generated images against expected content.
---

# image-understand

Understand single images via `image_understand_openai` — uses the Moonshot vision API for description/QA/classification (requires `OPENAI_VISION_API_KEY`) and fully local quality assessment (no API key needed for `quality` mode).

## Prerequisites

- `ffmpeg` (optional, only for exotic formats; common formats need no preprocessing)
- `openai` SDK for API modes: `pip install openai`
- `OPENAI_VISION_API_KEY` set in `.env` for `describe` / `qa` / `classify` modes
- `quality` mode runs offline (ffmpeg frames + numpy/PIL metrics), no API key required

## Tool: image_understand_openai

Handles IMAGES only. For videos use `video_understand_openai` (agent skill: `video-understand`).

| Mode | What It Does | Needs API Key? |
|------|-------------|----------------|
| `describe` | Captions the image content | Yes |
| `qa` | Answers a question about the image (e.g. "Is the speaker's face visible?") | Yes |
| `quality` | Measures blur, brightness, contrast numerically | **No — local** |
| `classify` | Categorizes scene type | Yes |

### Quality mode metrics

| Metric | What It Measures | Bad | Good |
|--------|-----------------|-----|------|
| `blur_score` | Laplacian variance | Below 100 = blurry | Above 500 = sharp |
| `brightness` | Mean pixel value (0-255) | Below 50 = too dark, above 200 = overexposed | 50-200 |
| `contrast` | Pixel standard deviation | Below 30 = flat/washed out | Above 80 = good contrast |

## When to Use

- **Asset validation** — verify a generated image matches the intended scene description (`qa`)
- **Pre-planning footage analysis** — understand user-provided stills before planning
- **Quality gating** — programmatic blur/exposure/contrast check on stills (`quality`, free)
- **Scene classification** — sort images into categories (`classify`)

## Common Workflows

### 1. Generated Asset Validation

```
image_understand_openai (qa, "Does this show: [scene description]?") → confirm or regenerate
```

### 2. Quality Gate on Stills

```
image_understand_openai (quality) → fail if blur_score < 100, brightness outside 50-200, contrast < 30
```

## Quality Checklist

- Descriptions accurately match image content
- QA answers are consistent for similar images
- Quality scores correlate with visual inspection (spot-check manually)

## Applying to OpenMontage

1. Use `qa` mode to validate generated images before they enter composition
2. Use `quality` mode (free, local) as an automated gate on keyframes and stills
3. Combine with the reviewer skill's rubric during review
4. For full videos use `video_understand_openai` instead — this skill covers single images only

## References

- `skills/creative/image-understand-usage.md` — detailed usage guide
