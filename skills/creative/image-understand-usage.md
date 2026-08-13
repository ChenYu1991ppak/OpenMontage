# Image Understanding Usage for OpenMontage

> Sources: OpenMontage image_understand_openai tool implementation, Moonshot vision API,
> OpenCV image quality metrics

## Quick Reference Card

```
DEFAULT MODE:     describe — generates captions for the image
FOR REVIEW:       quality — assesses blur, brightness, contrast (FREE, local)
FOR Q&A:          qa mode with a query — "Is the speaker visible?" "Is the text readable?"
FOR SORTING:      classify — categorizes scene type
IMAGES ONLY:      this tool handles single images; use video_understand_openai for videos
```

## When to Use image_understand_openai

- **Generated asset validation** — verify an image matches the intended scene description
- **Footage/still analysis** — understand what's in user-provided images before planning
- **Quality gating** — programmatic check for blur, exposure, contrast on stills and keyframes
- **Scene classification** — categorize images by content type
- **Draft review** — check storyboard frames match scene plans before committing to render

## Mode Selection

| Mode | What It Does | When to Use |
|------|-------------|-------------|
| `describe` | Generates a text description of the image | Understanding content, logging |
| `qa` | Answers a specific question about the image | Targeted checks ("Is text readable?", "Is face visible?") |
| `quality` | Measures blur, brightness, contrast numerically | Automated quality gating, comparing drafts |
| `classify` | Categorizes the scene type | Sorting images, pipeline routing |

### Quality Mode Metrics

| Metric | What It Measures | Bad | Good |
|--------|-----------------|-----|------|
| `blur_score` | Laplacian variance | Below 100 = blurry | Above 500 = sharp |
| `brightness` | Mean pixel value (0-255) | Below 50 = too dark, above 200 = overexposed | 50-200 |
| `contrast` | Pixel standard deviation | Below 30 = flat/washed out | Above 80 = good contrast |

> `quality` mode runs fully offline (no API key, no network). All other modes require
> `OPENAI_VISION_API_KEY` to be set.

## Common Workflows

### 1. Asset Validation

```
image_understand_openai (qa, "Does this match: [scene description]?") → confirm or regenerate
```

After generating an image, verify it matches the intended scene description before proceeding.

### 2. Post-Render Quality Gate (stills & keyframes)

```
image_understand_openai (quality) → pass/fail → re-render if needed
```

Fail if any image has blur_score < 100, brightness outside 50-200, or contrast < 30.

### 3. Draft Storyboard Review

```
image_understand_openai (describe/qa on storyboard frames) → adjust scene plan before render
```

### 4. Talking-Head Face Check

```
image_understand_openai (qa, "Is the speaker's face clearly visible?") → face_enhance if needed
```

Check face visibility and framing before applying lip-sync or face restoration tools.

## Quality Checklist

- Descriptions accurately match what's in the image
- Quality scores correlate with visual inspection (manually spot-check)
- QA answers are consistent across similar images
- Classification categories are stable across similar images
- No false positives in quality gating (good images passing, bad images failing)

## Applying to OpenMontage

When using the `image_understand_openai` tool:

1. **Use `quality` mode as a post-render gate** on stills and keyframes — reject outputs below thresholds
2. **Use `qa` mode to validate generated assets:** "Does this image show [expected content]?"
3. **Use `describe` mode to analyze user-provided stills** at the start of a pipeline
4. **Quality thresholds for passing:** blur_score > 100, brightness 50-200, contrast > 30
5. **In the review stage**, combine quality data with the reviewer skill's rubric
6. **For videos, use `video_understand_openai`** (see `skills/creative/video-understand-usage.md`)
7. **Prefer `quality` mode when API key is unavailable** — it is free and fully local
