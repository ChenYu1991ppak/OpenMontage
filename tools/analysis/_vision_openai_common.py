"""Shared helpers for the OpenAI-compatible vision-understanding tools.

This module is NOT a tool itself (it contains no BaseTool subclass, so the
registry's pkgutil discovery will import it but never register anything).
It hosts everything the two sibling tools share:

  - image_understand_openai  (tools/analysis/image_understand_openai.py)
  - video_understand_openai  (tools/analysis/video_understand_openai.py)

Shared pieces: vision env-var handling, OpenAI client creation, image/video
base64 data-URL encoding (with size limits), prompt construction, a single
retry wrapper around chat.completions.create(), the local quality-assessment
algorithm, ffmpeg frame extraction, and human-readable summary building.

The output field names and thresholds mirror tools/analysis/video_understand.py
so downstream parsers keep working unchanged.
"""

from __future__ import annotations

import base64
import io
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Shared constants (identical to video_understand.py)
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

SCENE_CATEGORIES = [
    "indoor", "outdoor", "landscape", "cityscape", "portrait",
    "action", "close-up", "aerial", "underwater", "night",
    "studio", "nature", "urban", "abstract", "text-overlay",
]

# ---------------------------------------------------------------------------
# Vision API configuration
# ---------------------------------------------------------------------------

ENV_API_KEY = "OPENAI_VISION_API_KEY"
ENV_BASE_URL = "OPENAI_VISION_BASE_URL"
ENV_MODEL = "OPENAI_VISION_MODEL"

DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "kimi-k3"

# Upload / request limits
MAX_IMAGE_EDGE = 1024          # longest edge in px after downscale
MAX_VIDEO_BYTES = 70 * 1024 * 1024  # 70MB source -> ~95MB base64 payload (< 100MB)
CONNECT_TIMEOUT = 30.0
READ_TIMEOUT = 120.0           # video inference can take a while
RETRY_BACKOFF_SECONDS = 1.0
RETRY_COUNT = 1                # 1 retry on top of the initial attempt


def get_vision_env() -> tuple[str, str, str]:
    """Return (api_key, base_url, model) resolved from OPENAI_VISION_* vars.

    api_key may be empty; base_url/model fall back to Moonshot/Kimi defaults.
    The project .env is loaded automatically by tools.base_tool at import time.
    """
    api_key = (os.environ.get(ENV_API_KEY) or "").strip()
    base_url = (os.environ.get(ENV_BASE_URL) or "").strip() or DEFAULT_BASE_URL
    model = (os.environ.get(ENV_MODEL) or "").strip() or DEFAULT_MODEL
    return api_key, base_url, model


def is_api_configured() -> bool:
    """True when an API key is present in the environment."""
    api_key, _, _ = get_vision_env()
    return bool(api_key)


def create_vision_client() -> Any:
    """Create an OpenAI-compatible client (imports openai lazily).

    Raises ValueError when the API key is missing, ImportError when the
    openai SDK is not installed.
    """
    api_key, base_url, _ = get_vision_env()
    if not api_key:
        raise ValueError(
            f"{ENV_API_KEY} is not set. Add it to the project .env file."
        )
    from openai import OpenAI

    try:
        from httpx import Timeout

        timeout = Timeout(connect=CONNECT_TIMEOUT, read=READ_TIMEOUT)
    except Exception:  # pragma: no cover - fallback to a plain float
        timeout = READ_TIMEOUT
    # max_retries=0: retries are handled explicitly in call_chat_once so the
    # backoff/error classification stays uniform across both tools.
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def encode_image_to_data_url(img: Any) -> str:
    """Downscale (longest edge <= 1024px) + JPEG-encode a PIL image to a data URL.

    Keeps payload size and token billing low; never touches disk.
    """
    img = img.convert("RGB")
    width, height = img.size
    longest = max(width, height)
    if longest > MAX_IMAGE_EDGE:
        scale = MAX_IMAGE_EDGE / longest
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        img = img.resize(new_size)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _video_mime(suffix: str) -> str:
    return {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".webm": "video/webm",
    }.get(suffix.lower(), "video/mp4")


def encode_video_to_data_url(video_path: Path) -> str:
    """Encode a whole video file to a base64 data URL (<= 70MB source).

    Base64 inflates by ~37%, so 70MB source stays under the ~100MB request
    body limit. Oversized videos raise ValueError with a clear message
    instead of silently degrading.
    """
    size_bytes = video_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    if size_bytes > MAX_VIDEO_BYTES:
        raise ValueError(
            f"Video file is {size_mb:.1f}MB, exceeding the {MAX_VIDEO_BYTES // (1024 * 1024)}MB "
            f"whole-video upload limit. Compress or trim it first, e.g.:\n"
            f"  ffmpeg -i input.mp4 -vf 'scale=-2:720' -crf 28 -preset fast output.mp4\n"
            f"The agent should decide how to proceed (no automatic fallback)."
        )
    mime = _video_mime(video_path.suffix)
    encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(mode: str, query: str | None, is_video: bool) -> tuple[str, str]:
    """Return (system_message, user_text) for a Chat Completions request.

    mode must be one of describe/qa/classify (quality is handled locally and
    never reaches the API).
    """
    if is_video:
        base = (
            "You are analyzing a video clip. You see the full sequence with "
            "motion, actions, and timing."
        )
    else:
        base = "You are an expert image analyst."

    if mode == "describe":
        system = (
            base + " Describe the visual content accurately and concisely, "
            "covering subjects, setting, actions, and notable details."
        )
        user = "Describe what is visible in this content."
    elif mode == "qa":
        system = (
            base + " Answer the user's question about the visual content "
            "accurately, based only on what is visible."
        )
        user = f"Question: {query}"
    elif mode == "classify":
        categories = ", ".join(SCENE_CATEGORIES)
        system = (
            base + " Classify the visual content into exactly one of the "
            "given scene categories."
        )
        user = (
            f"Classify this content into one of these categories: {categories}. "
            "Respond with ONLY the category name."
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return system, user


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------


def call_chat_once(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 1024,
) -> str:
    """Single chat.completions.create() with one automatic retry.

    Retries transient failures (rate limit, timeout, connection, 5xx). All
    other errors propagate to the caller. Returns the assistant text trimmed.
    """
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    retryable = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}

    last_exc: Exception | None = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Vision API returned an empty response.")
            return text
        except retryable as exc:
            last_exc = exc
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Local quality assessment (offline, zero API cost)
# ---------------------------------------------------------------------------


def analyze_quality_image(img: Any) -> dict[str, Any]:
    """Assess technical quality of one PIL image (fields match video_understand).

    Only numpy is strictly required; scipy is used when available and a pure
    numpy sliding-window convolution provides an identical fallback so the
    tool stays offline-capable even without scipy installed.
    """
    import numpy as np

    try:
        from scipy.signal import convolve2d
    except ImportError:  # pragma: no cover - numpy-only fallback
        convolve2d = None

    arr = np.array(img, dtype=np.float64)
    gray = np.mean(arr, axis=2)

    # Blur detection: Laplacian variance (low = blurry)
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    if convolve2d is not None:
        laplacian = convolve2d(gray, laplacian_kernel, mode="valid")
    else:
        from numpy.lib.stride_tricks import sliding_window_view

        patches = sliding_window_view(gray, (3, 3))
        laplacian = np.einsum("hwij,ij->hw", patches, laplacian_kernel)
    blur_score = float(np.var(laplacian))

    # Brightness: mean pixel value (0-255 scale)
    brightness = float(np.mean(arr))

    # Contrast: standard deviation of pixel values
    contrast = float(np.std(arr))

    quality_issues = []
    if blur_score < 100:
        quality_issues.append("blurry")
    if brightness < 40:
        quality_issues.append("underexposed")
    elif brightness > 220:
        quality_issues.append("overexposed")
    if contrast < 30:
        quality_issues.append("low_contrast")

    quality_label = "good" if not quality_issues else "issues_detected"

    return {
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "quality": quality_label,
        "issues": quality_issues,
        "resolution": f"{img.width}x{img.height}",
    }


# ---------------------------------------------------------------------------
# Video frame extraction (ffmpeg, same filter logic as video_understand)
# ---------------------------------------------------------------------------


def extract_video_frames(
    video_path: Path,
    frame_indices: list[int] | None = None,
    max_frames: int = 5,
) -> list[Any]:
    """Extract up to max_frames PIL images from a video via ffmpeg.

    frame_indices selects exact frame numbers (select filter + vsync vfr);
    otherwise evenly samples with the thumbnail filter. Frames are decoded
    into memory inside a TemporaryDirectory that cleans up after itself.
    """
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        if frame_indices:
            frames_to_extract = list(frame_indices)[:max_frames]
            select_expr = "+".join(f"eq(n\\,{idx})" for idx in frames_to_extract)
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-vf", f"select='{select_expr}'",
                "-vsync", "vfr",
                str(tmp / "frame_%04d.png"),
                "-y", "-loglevel", "error",
            ]
        else:
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-frames:v", str(max_frames),
                "-vf", f"thumbnail={max_frames}",
                str(tmp / "frame_%04d.png"),
                "-y", "-loglevel", "error",
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"ffmpeg frame extraction failed: {detail}")

        frame_files = sorted(tmp.glob("frame_*.png"))
        images = []
        for f in frame_files[:max_frames]:
            # convert("RGB") forces decode into memory; safe after tmp cleanup.
            images.append(Image.open(f).convert("RGB"))
        return images


# ---------------------------------------------------------------------------
# Summary building (wording mirrors video_understand._build_summary)
# ---------------------------------------------------------------------------


def _frame_text(r: dict[str, Any], mode: str) -> str:
    """Pull the human-readable text out of a frame/whole-video result entry."""
    key = {"describe": "description", "qa": "answer", "classify": "top_category"}.get(mode)
    if key and r.get(key):
        return str(r[key])
    return str(r.get("analysis", ""))


def build_summary(frame_results: list[dict[str, Any]], mode: str) -> str:
    """Build a human-readable summary from per-frame/whole-video results."""
    n = len(frame_results)

    if mode == "describe":
        texts = [_frame_text(r, mode) for r in frame_results]
        if n == 1:
            return texts[0]
        return f"Analyzed {n} frames. Descriptions: " + "; ".join(texts)

    if mode == "qa":
        texts = [_frame_text(r, mode) for r in frame_results]
        if n == 1:
            return texts[0]
        return f"Analyzed {n} frames. Answers: " + "; ".join(texts)

    if mode == "quality":
        issues_all = []
        for r in frame_results:
            issues_all.extend(r.get("issues", []))
        if not issues_all:
            return f"All {n} frame(s) passed quality checks."
        unique_issues = sorted(set(issues_all))
        return (
            f"Analyzed {n} frame(s). Issues found: {', '.join(unique_issues)}."
        )

    if mode == "classify":
        texts = [_frame_text(r, mode) for r in frame_results]
        if n == 1:
            return f"Scene classified as: {texts[0]}"
        return (
            f"Analyzed {n} frames. Scene categories: "
            + ", ".join(texts)
        )

    return f"Analyzed {n} frame(s)."
