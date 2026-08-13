"""Video understanding tool backed by an OpenAI-compatible vision API.

This is the VIDEO-only counterpart of image_understand_openai. For
describe / qa / classify it uploads the WHOLE video as a base64 video_url in
a single request, so the model sees motion, actions and timing in one pass.
No per-frame fallback: if the upload or request fails, the tool reports an
error directly and the agent decides what to do next. Quality mode extracts
frames with ffmpeg and assesses each one locally (offline, zero API cost).

The IDE agent should pick this tool when the resource is a video, and
image_understand_openai when it is an image. Input validation rejects the
wrong type with a pointer to the correct tool.

Configuration (project .env, loaded automatically by tools.base_tool):
  OPENAI_VISION_API_KEY   required
  OPENAI_VISION_BASE_URL  optional, default https://api.moonshot.cn/v1
  OPENAI_VISION_MODEL     optional, default kimi-k3

Output keeps the same top-level keys as video_understand (frames / summary /
mode / model / frame_count). Whole-video results use a single entry with
scope="whole_video" and frame_index=None; input_type="video" plus
strategy ("whole_video_upload" for API modes, "frame_by_frame" for quality)
let agents parse results unambiguously.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tools.analysis import _vision_openai_common as common
from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class VideoUnderstandOpenAI(BaseTool):
    name = "video_understand_openai"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "analysis"
    provider = "moonshot"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    # API key presence is checked manually in get_status(); the openai SDK
    # import is also verified there so UNAVAILABLE surfaces the right reason.
    dependencies: list[str] = []
    install_instructions = (
        "Set OPENAI_VISION_API_KEY in the project .env file "
        "(optionally OPENAI_VISION_BASE_URL / OPENAI_VISION_MODEL). "
        "Install the SDK with: pip install openai. "
        "ffmpeg is required for mode=quality (frame extraction)."
    )

    agent_skills = ["video-understand"]

    capabilities = [
        "video_description",
        "visual_qa",
        "quality_assessment",
        "scene_classification",
        "temporal_analysis",
    ]

    best_for = [
        "describing a whole video (actions, motion, timing in one pass)",
        "answering questions about a video (qa)",
        "classifying a video into a scene category",
        "assessing video quality frame by frame (offline)",
        "videos up to 70MB / ~1080p",
    ]
    not_good_for = [
        "image files — use image_understand_openai instead",
        "videos larger than 70MB (upload fails on purpose, no fallback)",
        "extremely long videos where a single request exceeds the API limit",
    ]

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {
                "type": "string",
                "description": (
                    "Path to a VIDEO file (.mp4/.mov/.avi/.webm). "
                    "Image files are rejected — use image_understand_openai for images."
                ),
            },
            "query": {
                "type": "string",
                "description": "Question to answer about the video (required for mode=qa)",
            },
            "mode": {
                "type": "string",
                "enum": ["describe", "qa", "quality", "classify"],
                "default": "describe",
                "description": (
                    "describe: whole-video caption; qa: answer query about the whole video; "
                    "quality: ffmpeg frame extraction + local assessment (offline); "
                    "classify: pick one of 15 scene categories"
                ),
            },
            "model": {
                "type": "string",
                "default": "kimi-k3",
                "description": (
                    "OpenAI-compatible multimodal model id; "
                    "defaults to OPENAI_VISION_MODEL or kimi-k3"
                ),
            },
            "frame_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Specific frame numbers to extract for mode=quality. Ignored for API modes.",
            },
            "max_frames": {
                "type": "integer",
                "default": 5,
                "description": "Number of frames to extract for mode=quality (default 5).",
            },
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "description": (
                    "One entry with scope=whole_video for API modes, or one "
                    "entry per frame for mode=quality"
                ),
            },
            "summary": {"type": "string"},
            "mode": {"type": "string"},
            "model": {"type": "string"},
            "frame_count": {"type": "integer"},
            "input_type": {
                "type": "string",
                "description": "Always \"video\" for this tool",
            },
            "strategy": {
                "type": "string",
                "description": "whole_video_upload (API modes) or frame_by_frame (quality)",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1,
        ram_mb=512,
        disk_mb=200,
        network_required=True,
    )
    retry_policy = RetryPolicy(
        max_retries=1,
        backoff_seconds=1.0,
        retryable_errors=["RateLimitError", "APITimeoutError", "APIConnectionError"],
    )

    idempotency_key_fields = ["input_path", "mode", "query", "model"]
    side_effects: list[str] = []
    fallback = "visual_qa"
    fallback_tools = ["visual_qa", "video_understand"]
    user_visible_verification = [
        "Confirm the whole-video analysis captures actions and timing",
        "Confirm quality metrics match perceived video quality",
    ]

    # ------------------------------------------------------------------
    # Status / cost
    # ------------------------------------------------------------------

    def get_status(self) -> ToolStatus:
        """AVAILABLE only when openai is importable AND a vision key is set."""
        try:
            import openai  # noqa: F401

            if common.is_api_configured():
                return ToolStatus.AVAILABLE
            return ToolStatus.UNAVAILABLE
        except ImportError:
            return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        """Rough API cost: quality is free (local); whole-video ~$0.01."""
        if inputs.get("mode", "describe") == "quality":
            return 0.0
        return 0.01

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        """Quality extracts a few frames (<3s); whole-video upload+infer ~15s."""
        if inputs.get("mode", "describe") == "quality":
            return 3.0
        return 15.0

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        input_path = Path(inputs["input_path"])
        mode = inputs.get("mode", "describe")
        query = inputs.get("query")
        model = inputs.get("model")

        if mode not in ("describe", "qa", "quality", "classify"):
            return ToolResult(success=False, error=f"Unknown mode: {mode}")

        if not input_path.exists():
            return ToolResult(success=False, error=f"Input file not found: {input_path}")

        suffix = input_path.suffix.lower()
        if suffix not in common.VIDEO_EXTENSIONS:
            if suffix in common.IMAGE_EXTENSIONS:
                return ToolResult(
                    success=False,
                    error=(
                        f"Input is an image file ({suffix}). This tool only accepts VIDEOS. "
                        "Use image_understand_openai for image understanding."
                    ),
                )
            return ToolResult(
                success=False,
                error=(
                    f"Unsupported file type: {suffix}. "
                    f"Supported video types: {sorted(common.VIDEO_EXTENSIONS)}. "
                    "For images use image_understand_openai."
                ),
            )

        if mode == "qa" and not query:
            return ToolResult(
                success=False,
                error="Query is required for 'qa' mode.",
            )

        start = time.time()
        try:
            if mode == "quality":
                return self._run_quality(input_path, inputs, start)
            return self._run_whole_video(input_path, mode, query, model, start)
        except ImportError as exc:  # pragma: no cover - openai/PIL missing
            return ToolResult(
                success=False,
                error=f"Missing dependency: {exc}. {self.install_instructions}",
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Video understanding failed: {exc}")

    # ------------------------------------------------------------------
    # Local quality mode (ffmpeg frames + offline assessment)
    # ------------------------------------------------------------------

    def _run_quality(self, input_path: Path, inputs: dict[str, Any], start: float) -> ToolResult:
        frame_indices = inputs.get("frame_indices") or None
        max_frames = int(inputs.get("max_frames") or 5)

        try:
            images = common.extract_video_frames(input_path, frame_indices, max_frames)
        except RuntimeError as exc:
            return ToolResult(
                success=False,
                error=f"Could not extract frames: {exc}",
            )

        if not images:
            return ToolResult(
                success=False,
                error="ffmpeg produced no frames; cannot assess quality.",
            )

        frames = []
        for idx, img in enumerate(images):
            entry = common.analyze_quality_image(img)
            entry["frame_index"] = idx
            frames.append(entry)

        summary = common.build_summary(frames, "quality")
        return ToolResult(
            success=True,
            data={
                "frames": frames,
                "summary": summary,
                "mode": "quality",
                "model": "metrics",
                "frame_count": len(frames),
                "input_type": "video",
                "strategy": "frame_by_frame",
            },
            duration_seconds=round(time.time() - start, 2),
            model=None,
        )

    # ------------------------------------------------------------------
    # Whole-video upload: describe / qa / classify
    # ------------------------------------------------------------------

    def _run_whole_video(
        self,
        input_path: Path,
        mode: str,
        query: str | None,
        model: str | None,
        start: float,
    ) -> ToolResult:
        # Local, deterministic failure first: oversized videos fail regardless
        # of API configuration, so surface that before requiring a key.
        try:
            data_url = common.encode_video_to_data_url(input_path)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        if not common.is_api_configured():
            return ToolResult(
                success=False,
                error=(
                    f"{common.ENV_API_KEY} is not set. Add it to the project .env file "
                    "to enable API-based understanding, or use mode=quality which "
                    "runs locally without any key."
                ),
            )

        client = common.create_vision_client()
        model = model or common.get_vision_env()[2]

        system, user_text = common.build_prompt(mode, query, is_video=True)
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "video_url", "video_url": {"url": data_url}},
                ],
            },
        ]
        text = common.call_chat_once(client, model, messages)

        # Whole-video result: a single entry with no frame index.
        entry = {
            "frame_index": None,
            "scope": "whole_video",
            "analysis": text,
        }
        if mode == "describe":
            entry["description"] = text
        elif mode == "qa":
            entry["query"] = query
            entry["answer"] = text
        else:  # classify
            entry["top_category"] = text

        frames = [entry]
        summary = common.build_summary(frames, mode)
        return ToolResult(
            success=True,
            data={
                "frames": frames,
                "summary": summary,
                "mode": mode,
                "model": model,
                "frame_count": 1,
                "input_type": "video",
                "strategy": "whole_video_upload",
            },
            duration_seconds=round(time.time() - start, 2),
            model=model,
        )
