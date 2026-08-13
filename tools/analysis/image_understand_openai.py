"""Image understanding tool backed by an OpenAI-compatible vision API.

This is the IMAGE-only counterpart of video_understand_openai. It accepts a
single image file and runs describe / qa / classify through the multimodal
API (default Kimi K3), or quality locally and offline.

The IDE agent should pick this tool when the resource is an image, and
video_understand_openai when it is a video. Input validation rejects the
wrong type with a pointer to the correct tool.

Configuration (project .env, loaded automatically by tools.base_tool):
  OPENAI_VISION_API_KEY   required
  OPENAI_VISION_BASE_URL  optional, default https://api.moonshot.cn/v1
  OPENAI_VISION_MODEL     optional, default kimi-k3

Output keeps the same top-level keys as video_understand (frames / summary /
mode / model / frame_count) plus input_type="image" and
strategy="frame_by_frame" so agents can parse results unambiguously.
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


class ImageUnderstandOpenAI(BaseTool):
    name = "image_understand_openai"
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
        "Install the SDK with: pip install openai"
    )

    agent_skills = ["image-understand"]

    capabilities = [
        "image_description",
        "visual_qa",
        "quality_assessment",
        "scene_classification",
    ]

    best_for = [
        "describing a single image",
        "answering questions about an image (qa)",
        "classifying an image into a scene category",
        "assessing technical quality of one image (offline)",
    ]
    not_good_for = [
        "video files — use video_understand_openai instead",
        "multi-frame temporal analysis",
        "understanding motion or timing across frames",
    ]

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {
                "type": "string",
                "description": (
                    "Path to an IMAGE file (.png/.jpg/.jpeg/.bmp/.tiff/.webp). "
                    "Video files are rejected — use video_understand_openai for videos."
                ),
            },
            "query": {
                "type": "string",
                "description": "Question to answer about the image (required for mode=qa)",
            },
            "mode": {
                "type": "string",
                "enum": ["describe", "qa", "quality", "classify"],
                "default": "describe",
                "description": (
                    "describe: generate caption; qa: answer query; "
                    "quality: local technical assessment (offline); "
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
                "description": "Ignored for images (single frame). Kept for video_understand compatibility.",
            },
            "max_frames": {
                "type": "integer",
                "default": 1,
                "description": "Ignored for images (single frame). Kept for video_understand compatibility.",
            },
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "description": "Per-frame analysis results (one entry for an image)",
            },
            "summary": {"type": "string"},
            "mode": {"type": "string"},
            "model": {"type": "string"},
            "frame_count": {"type": "integer"},
            "input_type": {
                "type": "string",
                "description": "Always \"image\" for this tool",
            },
            "strategy": {
                "type": "string",
                "description": "Always \"frame_by_frame\" (single image = single frame)",
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
        "Compare the description against the actual image content",
        "Confirm quality metrics match perceived image quality",
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
        """Rough API cost: quality is free (local); API modes ~$0.002/image."""
        if inputs.get("mode", "describe") == "quality":
            return 0.0
        return 0.002

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        """Quality runs offline in well under a second; API modes ~5s."""
        if inputs.get("mode", "describe") == "quality":
            return 0.5
        return 5.0

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
        if suffix not in common.IMAGE_EXTENSIONS:
            if suffix in common.VIDEO_EXTENSIONS:
                return ToolResult(
                    success=False,
                    error=(
                        f"Input is a video file ({suffix}). This tool only accepts IMAGES. "
                        "Use video_understand_openai for video understanding."
                    ),
                )
            return ToolResult(
                success=False,
                error=(
                    f"Unsupported file type: {suffix}. "
                    f"Supported image types: {sorted(common.IMAGE_EXTENSIONS)}. "
                    "For videos use video_understand_openai."
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
                return self._run_quality(input_path, start)
            return self._run_api(input_path, mode, query, model, start)
        except ImportError as exc:  # pragma: no cover - openai/PIL missing
            return ToolResult(
                success=False,
                error=f"Missing dependency: {exc}. {self.install_instructions}",
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Image understanding failed: {exc}")

    # ------------------------------------------------------------------
    # Local quality mode (offline, zero API cost)
    # ------------------------------------------------------------------

    def _run_quality(self, input_path: Path, start: float) -> ToolResult:
        from PIL import Image

        img = Image.open(input_path).convert("RGB")
        entry = common.analyze_quality_image(img)
        entry["frame_index"] = 0
        frames = [entry]
        summary = common.build_summary(frames, "quality")
        return ToolResult(
            success=True,
            data={
                "frames": frames,
                "summary": summary,
                "mode": "quality",
                "model": "metrics",
                "frame_count": 1,
                "input_type": "image",
                "strategy": "frame_by_frame",
            },
            duration_seconds=round(time.time() - start, 2),
            model=None,
        )

    # ------------------------------------------------------------------
    # API modes: describe / qa / classify
    # ------------------------------------------------------------------

    def _run_api(
        self,
        input_path: Path,
        mode: str,
        query: str | None,
        model: str | None,
        start: float,
    ) -> ToolResult:
        if not common.is_api_configured():
            return ToolResult(
                success=False,
                error=(
                    f"{common.ENV_API_KEY} is not set. Add it to the project .env file "
                    "to enable API-based understanding, or use mode=quality which "
                    "runs locally without any key."
                ),
            )

        from PIL import Image

        client = common.create_vision_client()
        model = model or common.get_vision_env()[2]

        img = Image.open(input_path).convert("RGB")
        data_url = common.encode_image_to_data_url(img)
        system, user_text = common.build_prompt(mode, query, is_video=False)

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        text = common.call_chat_once(client, model, messages)

        if mode == "describe":
            entry = {"frame_index": 0, "description": text}
        elif mode == "qa":
            entry = {"frame_index": 0, "query": query, "answer": text}
        else:  # classify
            entry = {"frame_index": 0, "top_category": text}

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
                "input_type": "image",
                "strategy": "frame_by_frame",
            },
            duration_seconds=round(time.time() - start, 2),
            model=model,
        )
