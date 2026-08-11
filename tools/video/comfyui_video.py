"""ComfyUI video generation via a local or remote ComfyUI server.

Supports text-to-video and image-to-video using MiniMax H3 (FL2VA)
native ComfyUI workflows with synchronized audio.  Custom workflows
are accepted via the ``workflow_json`` input.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

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
from tools._comfyui.client import ComfyUIClient, ComfyUIError
from tools._comfyui.metadata import (
    BUNDLED_MODEL_STACKS,
    COMFYUI_SETUP_OFFER,
    missing_models_payload,
    model_stack,
    workflow_hash,
)

_WORKFLOWS = Path(__file__).resolve().parent.parent / "_comfyui" / "workflows"

# Output node IDs in the bundled workflows
_T2V_OUTPUT_NODE = "92"
_I2V_OUTPUT_NODE = "92"

# Bundled MiniMax H3 workflows (API format)
_T2V_WORKFLOW_KEY = "minimax-h3-t2v"
_I2V_WORKFLOW_KEY = "minimax-h3-i2v"
_T2V_WORKFLOW_FILE = "video_minimax_h3_t2v_api.json"
_I2V_WORKFLOW_FILE = "video_minimax_h3_i2v_api.json"

# MiniMax H3 native framerate and length grid (17k + 5 frames)
_H3_FPS = 24
_H3_LENGTH_STEP = 17
_H3_LENGTH_OFFSET = 5

# Models required by the bundled MiniMax H3 workflows (shared by T2V/I2V)
_REQUIRED_MODELS_H3 = [
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors",
]
_REQUIRED_MODELS_I2V = list(_REQUIRED_MODELS_H3)
_REQUIRED_MODELS_T2V = list(_REQUIRED_MODELS_H3)

_RESOURCE_PROFILES = {
    "provider_floor": {
        "vram_mb": 8000,
        "ram_mb": 16000,
        "applies_to": (
            "ComfyUI provider availability and low-VRAM custom workflows. "
            "Actual requirements depend on workflow_json/workflow_path."
        ),
    },
    "bundled_minimax_h3_fl2va": {
        "vram_mb": 32000,
        "ram_mb": 64000,
        "applies_to": (
            "Bundled MiniMax H3 FL2VA T2V/I2V workflows. This is not a "
            "ComfyUI provider-wide requirement."
        ),
    },
    "low_vram_custom_workflows": {
        "vram_mb": "8000-12000",
        "ram_mb": "16000-32000",
        "examples": [
            "Wan 2.1 1.3B",
            "LTX-Video / LTXV FP8 or quantized workflows",
            "Wan 2.2 GGUF / quantized community workflows",
        ],
    },
}


class ComfyUIVideo(BaseTool):
    name = "comfyui_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "comfyui"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.LOCAL_GPU

    dependencies = []
    setup_offer = COMFYUI_SETUP_OFFER
    install_instructions = (
        "Start a ComfyUI server and set COMFYUI_SERVER_URL "
        "(default http://localhost:8188).\n"
        "Requires MiniMax H3 models (FL2VA diffusion, Qwen3VL-32B CLIP, "
        "video + audio VAEs) in ComfyUI's model directories."
    )
    agent_skills = ["comfyui", "ai-video-gen"]

    capabilities = ["text_to_video", "image_to_video"]
    supports = {
        "seed": True,
        "reference_image": True,
        "custom_workflow": True,
        "custom_output_node": True,
        "offline": True,
    }
    best_for = [
        "local GPU video generation without API costs",
        "high-quality video with synchronized audio (MiniMax H3 native)",
        "image-to-video with MiniMax H3 FL2VA",
        "text-to-video with MiniMax H3 FL2VA",
        "custom low-VRAM ComfyUI workflows on 8GB-12GB GPUs",
    ]
    not_good_for = [
        "setups without a running ComfyUI server",
        "CPU-only machines",
        "running the bundled MiniMax H3 FL2VA workflows on GPUs below 24GB VRAM",
    ]
    fallback = "wan_video"
    fallback_tools = ["wan_video", "hunyuan_video", "ltx_video_local", "kling_video"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Text prompt for video generation"},
            "operation": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video"],
                "default": "text_to_video",
            },
            "reference_image_path": {
                "type": "string",
                "description": "Local path to reference image (for image_to_video)",
            },
            "reference_image_url": {
                "type": "string",
                "description": "URL of reference image (for image_to_video, downloaded first)",
            },
            "width": {"type": "integer", "default": 832, "description": "T2V default 832, I2V default 640"},
            "height": {"type": "integer", "default": 480, "description": "T2V default 480, I2V default 640"},
            "num_frames": {
                "type": "integer",
                "default": 124,
                "description": "124 frames = ~5s at 24fps (snapped up to the model's 17k+5 grid)",
            },
            "seed": {"type": "integer", "description": "Random if omitted"},
            "output_path": {"type": "string", "description": "Where to save the video"},
            "workflow_json": {
                "type": "string",
                "description": "Optional full ComfyUI workflow JSON. Requires output_node.",
            },
            "workflow_path": {
                "type": "string",
                "description": "Optional path to a ComfyUI workflow JSON file. Requires output_node.",
            },
            "output_node": {
                "type": "string",
                "description": "ComfyUI output node ID for custom workflow_json/workflow_path.",
            },
            "workflow_name": {
                "type": "string",
                "description": "Optional human-readable provenance label for a custom workflow.",
            },
            "workflow_model": {
                "type": "string",
                "description": "Optional model/provenance label for a custom workflow.",
            },
            "workflow_model_stack": {
                "type": "array",
                "description": (
                    "Optional provenance metadata for custom workflow dependencies. "
                    "Items should include name, role, quantization, scheduler, "
                    "and LoRA strengths when known."
                ),
                "items": {"type": "object"},
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=16000, vram_mb=8000, disk_mb=2000, network_required=False,
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["timeout"])
    idempotency_key_fields = ["prompt", "operation", "width", "height", "num_frames", "seed"]
    side_effects = ["writes video file to output_path"]
    user_visible_verification = ["Watch generated clip for motion coherence and artifacts"]

    def __init__(self) -> None:
        self._client = ComfyUIClient()

    def get_status(self) -> ToolStatus:
        if not self._client.is_available():
            return ToolStatus.UNAVAILABLE
        statuses = self.operation_statuses()
        if any(status == "available" for status in statuses.values()):
            return ToolStatus.AVAILABLE
        if statuses:
            return ToolStatus.DEGRADED
        return ToolStatus.UNAVAILABLE

    def operation_statuses(self) -> dict[str, str]:
        """Return per-operation readiness for selector routing and preflight."""
        if not self._client.is_available():
            return {
                "text_to_video": "unavailable",
                "image_to_video": "unavailable",
            }

        _, missing_t2v = self._client.check_models(_REQUIRED_MODELS_T2V)
        _, missing_i2v = self._client.check_models(_REQUIRED_MODELS_I2V)
        return {
            "text_to_video": "available" if not missing_t2v else "degraded",
            "image_to_video": "available" if not missing_i2v else "degraded",
        }

    def is_operation_available(self, operation: str) -> bool:
        if operation not in {"text_to_video", "image_to_video"}:
            return False
        return self.operation_statuses().get(operation) == "available"

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["operation_statuses"] = self.operation_statuses()
        info["resource_profiles"] = _RESOURCE_PROFILES
        info["setup_offer"] = self.setup_offer
        info["bundled_model_stacks"] = {
            "text_to_video": BUNDLED_MODEL_STACKS[_T2V_WORKFLOW_KEY],
            "image_to_video": BUNDLED_MODEL_STACKS[_I2V_WORKFLOW_KEY],
        }
        info["resource_profile_note"] = (
            "The top-level resource_profile is a ComfyUI provider floor, not a "
            "promise that every workflow fits 8GB VRAM. Bundled MiniMax H3 FL2VA "
            "workflows recommend 32GB VRAM; custom low-VRAM workflows can target "
            "8GB-12GB depending on model, quantization, resolution, and frame count."
        )
        return info

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        operation = inputs.get("operation", "text_to_video")
        if operation == "image_to_video":
            return 600.0  # ~10 min (20-step FL2VA + 32B CLIP)
        return 600.0  # ~10 min

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        custom_workflow = bool(inputs.get("workflow_json") or inputs.get("workflow_path"))
        if custom_workflow and not inputs.get("output_node"):
            return ToolResult(
                success=False,
                error=(
                    "Custom ComfyUI workflows require output_node so OpenMontage "
                    "knows which ComfyUI node to download artifacts from."
                ),
            )

        if not self._client.is_available():
            return ToolResult(
                success=False,
                error=self._client.unavailable_reason(),
            )

        operation = inputs.get("operation", "text_to_video")

        if not custom_workflow:
            required = _REQUIRED_MODELS_I2V if operation == "image_to_video" else _REQUIRED_MODELS_T2V
            _, missing = self._client.check_models(required)
            if missing:
                workflow_key = (
                    _I2V_WORKFLOW_KEY
                    if operation == "image_to_video"
                    else _T2V_WORKFLOW_KEY
                )
                workflow_name = (
                    _I2V_WORKFLOW_FILE
                    if operation == "image_to_video"
                    else _T2V_WORKFLOW_FILE
                )
                return ToolResult(
                    success=False,
                    data=missing_models_payload(
                        missing,
                        workflow_key=workflow_key,
                        workflow_name=workflow_name,
                        operation=operation,
                    ),
                    error=(
                        f"ComfyUI server is running but missing models for {operation}: "
                        f"{', '.join(missing)}.\n"
                        f"See data.missing_models for destination hints and download URLs."
                    ),
                )
        start = time.time()
        seed = inputs.get("seed") or ComfyUIClient.random_seed()
        output_path = Path(
            inputs.get("output_path", f"comfyui_video_{operation}_{seed}.mp4")
        )

        try:
            if custom_workflow:
                workflow = self._load_custom_workflow(inputs)
                output_node = str(inputs["output_node"])
            elif operation == "image_to_video":
                workflow, output_node = self._build_i2v(inputs, seed, output_path)
            else:
                workflow, output_node = self._build_t2v(inputs, seed, output_path)

            provenance = self._workflow_provenance(
                inputs, custom_workflow, output_node, operation, workflow
            )
            paths = self._client.generate(
                workflow,
                output_node=output_node,
                dest=output_path,
                timeout=1500,
                interval=10,
            )

        except ComfyUIError as exc:
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:
            return ToolResult(success=False, error=f"ComfyUI video generation failed: {exc}")

        width = inputs.get("width", 832 if operation == "text_to_video" else 640)
        height = inputs.get("height", 480 if operation == "text_to_video" else 640)
        requested_frames = inputs.get("num_frames", 124)
        num_frames = (
            requested_frames
            if custom_workflow
            else ComfyUIVideo._h3_length(requested_frames)
        )
        fps = _H3_FPS

        model_name = self._model_name(inputs, custom_workflow)
        return ToolResult(
            success=True,
            data={
                "provider": "comfyui",
                "model": model_name,
                "prompt": inputs["prompt"],
                "operation": operation,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "duration_seconds": round(num_frames / fps, 2),
                "output": str(paths[0]),
                "format": "mp4",
                "workflow_provenance": provenance,
            },
            artifacts=[str(p) for p in paths],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            seed=seed,
            model=model_name,
        )

    # ------------------------------------------------------------------
    # Workflow builders
    # ------------------------------------------------------------------

    def _build_t2v(
        self, inputs: dict[str, Any], seed: int, output_path: Path
    ) -> tuple[dict, str]:
        width = ComfyUIVideo._snap_multiple(inputs.get("width", 832), 32, 32)
        height = ComfyUIVideo._snap_multiple(inputs.get("height", 480), 32, 32)
        length = ComfyUIVideo._h3_length(inputs.get("num_frames", 124))

        workflow = ComfyUIClient.load_workflow(_WORKFLOWS / _T2V_WORKFLOW_FILE)
        workflow = ComfyUIClient.patch_workflow(workflow, {
            "105:104": {
                "prompt": inputs["prompt"],
                "width": width,
                "height": height,
                "length": length,
            },
            "105:15": {"noise_seed": seed},
            "92": {"filename_prefix": output_path.stem},
        })
        # Length is patched directly, so the duration-math helper nodes are no
        # longer wired; drop them to avoid depending on extra custom nodes.
        workflow.pop("105:107", None)
        workflow.pop("105:111", None)
        return workflow, _T2V_OUTPUT_NODE

    def _build_i2v(
        self, inputs: dict[str, Any], seed: int, output_path: Path
    ) -> tuple[dict, str]:
        width = ComfyUIVideo._snap_multiple(inputs.get("width", 640), 32, 32)
        height = ComfyUIVideo._snap_multiple(inputs.get("height", 640), 32, 32)
        length = ComfyUIVideo._h3_length(inputs.get("num_frames", 124))

        # Resolve reference image
        ref_path = inputs.get("reference_image_path")
        ref_url = inputs.get("reference_image_url")

        if ref_url and not ref_path:
            # Download to a temp location
            resp = requests.get(ref_url, timeout=60)
            resp.raise_for_status()
            ref_path = str(output_path.with_suffix(".ref.png"))
            Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
            Path(ref_path).write_bytes(resp.content)

        if not ref_path:
            raise ComfyUIError(
                "image_to_video requires reference_image_path or reference_image_url"
            )

        # Upload to ComfyUI
        upload_name = f"om_{output_path.stem}.png"
        server_name = self._client.upload_image(Path(ref_path), upload_name)

        workflow = ComfyUIClient.load_workflow(_WORKFLOWS / _I2V_WORKFLOW_FILE)
        workflow = ComfyUIClient.patch_workflow(workflow, {
            "114": {"image": server_name},
            "105:104": {
                "prompt": inputs["prompt"],
                "width": width,
                "height": height,
                "length": length,
            },
            "105:15": {"noise_seed": seed},
            "92": {"filename_prefix": output_path.stem},
        })
        # Length is patched directly, so the duration-math helper nodes are no
        # longer wired; drop them to avoid depending on extra custom nodes.
        workflow.pop("105:107", None)
        workflow.pop("105:111", None)
        return workflow, _I2V_OUTPUT_NODE

    @staticmethod
    def _snap_multiple(value: int, step: int, minimum: int = 1) -> int:
        """Snap a dimension down to the nearest multiple of ``step``."""
        return max(minimum, (int(value) // step) * step)

    @staticmethod
    def _h3_length(num_frames: int) -> int:
        """MiniMax H3 frame length snapped to the model's 17k+5 grid."""
        base = max(_H3_LENGTH_OFFSET, int(num_frames))
        return base + (_H3_LENGTH_OFFSET - base % _H3_LENGTH_STEP) % _H3_LENGTH_STEP

    @staticmethod
    def _load_custom_workflow(inputs: dict[str, Any]) -> dict:
        if inputs.get("workflow_json"):
            return json.loads(inputs["workflow_json"])
        return ComfyUIClient.load_workflow(Path(inputs["workflow_path"]))

    @staticmethod
    def _model_name(inputs: dict[str, Any], custom_workflow: bool) -> str:
        if not custom_workflow:
            return "minimax-h3-fl2va-int8"
        return (
            inputs.get("workflow_model")
            or inputs.get("model")
            or inputs.get("workflow_name")
            or "custom-comfyui-workflow"
        )

    @staticmethod
    def _workflow_provenance(
        inputs: dict[str, Any],
        custom_workflow: bool,
        output_node: str,
        operation: str,
        workflow: dict[str, Any],
    ) -> dict[str, Any]:
        if not custom_workflow:
            workflow_key = (
                _I2V_WORKFLOW_KEY
                if operation == "image_to_video"
                else _T2V_WORKFLOW_KEY
            )
            return {
                "source": "bundled",
                "workflow": (
                    _I2V_WORKFLOW_FILE
                    if operation == "image_to_video"
                    else _T2V_WORKFLOW_FILE
                ),
                "workflow_hash_sha256": workflow_hash(workflow),
                "model_stack": model_stack(workflow_key, inputs),
                "output_node": output_node,
            }
        return {
            "source": "user_supplied",
            "workflow_name": inputs.get("workflow_name"),
            "workflow_path": inputs.get("workflow_path"),
            "model": inputs.get("workflow_model") or inputs.get("model"),
            "workflow_hash_sha256": workflow_hash(workflow),
            "model_stack": model_stack(None, inputs),
            "model_stack_source": (
                "caller_supplied"
                if inputs.get("workflow_model_stack")
                else "unknown_custom_workflow"
            ),
            "output_node": output_node,
        }
