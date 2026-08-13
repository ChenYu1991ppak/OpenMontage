"""ComfyUI image generation via a local or remote ComfyUI server.

Bundled workflows:
- text_to_image: FLUX 2 Dev (FP8 mixed) with Mistral text encoder.
- image_edit: Qwen-Image-Edit-2511 instruction-based image editing
  (upload a reference image + Chinese edit instruction, get an edited image).

Supports custom workflows via the ``workflow_json`` input.
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

# Models required by the bundled flux2-txt2img workflow
_REQUIRED_MODELS = [
    "flux2_dev_fp8mixed.safetensors",
    "mistral_3_small_flux2_fp4_mixed.safetensors",
    "flux2-vae.safetensors",
]

# Bundled Qwen-Image-Edit-2511 instruction-based image editing workflow
_QWEN_EDIT_WORKFLOW_KEY = "qwen-image-edit-2511"
_QWEN_EDIT_WORKFLOW_FILE = "image_qwen_image_edit_2511_api.json"
_QWEN_EDIT_OUTPUT_NODE = "9"
_QWEN_EDIT_MODELS = [
    "qwen_image_edit_2511_fp8mixed.safetensors",
    "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "qwen_image_vae.safetensors",
    "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors",
]
# Node IDs inside the Qwen edit workflow (verified from the bundled JSON)
_QWEN_EDIT_IMAGE_NODE = "41"  # LoadImage
_QWEN_EDIT_POSITIVE_NODE = "170:151"  # TextEncodeQwenImageEditPlus
_QWEN_EDIT_NEGATIVE_NODE = "170:149"  # TextEncodeQwenImageEditPlus
_QWEN_EDIT_SEED_NODE = "170:169"  # KSampler


class ComfyUIImage(BaseTool):
    name = "comfyui_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "comfyui"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.LOCAL_GPU

    dependencies = []  # checked at runtime via server health
    setup_offer = COMFYUI_SETUP_OFFER
    install_instructions = (
        "Start a ComfyUI server and set COMFYUI_SERVER_URL "
        "(default http://localhost:8188).\n"
        "See https://github.com/comfyanonymous/ComfyUI for setup."
    )
    agent_skills = ["comfyui", "flux-best-practices"]

    capabilities = ["text_to_image", "image_edit"]
    supports = {
        "seed": True,
        "custom_size": True,
        "custom_workflow": True,
        "custom_output_node": True,
        "offline": True,
        "image_edit": True,
    }
    best_for = [
        "local GPU generation without API costs",
        "Blackwell / DGX Spark hardware where diffusers is unsupported",
        "full control over sampling via custom ComfyUI workflows",
        "instruction-based image editing (clothing swap, background replacement, product/portrait retouch)",
    ]
    not_good_for = [
        "setups without a running ComfyUI server",
        "CPU-only machines",
    ]
    fallback = "flux_image"
    fallback_tools = ["flux_image", "local_diffusion", "openai_image"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Text prompt for image generation, or the edit instruction "
                    "(in Chinese for best results) when image_path/image_url is provided "
                    "for instruction-based image editing, e.g. \"给人物换一套西装\"."
                ),
            },
            "image_path": {
                "type": "string",
                "description": (
                    "Local path to the source image for image_edit. "
                    "When provided together with prompt, routes to the bundled "
                    "Qwen-Image-Edit-2511 instruction-based editing workflow."
                ),
            },
            "image_url": {
                "type": "string",
                "description": (
                    "URL of the source image for image_edit (downloaded first). "
                    "Mutually exclusive with image_path."
                ),
            },
            "width": {"type": "integer", "default": 1024},
            "height": {"type": "integer", "default": 1024},
            "steps": {"type": "integer", "default": 20},
            "guidance": {"type": "number", "default": 3.5},
            "seed": {"type": "integer", "description": "Random if omitted"},
            "output_path": {"type": "string", "description": "Where to save the image"},
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
                    "Items should include name, role, quantization, and LoRA strengths when known."
                ),
                "items": {"type": "object"},
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=8000, vram_mb=8000, disk_mb=500, network_required=False,
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["timeout"])
    idempotency_key_fields = ["prompt", "width", "height", "steps", "seed", "image_path", "image_url"]
    side_effects = ["writes image file to output_path"]
    user_visible_verification = ["Inspect generated image for quality and prompt adherence"]

    def __init__(self) -> None:
        self._client = ComfyUIClient()

    def get_status(self) -> ToolStatus:
        if not self._client.is_available():
            return ToolStatus.UNAVAILABLE
        # Available if either bundled operation's model stack is complete
        # (flux2-txt2img for text_to_image, qwen-image-edit-2511 for image_edit).
        _, missing_t2i = self._client.check_models(_REQUIRED_MODELS)
        _, missing_edit = self._client.check_models(_QWEN_EDIT_MODELS)
        if not missing_t2i or not missing_edit:
            return ToolStatus.AVAILABLE
        return ToolStatus.DEGRADED

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        if inputs.get("image_path") or inputs.get("image_url"):
            return 120.0  # Qwen-Image-Edit-2511 (8/40 steps with Lightning LoRA)
        return float(inputs.get("steps", 20)) * 1.5

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["setup_offer"] = self.setup_offer
        info["bundled_model_stack"] = BUNDLED_MODEL_STACKS["flux2-txt2img"]
        info["bundled_model_stacks"] = {
            "text_to_image": BUNDLED_MODEL_STACKS["flux2-txt2img"],
            "image_edit": BUNDLED_MODEL_STACKS[_QWEN_EDIT_WORKFLOW_KEY],
        }
        return info

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        custom_workflow = bool(inputs.get("workflow_json") or inputs.get("workflow_path"))
        # Instruction-based image editing: a source image + prompt routes to the
        # bundled Qwen-Image-Edit-2511 workflow (only when no custom workflow).
        wants_edit = not custom_workflow and bool(
            inputs.get("image_path") or inputs.get("image_url")
        )
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

        if not custom_workflow:
            _, missing = self._client.check_models(
                _QWEN_EDIT_MODELS if wants_edit else _REQUIRED_MODELS
            )
            if missing:
                return ToolResult(
                    success=False,
                    data=missing_models_payload(
                        missing,
                        workflow_key=(
                            _QWEN_EDIT_WORKFLOW_KEY if wants_edit else "flux2-txt2img"
                        ),
                        workflow_name=(
                            _QWEN_EDIT_WORKFLOW_FILE if wants_edit else "flux2-txt2img.json"
                        ),
                        operation="image_edit" if wants_edit else "text_to_image",
                    ),
                    error=(
                        f"ComfyUI server is running but missing required models: "
                        f"{', '.join(missing)}.\n"
                        f"See data.missing_models for destination hints and download URLs."
                    ),
                )

        start = time.time()
        seed = inputs.get("seed") or ComfyUIClient.random_seed()
        width = inputs.get("width", 1024)
        height = inputs.get("height", 1024)
        steps = inputs.get("steps", 20)
        guidance = inputs.get("guidance", 3.5)
        output_path = Path(inputs.get("output_path", f"comfyui_image_{seed}.png"))

        tmp_source: Path | None = None
        try:
            if custom_workflow:
                workflow = self._load_custom_workflow(inputs)
                output_node = str(inputs["output_node"])
            elif wants_edit:
                workflow = ComfyUIClient.load_workflow(_WORKFLOWS / _QWEN_EDIT_WORKFLOW_FILE)
                source = Path(inputs["image_path"]) if inputs.get("image_path") else None
                if source is None:
                    resp = requests.get(inputs["image_url"], timeout=60)
                    resp.raise_for_status()
                    tmp_source = output_path.parent / f"{output_path.stem}_source.png"
                    tmp_source.write_bytes(resp.content)
                    source = tmp_source
                server_name = self._client.upload_image(source, source.name)
                workflow = ComfyUIClient.patch_workflow(workflow, {
                    _QWEN_EDIT_IMAGE_NODE: {"image": server_name},
                    _QWEN_EDIT_POSITIVE_NODE: {"prompt": inputs["prompt"]},
                    _QWEN_EDIT_NEGATIVE_NODE: {"prompt": ""},
                    _QWEN_EDIT_SEED_NODE: {"seed": seed},
                    "9": {"filename_prefix": output_path.stem},
                })
                output_node = _QWEN_EDIT_OUTPUT_NODE
            else:
                workflow = ComfyUIClient.load_workflow(_WORKFLOWS / "flux2-txt2img.json")
                workflow = ComfyUIClient.patch_workflow(workflow, {
                    "4": {"text": inputs["prompt"]},
                    "5": {"guidance": guidance},
                    "6": {"width": width, "height": height, "batch_size": 1},
                    "7": {"noise_seed": seed},
                    "10": {"steps": steps, "width": width, "height": height},
                    "13": {"filename_prefix": output_path.stem},
                })
                output_node = "13"

            provenance = self._workflow_provenance(
                inputs, custom_workflow, output_node, workflow
            )
            paths = self._client.generate(
                workflow, output_node=output_node, dest=output_path, timeout=600,
            )

        except ComfyUIError as exc:
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:
            return ToolResult(success=False, error=f"ComfyUI image generation failed: {exc}")
        finally:
            if tmp_source is not None:
                tmp_source.unlink(missing_ok=True)

        model_name = self._model_name(inputs, custom_workflow)
        cleanup = self._client.last_cleanup
        data = {
            "provider": "comfyui",
            "model": model_name,
            "prompt": inputs["prompt"],
            "output": str(paths[0]),
            "format": "png",
            "workflow_provenance": provenance,
            "server_cleanup": {
                "attempted": cleanup["attempted"],
                "history_deleted": cleanup["attempted"] and cleanup["ok"],
                "warnings": self._client.last_cleanup_warnings,
            },
        }
        if wants_edit:
            data["mode"] = "image_edit"
            data["source_image"] = inputs.get("image_path") or inputs.get("image_url")
        else:
            data.update({"width": width, "height": height, "steps": steps, "guidance": guidance})
        return ToolResult(
            success=True,
            data=data,
            artifacts=[str(p) for p in paths],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            seed=seed,
            model=model_name,
        )

    @staticmethod
    def _load_custom_workflow(inputs: dict[str, Any]) -> dict:
        if inputs.get("workflow_json"):
            return json.loads(inputs["workflow_json"])
        return ComfyUIClient.load_workflow(Path(inputs["workflow_path"]))

    @staticmethod
    def _model_name(inputs: dict[str, Any], custom_workflow: bool) -> str:
        if not custom_workflow:
            if inputs.get("image_path") or inputs.get("image_url"):
                return "qwen-image-edit-2511-fp8mixed"
            return "flux2-dev-fp8mixed"
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
        workflow: dict[str, Any],
    ) -> dict[str, Any]:
        if not custom_workflow:
            if inputs.get("image_path") or inputs.get("image_url"):
                return {
                    "source": "bundled",
                    "workflow": _QWEN_EDIT_WORKFLOW_FILE,
                    "workflow_hash_sha256": workflow_hash(workflow),
                    "model_stack": model_stack(_QWEN_EDIT_WORKFLOW_KEY, inputs),
                    "output_node": output_node,
                }
            return {
                "source": "bundled",
                "workflow": "flux2-txt2img.json",
                "workflow_hash_sha256": workflow_hash(workflow),
                "model_stack": model_stack("flux2-txt2img", inputs),
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
