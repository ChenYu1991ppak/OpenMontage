"""Shared metadata helpers for ComfyUI provider tools."""

from __future__ import annotations

import hashlib
import json
from typing import Any


COMFYUI_SETUP_OFFER: dict[str, Any] = {
    "kind": "local_server",
    "fix_complexity": "1-minute env-var if ComfyUI is already running; otherwise local install",
    "env_var": "COMFYUI_SERVER_URL",
    "default_url": "http://localhost:8188",
    "health_check": "GET /system_stats",
    "what_it_unlocks": [
        "free local image generation through ComfyUI workflows",
        "free local video generation through ComfyUI workflows",
        "community workflow_json/workflow_path execution",
    ],
}


_MINIMAX_H3_STACK: list[dict[str, Any]] = [
    {
        "role": "diffusion_model",
        "name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "quantization": "INT8 pruned",
        "destination_hint": "ComfyUI/models/diffusion_models/",
        "download_url": "https://huggingface.co/Comfy-Org/MiniMax_H3",
    },
    {
        "role": "text_encoder",
        "name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "quantization": "NVFP4 AWQ",
        "destination_hint": "ComfyUI/models/clip/",
        "download_url": "https://huggingface.co/Comfy-Org/MiniMax_H3",
    },
    {
        "role": "vae",
        "name": "minimax_h3_video_vae_fp16.safetensors",
        "quantization": "FP16",
        "destination_hint": "ComfyUI/models/vae/",
        "download_url": "https://huggingface.co/Comfy-Org/MiniMax_H3",
    },
    {
        "role": "vae",
        "name": "minimax_h3_audio_vae_fp32.safetensors",
        "quantization": "FP32",
        "destination_hint": "ComfyUI/models/vae/",
        "download_url": "https://huggingface.co/Comfy-Org/MiniMax_H3",
    },
]


BUNDLED_MODEL_STACKS: dict[str, list[dict[str, Any]]] = {
    "flux2-txt2img": [
        {
            "role": "diffusion_model",
            "name": "flux2_dev_fp8mixed.safetensors",
            "quantization": "FP8 mixed",
            "destination_hint": "ComfyUI/models/diffusion_models/",
            "download_url": (
                "https://huggingface.co/Comfy-Org/flux2-dev/tree/main/"
                "split_files/diffusion_models"
            ),
        },
        {
            "role": "text_encoder",
            "name": "mistral_3_small_flux2_fp4_mixed.safetensors",
            "quantization": "FP4 mixed",
            "destination_hint": "ComfyUI/models/text_encoders/",
            "download_url": (
                "https://huggingface.co/Comfy-Org/flux2-dev/tree/main/"
                "split_files/text_encoders"
            ),
        },
        {
            "role": "vae",
            "name": "flux2-vae.safetensors",
            "destination_hint": "ComfyUI/models/vae/",
            "download_url": (
                "https://huggingface.co/Comfy-Org/flux2-dev/blob/main/"
                "split_files/vae/flux2-vae.safetensors"
            ),
        },
    ],
    "minimax-h3-t2v": _MINIMAX_H3_STACK,
    "minimax-h3-i2v": _MINIMAX_H3_STACK,
    "qwen-image-edit-2511": [
        {
            "role": "diffusion_model",
            "name": "qwen_image_edit_2511_fp8mixed.safetensors",
            "quantization": "FP8 mixed",
            "destination_hint": "ComfyUI/models/diffusion_models/",
            "download_url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI",
        },
        {
            "role": "text_encoder",
            "name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "quantization": "FP8 scaled",
            "destination_hint": "ComfyUI/models/text_encoders/",
            "download_url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI",
        },
        {
            "role": "vae",
            "name": "qwen_image_vae.safetensors",
            "destination_hint": "ComfyUI/models/vae/",
            "download_url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI",
        },
        {
            "role": "lora",
            "name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors",
            "quantization": "FP32",
            "destination_hint": "ComfyUI/models/loras/",
            "download_url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI",
        },
    ],
}


def workflow_hash(workflow: dict[str, Any]) -> str:
    """Return a stable hash of the final workflow JSON submitted to ComfyUI."""
    payload = json.dumps(workflow, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_stack(workflow_key: str | None, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Return bundled or caller-supplied model stack metadata."""
    if workflow_key:
        return [dict(item) for item in BUNDLED_MODEL_STACKS[workflow_key]]
    stack = inputs.get("workflow_model_stack")
    return stack if isinstance(stack, list) else []


def missing_models_payload(
    missing: list[str],
    *,
    workflow_key: str,
    workflow_name: str,
    operation: str | None = None,
) -> dict[str, Any]:
    """Build a machine-readable missing-model error payload."""
    stack_by_name = {
        item["name"]: item for item in BUNDLED_MODEL_STACKS.get(workflow_key, [])
    }
    items = []
    for name in missing:
        meta = dict(stack_by_name.get(name, {}))
        meta.setdefault("name", name)
        meta.setdefault("role", "unknown")
        meta.setdefault("destination_hint", "ComfyUI/models/ matching the workflow node")
        meta.setdefault("download_url", None)
        items.append(meta)

    return {
        "provider": "comfyui",
        "workflow": workflow_name,
        "operation": operation,
        "missing_models": items,
        "setup_offer": COMFYUI_SETUP_OFFER,
    }
