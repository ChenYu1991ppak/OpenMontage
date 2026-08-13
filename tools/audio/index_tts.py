"""IndexTTS25 zero-shot voice cloning TTS via a local ComfyUI server.

Runs the official IndexTTS25 ComfyUI workflow
(tools/_comfyui/workflows/IndexTTS25_官方工作流_api.json) against a local or
remote ComfyUI server with the IndexTTS25 custom node installed.

Supported modes:
- basic:        IndexTTS25TTS -- zero-shot voice cloning from a reference clip
- emotion_text: IndexTTS25TTSEmotionText -- adds emotion text + emo_alpha

The tool trims the official workflow down to the requested mode's nodes,
uploads the reference audio, patches text/sampling params, and downloads the
resulting mp3.
"""

from __future__ import annotations

import os
import time
import uuid
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
from tools._comfyui.metadata import COMFYUI_SETUP_OFFER, workflow_hash

_WORKFLOWS = Path(__file__).resolve().parent.parent / "_comfyui" / "workflows"
_WORKFLOW_FILE = "IndexTTS25_官方工作流_api.json"

# Languages supported by IndexTTS25 (official model).
_LANGS = ("ZH", "EN", "JA", "ES", "AR")

# Node IDs inside the official workflow, per mode (verified from the JSON).
_MODE_NODES = {
    "basic": {"tts": "2", "save": "3", "prefix": "indextts25_basic"},
    "emotion_text": {"tts": "6", "save": "7", "prefix": "indextts25_emotion_text"},
}

# Shared workflow nodes.
_REFERENCE_NODE = "1"  # LoadAudio
_LOADER_NODE = "12"  # IndexTTS25Loader (starts / reuses the 8108 service)

# The official workflow JSON carries UI-export placeholder sampling values
# (e.g. temperature=30, top_p=3). We always overwrite these with sane defaults
# so the TTS node runs with real sampling parameters.
_SAMPLING_FIELDS = (
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "num_beams",
    "repetition_penalty",
    "max_mel_tokens",
    "interval_silence",
)
_DEFAULT_SAMPLING: dict[str, Any] = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 50,
    "num_beams": 1,
    "repetition_penalty": 1.15,
    "max_mel_tokens": 1500,
    "interval_silence": 200,
}


class IndexTTSTTS(BaseTool):
    name = "index_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "indextts"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.LOCAL_GPU

    dependencies = []  # checked at runtime via ComfyUI server health
    setup_offer = COMFYUI_SETUP_OFFER
    install_instructions = (
        "Start a ComfyUI server with the IndexTTS25 custom node installed "
        "(https://github.com/index-tts/index-tts) and set COMFYUI_SERVER_URL "
        "(default http://localhost:8188).\n"
        "The IndexTTS25Loader node starts the model service on INDEXTTS_HOST:"
        "INDEXTTS_PORT (default 127.0.0.1:8108) using INDEXTTS_MODEL_DIR and "
        "INDEXTTS_VENV."
    )
    agent_skills = ["index-tts"]

    capabilities = ["text_to_speech", "voice_cloning", "emotion_text", "multilingual"]
    supports = {
        "voice_cloning": True,
        "multilingual": True,
        "offline": True,
        "native_audio": True,
        "emotion_control": True,
        "languages": list(_LANGS),
    }
    best_for = [
        "zero-shot voice cloning from a reference audio sample",
        "expressive narration with emotion_text control",
        "privacy-sensitive local-only TTS with no API cost",
        "multilingual narration (ZH/EN/JA/ES/AR) in one voice",
    ]
    not_good_for = [
        "setups without a running ComfyUI server with the IndexTTS25 node",
        "very short throwaway clips when a lighter local TTS (piper) suffices",
    ]
    fallback_tools = ["doubao_tts", "elevenlabs_tts", "openai_tts", "piper_tts"]

    input_schema = {
        "type": "object",
        "required": ["text", "reference_audio"],
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "Text to synthesize. Supports inline pronunciation hints: "
                    "<拼音|PINYIN>, <单词|CMU音素>, <汉字|假名>."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["basic", "emotion_text"],
                "default": "basic",
                "description": (
                    "basic: plain synthesis from a reference voice. "
                    "emotion_text: adds emotion_text + emo_alpha emotional control."
                ),
            },
            "reference_audio": {
                "type": "string",
                "description": (
                    "Voice to clone: a local wav/mp3 path, or a preset name "
                    "resolved under INDEXTTS_REFERENCE_DIR (default "
                    "assets/voices/index_tts)."
                ),
            },
            "lang": {
                "type": "string",
                "enum": list(_LANGS),
                "default": "ZH",
                "description": "Output language of text.",
            },
            "duration_factor": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 2.0,
                "default": 1.0,
                "description": "Speaking speed multiplier. <1 = faster, >1 = slower.",
            },
            "emotion_text": {
                "type": "string",
                "description": (
                    "Emotion description for mode=emotion_text, e.g. "
                    "'兴高采烈，欢呼雀跃'. Describes how the main text should sound."
                ),
            },
            "use_main_text": {
                "type": "boolean",
                "default": False,
                "description": (
                    "When True, emotion_text is prepended to the main text for "
                    "emotion extraction instead of using it standalone."
                ),
            },
            "emo_alpha": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 1.0,
                "description": "Emotion intensity (0 = neutral, 1 = strongest).",
            },
            "do_sample": {"type": "boolean", "default": True},
            "temperature": {"type": "number", "default": 0.6},
            "top_p": {"type": "number", "default": 0.95},
            "top_k": {"type": "integer", "default": 50},
            "num_beams": {"type": "integer", "default": 1},
            "repetition_penalty": {"type": "number", "default": 1.15},
            "max_mel_tokens": {"type": "integer", "default": 1500},
            "interval_silence": {"type": "integer", "default": 200},
            "output_path": {
                "type": "string",
                "description": "Where to save the mp3. Forced to .mp3 suffix.",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=4000, vram_mb=6000, disk_mb=200, network_required=False,
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["timeout"])
    idempotency_key_fields = ["text", "reference_audio", "lang", "mode", "duration_factor"]
    side_effects = ["writes audio file to output_path", "loads IndexTTS25 model on the ComfyUI host"]
    user_visible_verification = ["Listen to generated audio for voice match and emotion"]
    quality_score = 0.82
    latency_p50_seconds = 30.0

    def __init__(self) -> None:
        self._client = ComfyUIClient()

    def get_status(self) -> ToolStatus:
        if not self._client.is_available():
            return ToolStatus.UNAVAILABLE
        if not self._node_installed():
            return ToolStatus.UNAVAILABLE
        return ToolStatus.AVAILABLE

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["setup_offer"] = self.setup_offer
        return info

    def _node_installed(self) -> bool:
        """IndexTTS25 is usable if the custom node class is exposed by the server."""
        try:
            resp = requests.get(
                f"{self._client.server_url}/object_info/IndexTTS25TTS", timeout=10
            )
            resp.raise_for_status()
            return bool(resp.json())
        except Exception:
            return False

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 30.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        try:
            mode = self._validated_mode(inputs)
            lang = self._validated_lang(inputs)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))
        text = str(inputs.get("text") or "").strip()
        if not text:
            return ToolResult(success=False, error="text is required")
        ref = str(inputs.get("reference_audio") or "").strip()
        if not ref:
            return ToolResult(success=False, error="reference_audio is required (local path or preset name)")

        if not self._client.is_available():
            return ToolResult(success=False, error=self._client.unavailable_reason())
        if not self._node_installed():
            return ToolResult(
                success=False,
                error=(
                    "IndexTTS25 custom node not found on the ComfyUI server. "
                    "Install it from https://github.com/index-tts/index-tts and "
                    "restart ComfyUI."
                ),
            )

        start = time.time()
        try:
            ref_path = self._resolve_reference_audio(ref)
            server_name = self._client.upload_audio(ref_path, ref_path.name)

            seed = ComfyUIClient.random_seed()
            suffix = uuid.uuid4().hex[:8]
            output_path = Path(inputs.get("output_path", f"index_tts_{mode}_{suffix}.mp3"))
            if output_path.suffix.lower() != ".mp3":
                output_path = output_path.with_suffix(".mp3")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            workflow = self._build_workflow(mode)
            patch = self._build_patch(mode, text, lang, server_name, inputs, seed, suffix)
            workflow = ComfyUIClient.patch_workflow(workflow, patch)
            paths = self._client.generate(
                workflow,
                output_node=_MODE_NODES[mode]["save"],
                dest=output_path,
                timeout=600,
            )
        except ComfyUIError as exc:
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:
            return ToolResult(success=False, error=f"IndexTTS TTS failed: {exc}")

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": "IndexTTS-2.5",
                "mode": mode,
                "lang": lang,
                "text_length": len(text),
                "reference_audio": str(ref_path),
                "output": str(paths[0]),
                "format": "mp3",
                "workflow_provenance": {
                    "source": "bundled",
                    "workflow": _WORKFLOW_FILE,
                    "workflow_hash_sha256": workflow_hash(workflow),
                    "output_node": _MODE_NODES[mode]["save"],
                },
            },
            artifacts=[str(p) for p in paths],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            seed=seed,
            model="IndexTTS-2.5",
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _validated_mode(inputs: dict[str, Any]) -> str:
        mode = str(inputs.get("mode", "basic") or "basic").strip()
        if mode not in _MODE_NODES:
            raise ValueError(f"mode must be one of: {', '.join(_MODE_NODES)}")
        return mode

    @staticmethod
    def _validated_lang(inputs: dict[str, Any]) -> str:
        lang = str(inputs.get("lang", "ZH") or "ZH").strip().upper()
        if lang not in _LANGS:
            raise ValueError(f"lang must be one of: {', '.join(_LANGS)}")
        return lang

    @staticmethod
    def _resolve_reference_audio(ref: str) -> Path:
        """Local path wins; otherwise treat ref as a preset name in the preset dir."""
        candidate = Path(ref).expanduser()
        if candidate.is_file():
            return candidate
        preset_dir = Path(
            os.environ.get("INDEXTTS_REFERENCE_DIR", "assets/voices/index_tts")
        )
        for name in (ref, f"{ref}.wav", f"{ref}.mp3"):
            preset = preset_dir / name
            if preset.is_file():
                return preset
        raise FileNotFoundError(
            f"Reference audio {ref!r} not found (checked local path and preset dir {preset_dir}). "
            "Provide a local file path or place the preset in the preset dir."
        )

    @staticmethod
    def _loader_inputs() -> dict[str, Any]:
        return {
            "action": "load",
            "host": os.environ.get("INDEXTTS_HOST", "127.0.0.1"),
            "port": int(os.environ.get("INDEXTTS_PORT", "8108")),
            "model_dir": os.environ.get("INDEXTTS_MODEL_DIR", "/mnt/models/IndexTTS-2.5"),
            "venv": os.environ.get("INDEXTTS_VENV", "/mnt/indextts25-venv"),
        }

    def _build_workflow(self, mode: str) -> dict:
        wf = ComfyUIClient.load_workflow(_WORKFLOWS / _WORKFLOW_FILE)
        keep = {_REFERENCE_NODE, _MODE_NODES[mode]["tts"], _MODE_NODES[mode]["save"], _LOADER_NODE}
        return {nid: wf[nid] for nid in keep}

    def _build_patch(
        self,
        mode: str,
        text: str,
        lang: str,
        server_audio_name: str,
        inputs: dict[str, Any],
        seed: int,
        suffix: str,
    ) -> dict[str, Any]:
        tts_node = _MODE_NODES[mode]["tts"]
        patch: dict[str, Any] = {
            _REFERENCE_NODE: {"audio": server_audio_name},
            tts_node: {
                "text": text,
                "lang": lang,
                "reference_audio": [_REFERENCE_NODE, 0],
                "duration_factor": float(inputs.get("duration_factor", 1.0)),
                **{field: inputs.get(field, _DEFAULT_SAMPLING[field]) for field in _SAMPLING_FIELDS},
            },
            _MODE_NODES[mode]["save"]: {
                "filename_prefix": f"{_MODE_NODES[mode]['prefix']}_{suffix}",
                "quality": "V0",
            },
            _LOADER_NODE: self._loader_inputs(),
        }
        if mode == "emotion_text":
            patch[tts_node].update(
                {
                    "emotion_text": str(inputs.get("emotion_text") or "").strip()
                    or "平静自然",
                    "use_main_text": bool(inputs.get("use_main_text", False)),
                    "emo_alpha": float(inputs.get("emo_alpha", 1.0)),
                }
            )
        return patch
