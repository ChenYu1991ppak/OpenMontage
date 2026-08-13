"""Thin REST client for a running ComfyUI server.

Handles the full generation cycle: submit workflow, poll for completion,
download artifacts.  Used by comfyui_image, comfyui_video, and comfyui_music.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import requests


class ComfyUIError(Exception):
    """Raised when ComfyUI returns an error or times out."""


class ComfyUIClient:
    """Client for the ComfyUI REST API.

    The protocol is simple and battle-tested:
      1. POST /prompt           → queue a workflow, get a prompt_id
      2. GET  /history/{id}     → poll until outputs appear
      3. GET  /view?filename=…  → download the generated artifact
      4. POST /upload/image     → stage a local image for I2V workflows
      5. POST /upload/audio     → stage a local audio clip for LoadAudio nodes
      6. POST /history/{id}     → delete the finished task record (best-effort)
    """

    def __init__(self, server_url: str | None = None) -> None:
        self.server_url = (
            server_url
            or os.environ.get("COMFYUI_SERVER_URL", "http://localhost:8188")
        ).rstrip("/")
        self._last_cleanup_warnings: list[str] = []
        self._last_cleanup: dict = {
            "attempted": False,
            "ok": False,
            "message": "",
        }

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @property
    def is_default_url(self) -> bool:
        """True if using the fallback URL (user didn't set COMFYUI_SERVER_URL)."""
        return not os.environ.get("COMFYUI_SERVER_URL")

    def is_available(self) -> bool:
        """Return True if the ComfyUI server is reachable."""
        try:
            resp = requests.get(
                f"{self.server_url}/system_stats", timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False

    def unavailable_reason(self) -> str:
        """Human-readable explanation of why the server can't be reached."""
        if self.is_default_url:
            return (
                f"No ComfyUI server found at {self.server_url} "
                f"(default — no COMFYUI_SERVER_URL configured).\n"
                f"Set COMFYUI_SERVER_URL in your .env file to the address of "
                f"your ComfyUI server (e.g. http://localhost:8188)."
            )
        return (
            f"ComfyUI server not reachable at {self.server_url}.\n"
            f"Check that ComfyUI is running and the URL is correct."
        )

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    def list_models(self) -> dict[str, list[str]]:
        """Query ComfyUI for available models, grouped by type.

        Returns a dict like::

            {
                "checkpoints": ["sd_xl_base.safetensors", ...],
                "diffusion_models": ["flux2-dev-nvfp4.safetensors", ...],
                "vae": ["ae.safetensors", ...],
                "clip": ["clip_l.safetensors", ...],
                "loras": ["my_lora.safetensors", ...],
            }
        """
        node_to_key = {
            "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
            "UNETLoader": ("unet_name", "diffusion_models"),
            "VAELoader": ("vae_name", "vae"),
            "CLIPLoader": ("clip_name", "clip"),
            "LoraLoaderModelOnly": ("lora_name", "loras"),
        }
        result: dict[str, list[str]] = {}
        for node_class, (field, group) in node_to_key.items():
            try:
                resp = requests.get(
                    f"{self.server_url}/object_info/{node_class}", timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                options = (
                    data.get(node_class, {})
                    .get("input", {})
                    .get("required", {})
                    .get(field, [[]])[0]
                )
                if isinstance(options, list):
                    result[group] = options
            except Exception:
                result[group] = []
        return result

    def check_models(
        self, required: list[str]
    ) -> tuple[list[str], list[str]]:
        """Check which of *required* model filenames are available.

        Returns ``(found, missing)`` — two lists of filenames.
        """
        all_models: set[str] = set()
        for names in self.list_models().values():
            all_models.update(names)

        found = [m for m in required if m in all_models]
        missing = [m for m in required if m not in all_models]
        return found, missing

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    def submit(self, workflow: dict) -> str:
        """Queue a workflow for execution.  Returns the ``prompt_id``."""
        resp = requests.post(
            f"{self.server_url}/prompt",
            json={"prompt": workflow},
            timeout=30,
        )
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if data.get("node_errors"):
            raise ComfyUIError(f"Node errors: {json.dumps(data['node_errors'])}")
        if data.get("error"):
            raise ComfyUIError(f"Prompt error: {json.dumps(data['error'])}")
        resp.raise_for_status()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"No prompt_id in response: {data}")
        return prompt_id

    def poll(
        self,
        prompt_id: str,
        *,
        timeout: int = 600,
        interval: int = 5,
    ) -> dict:
        """Block until *prompt_id* finishes.  Returns the history entry."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = requests.get(
                f"{self.server_url}/history/{prompt_id}", timeout=10
            )
            resp.raise_for_status()
            history = resp.json()
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise ComfyUIError(f"Execution error: {msgs}")
                return entry
            time.sleep(interval)
        raise ComfyUIError(
            f"Prompt {prompt_id} did not complete within {timeout}s"
        )

    def download(
        self,
        filename: str,
        subfolder: str,
        dest: Path,
        folder_type: str = "output",
    ) -> Path:
        """Download an output artifact from the ComfyUI server."""
        resp = requests.get(
            f"{self.server_url}/view",
            params={
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type,
            },
            timeout=120,
        )
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest

    def upload_image(self, local_path: Path, name: str) -> str:
        """Upload a local image so it can be referenced by LoadImage nodes.

        Returns the server-side filename.
        """
        with open(local_path, "rb") as f:
            resp = requests.post(
                f"{self.server_url}/upload/image",
                files={"image": (name, f, "image/png")},
                timeout=30,
            )
        resp.raise_for_status()
        return resp.json()["name"]

    def upload_audio(self, local_path: Path, name: str) -> str:
        """Upload a local audio file so it can be referenced by LoadAudio nodes.

        Returns the server-side filename.
        """
        with open(local_path, "rb") as f:
            resp = requests.post(
                f"{self.server_url}/upload/audio",
                files={"audio": (name, f, "audio/wav")},
                timeout=60,
            )
        resp.raise_for_status()
        return resp.json()["name"]

    def delete_history(self, prompt_id: str) -> tuple[bool, str]:
        """Delete a finished task record from the server history.

        Supports both API layouts in the wild:
          * new-style:  ``POST /history/{prompt_id}`` with ``{"delete": [id]}``
          * old-style:  ``POST /history``          with ``{"delete": [id]}``
        Falls back automatically when the server rejects the new-style URL.

        Best-effort: never raises — the caller decides what to do with the
        outcome.  Returns ``(ok, message)``.
        """
        payload = {"delete": [prompt_id]}
        attempts = (
            f"{self.server_url}/history/{prompt_id}",
            f"{self.server_url}/history",
        )
        last_error = None
        for url in attempts:
            try:
                resp = requests.post(url, json=payload, timeout=15)
                if resp.status_code == 405:
                    # Endpoint not supported on this server — try the other.
                    last_error = f"{resp.status_code} {resp.reason} for {url}"
                    continue
                resp.raise_for_status()
                return True, f"Deleted server history record {prompt_id}"
            except Exception as exc:  # noqa: BLE001 - cleanup must never block
                last_error = str(exc)
        return False, (
            f"Could not delete server history record {prompt_id}: {last_error}"
        )

    # ------------------------------------------------------------------
    # High-level helper
    # ------------------------------------------------------------------

    def generate(
        self,
        workflow: dict,
        output_node: str,
        dest: Path,
        *,
        timeout: int = 600,
        interval: int = 5,
        cleanup_history: bool | None = None,
    ) -> list[Path]:
        """Submit → poll → download.  Returns list of artifact paths.

        After all artifacts are downloaded successfully, the finished task
        record is deleted from the server history (best-effort).  Cleanup
        failures are recorded in :attr:`last_cleanup_warnings` and never
        raise, so a successful download is never turned into an error.

        Set *cleanup_history* to ``True``/``False`` to force the behaviour;
        the default (``None``) honours the ``COMFYUI_KEEP_SERVER_OUTPUTS``
        environment variable — when set to ``1``/``true``/``yes`` the server
        history is kept instead of deleted.
        """
        self._last_cleanup_warnings = []
        self._last_cleanup = {"attempted": False, "ok": False, "message": ""}
        prompt_id = self.submit(workflow)
        entry = self.poll(prompt_id, timeout=timeout, interval=interval)

        outputs = entry.get("outputs", {})
        node_output = outputs.get(output_node, {})

        # ComfyUI stores outputs under different keys: "images" (PNG/WEBP),
        # "gifs" (VHS_VideoCombine), "videos" (core SaveVideo), or "audio"
        # (SaveAudio / SaveAudioMP3).
        items = (
            node_output.get("videos")
            or node_output.get("images")
            or node_output.get("gifs")
            or node_output.get("audio")
        )
        if not items:
            raise ComfyUIError(
                f"No output artifacts on node {output_node}. "
                f"Available nodes: {list(outputs.keys())}"
            )

        paths: list[Path] = []
        for i, item in enumerate(items):
            suffix = Path(item["filename"]).suffix
            if len(items) == 1:
                target = dest
            else:
                target = dest.with_stem(f"{dest.stem}_{i:03d}").with_suffix(suffix)
            self.download(
                item["filename"],
                item.get("subfolder", ""),
                target,
                item.get("type", "output"),
            )
            paths.append(target)

        if (
            cleanup_history
            if cleanup_history is not None
            else not self._keep_server_outputs()
        ):
            ok, msg = self.delete_history(prompt_id)
            self._last_cleanup = {"attempted": True, "ok": ok, "message": msg}
            if not ok:
                self._last_cleanup_warnings.append(msg)
        else:
            self._last_cleanup = {
                "attempted": False,
                "ok": False,
                "message": "Skipped: server cleanup disabled",
            }
        return paths

    @staticmethod
    def _keep_server_outputs() -> bool:
        """True when COMFYUI_KEEP_SERVER_OUTPUTS=1 disables cleanup."""
        return os.environ.get("COMFYUI_KEEP_SERVER_OUTPUTS", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    @property
    def last_cleanup_warnings(self) -> list[str]:
        """Warnings from the most recent ``generate()`` server cleanup."""
        return list(self._last_cleanup_warnings)

    @property
    def last_cleanup(self) -> dict:
        """Outcome of the server-cleanup step of the last ``generate()``.

        Returns ``{"attempted": bool, "ok": bool, "message": str}``.
        """
        return dict(self._last_cleanup)

    # ------------------------------------------------------------------
    # Workflow helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_workflow(path: Path) -> dict:
        """Load a workflow JSON template from disk."""
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def patch_workflow(
        workflow: dict, patches: dict[str, dict[str, Any]]
    ) -> dict:
        """Deep-copy *workflow* and apply *patches*.

        *patches* maps ``node_id`` → ``{input_name: value, ...}``.
        """
        w = copy.deepcopy(workflow)
        for node_id, values in patches.items():
            if node_id not in w:
                raise ComfyUIError(
                    f"Node {node_id!r} not found in workflow. "
                    f"Available: {list(w.keys())}"
                )
            for key, val in values.items():
                w[node_id]["inputs"][key] = val
        return w

    @staticmethod
    def random_seed() -> int:
        """Return a random seed suitable for ComfyUI noise nodes."""
        return random.randint(0, 2**32 - 1)
