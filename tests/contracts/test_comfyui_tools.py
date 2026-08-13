"""Contract tests for ComfyUI provider tools.

These tests verify that the tools satisfy the BaseTool contract without
requiring a running ComfyUI server.  They check class attributes,
schemas, status reporting, and cost estimates.
"""

import json
from pathlib import Path

import pytest

from tools.base_tool import (
    BaseTool,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools.graphics.comfyui_image import ComfyUIImage
from tools.graphics.image_selector import ImageSelector
from tools.tool_registry import ToolRegistry
from tools.video.video_selector import VideoSelector
from tools.video.comfyui_video import ComfyUIVideo

TOOLS = [ComfyUIImage, ComfyUIVideo]
WORKFLOW_DIR = Path(__file__).resolve().parent.parent.parent / "tools" / "_comfyui" / "workflows"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ------------------------------------------------------------------
# Contract compliance
# ------------------------------------------------------------------

@pytest.mark.parametrize("cls", TOOLS, ids=lambda c: c.name)
class TestContract:

    def test_inherits_base_tool(self, cls):
        assert issubclass(cls, BaseTool)

    def test_has_required_identity(self, cls):
        tool = cls()
        assert tool.name
        assert tool.version
        assert tool.capability
        assert tool.provider == "comfyui"
        assert tool.tier == ToolTier.GENERATE
        assert tool.stability == ToolStability.EXPERIMENTAL
        assert tool.runtime == ToolRuntime.LOCAL_GPU

    def test_has_input_schema(self, cls):
        tool = cls()
        schema = tool.input_schema
        assert schema.get("type") == "object"
        assert "prompt" in schema.get("properties", {})
        assert "prompt" in schema.get("required", [])

    def test_has_capabilities(self, cls):
        tool = cls()
        assert len(tool.capabilities) > 0

    def test_has_agent_skills(self, cls):
        tool = cls()
        assert tool.agent_skills
        assert "comfyui" in tool.agent_skills

    def test_comfyui_layer3_skill_exists(self, cls):
        skill_path = PROJECT_ROOT / ".agents" / "skills" / "comfyui" / "SKILL.md"
        assert skill_path.exists()
        assert "output_node" in skill_path.read_text(encoding="utf-8")

    def test_has_fallbacks(self, cls):
        tool = cls()
        assert tool.fallback or tool.fallback_tools

    def test_cost_is_zero(self, cls):
        tool = cls()
        assert tool.estimate_cost({"prompt": "test"}) == 0.0

    def test_runtime_estimate_positive(self, cls):
        tool = cls()
        assert tool.estimate_runtime({"prompt": "test"}) > 0

    def test_get_info_returns_dict(self, cls):
        tool = cls()
        info = tool.get_info()
        assert isinstance(info, dict)
        assert info["name"] == tool.name
        assert info["provider"] == "comfyui"
        assert info["runtime"] == "local_gpu"
        assert info["setup_offer"]["env_var"] == "COMFYUI_SERVER_URL"

    def test_video_resource_profile_does_not_mandate_16gb(self, cls):
        if cls is not ComfyUIVideo:
            return
        tool = ComfyUIVideo()
        info = tool.get_info()
        assert info["resource_profile"]["vram_mb"] == 8000
        assert info["resource_profiles"]["provider_floor"]["vram_mb"] == 8000
        assert info["resource_profiles"]["bundled_minimax_h3_fl2va"]["vram_mb"] == 32000
        assert "not a ComfyUI provider-wide requirement" in (
            info["resource_profiles"]["bundled_minimax_h3_fl2va"]["applies_to"]
        )

    def test_status_unavailable_without_server(self, cls):
        """Without a running server, status should be UNAVAILABLE."""
        tool = cls()
        # Point to a port that's almost certainly not running ComfyUI
        tool._client.server_url = "http://127.0.0.1:19999"
        assert tool.get_status() == ToolStatus.UNAVAILABLE

    def test_idempotency_key_fields(self, cls):
        tool = cls()
        assert len(tool.idempotency_key_fields) > 0
        assert "prompt" in tool.idempotency_key_fields

    def test_custom_workflow_schema_requires_output_node_contract(self, cls):
        tool = cls()
        props = tool.input_schema.get("properties", {})
        assert "workflow_json" in props
        assert "workflow_path" in props
        assert "output_node" in props

    def test_custom_workflow_requires_output_node(self, cls):
        tool = cls()
        result = tool.execute({"prompt": "test", "workflow_json": "{}"})
        assert result.success is False
        assert "output_node" in result.error


# ------------------------------------------------------------------
# Workflow files
# ------------------------------------------------------------------

EXPECTED_WORKFLOWS = [
    "flux2-txt2img.json",
    "image_qwen_image_edit_2511_api.json",
    "video_minimax_h3_i2v_api.json",
    "video_minimax_h3_t2v_api.json",
]


@pytest.mark.parametrize("filename", EXPECTED_WORKFLOWS)
def test_workflow_exists_and_valid_json(filename):
    path = WORKFLOW_DIR / filename
    assert path.exists(), f"Missing workflow: {path}"
    with open(path) as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) > 0


def test_flux2_workflow_has_templated_nodes():
    with open(WORKFLOW_DIR / "flux2-txt2img.json") as f:
        w = json.load(f)
    assert "4" in w  # CLIPTextEncode (prompt)
    assert "7" in w  # RandomNoise (seed)
    assert "13" in w  # SaveImage (output)


def test_qwen_edit_workflow_has_templated_nodes():
    with open(WORKFLOW_DIR / "image_qwen_image_edit_2511_api.json") as f:
        w = json.load(f)
    assert "41" in w       # LoadImage (source image)
    assert "170:151" in w  # positive edit instruction
    assert "170:149" in w  # negative prompt
    assert "170:169" in w  # KSampler (seed)
    assert "9" in w        # SaveImage (output)
    assert w["41"]["class_type"] == "LoadImage"
    assert w["170:151"]["inputs"]["prompt"] == "给人物换一套西装"


def test_i2v_workflow_has_templated_nodes():
    with open(WORKFLOW_DIR / "video_minimax_h3_i2v_api.json") as f:
        w = json.load(f)
    assert "114" in w     # LoadImage (reference)
    assert "105:104" in w  # MiniMaxH3ImageToVideo (prompt + first_frame)
    assert "105:15" in w   # RandomNoise (seed)
    assert "92" in w       # SaveVideo (output)
    assert w["105:104"]["inputs"]["first_frame"] == ["114", 0]


def test_t2v_workflow_has_templated_nodes():
    with open(WORKFLOW_DIR / "video_minimax_h3_t2v_api.json") as f:
        w = json.load(f)
    assert "105:104" in w  # MiniMaxH3ImageToVideo (prompt)
    assert "105:15" in w   # RandomNoise (seed)
    assert "92" in w       # SaveVideo (output)
    assert "first_frame" not in w["105:104"]["inputs"]


def test_t2v_workflow_uses_h3_video_vae():
    with open(WORKFLOW_DIR / "video_minimax_h3_t2v_api.json") as f:
        w = json.load(f)
    assert w["105:11"]["inputs"]["vae_name"] == "minimax_h3_video_vae_fp16.safetensors"


def test_t2v_metadata_stack_uses_h3_video_vae():
    from tools._comfyui.metadata import BUNDLED_MODEL_STACKS

    vae_entry = next(item for item in BUNDLED_MODEL_STACKS["minimax-h3-t2v"] if item["role"] == "vae")
    assert vae_entry["name"] == "minimax_h3_video_vae_fp16.safetensors"
    assert "MiniMax_H3" in vae_entry["download_url"]


# ------------------------------------------------------------------
# Client unit tests
# ------------------------------------------------------------------

class TestClientHelpers:

    def test_load_workflow(self):
        from tools._comfyui.client import ComfyUIClient
        w = ComfyUIClient.load_workflow(WORKFLOW_DIR / "flux2-txt2img.json")
        assert isinstance(w, dict)
        assert "1" in w

    def test_patch_workflow(self):
        from tools._comfyui.client import ComfyUIClient
        w = ComfyUIClient.load_workflow(WORKFLOW_DIR / "flux2-txt2img.json")
        patched = ComfyUIClient.patch_workflow(w, {
            "4": {"text": "hello world"},
            "7": {"noise_seed": 123},
        })
        assert patched["4"]["inputs"]["text"] == "hello world"
        assert patched["7"]["inputs"]["noise_seed"] == 123
        # Original unchanged
        assert w["4"]["inputs"]["text"] == ""

    def test_patch_workflow_bad_node(self):
        from tools._comfyui.client import ComfyUIClient, ComfyUIError
        w = {"1": {"inputs": {"x": 1}}}
        with pytest.raises(ComfyUIError, match="not found"):
            ComfyUIClient.patch_workflow(w, {"99": {"x": 2}})

    def test_submit_surfaces_node_errors_before_http_error(self, monkeypatch):
        from tools._comfyui.client import ComfyUIClient, ComfyUIError

        class FakeResponse:
            status_code = 400

            def json(self):
                return {
                    "error": {"message": "Prompt outputs failed validation"},
                    "node_errors": {"4": {"class_type": "MissingNode"}},
                }

            def raise_for_status(self):
                raise AssertionError("HTTPError should not hide node_errors")

        monkeypatch.setattr(
            "tools._comfyui.client.requests.post",
            lambda *args, **kwargs: FakeResponse(),
        )

        with pytest.raises(ComfyUIError, match="Node errors"):
            ComfyUIClient("http://comfy.test").submit({})

    def test_random_seed_range(self):
        from tools._comfyui.client import ComfyUIClient
        for _ in range(100):
            s = ComfyUIClient.random_seed()
            assert 0 <= s < 2**32

    def test_generate_passes_history_item_type_to_view(self, monkeypatch, tmp_path):
        from tools._comfyui.client import ComfyUIClient

        client = ComfyUIClient("http://comfy.test")
        seen = {}

        monkeypatch.setattr(client, "submit", lambda workflow: "prompt-1")
        monkeypatch.setattr(client, "poll", lambda prompt_id, **kwargs: {
            "outputs": {
                "9": {
                    "images": [{
                        "filename": "preview.png",
                        "subfolder": "previews",
                        "type": "temp",
                    }]
                }
            }
        })

        def fake_download(filename, subfolder, dest, folder_type="output"):
            seen["filename"] = filename
            seen["subfolder"] = subfolder
            seen["folder_type"] = folder_type
            return Path(dest)

        monkeypatch.setattr(client, "download", fake_download)

        client.generate(
            {"9": {"inputs": {}}},
            "9",
            tmp_path / "preview.png",
            cleanup_history=False,
        )

        assert seen == {
            "filename": "preview.png",
            "subfolder": "previews",
            "folder_type": "temp",
        }

    def test_is_default_url_when_env_not_set(self, monkeypatch):
        from tools._comfyui.client import ComfyUIClient
        monkeypatch.delenv("COMFYUI_SERVER_URL", raising=False)
        client = ComfyUIClient()
        assert client.is_default_url is True

    def test_is_not_default_url_when_env_set(self, monkeypatch):
        from tools._comfyui.client import ComfyUIClient
        monkeypatch.setenv("COMFYUI_SERVER_URL", "http://myhost:9999")
        client = ComfyUIClient()
        assert client.is_default_url is False

    def test_unavailable_reason_default_url(self, monkeypatch):
        from tools._comfyui.client import ComfyUIClient
        monkeypatch.delenv("COMFYUI_SERVER_URL", raising=False)
        client = ComfyUIClient()
        msg = client.unavailable_reason()
        assert "COMFYUI_SERVER_URL" in msg
        assert ".env" in msg

    def test_unavailable_reason_custom_url(self, monkeypatch):
        from tools._comfyui.client import ComfyUIClient
        monkeypatch.setenv("COMFYUI_SERVER_URL", "http://myhost:9999")
        client = ComfyUIClient()
        msg = client.unavailable_reason()
        assert "myhost:9999" in msg
        assert "COMFYUI_SERVER_URL" not in msg

    # --- server history cleanup -------------------------------------------------

    def _make_generate_client(self, monkeypatch, tmp_path):
        """ComfyUIClient whose submit/poll/download hit no network."""
        from tools._comfyui.client import ComfyUIClient

        client = ComfyUIClient("http://comfy.test")
        monkeypatch.setattr(client, "submit", lambda workflow: "prompt-1")
        monkeypatch.setattr(
            client,
            "poll",
            lambda prompt_id, **kwargs: {
                "outputs": {
                    "9": {
                        "images": [{
                            "filename": "out.png",
                            "subfolder": "",
                            "type": "output",
                        }]
                    }
                }
            },
        )
        monkeypatch.setattr(
            client,
            "download",
            lambda filename, subfolder, dest, folder_type="output": Path(dest),
        )
        return client

    def test_generate_deletes_history_after_success(self, monkeypatch, tmp_path):
        client = self._make_generate_client(monkeypatch, tmp_path)
        deleted = []
        monkeypatch.setattr(
            client,
            "delete_history",
            lambda prompt_id: deleted.append(prompt_id) or (True, "ok"),
        )

        paths = client.generate({}, "9", tmp_path / "out.png")

        assert paths
        assert deleted == ["prompt-1"]
        assert client.last_cleanup == {
            "attempted": True,
            "ok": True,
            "message": "ok",
        }
        assert client.last_cleanup_warnings == []

    def test_generate_keeps_history_when_env_keep_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("COMFYUI_KEEP_SERVER_OUTPUTS", "1")
        client = self._make_generate_client(monkeypatch, tmp_path)
        deleted = []
        monkeypatch.setattr(
            client,
            "delete_history",
            lambda prompt_id: deleted.append(prompt_id) or (True, "ok"),
        )

        paths = client.generate({}, "9", tmp_path / "out.png")

        assert paths
        assert deleted == []
        assert client.last_cleanup["attempted"] is False

    def test_generate_cleanup_failure_is_best_effort(self, monkeypatch, tmp_path):
        client = self._make_generate_client(monkeypatch, tmp_path)
        monkeypatch.setattr(
            client, "delete_history", lambda prompt_id: (False, "boom")
        )

        paths = client.generate({}, "9", tmp_path / "out.png")

        # Download result is returned regardless of cleanup failure.
        assert paths
        assert client.last_cleanup["attempted"] is True
        assert client.last_cleanup["ok"] is False
        assert client.last_cleanup_warnings == ["boom"]

    def test_generate_cleanup_history_false_forces_skip(self, monkeypatch, tmp_path):
        client = self._make_generate_client(monkeypatch, tmp_path)
        deleted = []
        monkeypatch.setattr(
            client,
            "delete_history",
            lambda prompt_id: deleted.append(prompt_id) or (True, "ok"),
        )

        paths = client.generate(
            {}, "9", tmp_path / "out.png", cleanup_history=False
        )

        assert paths
        assert deleted == []
        assert client.last_cleanup["attempted"] is False

    def test_delete_history_posts_correct_payload(self, monkeypatch):
        from tools._comfyui.client import ComfyUIClient

        seen = {}

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

        def fake_post(url, json=None, timeout=15):
            seen["url"] = url
            seen["json"] = json
            return FakeResponse()

        monkeypatch.setattr("tools._comfyui.client.requests.post", fake_post)

        client = ComfyUIClient("http://comfy.test")
        ok, msg = client.delete_history("prompt-9")

        assert ok is True
        assert "Deleted" in msg
        assert seen["url"] == "http://comfy.test/history/prompt-9"
        assert seen["json"] == {"delete": ["prompt-9"]}

    def test_delete_history_failure_never_raises(self, monkeypatch):
        from tools._comfyui.client import ComfyUIClient

        def fake_post(url, json=None, timeout=15):
            raise ConnectionError("down")

        monkeypatch.setattr("tools._comfyui.client.requests.post", fake_post)

        client = ComfyUIClient("http://comfy.test")
        ok, msg = client.delete_history("prompt-9")

        assert ok is False
        assert "prompt-9" in msg

    def test_delete_history_falls_back_on_405(self, monkeypatch):
        from tools._comfyui.client import ComfyUIClient

        calls = []

        class Fake405:
            status_code = 405
            reason = "Method Not Allowed"

            def raise_for_status(self):
                raise RuntimeError("405")

        class FakeOK:
            status_code = 200

            def raise_for_status(self):
                return None

        def fake_post(url, json=None, timeout=15):
            calls.append(url)
            if url.endswith("/history/prompt-9"):
                return Fake405()
            return FakeOK()

        monkeypatch.setattr("tools._comfyui.client.requests.post", fake_post)

        client = ComfyUIClient("http://comfy.test")
        ok, msg = client.delete_history("prompt-9")

        assert ok is True
        assert calls == [
            "http://comfy.test/history/prompt-9",
            "http://comfy.test/history",
        ]


# ------------------------------------------------------------------
# Model discovery (offline, no server needed)
# ------------------------------------------------------------------

class TestModelRequirements:

    def test_image_tool_has_required_models(self):
        from tools.graphics.comfyui_image import _REQUIRED_MODELS
        assert len(_REQUIRED_MODELS) > 0
        assert any("flux" in m.lower() for m in _REQUIRED_MODELS)

    def test_video_tool_has_required_models_i2v(self):
        from tools.video.comfyui_video import _REQUIRED_MODELS_I2V
        assert len(_REQUIRED_MODELS_I2V) > 0
        assert any("minimax" in m.lower() for m in _REQUIRED_MODELS_I2V)
        assert "minimax_h3_fl2va_pruned_int8_convrot.safetensors" in _REQUIRED_MODELS_I2V

    def test_video_tool_has_required_models_t2v(self):
        from tools.video.comfyui_video import _REQUIRED_MODELS_T2V
        assert len(_REQUIRED_MODELS_T2V) > 0
        assert any("minimax" in m.lower() for m in _REQUIRED_MODELS_T2V)
        assert "minimax_h3_fl2va_pruned_int8_convrot.safetensors" in _REQUIRED_MODELS_T2V

    def test_image_tool_has_qwen_edit_required_models(self):
        from tools.graphics.comfyui_image import _QWEN_EDIT_MODELS
        assert len(_QWEN_EDIT_MODELS) == 4
        assert "qwen_image_edit_2511_fp8mixed.safetensors" in _QWEN_EDIT_MODELS
        assert "qwen_2.5_vl_7b_fp8_scaled.safetensors" in _QWEN_EDIT_MODELS
        assert "qwen_image_vae.safetensors" in _QWEN_EDIT_MODELS
        assert (
            "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors"
            in _QWEN_EDIT_MODELS
        )

    def test_qwen_edit_metadata_stack_has_all_roles(self):
        from tools._comfyui.metadata import BUNDLED_MODEL_STACKS

        stack = BUNDLED_MODEL_STACKS["qwen-image-edit-2511"]
        roles = {item["role"] for item in stack}
        assert roles == {"diffusion_model", "text_encoder", "vae", "lora"}
        for item in stack:
            assert item["name"]
            assert item["destination_hint"]
            assert item["download_url"]
        vae_entry = next(item for item in stack if item["role"] == "vae")
        assert vae_entry["name"] == "qwen_image_vae.safetensors"
        assert vae_entry["destination_hint"] == "ComfyUI/models/vae/"
        assert "Qwen-Image-Edit" in vae_entry["download_url"]


# ------------------------------------------------------------------
# Custom workflow contract and provenance
# ------------------------------------------------------------------

class TestCustomWorkflowContract:

    def test_image_custom_workflow_uses_caller_output_node_and_provenance(self, tmp_path):
        tool = ComfyUIImage()
        tool._client.is_available = lambda: True
        seen = {}

        def fake_generate(workflow, output_node, dest, **kwargs):
            seen["workflow"] = workflow
            seen["output_node"] = output_node
            return [Path(dest)]

        tool._client.generate = fake_generate

        result = tool.execute({
            "prompt": "test",
            "workflow_json": json.dumps({"99": {"inputs": {}}}),
            "output_node": "99",
            "workflow_model": "custom-flux",
            "output_path": str(tmp_path / "image.png"),
        })

        assert result.success is True
        assert seen["output_node"] == "99"
        assert result.model == "custom-flux"
        assert result.data["model"] == "custom-flux"
        assert result.data["workflow_provenance"]["source"] == "user_supplied"
        assert result.data["workflow_provenance"]["output_node"] == "99"
        assert result.data["workflow_provenance"]["workflow_hash_sha256"]
        assert result.data["workflow_provenance"]["model_stack_source"] == (
            "unknown_custom_workflow"
        )

    def test_video_custom_workflow_uses_caller_output_node_and_provenance(self, tmp_path):
        tool = ComfyUIVideo()
        tool._client.is_available = lambda: True
        seen = {}

        def fake_generate(workflow, output_node, dest, **kwargs):
            seen["workflow"] = workflow
            seen["output_node"] = output_node
            return [Path(dest)]

        tool._client.generate = fake_generate

        result = tool.execute({
            "prompt": "test",
            "workflow_json": json.dumps({"42": {"inputs": {}}}),
            "output_node": "42",
            "workflow_model": "custom-wan",
            "output_path": str(tmp_path / "video.mp4"),
        })

        assert result.success is True
        assert seen["output_node"] == "42"
        assert result.model == "custom-wan"
        assert result.data["model"] == "custom-wan"
        assert result.data["workflow_provenance"]["source"] == "user_supplied"
        assert result.data["workflow_provenance"]["output_node"] == "42"
        assert result.data["workflow_provenance"]["workflow_hash_sha256"]
        assert result.data["workflow_provenance"]["model_stack_source"] == (
            "unknown_custom_workflow"
        )

    def test_result_reports_server_cleanup_outcome(self, tmp_path):
        tool = ComfyUIImage()
        tool._client.is_available = lambda: True
        tool._client.generate = lambda workflow, output_node, dest, **kwargs: [Path(dest)]

        result = tool.execute({
            "prompt": "test",
            "workflow_json": json.dumps({"99": {"inputs": {}}}),
            "output_node": "99",
            "output_path": str(tmp_path / "image.png"),
        })

        assert result.success is True
        assert result.data["server_cleanup"] == {
            "attempted": False,
            "history_deleted": False,
            "warnings": [],
        }

    def test_custom_workflow_accepts_model_stack_provenance(self, tmp_path):
        tool = ComfyUIVideo()
        tool._client.is_available = lambda: True
        tool._client.generate = lambda workflow, output_node, dest, **kwargs: [Path(dest)]

        result = tool.execute({
            "prompt": "test",
            "workflow_json": json.dumps({"42": {"inputs": {}}}),
            "output_node": "42",
            "workflow_model_stack": [{"role": "lora", "name": "style.safetensors"}],
            "output_path": str(tmp_path / "video.mp4"),
        })

        provenance = result.data["workflow_provenance"]
        assert provenance["model_stack"] == [{"role": "lora", "name": "style.safetensors"}]
        assert provenance["model_stack_source"] == "caller_supplied"

    def test_image_missing_models_are_structured(self):
        tool = ComfyUIImage()
        tool._client.is_available = lambda: True
        tool._client.check_models = lambda required: (
            [],
            ["flux2-vae.safetensors"],
        )

        result = tool.execute({"prompt": "test"})

        assert result.success is False
        assert result.data["provider"] == "comfyui"
        assert result.data["missing_models"][0]["name"] == "flux2-vae.safetensors"
        assert result.data["missing_models"][0]["destination_hint"] == "ComfyUI/models/vae/"
        assert result.data["missing_models"][0]["download_url"]

    def test_video_missing_models_are_structured(self):
        tool = ComfyUIVideo()
        tool._client.is_available = lambda: True
        tool._client.check_models = lambda required: (
            [],
            ["minimax_h3_fl2va_pruned_int8_convrot.safetensors"],
        )

        result = tool.execute({"prompt": "test", "operation": "text_to_video"})

        assert result.success is False
        assert result.data["operation"] == "text_to_video"
        assert result.data["missing_models"][0]["role"] == "diffusion_model"
        assert result.data["missing_models"][0]["download_url"]

    def test_bundled_workflow_provenance_records_hash_and_stack(self, tmp_path):
        tool = ComfyUIImage()
        tool._client.is_available = lambda: True
        tool._client.check_models = lambda required: (list(required), [])
        tool._client.generate = lambda workflow, output_node, dest, **kwargs: [Path(dest)]

        result = tool.execute({
            "prompt": "test",
            "output_path": str(tmp_path / "image.png"),
        })

        provenance = result.data["workflow_provenance"]
        assert provenance["source"] == "bundled"
        assert provenance["workflow_hash_sha256"]
        assert any(item["role"] == "vae" for item in provenance["model_stack"])


# ------------------------------------------------------------------
# Qwen-Image-Edit-2511 image editing contract
# ------------------------------------------------------------------

class TestQwenImageEditContract:

    def test_declares_image_edit_capability(self):
        tool = ComfyUIImage()
        assert "image_edit" in tool.capabilities
        assert tool.supports.get("image_edit") is True

    def test_schema_exposes_source_image_fields(self):
        tool = ComfyUIImage()
        props = tool.input_schema["properties"]
        assert "image_path" in props
        assert "image_url" in props
        assert "image editing" in props["prompt"]["description"]

    def test_edit_branch_uploads_image_and_patches_nodes(self, tmp_path, monkeypatch):
        from tools._comfyui.client import ComfyUIClient

        tool = ComfyUIImage()
        tool._client.is_available = lambda: True
        tool._client.check_models = lambda required: (list(required), [])
        seen = {}

        def fake_upload(local_path, name):
            seen["uploaded"] = name
            return "server_ref.png"

        def fake_generate(workflow, output_node, dest, **kwargs):
            seen["workflow"] = workflow
            seen["output_node"] = output_node
            return [Path(dest)]

        tool._client.upload_image = fake_upload
        tool._client.generate = fake_generate

        src = tmp_path / "ref.png"
        src.write_bytes(b"fake-png")
        out = tmp_path / "edited.png"

        result = tool.execute({
            "prompt": "给人物换一套西装",
            "image_path": str(src),
            "seed": 123,
            "output_path": str(out),
        })

        assert result.success is True
        assert seen["uploaded"] == "ref.png"
        assert seen["output_node"] == "9"
        w = seen["workflow"]
        assert w["41"]["inputs"]["image"] == "server_ref.png"
        assert w["170:151"]["inputs"]["prompt"] == "给人物换一套西装"
        assert w["170:149"]["inputs"]["prompt"] == ""
        assert w["170:169"]["inputs"]["seed"] == 123
        assert w["9"]["inputs"]["filename_prefix"] == "edited"
        assert result.model == "qwen-image-edit-2511-fp8mixed"
        assert result.data["mode"] == "image_edit"
        assert result.data["source_image"] == str(src)
        provenance = result.data["workflow_provenance"]
        assert provenance["source"] == "bundled"
        assert provenance["workflow"] == "image_qwen_image_edit_2511_api.json"
        assert provenance["output_node"] == "9"
        assert any(item["role"] == "lora" for item in provenance["model_stack"])

    def test_edit_branch_downloads_image_url(self, tmp_path, monkeypatch):
        tool = ComfyUIImage()
        tool._client.is_available = lambda: True
        tool._client.check_models = lambda required: (list(required), [])
        seen = {}

        class FakeResp:
            content = b"fake-png"

            def raise_for_status(self):
                return None

        def fake_get(url, timeout=60):
            seen["url"] = url
            return FakeResp()

        monkeypatch.setattr("tools.graphics.comfyui_image.requests.get", fake_get)

        def fake_upload(local_path, name):
            seen["uploaded"] = name
            seen["local"] = Path(local_path)
            return "server_ref.png"

        tool._client.upload_image = fake_upload
        tool._client.generate = lambda workflow, output_node, dest, **kwargs: [Path(dest)]

        out = tmp_path / "edited.png"
        result = tool.execute({
            "prompt": "把背景换成纯白色",
            "image_url": "https://example.com/ref.png",
            "output_path": str(out),
        })

        assert result.success is True
        assert seen["url"] == "https://example.com/ref.png"
        assert seen["uploaded"] == "edited_source.png"
        # Downloaded temp file is cleaned up.
        assert not seen["local"].exists()

    def test_edit_missing_models_are_structured(self):
        tool = ComfyUIImage()
        tool._client.is_available = lambda: True
        tool._client.check_models = lambda required: (
            [],
            ["qwen_image_vae.safetensors"],
        )

        result = tool.execute({"prompt": "x", "image_path": "/tmp/ref.png"})

        assert result.success is False
        assert result.data["missing_models"][0]["name"] == "qwen_image_vae.safetensors"
        assert result.data["missing_models"][0]["destination_hint"] == "ComfyUI/models/vae/"
        assert result.data["missing_models"][0]["download_url"]
        assert result.data["workflow"] == "image_qwen_image_edit_2511_api.json"
        assert result.data["operation"] == "image_edit"

    def test_selector_keeps_image_tool_for_edit_request(self):
        selector = ImageSelector()
        candidates = [ComfyUIImage()]
        inputs = {"prompt": "x", "generation_mode": "edit", "image_path": "/tmp/ref.png"}
        filtered = selector._filter_candidates(inputs, candidates)
        assert [t.name for t in filtered] == ["comfyui_image"]

    def test_selector_keeps_image_tool_for_image_url_request(self):
        selector = ImageSelector()
        candidates = [ComfyUIImage()]
        inputs = {"prompt": "x", "image_url": "https://example.com/ref.png"}
        filtered = selector._filter_candidates(inputs, candidates)
        assert [t.name for t in filtered] == ["comfyui_image"]


class TestComfyUISetupOffer:

    def test_provider_menu_summary_includes_structured_setup_offer(self):
        registry = ToolRegistry()
        tool = ComfyUIImage()
        tool._client.is_available = lambda: False
        registry.register(tool)
        registry._discovered_packages.add("tools")

        summary = registry.provider_menu_summary()

        offer = summary["setup_offers"][0]
        assert offer["tool"] == "comfyui_image"
        assert offer["env_var"] == "COMFYUI_SERVER_URL"
        assert offer["default_url"] == "http://localhost:8188"
        assert offer["health_check"] == "GET /system_stats"


# ------------------------------------------------------------------
# Operation-specific video readiness
# ------------------------------------------------------------------

class TestVideoOperationReadiness:

    def test_video_tool_reports_partial_operation_readiness(self):
        from tools.video.comfyui_video import _REQUIRED_MODELS_I2V, _REQUIRED_MODELS_T2V

        tool = ComfyUIVideo()
        tool._client.is_available = lambda: True

        def fake_check_models(required):
            # H3 T2V/I2V share the same model set, so distinguish by identity.
            if required is _REQUIRED_MODELS_T2V:
                return list(required), []
            if required is _REQUIRED_MODELS_I2V:
                return [], list(required)
            return [], list(required)

        tool._client.check_models = fake_check_models

        assert tool.get_status() == ToolStatus.AVAILABLE
        assert tool.is_operation_available("text_to_video") is True
        assert tool.is_operation_available("image_to_video") is False
        assert tool.operation_statuses() == {
            "text_to_video": "available",
            "image_to_video": "degraded",
        }

    def test_video_selector_filters_operation_unready_tools(self):
        class PartialVideoTool(BaseTool):
            name = "partial_video"
            capability = "video_generation"
            provider = "partial"
            supports = {"image_to_video": True}
            input_schema = {"type": "object", "properties": {}}

            def is_operation_available(self, operation):
                return operation == "text_to_video"

            def execute(self, inputs):
                raise AssertionError("not used")

        selector = VideoSelector()
        candidates = [PartialVideoTool()]

        assert selector._filter_candidates(
            {"operation": "image_to_video"}, candidates
        ) == []

    def test_video_selector_rank_uses_target_operation_for_readiness(self):
        class PartialVideoTool(BaseTool):
            name = "partial_video"
            capability = "video_generation"
            provider = "partial"
            supports = {"image_to_video": True}
            input_schema = {"type": "object", "properties": {}}

            def is_operation_available(self, operation):
                return operation == "text_to_video"

            def execute(self, inputs):
                raise AssertionError("not used")

        selector = VideoSelector()
        candidates = [PartialVideoTool()]
        rank_inputs = selector._rank_inputs({
            "operation": "rank",
            "target_operation": "image_to_video",
        })

        assert rank_inputs["operation"] == "image_to_video"
        assert selector._filter_candidates(rank_inputs, candidates) == []


# ------------------------------------------------------------------
# Custom-workflow selector eligibility
# ------------------------------------------------------------------

class _DegradedComfyVideo(BaseTool):
    """Server reachable, but bundled MiniMax H3 models missing -> DEGRADED,
    no operation ready. Stands in for comfyui_video on a low-VRAM box."""

    name = "comfyui_video"
    capability = "video_generation"
    provider = "comfyui"
    supports = {"custom_workflow": True, "image_to_video": True}
    input_schema = {"type": "object", "properties": {"workflow_json": {"type": "string"}}}

    def get_status(self):
        return ToolStatus.DEGRADED

    def is_operation_available(self, operation):
        return False

    def execute(self, inputs):
        raise AssertionError("not used")


class _DegradedComfyImage(BaseTool):
    name = "comfyui_image"
    capability = "image_generation"
    provider = "comfyui"
    supports = {"custom_workflow": True}
    input_schema = {"type": "object", "properties": {"workflow_json": {"type": "string"}}}

    def get_status(self):
        return ToolStatus.DEGRADED

    def execute(self, inputs):
        raise AssertionError("not used")


class TestCustomWorkflowSelectorEligibility:

    def test_video_selector_passes_degraded_tool_for_custom_workflow(self):
        selector = VideoSelector()
        candidates = [_DegradedComfyVideo()]
        inputs = {
            "prompt": "x",
            "operation": "text_to_video",
            "workflow_json": "{}",
            "output_node": "14",
        }
        # Without the custom-workflow path this DEGRADED, operation-unready tool
        # would be filtered out; with it, it is eligible and selectable.
        filtered = selector._filter_candidates(inputs, candidates)
        assert [t.name for t in filtered] == ["comfyui_video"]
        assert selector._tool_selectable(candidates[0], inputs) is True

    def test_video_selector_custom_workflow_requires_output_node(self):
        selector = VideoSelector()
        candidates = [_DegradedComfyVideo()]
        inputs = {"prompt": "x", "operation": "text_to_video", "workflow_json": "{}"}
        # output_node missing -> not eligible -> filtered out.
        assert selector._filter_candidates(inputs, candidates) == []
        assert selector._tool_selectable(candidates[0], inputs) is False

    def test_video_selector_custom_workflow_needs_server(self):
        class _OfflineComfyVideo(_DegradedComfyVideo):
            def get_status(self):
                return ToolStatus.UNAVAILABLE

        selector = VideoSelector()
        candidates = [_OfflineComfyVideo()]
        inputs = {
            "prompt": "x",
            "operation": "text_to_video",
            "workflow_json": "{}",
            "output_node": "14",
        }
        assert selector._filter_candidates(inputs, candidates) == []

    def test_image_selector_passes_degraded_tool_for_custom_workflow(self):
        selector = ImageSelector()
        candidates = [_DegradedComfyImage()]
        inputs = {"prompt": "x", "workflow_json": "{}", "output_node": "13"}
        filtered = selector._filter_candidates(inputs, candidates)
        assert [t.name for t in filtered] == ["comfyui_image"]
        assert selector._tool_selectable(candidates[0], inputs) is True

    def test_image_selector_custom_workflow_requires_output_node(self):
        selector = ImageSelector()
        candidates = [_DegradedComfyImage()]
        inputs = {"prompt": "x", "workflow_json": "{}"}
        assert selector._filter_candidates(inputs, candidates) == []
        assert selector._tool_selectable(candidates[0], inputs) is False

    def test_selector_schemas_expose_custom_workflow_inputs(self):
        for selector in (VideoSelector(), ImageSelector()):
            props = selector.input_schema["properties"]
            for field in (
                "workflow_json",
                "workflow_path",
                "output_node",
                "workflow_name",
                "workflow_model",
                "workflow_model_stack",
            ):
                assert field in props, f"{selector.name} missing {field}"
