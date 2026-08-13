"""Contract tests for the IndexTTS25 TTS tool (tools/audio/index_tts.py).

Verifies the BaseTool contract plus workflow trimming, reference-audio
resolution, get_status branching, execute output structure, and the ComfyUI
client's audio output support -- all without a live ComfyUI server (client
methods are mocked).
"""

from pathlib import Path

import pytest

from tools.audio.index_tts import (
    IndexTTSTTS,
    _SAMPLING_FIELDS,
    _WORKFLOW_FILE,
)
from tools.base_tool import (
    BaseTool,
    ToolRuntime,
    ToolStatus,
    ToolTier,
)
from tools._comfyui.client import ComfyUIClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW_DIR = PROJECT_ROOT / "tools" / "_comfyui" / "workflows"


# ------------------------------------------------------------------
# Contract compliance
# ------------------------------------------------------------------


class TestContract:

    def test_inherits_base_tool(self):
        assert issubclass(IndexTTSTTS, BaseTool)

    def test_has_required_identity(self):
        tool = IndexTTSTTS()
        assert tool.name == "index_tts"
        assert tool.version
        assert tool.capability == "tts"
        assert tool.provider == "indextts"
        assert tool.tier == ToolTier.VOICE
        assert tool.runtime == ToolRuntime.LOCAL_GPU

    def test_has_input_schema(self):
        tool = IndexTTSTTS()
        schema = tool.input_schema
        assert schema.get("type") == "object"
        assert "text" in schema.get("properties", {})
        assert "reference_audio" in schema.get("properties", {})
        assert {"text", "reference_audio"} <= set(schema.get("required", []))

    def test_has_capabilities(self):
        tool = IndexTTSTTS()
        assert len(tool.capabilities) > 0
        assert "voice_cloning" in tool.capabilities
        assert "emotion_text" in tool.capabilities

    def test_has_agent_skills(self):
        tool = IndexTTSTTS()
        assert tool.agent_skills == ["index-tts"]

    def test_layer3_skill_exists(self):
        skill = PROJECT_ROOT / ".agents" / "skills" / "index-tts" / "SKILL.md"
        assert skill.exists()
        text = skill.read_text(encoding="utf-8")
        assert "IndexTTS" in text
        assert "emotion_text" in text
        assert "reference_audio" in text

    def test_has_fallbacks(self):
        tool = IndexTTSTTS()
        assert tool.fallback_tools

    def test_cost_is_zero(self):
        assert IndexTTSTTS().estimate_cost({"text": "hi"}) == 0.0

    def test_runtime_estimate_positive(self):
        assert IndexTTSTTS().estimate_runtime({"text": "hi"}) > 0

    def test_get_info_returns_dict(self):
        tool = IndexTTSTTS()
        info = tool.get_info()
        assert isinstance(info, dict)
        assert info["name"] == "index_tts"
        assert info["provider"] == "indextts"
        assert info["runtime"] == "local_gpu"
        assert info["setup_offer"]["env_var"] == "COMFYUI_SERVER_URL"

    def test_idempotency_key_fields(self):
        tool = IndexTTSTTS()
        assert "text" in tool.idempotency_key_fields
        assert "reference_audio" in tool.idempotency_key_fields


# ------------------------------------------------------------------
# Workflow trimming
# ------------------------------------------------------------------


class TestWorkflowTrimming:

    def _workflow(self, mode):
        return IndexTTSTTS()._build_workflow(mode)

    def test_basic_keeps_only_expected_nodes(self):
        wf = self._workflow("basic")
        assert set(wf) == {"1", "2", "3", "12"}
        assert wf["2"]["class_type"] == "IndexTTS25TTS"
        assert wf["3"]["class_type"] == "SaveAudioMP3"
        assert wf["1"]["class_type"] == "LoadAudio"
        assert wf["12"]["class_type"] == "IndexTTS25Loader"

    def test_emotion_text_keeps_only_expected_nodes(self):
        wf = self._workflow("emotion_text")
        assert set(wf) == {"1", "6", "7", "12"}
        assert wf["6"]["class_type"] == "IndexTTS25TTSEmotionText"
        assert wf["7"]["class_type"] == "SaveAudioMP3"

    def test_no_emotion_vector_or_audio_nodes_left(self):
        for mode in ("basic", "emotion_text"):
            classes = {n["class_type"] for n in self._workflow(mode).values()}
            assert "IndexTTS25TTSEmotionVector" not in classes
            assert "IndexTTS25TTSEmotionAudio" not in classes

    def test_workflow_file_has_all_templated_nodes(self):
        import json

        with open(WORKFLOW_DIR / _WORKFLOW_FILE, encoding="utf-8") as f:
            w = json.load(f)
        assert w["1"]["class_type"] == "LoadAudio"
        assert w["2"]["class_type"] == "IndexTTS25TTS"
        assert w["6"]["class_type"] == "IndexTTS25TTSEmotionText"
        assert w["12"]["class_type"] == "IndexTTS25Loader"
        assert w["2"]["inputs"]["reference_audio"] == ["1", 0]


# ------------------------------------------------------------------
# Reference audio resolution
# ------------------------------------------------------------------


def _ready_tool(seen=None):
    tool = IndexTTSTTS()
    tool._client.is_available = lambda: True
    tool._node_installed = lambda: True
    seen = {} if seen is None else seen
    tool._client.upload_audio = lambda local_path, name: (
        seen.setdefault("uploads", []).append(name),
        seen.setdefault("paths", []).append(str(local_path)),
    ) and "server_ref.wav"
    def fake_generate(workflow, output_node, dest, **kwargs):
        Path(dest).write_bytes(b"ID3")
        seen.update({"workflow": workflow, "output_node": output_node})
        return [Path(dest)]

    tool._client.generate = fake_generate
    return tool, seen


class TestReferenceAudio:

    def test_local_path_uploaded(self, tmp_path):
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"RIFF")
        tool, seen = _ready_tool()
        out = tmp_path / "out.mp3"
        result = tool.execute(
            {"text": "hello", "reference_audio": str(ref), "output_path": str(out)}
        )
        assert result.success is True
        assert seen["uploads"] == ["ref.wav"]
        assert seen["workflow"]["1"]["inputs"]["audio"] == "server_ref.wav"

    def test_preset_resolved_in_preset_dir(self, tmp_path, monkeypatch):
        preset_dir = tmp_path / "voices"
        preset_dir.mkdir()
        (preset_dir / "narrator.wav").write_bytes(b"RIFF")
        monkeypatch.setenv("INDEXTTS_REFERENCE_DIR", str(preset_dir))
        tool, seen = _ready_tool()
        result = tool.execute(
            {"text": "hi", "reference_audio": "narrator", "output_path": str(tmp_path / "out.mp3")}
        )
        assert result.success is True
        assert seen["paths"] == [str(preset_dir / "narrator.wav")]

    def test_missing_reference_fails(self):
        tool = IndexTTSTTS()
        tool._client.is_available = lambda: True
        tool._node_installed = lambda: True
        result = tool.execute({"text": "hi", "reference_audio": "nope-not-there"})
        assert result.success is False
        assert "Reference audio" in result.error


# ------------------------------------------------------------------
# get_status branching
# ------------------------------------------------------------------


class TestGetStatus:

    def test_available(self):
        tool = IndexTTSTTS()
        tool._client.is_available = lambda: True
        tool._node_installed = lambda: True
        assert tool.get_status() == ToolStatus.AVAILABLE

    def test_unavailable_when_server_down(self):
        tool = IndexTTSTTS()
        tool._client.is_available = lambda: False
        assert tool.get_status() == ToolStatus.UNAVAILABLE

    def test_unavailable_when_node_missing(self):
        tool = IndexTTSTTS()
        tool._client.is_available = lambda: True
        tool._node_installed = lambda: False
        assert tool.get_status() == ToolStatus.UNAVAILABLE


# ------------------------------------------------------------------
# execute output
# ------------------------------------------------------------------


class TestExecute:

    def test_basic_execute_result_structure(self, tmp_path):
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"RIFF")
        tool, seen = _ready_tool()
        out = tmp_path / "out.mp3"
        result = tool.execute(
            {"text": "你好", "reference_audio": str(ref), "output_path": str(out)}
        )
        assert result.success is True
        assert result.model == "IndexTTS-2.5"
        assert result.data["provider"] == "indextts"
        assert result.data["mode"] == "basic"
        assert result.data["format"] == "mp3"
        assert result.data["output"] == str(out)
        assert result.artifacts == [str(out)]
        assert seen["output_node"] == "3"
        provenance = result.data["workflow_provenance"]
        assert provenance["source"] == "bundled"
        assert provenance["workflow"] == _WORKFLOW_FILE
        assert provenance["output_node"] == "3"
        assert provenance["workflow_hash_sha256"]
        assert out.exists()

    def test_emotion_text_patch_injects_fields(self, tmp_path):
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"RIFF")
        tool, seen = _ready_tool()
        result = tool.execute(
            {
                "text": "太开心了",
                "mode": "emotion_text",
                "reference_audio": str(ref),
                "emotion_text": "兴高采烈",
                "use_main_text": True,
                "emo_alpha": 0.6,
                "output_path": str(tmp_path / "out.mp3"),
            }
        )
        assert result.success is True
        assert seen["output_node"] == "7"
        tts = seen["workflow"]["6"]["inputs"]
        assert tts["emotion_text"] == "兴高采烈"
        assert tts["use_main_text"] is True
        assert tts["emo_alpha"] == 0.6
        assert tts["lang"] == "ZH"
        assert tts["reference_audio"] == ["1", 0]

    def test_sampling_defaults_inject_sane_values(self, tmp_path):
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"RIFF")
        tool, seen = _ready_tool()
        result = tool.execute(
            {"text": "hi", "reference_audio": str(ref), "output_path": str(tmp_path / "out.mp3")}
        )
        assert result.success is True
        tts = seen["workflow"]["2"]["inputs"]
        for field in _SAMPLING_FIELDS:
            assert field in tts, f"missing sampling field {field}"
        # Placeholder values from the raw workflow must be overwritten.
        assert tts["temperature"] != 30
        assert tts["top_p"] != 3
        assert tts["do_sample"] is True

    def test_output_suffix_forced_to_mp3(self, tmp_path):
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"RIFF")
        tool, seen = _ready_tool()
        result = tool.execute(
            {"text": "hi", "reference_audio": str(ref), "output_path": str(tmp_path / "out.wav")}
        )
        assert result.success is True
        assert result.data["output"].endswith(".mp3")

    def test_validation_errors(self, tmp_path):
        tool = IndexTTSTTS()
        tool._client.is_available = lambda: True
        tool._node_installed = lambda: True
        # missing text
        r = tool.execute({"reference_audio": "/tmp/x.wav"})
        assert r.success is False and "text" in r.error
        # missing reference
        r = tool.execute({"text": "hi"})
        assert r.success is False and "reference_audio" in r.error
        # bad mode
        r = tool.execute({"text": "hi", "mode": "emo_vector", "reference_audio": "/tmp/x.wav"})
        assert r.success is False and "mode" in r.error
        # bad lang
        ref = tmp_path / "r.wav"
        ref.write_bytes(b"R")
        r = tool.execute({"text": "hi", "lang": "XX", "reference_audio": str(ref)})
        assert r.success is False and "lang" in r.error

    def test_server_unavailable_reports_reason(self):
        tool = IndexTTSTTS()
        tool._client.is_available = lambda: False
        tool._client.unavailable_reason = lambda: "no server"
        result = tool.execute({"text": "hi", "reference_audio": "/tmp/x.wav"})
        assert result.success is False
        assert "no server" in result.error


# ------------------------------------------------------------------
# ComfyUI client audio support (regression guard)
# ------------------------------------------------------------------


class TestClientAudioSupport:

    def test_generate_downloads_audio_outputs(self, monkeypatch, tmp_path):
        """generate() must handle SaveAudioMP3's "audio" output key."""
        client = ComfyUIClient("http://comfy.test")
        monkeypatch.setattr(client, "submit", lambda workflow: "p-1")
        monkeypatch.setattr(
            client,
            "poll",
            lambda prompt_id, **kwargs: {
                "outputs": {
                    "3": {
                        "audio": [{
                            "filename": "clip.mp3",
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
        dest = tmp_path / "clip.mp3"
        paths = client.generate({}, "3", dest, cleanup_history=False)
        assert paths == [dest]

    def test_upload_audio_posts_to_upload_audio_endpoint(self, monkeypatch, tmp_path):
        seen = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"name": "ref.wav"}

            def raise_for_status(self):
                return None

        def fake_post(url, files=None, timeout=60):
            seen["url"] = url
            seen["files"] = files
            return FakeResponse()

        monkeypatch.setattr("tools._comfyui.client.requests.post", fake_post)

        audio = tmp_path / "ref.wav"
        audio.write_bytes(b"RIFF")
        client = ComfyUIClient("http://comfy.test")
        name = client.upload_audio(audio, "ref.wav")
        assert name == "ref.wav"
        assert seen["url"] == "http://comfy.test/upload/audio"
        assert "audio" in seen["files"]
