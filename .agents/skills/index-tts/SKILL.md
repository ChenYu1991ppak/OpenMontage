---
name: index-tts
description: Zero-shot voice cloning TTS via IndexTTS25 on a local ComfyUI server. Use when the user wants narration in a specific cloned voice, needs local/privacy-safe TTS with no API cost, wants emotion-controlled speech (emotion_text), or speaks Chinese/English/Japanese/Spanish/Arabic (ZH/EN/JA/ES/AR). Requires a running ComfyUI server with the IndexTTS25 custom node.
---

# IndexTTS25 TTS

Local zero-shot voice cloning TTS (bilibili IndexTeam, 0.8B). Runs the official
`IndexTTS25_官方工作流_api.json` ComfyUI workflow through the `index_tts` tool.

Requires:
- A running ComfyUI server (`COMFYUI_SERVER_URL`, default `http://localhost:8188`)
  with the IndexTTS25 custom node installed.
- The IndexTTS25 model service reachable from the ComfyUI host
  (`INDEXTTS_HOST:INDEXTTS_PORT`, default `127.0.0.1:8108`), using
  `INDEXTTS_MODEL_DIR` and `INDEXTTS_VENV`.
- No API key or cloud cost: everything runs locally on the ComfyUI host's GPU.

## OpenMontage Usage

Generate with the TTS selector:

```python
from tools.audio.tts_selector import TTSSelector

result = TTSSelector().execute({
    "preferred_provider": "indextts",
    "text": "今天收到录取通知，太开心了！",
    "reference_audio": "assets/voices/index_tts/female_calm.wav",  # or preset name
    "output_path": "projects/my-video/assets/audio/narration.mp3",
})
```

Or call the provider directly (recommended when you need full control):

```python
from tools.audio.index_tts import IndexTTSTTS

result = IndexTTSTTS().execute({
    "text": "用冷静而坚定的语气，讲述接下来发生的事情。",
    "mode": "emotion_text",
    "reference_audio": "assets/voices/index_tts/female_calm.wav",
    "lang": "ZH",
    "emotion_text": "平静自然，略带坚定",
    "emo_alpha": 0.7,
    "output_path": "projects/my-video/assets/audio/narration.mp3",
})
```

The tool writes the mp3 to `output_path` (suffix forced to `.mp3`).

## Parameters

- `mode`: `basic` (default) or `emotion_text`. `basic` clones the reference
  voice; `emotion_text` adds `emotion_text` + `emo_alpha` emotional control.
- `text` (required): text to synthesize. Supports inline pronunciation hints
  (see "Pronunciation control").
- `reference_audio` (required): local wav/mp3 path, or a preset name resolved
  under `INDEXTTS_REFERENCE_DIR` (default `assets/voices/index_tts/`).
- `lang`: `ZH` | `EN` | `JA` | `ES` | `AR` (default `ZH`). Must match the
  language of `text`.
- `duration_factor`: speaking speed, 0.5–2.0. Lower = faster, higher = slower.
  Default 1.0. The bundled workflow ships 0.8 (slightly slower); tune per project.
- `emotion_text` (emotion_text mode): Chinese emotion description, e.g.
  `"兴高采烈，欢呼雀跃"`, `"平静自然"`, `"悲伤低沉"`.
- `use_main_text` (emotion_text mode, default False): when True the emotion
  text is prepended to the main text for emotion extraction.
- `emo_alpha`: emotion intensity 0–1 (default 1.0).
- Sampling (optional, sane defaults injected automatically — do NOT copy
  placeholder values from the raw workflow JSON): `do_sample`, `temperature`,
  `top_p`, `top_k`, `num_beams`, `repetition_penalty`, `max_mel_tokens`,
  `interval_silence`.

## Recommended Workflow

1. Pick or record a 5–15 s clean reference clip with no background music,
   reverb, or overlapping speech; the clearer the reference, the closer the clone.
2. Generate a 10–15 s sample first, verify voice match, pace, and emotion.
3. Generate the full narration only after approval.
4. For emotion control, describe the emotion in Chinese — short vivid phrases
   work better than long sentences, e.g. `"温柔舒缓"` over `"用非常温柔和缓的
   语调说"`.
5. Prefer `duration_factor` for pace instead of editing audio later.

## Pronunciation Control

IndexTTS25 accepts inline pronunciation hints to fix rare-word readings:

- Pinyin for Chinese: `<拼音|PINYIN>` e.g. `<重庆|Chóng Qìng>`
- CMU phonemes for English words: `<单词|CMU音素>`
- Kana for Japanese kanji: `<汉字|假名>`

## Language Notes

- One model covers ZH/EN/JA/ES/AR. Set `lang` to match `text`; mixing languages
  in one utterance generally works best when the primary language matches.
- For Chinese narration, keep punctuation natural; long uninterrupted runs can
  produce monotone pacing.

## Troubleshooting

- `get_status()` returns UNAVAILABLE: check `COMFYUI_SERVER_URL`, whether
  ComfyUI is running, and whether the IndexTTS25 custom node is installed
  (`/object_info/IndexTTS25TTS` must return a definition).
- `Reference audio ... not found`: pass an absolute/relative local path, or put
  the preset file under `INDEXTTS_REFERENCE_DIR`.
- Slow first request: the loader node starts the 8108 model service on first
  use; subsequent calls reuse it. Keep the ComfyUI server alive between calls.
- Server-side HTTP errors: check the ComfyUI host's logs and confirm the model
  service on `INDEXTTS_PORT` is reachable from the ComfyUI host, not just your
  machine.

## Safety

Never write reference audio content into logs or metadata. Reference clips are
user-provided; do not print, transcribe, or redistribute them.
