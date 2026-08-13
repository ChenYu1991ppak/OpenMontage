---
name: comfyui
description: Use when working with ComfyUI workflows in OpenMontage, including comfyui_image/comfyui_video, custom workflow_json/workflow_path inputs, output_node selection, missing model setup, LoRAs, low-VRAM workflow choices, and community workflow imports.
---

# ComfyUI Workflows in OpenMontage

Use this skill before calling `comfyui_image` or `comfyui_video`, and when converting a community ComfyUI workflow into an OpenMontage tool call.

## Server Contract

- ComfyUI must be running before the tool can generate. The default server is `http://localhost:8188`; override it with `COMFYUI_SERVER_URL`.
- Health and hardware status come from `GET /system_stats`.
- Jobs are submitted to `POST /prompt`, completed outputs are read from `GET /history/{prompt_id}`, and artifact bytes are downloaded with `GET /view`.
- Once every artifact of a finished job has been downloaded, the tool deletes that job's record from the server with `POST /history/{prompt_id}` (body `{"delete": [prompt_id]}`) by default. The deletion is best-effort — a failure only surfaces as a warning in `data.server_cleanup.warnings[]`. Set `COMFYUI_KEEP_SERVER_OUTPUTS=1` to keep the server record instead.
- Export workflows with ComfyUI's API-format JSON, not the UI layout format. If a downloaded workflow will not submit, re-export it from ComfyUI with API format enabled.

## Choosing a Workflow

- Use bundled workflows when the requested operation matches and the local machine has the required models and VRAM.
- Use a custom `workflow_json` or `workflow_path` when the user needs a community recipe, a lower-VRAM model, a different style family, or custom nodes.
- For 8GB-12GB GPUs, prefer lower-footprint workflows such as Wan 2.1 1.3B, LTXV FP8 or quantized workflows, or Wan 2.2 GGUF/quantized community workflows. The bundled Wan 2.2 14B FP8 video workflows are a 16GB-class path, not a provider-wide floor.
- Do not promise that arbitrary custom workflows will fit a machine. The workflow, quantization, resolution, frame count, and offload settings determine the real resource envelope.

## Image Editing with Qwen-Image-Edit-2511

`comfyui_image` bundles an instruction-based image editing workflow (Qwen-Image-Edit-2511) alongside its default FLUX 2 text-to-image workflow.

- **Trigger**: pass `image_path` (local file) or `image_url` (remote, downloaded first) together with `prompt`. The edit instruction goes in `prompt` and should be a plain Chinese sentence (e.g. `给人物换一套西装`, `把背景换成纯白色，人物保持完全不动`). With no source image the tool falls back to FLUX text-to-image; a custom `workflow_json`/`workflow_path` still takes precedence.
- **Nodes injected by the tool** (from the bundled `image_qwen_image_edit_2511_api.json`): node `41` = LoadImage (source image), `170:151` = positive edit instruction, `170:149` = negative prompt (left empty), `170:169` = KSampler seed, `9` = SaveImage output. Everything else (UNET/CLIP/VAE loaders, Lightning LoRA, 4-steps switch, tile settings) is fixed by the workflow.
- **Required models** (see `data.missing_models` when incomplete): `qwen_image_edit_2511_fp8mixed.safetensors` → `models/diffusion_models/`, `qwen_2.5_vl_7b_fp8_scaled.safetensors` → `models/text_encoders/`, `qwen_image_vae.safetensors` → `models/vae/`, `Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors` → `models/loras/`. Source: `Comfy-Org/Qwen-Image-Edit_ComfyUI`.
- **Capabilities vs FLUX 2509**: less image drift, stronger multi-person consistency (poses/clothing stay coherent), and Geometry Lock that preserves straight lines, parallel lines, right angles and circle centers — suited to PCB/CAD/mechanical drawings. LoRAs (style transfer: anime, watercolor, film, product photography) work out of the box.
- **Input guidance**: recommend 1024x1024 or 1280x720 source images; avoid 4K inputs. On low VRAM lower `tile_size` from 256 to 192 (saves ~30% VRAM, <15% slower) instead of dropping resolution below 512.
- **Prompt pitfalls**: do not use empty quality words like 高清/超清/8K; write `保持边缘锐利` instead of `不要模糊`; in multi-person scenes address people by position (`左边穿蓝衬衫的人`) rather than ordinal numbers (`第一个人`). If the output looks gray/washed out, check that SaveImage's `embed_workflow` option is off.

## Output Node Contract

- Custom workflows must pass `output_node`.
- Pick the node that writes the artifact, usually `SaveImage`, `SaveVideo`, `VHS_VideoCombine`, or another terminal saver node.
- Pass the node ID as a string, for example `"108"`. Do not pass the class name.
- If a workflow has multiple savers, choose the final deliverable node, not previews or intermediates.

## Templated vs Fixed Nodes

- Identify templated nodes before execution: prompt text, seed, dimensions, frame count, source image, sampler settings, and output filename prefix.
- Fixed nodes are model loaders, VAEs, text encoders, LoRA loaders, schedulers, and graph wiring. Do not mutate those unless the workflow author intended that customization.
- For community workflows, inspect each loader node and note every required model or custom node before running. Missing models should be handled through the tool's structured `missing_models` payload when available.

## Model and LoRA Setup

- Use ComfyUI Manager or the workflow author's model links when available, and respect model licenses.
- Place models in the folders expected by the loader nodes: diffusion models under `ComfyUI/models/diffusion_models/`, text encoders under `ComfyUI/models/text_encoders/`, VAEs under `ComfyUI/models/vae/`, and LoRAs under `ComfyUI/models/loras/`.
- For LoRA stacks, use `LoraLoader` or `LoraLoaderModelOnly` chains in the workflow. Record each LoRA name plus `strength_model` and `strength_clip` when applicable.
- The current ComfyUI tools do not inject LoRAs into arbitrary graphs. To use LoRAs, provide a workflow that already contains the LoRA loader chain and pass model-stack provenance.

## Provenance

- For custom workflows, provide `workflow_name` and `workflow_model` when known.
- Provide `workflow_model_stack` for reproducibility when the workflow is not bundled. Include base checkpoint or diffusion model, quantization, text encoder, VAE, LoRAs and strengths, sampler or scheduler, steps, and guidance if the workflow exposes them.
- The tools record the final workflow hash. Treat that hash plus the model stack, seed, dimensions, and prompt as the reproducibility contract.

## Failure Handling

- If the server is unavailable, surface the structured setup offer. Starting ComfyUI or setting `COMFYUI_SERVER_URL` is the first fix.
- If models are missing, read `data.missing_models[]`; each item should include the file name, role, destination hint, and download URL when OpenMontage knows it.
- If custom nodes are missing, ask the user to install them through ComfyUI Manager or the workflow author's documented install path, then restart ComfyUI.
- If a long render times out locally, check ComfyUI history before retrying from scratch; the server may still have completed the prompt. Note that by default a finished job's history record is deleted after download, so only jobs that failed to download keep a record to inspect. Set `COMFYUI_KEEP_SERVER_OUTPUTS=1` if you want every job's record retained for debugging.
