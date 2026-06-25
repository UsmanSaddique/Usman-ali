"""
ComfyUI API Client — submit workflows, monitor completion, retrieve outputs.
"""
import json
import time
import uuid
import shutil
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

COMFYUI_BASE = "http://127.0.0.1:8188"
COMFYUI_OUTPUT = Path(
    r"C:\ComfyUI_windows_portable_nvidia_cu126"
    r"\ComfyUI_windows_portable\ComfyUI\output"
)


class ComfyUIClient:
    def __init__(self, base_url: str = COMFYUI_BASE):
        self.base_url = base_url.rstrip("/")

    def ping(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/system_stats")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def wait_ready(self, timeout: float = 60.0, poll: float = 2.0) -> bool:
        """Block until ComfyUI answers, up to `timeout`. Handles the case where
        ComfyUI is still reloading its CUDA context after the LLM released VRAM."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.ping():
                return True
            time.sleep(poll)
        return False

    def submit(self, workflow: dict) -> str:
        client_id = uuid.uuid4().hex[:12]
        payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            error = result.get("error") or result.get("node_errors") or result
            raise RuntimeError(f"ComfyUI rejected workflow: {error}")
        logger.info(f"[ComfyUI] Submitted prompt {prompt_id}")
        return prompt_id

    def get_history(self, prompt_id: str) -> Optional[dict]:
        try:
            url = f"{self.base_url}/history/{prompt_id}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return data.get(prompt_id)
        except Exception:
            return None

    def wait_for_completion(
        self, prompt_id: str, timeout: int = 600, poll: float = 2.0
    ) -> dict:
        t0 = time.time()
        last_log_time = 0
        while time.time() - t0 < timeout:
            hist = self.get_history(prompt_id)
            if hist:
                status = hist.get("status", {})
                if status.get("completed"):
                    return hist
                status_str = status.get("status_str", "")
                if status_str == "error":
                    msgs = hist.get("status", {}).get("messages", [])
                    raise RuntimeError(f"ComfyUI error: {msgs}")
            
            # Log queue status periodically
            now = time.time()
            if now - last_log_time > 10.0:
                q_status = self.get_queue_status()
                queue_running = q_status.get("queue_running", [])
                queue_pending = q_status.get("queue_pending", [])
                
                is_running = any(item[1] == prompt_id for item in queue_running)
                is_pending = any(item[1] == prompt_id for item in queue_pending)
                
                if is_running:
                    logger.info(f"[ComfyUI] Generating video... (currently running)")
                elif is_pending:
                    logger.info(f"[ComfyUI] Waiting in queue... ({len(queue_pending)} pending)")
                else:
                    logger.info(f"[ComfyUI] Processing request...")
                last_log_time = now

            time.sleep(poll)
        raise TimeoutError(f"ComfyUI timed out after {timeout}s")

    def collect_output(self, history: dict, dest_path: str) -> str:
        """Find the video output from history and copy it to dest_path."""
        outputs = history.get("outputs", {})
        for node_id, node_out in outputs.items():
            for key in ("gifs", "images", "videos", "audio"):
                for entry in node_out.get(key, []):
                    fname = entry.get("filename", "")
                    subfolder = entry.get("subfolder", "")
                    if not fname:
                        continue
                    src = COMFYUI_OUTPUT / subfolder / fname if subfolder else COMFYUI_OUTPUT / fname
                    if src.exists():
                        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(src), dest_path)
                        logger.info(f"[ComfyUI] Copied output {src} -> {dest_path}")
                        return dest_path
        raise FileNotFoundError(
            f"No output file found in ComfyUI history. Outputs: {outputs}"
        )

    def free_vram(self) -> bool:
        """Ask ComfyUI to unload its models and free GPU memory.
        Critical on a single 16GB GPU: call before loading the LLM so the
        director model gets the whole card (avoids CPU-offload slowdown)."""
        try:
            payload = json.dumps({"unload_models": True, "free_memory": True}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/free", data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info("[ComfyUI] Requested VRAM free (unload models)")
                return resp.status == 200
        except Exception as e:
            logger.debug(f"[ComfyUI] free_vram failed (non-fatal): {e}")
            return False

    def get_queue_status(self) -> dict:
        try:
            req = urllib.request.Request(f"{self.base_url}/queue")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception:
            return {}


# ── Workflow Builders ─────────────────────────────────────────────────────


def detect_family(model_filename: str) -> str:
    fn = model_filename.lower()
    if "ltx" in fn:
        return "ltx"
    if "wan" in fn:
        return "wan"
    return "unknown"


def build_ltx_workflow(
    model_filename: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 97,
    steps: int = 8,
    cfg: float = 1.0,
    seed: int = 42,
    fps: int = 24,
    loras: list[tuple[str, float]] = None,
    output_prefix: str = "ai_director",
) -> dict:
    """Build a ComfyUI API-format workflow for LTX GGUF models."""
    is_gguf = model_filename.lower().endswith(".gguf")

    wf = {}
    next_id = 1

    def nid():
        nonlocal next_id
        s = str(next_id)
        next_id += 1
        return s

    # 1. Load diffusion model
    model_node = nid()
    if is_gguf:
        wf[model_node] = {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": model_filename},
        }
    else:
        wf[model_node] = {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": model_filename,
                "weight_dtype": "default",
            },
        }

    # 2. Load text encoders (DualCLIPLoader for LTX)
    clip_node = nid()
    wf[clip_node] = {
        "class_type": "DualCLIPLoader",
        "inputs": {
            "clip_name1": "gemma_3_12B_it_fp4_mixed.safetensors",
            "clip_name2": "ltx-2.3_text_projection_bf16.safetensors",
            "type": "ltxv",
        },
    }

    # 3. Load VAE
    vae_node = nid()
    wf[vae_node] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "LTX23_video_vae_bf16.safetensors"},
    }

    # Track current model/clip refs for LoRA chaining
    cur_model = [model_node, 0]
    cur_clip = [clip_node, 0]

    # 3b. Apply LoRAs (if any)
    if loras:
        for lora_file, weight in loras:
            lora_node = nid()
            wf[lora_node] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": cur_model,
                    "clip": cur_clip,
                    "lora_name": lora_file,
                    "strength_model": weight,
                    "strength_clip": weight,
                },
            }
            cur_model = [lora_node, 0]
            cur_clip = [lora_node, 1]

    # 4. Encode positive prompt
    pos_node = nid()
    wf[pos_node] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": prompt, "clip": cur_clip},
    }

    # 5. Encode negative prompt
    neg_node = nid()
    wf[neg_node] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative_prompt or "", "clip": cur_clip},
    }

    # 6. Empty video latent
    latent_node = nid()
    wf[latent_node] = {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {
            "width": width,
            "height": height,
            "length": num_frames,
            "batch_size": 1,
        },
    }

    # 7. KSampler
    sampler_node = nid()
    wf[sampler_node] = {
        "class_type": "KSampler",
        "inputs": {
            "model": cur_model,
            "positive": [pos_node, 0],
            "negative": [neg_node, 0],
            "latent_image": [latent_node, 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
        },
    }

    # 8. VAE Decode
    decode_node = nid()
    wf[decode_node] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": [sampler_node, 0],
            "vae": [vae_node, 0],
        },
    }

    # 9. Save video (VHS_VideoCombine)
    save_node = nid()
    wf[save_node] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": [decode_node, 0],
            "frame_rate": fps,
            "loop_count": 0,
            "filename_prefix": output_prefix,
            "format": "video/h264-mp4",
            "save_output": True,
            "pingpong": False,
        },
    }

    return wf


def build_ltx_img2vid_workflow(
    model_filename: str,
    image_filename: str,          # file already placed in ComfyUI/input/
    prompt: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 97,
    steps: int = 8,
    cfg: float = 1.0,
    seed: int = 42,
    fps: int = 24,
    strength: float = 0.5,   # LOWER = MORE motion. 1.0 glued the clip to the still (looked static);
                             # 0.5 gives strong visible movement while keeping the character's look (measured 22 vs 3.4).
    loras: list[tuple[str, float]] = None,
    output_prefix: str = "ai_director_i2v",
) -> dict:
    """LTX image-to-video: animate a still (an SDXL character frame) so the
    character/scene actually MOVES while keeping the still's look. Consistent
    character WITH motion (vs Ken Burns, which only pans a static image)."""
    is_gguf = model_filename.lower().endswith(".gguf")
    wf = {}
    n = [0]
    def nid():
        n[0] += 1; return str(n[0])

    model_node = nid()
    wf[model_node] = ({"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": model_filename}}
                      if is_gguf else
                      {"class_type": "UNETLoader", "inputs": {"unet_name": model_filename, "weight_dtype": "default"}})
    clip_node = nid()
    wf[clip_node] = {"class_type": "DualCLIPLoader", "inputs": {
        "clip_name1": "gemma_3_12B_it_fp4_mixed.safetensors",
        "clip_name2": "ltx-2.3_text_projection_bf16.safetensors", "type": "ltxv"}}
    vae_node = nid()
    wf[vae_node] = {"class_type": "VAELoader", "inputs": {"vae_name": "LTX23_video_vae_bf16.safetensors"}}

    cur_model = [model_node, 0]; cur_clip = [clip_node, 0]
    if loras:
        for lora_file, weight in loras:
            ln = nid()
            wf[ln] = {"class_type": "LoraLoader", "inputs": {
                "model": cur_model, "clip": cur_clip, "lora_name": lora_file,
                "strength_model": weight, "strength_clip": weight}}
            cur_model = [ln, 0]; cur_clip = [ln, 1]

    pos = nid(); wf[pos] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": cur_clip}}
    neg = nid(); wf[neg] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt or "", "clip": cur_clip}}
    img = nid(); wf[img] = {"class_type": "LoadImage", "inputs": {"image": image_filename}}

    i2v = nid()
    wf[i2v] = {"class_type": "LTXVImgToVideo", "inputs": {
        "positive": [pos, 0], "negative": [neg, 0], "vae": [vae_node, 0],
        "image": [img, 0], "width": width, "height": height,
        "length": num_frames, "batch_size": 1, "strength": strength}}

    samp = nid()
    wf[samp] = {"class_type": "KSampler", "inputs": {
        "model": cur_model, "positive": [i2v, 0], "negative": [i2v, 1],
        "latent_image": [i2v, 2], "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}}
    dec = nid(); wf[dec] = {"class_type": "VAEDecode", "inputs": {"samples": [samp, 0], "vae": [vae_node, 0]}}
    save = nid(); wf[save] = {"class_type": "VHS_VideoCombine", "inputs": {
        "images": [dec, 0], "frame_rate": fps, "loop_count": 0,
        "filename_prefix": output_prefix, "format": "video/h264-mp4",
        "save_output": True, "pingpong": False}}
    return wf


def build_wan_workflow(
    model_filename: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
    steps: int = 30,
    cfg: float = 5.0,
    seed: int = 42,
    fps: int = 16,
    loras: list[tuple[str, float]] = None,
    output_prefix: str = "ai_director",
) -> dict:
    """Build a ComfyUI API-format workflow for Wan 2.2 models."""
    # Determine weight dtype from filename
    weight_dtype = "default"
    fn_lower = model_filename.lower()
    if "fp8" in fn_lower:
        weight_dtype = "fp8_e4m3fn"
    elif "fp16" in fn_lower:
        weight_dtype = "default"

    wf = {}
    next_id = 1

    def nid():
        nonlocal next_id
        s = str(next_id)
        next_id += 1
        return s

    # 1. Load model
    model_node = nid()
    wf[model_node] = {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": model_filename,
            "weight_dtype": weight_dtype,
        },
    }

    # 2. Load text encoder (UMT5 for Wan)
    clip_node = nid()
    wf[clip_node] = {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan",
        },
    }

    # 3. Load VAE
    vae_node = nid()
    wf[vae_node] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "wan2.2_vae.safetensors"},
    }

    cur_model = [model_node, 0]
    cur_clip = [clip_node, 0]

    # 3b. Apply LoRAs
    if loras:
        for lora_file, weight in loras:
            lora_node = nid()
            wf[lora_node] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": cur_model,
                    "clip": cur_clip,
                    "lora_name": lora_file,
                    "strength_model": weight,
                    "strength_clip": weight,
                },
            }
            cur_model = [lora_node, 0]
            cur_clip = [lora_node, 1]

    # 4. Positive prompt
    pos_node = nid()
    wf[pos_node] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": prompt, "clip": cur_clip},
    }

    # 5. Negative prompt
    neg_node = nid()
    wf[neg_node] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative_prompt or "blurry, low quality", "clip": cur_clip},
    }

    # 6. Empty latent for Wan video
    latent_node = nid()
    wf[latent_node] = {
        "class_type": "EmptySD3LatentImage",
        "inputs": {
            "width": width,
            "height": height,
            "batch_size": num_frames,
        },
    }

    # 7. KSampler
    sampler_node = nid()
    wf[sampler_node] = {
        "class_type": "KSampler",
        "inputs": {
            "model": cur_model,
            "positive": [pos_node, 0],
            "negative": [neg_node, 0],
            "latent_image": [latent_node, 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
        },
    }

    # 8. VAE Decode
    decode_node = nid()
    wf[decode_node] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": [sampler_node, 0],
            "vae": [vae_node, 0],
        },
    }

    # 9. Save video
    save_node = nid()
    wf[save_node] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": [decode_node, 0],
            "frame_rate": fps,
            "loop_count": 0,
            "filename_prefix": output_prefix,
            "format": "video/h264-mp4",
            "save_output": True,
            "pingpong": False,
        },
    }

    return wf


def build_wan_workflow(
    model_filename: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
    steps: int = 30,
    cfg: float = 5.0,
    seed: int = 42,
    fps: int = 16,
    loras: list[tuple[str, float]] = None,
    output_prefix: str = "ai_director",
) -> dict:
    wf = {}
    next_id = 1
    def nid():
        nonlocal next_id
        s = str(next_id)
        next_id += 1
        return s

    model_node = nid()
    wf[model_node] = {"class_type": "UNETLoader", "inputs": {"unet_name": model_filename, "weight_dtype": "default"}}

    clip_node = nid()
    wf[clip_node] = {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "t5xxl_fp16.safetensors", "clip_name2": "clip_l.safetensors", "type": "wan"}}

    vae_node = nid()
    wf[vae_node] = {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}}

    cur_model = [model_node, 0]
    cur_clip = [clip_node, 0]

    pos_node = nid()
    wf[pos_node] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": cur_clip}}

    neg_node = nid()
    wf[neg_node] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": cur_clip}}

    latent_node = nid()
    wf[latent_node] = {"class_type": "EmptyWanVLatentVideo", "inputs": {"width": width, "height": height, "length": num_frames, "batch_size": 1}}

    sampler_node = nid()
    wf[sampler_node] = {"class_type": "KSampler", "inputs": {"model": cur_model, "positive": [pos_node, 0], "negative": [neg_node, 0], "latent_image": [latent_node, 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}}

    decode_node = nid()
    wf[decode_node] = {"class_type": "VAEDecode", "inputs": {"samples": [sampler_node, 0], "vae": [vae_node, 0]}}

    save_node = nid()
    wf[save_node] = {"class_type": "VHS_VideoCombine", "inputs": {"images": [decode_node, 0], "frame_rate": fps, "loop_count": 0, "filename_prefix": output_prefix, "format": "video/h264-mp4", "save_output": True, "pingpong": False}}

    return wf


def build_sdxl_workflow(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    cfg: float = 7.0,
    seed: int = 42,
    ckpt_name: str = "sd_xl_base_1.0.safetensors",
    output_prefix: str = "ai_director_img",
    loras: list[tuple[str, float]] = None,
    hires: bool = False,
    hires_scale: float = 1.5,
    hires_denoise: float = 0.45,
    hires_steps: int = 0,
) -> dict:
    """ComfyUI API workflow for SDXL still-image generation.
    Used instead of diffusers (which breaks with transformers 5.9).
    `loras`: optional [(filename, weight)] — e.g. a Pixar-style LoRA and/or a
    trained character LoRA, chained onto the model+clip.

    `hires=True` enables a two-pass "hires-fix": render the base latent, upscale
    it `hires_scale`x, then run a SECOND low-denoise KSampler pass that adds real
    facial/hand/texture detail (the single biggest still-quality lever on SDXL
    base — turns soft, mushy faces into clean ones). `hires_denoise` ~0.4-0.5 keeps
    composition while sharpening; higher invents more detail (and more drift)."""
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
    }
    cur_model = ["1", 0]
    cur_clip = ["1", 1]
    nid = 10
    for lora_file, weight in (loras or []):
        node = str(nid); nid += 1
        wf[node] = {"class_type": "LoraLoader", "inputs": {
            "model": cur_model, "clip": cur_clip, "lora_name": lora_file,
            "strength_model": weight, "strength_clip": weight}}
        cur_model = [node, 0]; cur_clip = [node, 1]
    wf.update({
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": cur_clip}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": cur_clip}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {
            "width": width, "height": height, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": cur_model, "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["4", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
    })
    final_latent = ["5", 0]
    if hires:
        # Pass 2: upscale the latent and refine — adds genuine detail to faces/hands.
        wf["8"] = {"class_type": "LatentUpscaleBy", "inputs": {
            "samples": ["5", 0], "upscale_method": "nearest-exact", "scale_by": hires_scale}}
        wf["9"] = {"class_type": "KSampler", "inputs": {
            "model": cur_model, "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["8", 0], "seed": seed,
            "steps": hires_steps or steps, "cfg": cfg,
            "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": hires_denoise}}
        final_latent = ["9", 0]
    wf["6"] = {"class_type": "VAEDecode", "inputs": {"samples": final_latent, "vae": ["1", 2]}}
    wf["7"] = {"class_type": "SaveImage", "inputs": {
        "images": ["6", 0], "filename_prefix": output_prefix}}
    return wf


def build_esrgan_video_upscale_workflow(
    video_filename: str,         # file already in ComfyUI/input/
    target_width: int = 1920,
    target_height: int = 1080,
    fps: float = 24.0,
    model_name: str = "4x-UltraSharp.pth",
    output_prefix: str = "ai_director_hd",
    frame_load_cap: int = 0,     # 0 = all; set small (e.g. 16) to chunk and avoid OOM
    skip_first_frames: int = 0,
) -> dict:
    """Real AI upscale of a video clip via ComfyUI: load frames -> 4x ESRGAN
    (adds genuine detail/sharpness) -> downscale to target (crisp) -> recombine.
    Far better than lanczos+sharpen (which just smears soft input).
    Process in frame chunks (frame_load_cap) so the 4x batch doesn't OOM 16GB."""
    return {
        "1": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": video_filename, "force_rate": 0, "custom_width": 0, "custom_height": 0,
            "frame_load_cap": frame_load_cap, "skip_first_frames": skip_first_frames,
            "select_every_nth": 1}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {
            "upscale_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "ImageScale", "inputs": {
            "image": ["3", 0], "upscale_method": "lanczos",
            "width": target_width, "height": target_height, "crop": "disabled"}},
        "5": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["4", 0], "frame_rate": fps, "loop_count": 0,
            "filename_prefix": output_prefix, "format": "video/h264-mp4",
            "pingpong": False, "save_output": True}},
    }


def build_wan22_ti2v_workflow(
    model_filename: str = "wan2.2_ti2v_5B_fp16.safetensors",
    prompt: str = "",
    negative_prompt: str = "",
    width: int = 832,
    height: int = 480,
    num_frames: int = 97,
    steps: int = 24,
    cfg: float = 5.0,
    seed: int = 42,
    fps: int = 24,
    clip_name: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    vae_name: str = "wan2.2_vae.safetensors",
    loras: list[tuple[str, float]] = None,
    output_prefix: str = "ai_director_wan",
) -> dict:
    """Correct ComfyUI API workflow for Wan 2.2 **ti2v 5B** text-to-video.
    The 5B model (~9.4GB) + UMT5 encoder (~6.3GB) + wan2.2 VAE fit resident in 16GB
    -> NO per-scene reload (unlike LTX-22B+Gemma-12B). Uses Wan22ImageToVideoLatent
    (the proper temporal video latent; omit start_image for pure txt2vid)."""
    wf = {}
    n = [0]
    def nid():
        n[0] += 1; return str(n[0])
    model_node = nid()
    wf[model_node] = {"class_type": "UNETLoader", "inputs": {
        "unet_name": model_filename, "weight_dtype": "default"}}
    clip_node = nid()
    wf[clip_node] = {"class_type": "CLIPLoader", "inputs": {
        "clip_name": clip_name, "type": "wan"}}
    vae_node = nid()
    wf[vae_node] = {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}}

    cur_model = [model_node, 0]; cur_clip = [clip_node, 0]
    for lora_file, weight in (loras or []):
        ln = nid()
        wf[ln] = {"class_type": "LoraLoader", "inputs": {
            "model": cur_model, "clip": cur_clip, "lora_name": lora_file,
            "strength_model": weight, "strength_clip": weight}}
        cur_model = [ln, 0]; cur_clip = [ln, 1]

    pos = nid(); wf[pos] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": cur_clip}}
    neg = nid(); wf[neg] = {"class_type": "CLIPTextEncode", "inputs": {
        "text": negative_prompt or "blurry, low quality, distorted", "clip": cur_clip}}
    lat = nid(); wf[lat] = {"class_type": "Wan22ImageToVideoLatent", "inputs": {
        "vae": [vae_node, 0], "width": width, "height": height,
        "length": num_frames, "batch_size": 1}}
    samp = nid(); wf[samp] = {"class_type": "KSampler", "inputs": {
        "model": cur_model, "positive": [pos, 0], "negative": [neg, 0],
        "latent_image": [lat, 0], "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1.0}}
    dec = nid(); wf[dec] = {"class_type": "VAEDecode", "inputs": {"samples": [samp, 0], "vae": [vae_node, 0]}}
    save = nid(); wf[save] = {"class_type": "VHS_VideoCombine", "inputs": {
        "images": [dec, 0], "frame_rate": fps, "loop_count": 0,
        "filename_prefix": output_prefix, "format": "video/h264-mp4",
        "save_output": True, "pingpong": False}}
    return wf


def build_heartmula_workflow(
    style_tags: str,
    lyrics: str = "",
    duration_sec: float = 60.0,
    seed: int = 42,
    temperature: float = 1.0,
    top_k: int = 50,
    cfg_scale: float = 1.5,
    genre: str = "",
    mood: str = "",
    instruments: str = "",
    vocal_type: str = "instrumental",
    gender: str = "none",
    tempo: str = "medium",
    output_prefix: str = "ai_director_music",
) -> dict:
    """ComfyUI API workflow for HeartMuLa 3B music generation.

    Tags must use HeartMuLa's training vocabulary (short comma-separated tokens).
    The TagsBuilder node expects values from its dropdown lists — not free-form text.
    """
    wf = {
        "1": {"class_type": "FL_HeartMuLa_ModelLoader", "inputs": {
            "model_version": "3B",
            "precision": "auto",
            "use_4bit": False,
            "force_reload": False,
            "memory_mode": "auto",
        }},
        "2": {"class_type": "FL_HeartMuLa_TagsBuilder", "inputs": {
            "genre": genre or "acoustic",
            "vocal_type": vocal_type,
            "mood": mood or "peaceful",
            "tempo": tempo,
            "secondary_genre": "cinematic",
            "instruments": instruments or "piano, strings, flute",
            "additional_tags": style_tags,
        }},
        "3": {"class_type": "FL_HeartMuLa_Conditioning", "inputs": {
            "model": ["1", 0],
            "tags": ["2", 0],
            "lyrics": lyrics,
            "cfg_scale": cfg_scale,
        }},
        "4": {"class_type": "FL_HeartMuLa_Sampler", "inputs": {
            "model": ["1", 0],
            "conditioning": ["3", 0],
            "max_duration_sec": int(duration_sec),
            "temperature": temperature,
            "top_k": top_k,
            "seed": seed,
        }},
        "5": {"class_type": "FL_HeartMuLa_Decode", "inputs": {
            "model": ["1", 0],
            "audio_tokens": ["4", 0],
        }},
        "6": {"class_type": "SaveAudio", "inputs": {
            "audio": ["5", 0],
            "filename_prefix": output_prefix,
        }},
    }
    return wf


def build_acestep15xl_workflow(
    style_tags: str,
    lyrics: str = "",
    seconds: float = 120.0,
    seed: int = 42,
    steps: int = 8,
    cfg: float = 1.0,
    bpm: int = 90,
    language: str = "en",
    keyscale: str = "C major",
    unet_name: str = "acestep_v1.5_xl_turbo_bf16.safetensors",
    clip_name: str = "qwen_1.7b_ace15.safetensors",
    vae_name: str = "ace_1.5_vae.safetensors",
    output_prefix: str = "ai_director_music",
) -> dict:
    """ComfyUI API workflow for ACE-Step 1.5 XL (4B DiT, turbo).
    Commercial-grade quality. Uses native ComfyUI nodes (no custom extensions).
    Turbo variant = 8 steps. Split-file loading for 16GB VRAM with auto offload.
    """
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": clip_name, "type": "ace"}},
        "3": {"class_type": "VAELoader", "inputs": {
            "vae_name": vae_name}},
        "4": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {
            "clip": ["2", 0],
            "tags": style_tags,
            "lyrics": lyrics,
            "seed": seed,
            "bpm": bpm,
            "duration": float(seconds),
            "timesignature": "4",
            "language": language,
            "keyscale": keyscale,
            "generate_audio_codes": True,
            "cfg_scale": 2.0,
            "temperature": 0.85,
            "top_p": 0.9,
            "top_k": 0,
            "min_p": 0.0,
        }},
        "5": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {
            "seconds": float(seconds), "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["4", 0],
            "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveAudio", "inputs": {
            "audio": ["7", 0], "filename_prefix": output_prefix}},
    }
    return wf


def build_acestep_workflow(
    style_tags: str,
    lyrics: str = "",
    seconds: float = 60.0,
    seed: int = 42,
    steps: int = 60,
    cfg: float = 5.0,
    ckpt_name: str = "ace_step_v1_3.5b.safetensors",
    output_prefix: str = "ai_director_music",
) -> dict:
    """ComfyUI API workflow for ACE-Step v1.0 (legacy fallback).
    `style_tags` = comma-separated style/genre/instrument/mood prompt.
    `lyrics` = empty string for instrumental."""
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "TextEncodeAceStepAudio", "inputs": {
            "clip": ["1", 1], "tags": style_tags, "lyrics": lyrics, "lyrics_strength": 1.0}},
        "3": {"class_type": "TextEncodeAceStepAudio", "inputs": {
            "clip": ["1", 1], "tags": "", "lyrics": "", "lyrics_strength": 1.0}},
        "4": {"class_type": "EmptyAceStepLatentAudio", "inputs": {
            "seconds": float(seconds), "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["4", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "6": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveAudioMP3", "inputs": {
            "audio": ["6", 0], "filename_prefix": output_prefix, "quality": "V0"}},
    }
    return wf


# Default generation params per model family
FAMILY_DEFAULTS = {
    "ltx": {
        # Measured sweet spot for LTX-22B on a 16GB card: 1152x640 x 97 frames
        # peaks ~15.8GB and renders in ~85s (no VRAM spill). Upscales cleanly to
        # 1920x1080. 720p (1280x720) spills VRAM and crawls — do not raise without
        # re-measuring peak VRAM.
        "width": 1152, "height": 640, "fps": 24, "num_frames": 97,
        "steps": 8, "cfg": 1.0,
    },
    "ltx_dev": {
        "width": 768, "height": 512, "fps": 24, "num_frames": 97,
        "steps": 20, "cfg": 3.5,
    },
    "wan": {
        "width": 832, "height": 480, "fps": 16, "num_frames": 81,
        "steps": 30, "cfg": 5.0,
    },
}


def get_defaults_for_model(model_filename: str) -> dict:
    family = detect_family(model_filename)
    fn = model_filename.lower()
    if family == "ltx":
        if "dev" in fn:
            return FAMILY_DEFAULTS["ltx_dev"].copy()
        return FAMILY_DEFAULTS["ltx"].copy()
    if family == "wan":
        return FAMILY_DEFAULTS["wan"].copy()
    return FAMILY_DEFAULTS["ltx"].copy()
