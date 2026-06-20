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


def build_acestep_workflow(
    style_tags: str,
    lyrics: str = "",
    seconds: float = 60.0,
    seed: int = 42,
    steps: int = 60,        # higher = better fidelity (ACE-Step sweet spot ~60)
    cfg: float = 5.0,
    ckpt_name: str = "ace_step_v1_3.5b.safetensors",
    output_prefix: str = "ai_director_music",
) -> dict:
    """ComfyUI API workflow for ACE-Step music generation.
    `style_tags` = comma-separated style/genre/instrument/mood prompt.
    `lyrics` = empty string for instrumental. Output is an MP3 in ComfyUI/output."""
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
