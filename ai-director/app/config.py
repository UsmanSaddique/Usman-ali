"""
AI Director — Configuration
All paths, model configs, and runtime settings.
"""
import shutil
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import Optional


def _resolve_ffmpeg() -> str:
    """Find a usable ffmpeg binary.
    Order: PATH → imageio_ffmpeg bundled binary (ships with ComfyUI's
    embedded python) → bare 'ffmpeg' as a last resort.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    return "ffmpeg"


# ── Paths ──────────────────────────────────────────────────────────────────

class PathConfig(BaseModel):
    base_dir: Path = Path(__file__).parent.parent
    models_dir: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models")
    assets_dir: Path = Path(__file__).parent.parent / "assets_generated"
    projects_dir: Path = Path(__file__).parent.parent / "projects"
    database: Path = Path(__file__).parent.parent / "ai_director.db"
    channels_dir: Path = Path(__file__).parent.parent / "channels"
    archetypes_dir: Path = Path(__file__).parent.parent / "archetypes"
    loras_dir: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\loras")
    text_encoders_dir: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\text_encoders")
    vae_dir: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\vae")
    ffmpeg_bin: str = _resolve_ffmpeg()  # auto-detected: PATH or bundled imageio_ffmpeg

    def ensure_dirs(self):
        for d in [self.base_dir, self.models_dir, self.assets_dir, self.projects_dir, self.channels_dir, self.archetypes_dir, self.loras_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ── Model Definitions ──────────────────────────────────────────────────────

class LLMModelConfig(BaseModel):
    """Director brain via Ollama or llama-cpp-python.

    VRAM tuning (16GB / RTX 5070 Ti): the 27B-Q3 weights are ~11.7GB. To fit ALL
    layers on the GPU (and avoid the slow 30%-GPU CPU-offload path), we keep the
    KV cache + compute buffers small: modest n_ctx, small n_batch. flash_attn
    shrinks the KV cache further. n_threads is raised to use the i5-14400F's cores
    for any work that does land on CPU.
    """
    name: str = "qwen2.5:32b"    # Ollama fallback model id (only if local GGUF missing)
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 8192       # a 12-15 scene script + SEO can exceed 4096 → was truncating
    n_ctx: int = 16384           # prompt (~3k) + full script output (~8k). This Qwen is a
                                 # Mamba-hybrid (tiny recurrent KV ~150MB) so large context is cheap.
    model_path: Optional[Path] = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\checkpoints\Qwen3.6-27B-Q3_K_S.gguf")
    n_gpu_layers: int = -1       # -1 = offload all layers to GPU
    n_batch: int = 512           # small batch → small compute buffer (frees VRAM for layers)
    n_ubatch: int = 512
    n_threads: int = 8           # CPU threads for any non-offloaded work (i5-14400F has 16)
    flash_attn: bool = True      # smaller KV cache, faster attention


class ImageModelConfig(BaseModel):
    """Still-image generation. Z-Image-Turbo (primary) with SDXL fallback.

    Z-Image-Turbo (6B, Apache-2.0) replaced SDXL+LoRAs as the still generator:
    faster (8 steps, cfg 1, no negative prompt) and far better character
    fidelity — correct taqiyah/kurta, consistent faces, clean 3D render."""
    engine: str = "zimage"       # "zimage" (best character fidelity) or "sdxl"
    zimage_unet: str = "z_image_turbo_bf16.safetensors"
    # Quality bump 2026-07-25 (user: "images are not good"): more steps + a higher
    # still floor. Z-Image-Turbo stays fast even at 12 steps; the crisper, larger
    # still is what seeds img2vid and is supersampled into the final frame, so
    # sharper stills = sharper video. 960 floor keeps us near the ~1MP budget.
    zimage_steps: int = 12       # was 9 — sharper faces/edges, still quick on Turbo
    zimage_shift: float = 3.0
    zimage_min_height: int = 960  # was 720 — bigger, crisper still seed for img2vid
    # "Extra effort" 2026-08-04 (user: match Deen Kids TV / Cocomelon polish): a
    # two-pass hires-fix refine on the Z-Image still. Pass 1 renders the base,
    # then the latent is upscaled ~1.4x and a low-denoise second pass paints in
    # real detail (glossy eyes, ornate textures, crisp edges) instead of just
    # interpolating pixels. Keeps well within 16GB at 960-floor base * 1.4.
    zimage_hires: bool = False   # OFF 2026-08-04 (user: 2nd-pass upscale spilled VRAM,
                                 # ~2x still time + RAM overflow). The style suffix is the
                                 # real quality win (realistic->cartoon); the refine pass
                                 # only added cost. Single-pass 1696x960 is clean + fast (~46s)
                                 # and still supersamples into the 832x480 clip.
    zimage_hires_scale: float = 1.4
    zimage_hires_denoise: float = 0.4   # low = keep composition, add detail
    zimage_hires_steps: int = 10        # refine pass length (Turbo stays fast)
    # Compulsory style suffix appended to every still prompt so the render lands
    # the target children's-3D look (glossy Pixar/Cocomelon) regardless of what
    # the director LLM wrote. Density/variety come from the director; this locks
    # the *finish*. See feedback-visual-director-master-prompt.
    quality_suffix: str = (
        "glossy Pixar-style 3D animation render, adorable character with big "
        "expressive glossy eyes, soft rounded features, ultra-detailed ornate "
        "textures, warm volumetric golden-hour lighting, soft rim light, "
        "cinematic depth of field, dreamy bokeh sparkles, vibrant saturated "
        "colors, smooth soft shadows, subsurface scattering skin, polished "
        "high-gloss surfaces, extremely detailed, crisp focus, 8k, masterpiece, "
        "professional children's animation studio quality"
    )
    name: str = "sdxl-base-1.0"
    path: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\checkpoints\sd_xl_base_1.0.safetensors")
    dtype: str = "float16"       # float16 or bfloat16
    scheduler: str = "euler_a"   # euler_a, dpm++_2m_karras, ddim
    default_steps: int = 36      # 20% more steps for better detail (faces, hands, eyes)
    default_cfg: float = 7.0
    default_width: int = 1280
    default_height: int = 720
    # "Extra effort" still quality. SDXL base is weak/soft at <768px and single-pass;
    # render the still at its native ~1MP budget and add a hires-fix refine pass so
    # faces/eyes/hands actually resolve. The video stage downsizes the bigger still
    # (supersampling = cleaner final frames).
    min_gen_height: int = 768    # floor the still height to an SDXL bucket (16:9 -> ~1368x768, avoids duplication)
    hires: bool = True           # two-pass hires-fix refine
    hires_scale: float = 1.5
    hires_denoise: float = 0.45
    hires_steps: int = 18        # refine pass; 20% more for sharper faces/details
    enable_vae_slicing: bool = False
    enable_vae_tiling: bool = False
    # Style + character LoRAs applied to SDXL stills, as [(filename, weight)].
    # pixar-style pushes SDXL base toward the 3D cartoon look; add trained
    # character LoRAs here later for a consistent recurring cast.
    # pixar-style for the 3D cartoon look + trained character LoRAs for a consistent cast.
    # Tuned weights: too high (0.9+) over-bakes and adds noise; ~0.5 = clean + consistent.
    # Samaritan 3D Cartoon gives a clean crisp 3D render (pixar-style.safetensors on
    # SDXL base looked like a smeared oil painting). Keep yusuf for character identity.
    # (yusuf_v1.safetensors was deleted — it hurt results; Z-Image needs no character LoRA)
    style_loras: list = [("samaritan-3d-cartoon-lora.safetensors", 0.9)]


class VideoModelConfig(BaseModel):
    """LTX-Video 2.3 distilled — uses ltx_pipelines directly via subprocess."""
    name: str = "ltx-2.3-22b-distilled-1.1-gguf"
    model_path: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf")
    spatial_upsampler_path: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\latent_upscale_models\ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
    output_dir: Path = Path(__file__).parent.parent / "assets_generated/ltx_output"
    ltx_script: Path = Path(__file__).parent.parent / "scripts/ltx_generate.py"
    use_fp8: bool = True
    default_width: int = 768
    default_height: int = 512
    default_num_frames: int = 81  # ~5s at 16fps
    default_fps: int = 16
    default_steps: int = 8        # 8 steps = faster generation while keeping good LTX quality
    default_cfg: float = 1.0      # distilled models use low cfg
    # Per-clip frame ceiling (must be 8n+1 for LTX). 121 = 5.04s @24fps,
    # bench-validated VRAM-safe at 832x480 on the 16GB card. Do not raise
    # without re-benching peak VRAM (test_img2vid_bench.py pattern).
    max_num_frames: int = 121
    seed: int = -1                # -1 = random
    # img2vid: higher = clip stays closer to the (clean) still -> less hair/shirt
    # edge warping, calmer motion. Raised 0.7 -> 0.75 (2026-07-15, user request):
    # slow-to-medium character movement — fast motion makes local models fumble
    # (warped limbs); calmer clips read cleaner and stay above the frozen-clip
    # QA threshold thanks to ambient-motion prompt cues.
    img2vid_strength: float = 0.75
    # ── Premium opening ──────────────────────────────────────────────────
    # Retention is decided in the first seconds: scenes that START inside
    # this window render at a higher resolution + more steps (slower per
    # clip, deliberately), so the hook always looks best. 0 disables.
    # Benched 2026-07-15 (121f img2vid, LTX-22B, 16GB): 832x480@8 = 46s,
    # 960x544@8 = 55s, 960x544@16 = 88s (VRAM-safe), 1024x576 SPILLS = 574s.
    # 960x544@16 is the sweet spot: +33% pixels, 2x sampling, ~1.5min/clip.
    # Quality bump 2026-07-25: widen the premium window (more of the video gets
    # the hi-res hook treatment) and a couple more steps. 960x544@18 stays
    # VRAM-safe on 16GB (benched 960x544@16 = 88s/clip). Deliberately slower —
    # the user asked for more quality/effort over speed.
    premium_open_seconds: float = 35.0   # was 20 — more early scenes render hi-res
    premium_width: int = 960             # (landscape ref; actual dims now orientation-driven)
    premium_height: int = 544
    premium_steps: int = 18   # was 16 (distilled LTX default is 8)


class VideoModelAltConfig(BaseModel):
    """Wan 2.2 14B — alternative video gen via diffusers."""
    name: str = "wan2.2_t2v_high_noise_14B_fp8"
    path: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors")
    dtype: str = "float16"
    default_width: int = 832
    default_height: int = 480
    default_num_frames: int = 81  # ~5s at 16fps
    default_fps: int = 16
    default_steps: int = 30
    default_cfg: float = 5.0


class UpscaleModelConfig(BaseModel):
    """Upscaler — supports Real-ESRGAN or compatible ESRGAN models (e.g. 4x-UltraSharp)."""
    name: str = "4x-UltraSharp"
    model_path: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\upscale_models\4x-UltraSharp.pth")
    anime_model_path: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\upscale_models\RealESRGAN_x4plus_anime_6B.pth")
    scale: int = 4
    tile_size: int = 512         # lower = less VRAM, slower
    tile_pad: int = 10
    use_anime_model: bool = True   # 3D-cartoon content: anime ESRGAN = clean hair/shirt edges (no ringing), faster


class MusicModelConfig(BaseModel):
    """Music generation — HeartMuLa 3B (primary) via ComfyUI, ACE-Step (fallback).
    HeartMuLa auto-downloads to ComfyUI/models/heartmula/ on first use.
    Uses 4-bit quantization to fit on 16GB VRAM (~4.9GB)."""
    name: str = "heartmula-3b"
    path: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\heartmula")
    default_duration: int = 60   # seconds
    default_sample_rate: int = 48000  # HeartMuLa outputs 48kHz


class TTSConfig(BaseModel):
    """Narration TTS.
    Primary engine: Kokoro-82M (hexgrad/Kokoro-82M, Apache-2.0) — near-instant
    on GPU, natural prosody, multiple EN voices. English narration routes here.
    Fallback: Meta MMS-TTS (per-language VITS) for Urdu/Hindi/etc. where
    Kokoro has no voice.
    Alignment/QA: faster-whisper word timestamps (no torch-version conflicts)."""
    engine: str = "kokoro"              # "kokoro" | "mms"
    kokoro_repo: str = "hexgrad/Kokoro-82M"
    kokoro_voice: str = "am_michael"    # deep US male — documentary default
    # Curated voice menu shown in the wizard (Kokoro ships ~50; these are the
    # consistently best for narration)
    kokoro_voices: list = [
        "am_michael",   # US male, deep, documentary
        "am_fenrir",    # US male, energetic explainer
        "am_puck",      # US male, warm storyteller
        "af_heart",     # US female, warm (Kokoro's best overall voice)
        "af_bella",     # US female, bright
        "bf_emma",      # UK female
        "bm_george",    # UK male, classic documentary
    ]
    kokoro_speed: float = 1.0
    kokoro_sample_rate: int = 24000
    # faster-whisper QA/alignment
    whisper_model: str = "small"        # word timestamps + WER transcribe-back
    wer_flag_threshold: float = 0.12    # per-beat WER above this → retry with normalized text
    # narration master loudness (voice bus; final mix master is -14 LUFS)
    narration_lufs: float = -16.0
    # legacy MMS settings
    name: str = "mms-tts"
    default_voice: str = "warm_female"
    default_speed: float = 0.95
    # pauses between beats when concatenating the master narration WAV
    pause_beat: float = 0.35
    pause_chapter: float = 0.9


# ── Generation Defaults ───────────────────────────────────────────────────

class GenerationConfig(BaseModel):
    max_retries: int = 3
    clip_duration_range: tuple[float, float] = (4.0, 7.0)
    transition_duration: float = 0.5     # crossfade seconds
    upscale_target: str = "1080p"        # 1080p or 2k
    max_concurrent_upscale: int = 1
    save_intermediate: bool = True       # keep raw clips before upscale
    auto_evaluate: bool = False          # use Qwen-VL to score clips
    ken_burns_zoom_range: tuple[float, float] = (1.0, 1.15)  # 0-15% zoom
    ken_burns_pan_speed: float = 0.02    # pixels per frame normalized


# ── App Settings ──────────────────────────────────────────────────────────

class Settings(BaseSettings):
    app_name: str = "AI Director"
    host: str = "0.0.0.0"
    port: int = 8000
    # MUST stay False for production use: True turns on uvicorn auto-reload,
    # which restarts the server (killing any in-flight generation run) every
    # time a .py file is edited.
    debug: bool = False
    # Extreme resumability: projects that were mid-pipeline when the server
    # died are always rolled back to their last checkpoint on startup (no work
    # is ever lost). auto_resume=False (user preference): recovered projects
    # WAIT for an explicit "Resume" click / POST /resume instead of restarting
    # GPU work by themselves on boot. Set AIDIR_AUTO_RESUME=1 to flip.
    auto_resume: bool = False

    paths: PathConfig = PathConfig()
    llm: LLMModelConfig = LLMModelConfig()
    image: ImageModelConfig = ImageModelConfig()
    video: VideoModelConfig = VideoModelConfig()
    video_alt: VideoModelAltConfig = VideoModelAltConfig()
    upscale: UpscaleModelConfig = UpscaleModelConfig()
    music: MusicModelConfig = MusicModelConfig()
    tts: TTSConfig = TTSConfig()
    generation: GenerationConfig = GenerationConfig()

    class Config:
        env_prefix = "AIDIR_"
        env_file = ".env"


# Singleton
settings = Settings()
settings.paths.ensure_dirs()
