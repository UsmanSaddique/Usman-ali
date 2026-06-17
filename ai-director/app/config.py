"""
AI Director — Configuration
All paths, model configs, and runtime settings.
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import Optional


# ── Paths ──────────────────────────────────────────────────────────────────

class PathConfig(BaseModel):
    base_dir: Path = Path(__file__).parent.parent
    models_dir: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models")
    assets_dir: Path = Path(__file__).parent.parent / "assets_generated"
    projects_dir: Path = Path(__file__).parent.parent / "projects"
    database: Path = Path(__file__).parent.parent / "ai_director.db"
    channels_dir: Path = Path(__file__).parent.parent / "channels"
    loras_dir: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\loras")
    ffmpeg_bin: str = "ffmpeg"  # or full path like "C:/ffmpeg/bin/ffmpeg.exe"

    def ensure_dirs(self):
        for d in [self.base_dir, self.models_dir, self.assets_dir, self.projects_dir, self.channels_dir, self.loras_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ── Model Definitions ──────────────────────────────────────────────────────

class LLMModelConfig(BaseModel):
    """Qwen director brain via llama-cpp-python."""
    name: str = "qwen3.6-27b"
    path: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\checkpoints\Qwen3.6-27B-Q3_K_S.gguf")
    n_ctx: int = 8192            # context window
    n_gpu_layers: int = 35       # how many layers on GPU (tune for VRAM)
    n_batch: int = 512
    n_threads: int = 8           # CPU threads for offloaded layers
    temperature: float = 0.7
    max_tokens: int = 4096
    rope_freq_base: float = 0.0  # 0 = auto
    verbose: bool = False


class ImageModelConfig(BaseModel):
    """SDXL image generation via diffusers."""
    name: str = "sdxl-base-1.0"
    path: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\checkpoints\stable-diffusion-xl-base-1.0")
    dtype: str = "float16"       # float16 or bfloat16
    scheduler: str = "euler_a"   # euler_a, dpm++_2m_karras, ddim
    default_steps: int = 30
    default_cfg: float = 7.0
    default_width: int = 1280
    default_height: int = 720
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True


class VideoModelConfig(BaseModel):
    """LTX-Video 2.3 distilled — uses custom pipeline (NOT diffusers)."""
    name: str = "ltx-2.3-22b-distilled-1.1-gguf"
    model_path: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf")
    output_dir: Path = Path(__file__).parent.parent / "assets_generated/ltx_output"
    ltx_python: Path = Path(r"C:\Users\PC\AppData\Local\LTXDesktop\python\python.exe")
    ltx_script: Path = Path(__file__).parent.parent / "scripts/ltx_generate.py"
    use_fp8: bool = True
    default_width: int = 768
    default_height: int = 512
    default_num_frames: int = 121  # ~5s at 24fps
    default_fps: int = 24
    default_steps: int = 8        # distilled = fewer steps
    default_cfg: float = 1.0      # distilled models use low cfg
    seed: int = -1                # -1 = random


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
    """Real-ESRGAN upscaler."""
    name: str = "realesrgan-x4plus"
    model_path: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\upscale_models\RealESRGAN_x4plus.pth")
    anime_model_path: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\upscale_models\RealESRGAN_x4plus_anime_6B.pth")
    scale: int = 4
    tile_size: int = 512         # lower = less VRAM, slower
    tile_pad: int = 10
    use_anime_model: bool = False  # per-channel override


class MusicModelConfig(BaseModel):
    """ACE-Step music generation."""
    name: str = "ace-step-v1.5"
    path: Path = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\checkpoints\ace-step-v1.5")
    default_duration: int = 60   # seconds
    default_sample_rate: int = 44100


class TTSConfig(BaseModel):
    """WanGP Omnivoice TTS — communicates via local API."""
    name: str = "wangp-omnivoice"
    api_url: str = "http://localhost:5000"  # WanGP local server
    default_voice: str = "warm_female"
    default_speed: float = 0.9


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
    debug: bool = True

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
