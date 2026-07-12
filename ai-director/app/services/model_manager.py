"""
AI Director — Model Manager
Single-GPU VRAM orchestrator. Only ONE model loaded at any time.
Handles load/unload lifecycle with proper memory cleanup.
"""
import gc
import time
import logging
import threading
from pathlib import Path
from enum import Enum
from typing import Optional, Any, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    LLM = "llm"              # Qwen via llama-cpp-python
    IMAGE_GEN = "image_gen"  # SDXL via diffusers
    VIDEO_GEN = "video_gen"  # LTX via custom pipeline
    VIDEO_ALT = "video_alt"  # Wan via diffusers
    UPSCALER = "upscaler"    # Real-ESRGAN
    MUSIC = "music"          # ACE-Step
    EVALUATOR = "evaluator"  # Qwen-VL for clip QA


@dataclass
class LoadedModel:
    model_type: ModelType
    name: str
    model: Any              # the actual model object
    extras: dict = field(default_factory=dict)  # pipeline, tokenizer, etc.
    loaded_at: float = 0.0
    vram_mb: int = 0


class ModelManager:
    """
    Ensures only one model occupies VRAM at a time.
    Thread-safe via a reentrant lock.

    Usage:
        manager = ModelManager()

        # Load LLM for script generation
        llm = manager.load(ModelType.LLM)
        result = llm.model.create_chat_completion(...)
        manager.unload()  # free VRAM

        # Now load image model
        img = manager.load(ModelType.IMAGE_GEN)
        image = img.extras["pipeline"](prompt=...).images[0]
        manager.unload()
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._current: Optional[LoadedModel] = None
        self._loaders: dict[ModelType, Callable] = {}
        self._unloaders: dict[ModelType, Callable] = {}

    # ── Loader Registration ────────────────────────────────────────────

    def register_loader(
        self,
        model_type: ModelType,
        loader: Callable[[], LoadedModel],
        unloader: Optional[Callable[[LoadedModel], None]] = None
    ):
        """
        Register a load/unload function pair for a model type.
        loader() -> LoadedModel
        unloader(LoadedModel) -> None  (optional, for custom cleanup)
        """
        self._loaders[model_type] = loader
        if unloader:
            self._unloaders[model_type] = unloader

    # ── Core Operations ────────────────────────────────────────────────

    def load(self, model_type: ModelType, force_reload: bool = False) -> LoadedModel:
        """
        Load a model into VRAM. If a different model is loaded, unload it first.
        If the SAME model is already loaded and force_reload=False, return it.
        """
        with self._lock:
            # Already loaded?
            if self._current and self._current.model_type == model_type and not force_reload:
                logger.info(f"[ModelManager] {model_type.value} already loaded, reusing")
                return self._current

            # Unload whatever is currently loaded
            if self._current:
                self.unload()

            # Load the new model
            if model_type not in self._loaders:
                raise ValueError(
                    f"No loader registered for {model_type.value}. "
                    f"Available: {list(self._loaders.keys())}"
                )

            logger.info(f"[ModelManager] Loading {model_type.value}...")
            t0 = time.time()

            loaded = self._loaders[model_type]()
            loaded.loaded_at = time.time()

            elapsed = time.time() - t0
            logger.info(
                f"[ModelManager] {model_type.value} loaded in {elapsed:.1f}s "
                f"(~{loaded.vram_mb}MB VRAM)"
            )

            self._current = loaded
            return loaded

    def unload(self):
        """Unload the current model and aggressively free VRAM."""
        with self._lock:
            if not self._current:
                return

            model_type = self._current.model_type
            logger.info(f"[ModelManager] Unloading {model_type.value}...")

            # Custom unloader?
            if model_type in self._unloaders:
                try:
                    self._unloaders[model_type](self._current)
                except Exception as e:
                    logger.warning(f"Custom unloader error: {e}")

            # Aggressively delete everything
            if hasattr(self._current.model, 'close'):
                try:
                    self._current.model.close()
                except Exception:
                    pass

            # Delete extras (pipelines, tokenizers, etc.)
            for key in list(self._current.extras.keys()):
                obj = self._current.extras.pop(key)
                del obj

            model_ref = self._current.model
            self._current.model = None
            self._current = None
            del model_ref

            # Force garbage collection + CUDA cache clear
            self._flush_gpu_memory()
            logger.info(f"[ModelManager] {model_type.value} unloaded, VRAM freed")

    def get_current(self) -> Optional[LoadedModel]:
        """Return the currently loaded model, if any."""
        with self._lock:
            return self._current

    @property
    def is_loaded(self) -> bool:
        return self._current is not None

    @property
    def current_type(self) -> Optional[ModelType]:
        return self._current.model_type if self._current else None

    # ── GPU Utilities ──────────────────────────────────────────────────

    @staticmethod
    def _flush_gpu_memory():
        """Nuclear option: GC + empty CUDA cache."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
        except ImportError:
            pass

    @staticmethod
    def get_gpu_stats() -> dict:
        """Get current GPU VRAM usage."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {"available": False}
            
            # Use mem_get_info to get actual system-wide free/total VRAM
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_mb = free_bytes // (1024 * 1024)
            total_mb = total_bytes // (1024 * 1024)
            allocated_mb = total_mb - free_mb
            
            return {
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "total_mb": total_mb,
                "allocated_mb": allocated_mb,
                "reserved_mb": torch.cuda.memory_reserved(0) // (1024 * 1024),
                "free_mb": free_mb,
            }
        except ImportError:
            return {"available": False, "error": "torch not installed"}


# ── Model Loader Factories ─────────────────────────────────────────────────
# Each returns a function that creates a LoadedModel.

def create_llm_loader(config) -> Callable[[], LoadedModel]:
    """Factory for Qwen LLM loader. Uses direct llama-cpp-python if local GGUF path is configured and exists; otherwise falls back to Ollama."""

    def loader() -> LoadedModel:
        import os
        model_path = getattr(config.llm, "model_path", None)
        if model_path and os.path.exists(model_path):
            logger.info(f"[ModelManager] Loading local GGUF model from {model_path} via llama-cpp-python...")
            try:
                from llama_cpp import Llama
                
                n_gpu_layers = getattr(config.llm, "n_gpu_layers", -1)
                n_ctx = getattr(config.llm, "n_ctx", 6144)
                n_batch = getattr(config.llm, "n_batch", 512)
                n_ubatch = getattr(config.llm, "n_ubatch", 512)
                n_threads = getattr(config.llm, "n_threads", 8)
                flash_attn = getattr(config.llm, "flash_attn", True)

                try:
                    logger.info(
                        f"[ModelManager] Instantiating Llama "
                        f"(n_ctx={n_ctx}, n_batch={n_batch}, n_threads={n_threads}, "
                        f"flash_attn={flash_attn}, n_gpu_layers={n_gpu_layers}) "
                        f"— tuned to fit all layers on 16GB GPU"
                    )
                    llm = Llama(
                        model_path=str(model_path),
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        n_batch=n_batch,
                        n_ubatch=n_ubatch,
                        n_threads=n_threads,
                        n_threads_batch=n_threads,
                        flash_attn=flash_attn,
                        verbose=True  # prints "offloaded N/M layers to GPU" so we can verify fit
                    )
                except TypeError:
                    logger.warning("[ModelManager] flash_attn=True or advanced parameters not supported by this llama-cpp-python version. Falling back to basic parameters.")
                    llm = Llama(
                        model_path=str(model_path),
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        verbose=False
                    )
                
                vram_est = _estimate_gguf_vram(model_path, n_gpu_layers if n_gpu_layers >= 0 else 64)
                
                return LoadedModel(
                    model_type=ModelType.LLM,
                    name=os.path.basename(model_path),
                    model=llm,
                    extras={"local_gguf": True},
                    vram_mb=vram_est,
                )
            except Exception as e:
                logger.error(f"[ModelManager] Failed to load local GGUF: {e}. Falling back to Ollama.")

        class OllamaClient:
            def __init__(self, model_name, base_url, n_ctx):
                self.model_name = model_name
                self.base_url = base_url
                self.n_ctx = n_ctx

            def create_chat_completion(self, messages, temperature=0.7, max_tokens=1024, response_format=None, stream=False):
                import urllib.request
                import json
                
                url = f"{self.base_url}/v1/chat/completions"
                
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": stream,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                if response_format and response_format.get("type") == "json_object":
                    payload["response_format"] = {"type": "json_object"}
                    
                req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
                
                try:
                    if stream:
                        def generator():
                            with urllib.request.urlopen(req) as response:
                                for line in response:
                                    if line:
                                        chunk = json.loads(line.decode('utf-8'))
                                        content = chunk.get("message", {}).get("content", "")
                                        yield {
                                            "choices": [
                                                {
                                                    "delta": {
                                                        "content": content
                                                    }
                                                }
                                            ]
                                        }
                        return generator()
                    else:
                        with urllib.request.urlopen(req) as response:
                            return json.loads(response.read().decode('utf-8'))
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Ollama API Error: {e}")
                    raise e
                    
        client = OllamaClient(config.llm.name, config.llm.base_url, config.llm.n_ctx)

        return LoadedModel(
            model_type=ModelType.LLM,
            name=config.llm.name,
            model=client,
            extras={},
            vram_mb=0, # Ollama manages its own VRAM; we free it via unloader
        )

    return loader


def create_llm_unloader(config) -> Callable[[LoadedModel], None]:
    """Factory for Ollama unloader to force VRAM release."""
    
    def unloader(loaded_model: LoadedModel):
        if loaded_model.extras.get("local_gguf"):
            logger.info("[ModelManager] Local GGUF model unloaded.")
            return

        import urllib.request
        import json
        import logging
        
        try:
            url = f"{config.llm.base_url}/api/generate"
            payload = {
                "model": config.llm.name,
                "keep_alive": 0
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                pass
            logging.getLogger(__name__).info("[Ollama] Sent keep_alive=0 to free VRAM.")
        except Exception as e:
            logging.getLogger(__name__).warning(f"[Ollama] Failed to free VRAM: {e}")
            
    return unloader


def create_image_gen_loader(config) -> Callable[[], LoadedModel]:
    """Factory for SDXL loader via diffusers."""

    def loader() -> LoadedModel:
        import os
        import torch
        from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

        dtype = torch.float16 if config.image.dtype == "float16" else torch.bfloat16

        model_path = str(config.image.path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[ModelManager] SDXL model not found at '{model_path}'.\n"
                f"Please set the correct path in config.py (image.path) or download SDXL first.\n"
                f"Download: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
            )

        # Detect single-file (.safetensors/.ckpt) vs diffusers folder
        is_single_file = model_path.endswith(('.safetensors', '.ckpt', '.pt'))

        load_kwargs = dict(
            torch_dtype=dtype,
            use_safetensors=True,
        )
        if is_single_file:
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=dtype,
            ).to("cuda")
        else:
            # Diffusers folder format — only request fp16 variant from HF Hub
            if os.path.exists(os.path.join(model_path, "model_index.json")):
                load_kwargs["variant"] = "fp16"
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                **load_kwargs
            ).to("cuda")

        # Optimizations for high-VRAM GPU (RTX 5070 Ti, 16GB)
        if config.image.enable_vae_slicing:
            pipe.enable_vae_slicing()
        if config.image.enable_vae_tiling:
            pipe.enable_vae_tiling()

        # Use scaled dot-product attention (PyTorch 2.0+) - no xformers needed
        try:
            pipe.unet.set_attn_processor(
                __import__('diffusers').models.attention_processor.AttnProcessor2_0()
            )
            logger.info("[ModelManager] SDXL: Using AttnProcessor2_0 (SDPA) for GPU attention")
        except Exception as e:
            logger.warning(f"[ModelManager] AttnProcessor2_0 not available: {e}")

        # Set scheduler
        if config.image.scheduler == "euler_a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config
            )

        # Ensure VAE is in fp16 too
        try:
            pipe.vae.to(dtype=dtype)
        except Exception:
            pass

        return LoadedModel(
            model_type=ModelType.IMAGE_GEN,
            name=config.image.name,
            model=pipe,
            extras={"pipeline": pipe},
            vram_mb=8000,
        )

    return loader


def create_video_gen_loader(config) -> Callable[[], LoadedModel]:
    """
    Factory for video generation loader.
    Tries LTX-Video first, falls back to Wan2.2 via diffusers.
    """

    def loader() -> LoadedModel:
        import os
        import torch
        import sys

        # ── Try 1: LTX-Video (custom pipeline) ──────────────────────
        # Legacy direct-LTX path. The primary backend is now ComfyUI
        # (see VideoGenService); this only runs if an ltx_python interpreter
        # is explicitly configured. getattr guards the now-removed config field.
        ltx_python = getattr(config.video, "ltx_python", None)
        if ltx_python is None:
            raise ImportError("LTX direct mode not configured (ltx_python unset)")

        ltx_base = Path(ltx_python).parent.parent
        ltx_src = ltx_base / "ltx_video"
        if str(ltx_src) not in sys.path:
            sys.path.insert(0, str(ltx_src))

        try:
            from ltx_video.pipelines.distilled_pipeline import DistilledPipeline
            from ltx_video.utils.model_builder import SingleGPUModelBuilder

            builder = SingleGPUModelBuilder(
                model_path=str(config.video.model_path),
                use_fp8=config.video.use_fp8,
            )
            pipeline = DistilledPipeline(builder)

            logger.info("[ModelManager] LTX-Video loaded successfully (direct mode)")
            return LoadedModel(
                model_type=ModelType.VIDEO_GEN,
                name=config.video.name,
                model=pipeline,
                extras={"builder": builder, "engine": "ltx"},
                vram_mb=14000,
            )
        except ImportError:
            logger.warning("[ModelManager] LTX Python modules not found, trying Wan2.2...")

        # ── Try 2: Wan2.2 via diffusers ──────────────────────────────
        wan_model_path = str(config.video_alt.path)
        if os.path.exists(wan_model_path):
            try:
                from diffusers import WanPipeline
                
                dtype = torch.float16 if config.video_alt.dtype == "float16" else torch.bfloat16
                
                logger.info(f"[ModelManager] Loading Wan2.2 from {wan_model_path}...")
                pipe = WanPipeline.from_single_file(
                    wan_model_path,
                    torch_dtype=dtype,
                ).to("cuda")
                
                logger.info("[ModelManager] Wan2.2 video pipeline loaded successfully")
                return LoadedModel(
                    model_type=ModelType.VIDEO_GEN,
                    name=config.video_alt.name,
                    model=pipe,
                    extras={"engine": "wan2.2", "pipeline": pipe},
                    vram_mb=14000,
                )
            except ImportError:
                logger.warning("[ModelManager] diffusers WanPipeline not available")
            except Exception as e:
                logger.error(f"[ModelManager] Wan2.2 load failed: {e}")

        # ── Try 3: LTX subprocess mode ───────────────────────────────
        ltx_script = getattr(config.video, "ltx_script", None)
        ltx_python = getattr(config.video, "ltx_python", None)
        if ltx_script and ltx_python and Path(ltx_script).exists() and Path(ltx_python).exists():
            logger.warning("[ModelManager] Falling back to LTX subprocess mode")
            return LoadedModel(
                model_type=ModelType.VIDEO_GEN,
                name=config.video.name + "-subprocess",
                model=None,
                extras={"subprocess_mode": True, "engine": "ltx-subprocess"},
                vram_mb=0,
            )

        raise RuntimeError(
            "[ModelManager] No video generation backend available!\n"
            "Install ltx_video, or ensure Wan2.2 model exists, or configure LTX subprocess."
        )

    return loader



def create_upscaler_loader(config) -> Callable[[], LoadedModel]:
    """Factory for Real-ESRGAN upscaler."""

    def loader() -> LoadedModel:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import torch

        model_path = str(config.upscale.anime_model_path
                         if config.upscale.use_anime_model
                         else config.upscale.model_path)

        # RRDBNet architecture params differ between models
        if config.upscale.use_anime_model:
            rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=6, num_grow_ch=32, scale=4)
        else:
            rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)

        upsampler = RealESRGANer(
            scale=config.upscale.scale,
            model_path=model_path,
            model=rrdb,
            tile=config.upscale.tile_size,
            tile_pad=config.upscale.tile_pad,
            half=True,  # fp16 to save VRAM
            device="cuda",
        )

        return LoadedModel(
            model_type=ModelType.UPSCALER,
            name=config.upscale.name,
            model=upsampler,
            extras={},
            vram_mb=500,
        )

    return loader


def _estimate_gguf_vram(model_path, n_gpu_layers: int) -> int:
    """Rough VRAM estimate for GGUF models."""
    try:
        from pathlib import Path
        size_gb = Path(model_path).stat().st_size / (1024**3)
        # Rough: if all layers on GPU, ~= file size. Partial = proportional.
        # Most 32B models have ~64 layers
        ratio = min(n_gpu_layers / 64.0, 1.0)
        return int(size_gb * ratio * 1024)  # MB
    except Exception:
        return 12000  # default estimate


# ── Registration Helper ────────────────────────────────────────────────────

def register_all_loaders(manager: ModelManager, config):
    """Register all model loaders with the manager."""
    manager.register_loader(ModelType.LLM, create_llm_loader(config), create_llm_unloader(config))
    manager.register_loader(ModelType.IMAGE_GEN, create_image_gen_loader(config))
    manager.register_loader(ModelType.VIDEO_GEN, create_video_gen_loader(config))
    manager.register_loader(ModelType.UPSCALER, create_upscaler_loader(config))
    # Music and TTS registered separately when those modules are ready
    logger.info("[ModelManager] All loaders registered")
