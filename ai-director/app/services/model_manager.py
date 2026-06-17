"""
AI Director — Model Manager
Single-GPU VRAM orchestrator. Only ONE model loaded at any time.
Handles load/unload lifecycle with proper memory cleanup.
"""
import gc
import time
import logging
import threading
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
            return {
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "total_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
                "allocated_mb": torch.cuda.memory_allocated(0) // (1024 * 1024),
                "reserved_mb": torch.cuda.memory_reserved(0) // (1024 * 1024),
                "free_mb": (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_reserved(0)
                ) // (1024 * 1024),
            }
        except ImportError:
            return {"available": False, "error": "torch not installed"}


# ── Model Loader Factories ─────────────────────────────────────────────────
# Each returns a function that creates a LoadedModel.

def create_llm_loader(config) -> Callable[[], LoadedModel]:
    """Factory for Qwen LLM loader via llama-cpp-python."""

    def loader() -> LoadedModel:
        import torch  # Import torch first so its CUDA DLLs are added to the search path
        from llama_cpp import Llama

        model = Llama(
            model_path=str(config.llm.path),
            n_ctx=config.llm.n_ctx,
            n_gpu_layers=config.llm.n_gpu_layers,
            n_batch=config.llm.n_batch,
            n_threads=config.llm.n_threads,
            verbose=config.llm.verbose,
            # Enable chat template for Qwen
            chat_format="chatml",
        )

        return LoadedModel(
            model_type=ModelType.LLM,
            name=config.llm.name,
            model=model,
            extras={},
            vram_mb=_estimate_gguf_vram(config.llm.path, config.llm.n_gpu_layers),
        )

    return loader


def create_image_gen_loader(config) -> Callable[[], LoadedModel]:
    """Factory for SDXL loader via diffusers."""

    def loader() -> LoadedModel:
        import torch
        from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

        dtype = torch.float16 if config.image.dtype == "float16" else torch.bfloat16

        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(config.image.path),
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")

        # Optimizations
        if config.image.enable_vae_slicing:
            pipe.enable_vae_slicing()
        if config.image.enable_vae_tiling:
            pipe.enable_vae_tiling()

        # Set scheduler
        if config.image.scheduler == "euler_a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config
            )

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
    Factory for LTX-Video 2.3 distilled loader.
    Uses DistilledPipeline + SingleGPUModelBuilder (NOT diffusers LTX2Pipeline).
    """

    def loader() -> LoadedModel:
        import torch
        # LTX uses its own pipeline — import from LTX Desktop or custom path
        # This needs the LTX codebase on sys.path
        import sys
        ltx_base = config.video.ltx_python.parent.parent
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

            return LoadedModel(
                model_type=ModelType.VIDEO_GEN,
                name=config.video.name,
                model=pipeline,
                extras={"builder": builder},
                vram_mb=14000,
            )
        except ImportError:
            # Fallback: call LTX via subprocess
            logger.warning("LTX Python modules not found, will use subprocess mode")
            return LoadedModel(
                model_type=ModelType.VIDEO_GEN,
                name=config.video.name + "-subprocess",
                model=None,  # signal to use subprocess
                extras={"subprocess_mode": True},
                vram_mb=0,
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
    manager.register_loader(ModelType.LLM, create_llm_loader(config))
    manager.register_loader(ModelType.IMAGE_GEN, create_image_gen_loader(config))
    manager.register_loader(ModelType.VIDEO_GEN, create_video_gen_loader(config))
    manager.register_loader(ModelType.UPSCALER, create_upscaler_loader(config))
    # Music and TTS registered separately when those modules are ready
    logger.info("[ModelManager] All loaders registered")
