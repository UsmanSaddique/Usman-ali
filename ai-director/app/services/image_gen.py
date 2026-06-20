"""
AI Director — Image Generation Service
SDXL image generation with LoRA support, optimized for single GPU.
"""
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from app.services.model_manager import ModelManager, ModelType

logger = logging.getLogger(__name__)


@dataclass
class ImageResult:
    path: str
    width: int
    height: int
    seed: int
    generation_time: float
    prompt_used: str


class ImageGenService:
    """Generate images via SDXL with LoRA support."""

    def __init__(self, model_manager: ModelManager, config):
        self.manager = model_manager
        self.config = config

    def _generate_comfyui(
        self, prompt, negative_prompt, width, height, steps, cfg_scale, seed, output_path,
    ) -> ImageResult:
        """Generate an SDXL still via ComfyUI (avoids diffusers/transformers breakage)."""
        from pathlib import Path as _P
        from app.services.comfyui_client import ComfyUIClient, build_sdxl_workflow

        client = ComfyUIClient()
        if not client.wait_ready(30):
            raise RuntimeError("ComfyUI not reachable for SDXL")
        # free the LLM/other VRAM so SDXL has room
        self.manager.unload()

        ckpt = _P(str(self.config.image.path)).name
        wf = build_sdxl_workflow(
            prompt=prompt, negative_prompt=negative_prompt,
            width=width, height=height, steps=steps, cfg=cfg_scale,
            seed=seed, ckpt_name=ckpt,
        )
        if not output_path:
            output_path = str(self.config.paths.assets_dir / "images" / f"img_{seed}.png")
        _P(output_path).parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        prompt_id = client.submit(wf)
        history = client.wait_for_completion(prompt_id, timeout=600, poll=1.5)
        final = client.collect_output(history, output_path)
        elapsed = time.time() - t0
        logger.info(f"[ImageGen] ComfyUI SDXL {width}x{height} in {elapsed:.1f}s -> {final}")
        return ImageResult(
            path=final, width=width, height=height, seed=seed,
            generation_time=elapsed, prompt_used=prompt,
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: int = -1,
        lora_paths: Optional[list[str]] = None,
        lora_weights: Optional[list[float]] = None,
        output_path: Optional[str] = None,
    ) -> ImageResult:
        """
        Generate a single image. Loads SDXL if not already loaded.
        """
        import torch
        from PIL import Image

        # Defaults from config
        width = width or self.config.image.default_width
        height = height or self.config.image.default_height
        steps = steps or self.config.image.default_steps
        cfg_scale = cfg_scale or self.config.image.default_cfg

        # Handle seed
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        # Preferred: generate via ComfyUI SDXL (same backend as video/music).
        # The diffusers path breaks on transformers 5.9 ('CLIPTextModel' has no
        # attribute 'text_model'), so ComfyUI is the reliable route.
        try:
            return self._generate_comfyui(
                prompt=prompt, negative_prompt=negative_prompt or self._default_negative(),
                width=width, height=height, steps=steps, cfg_scale=cfg_scale,
                seed=seed, output_path=output_path,
            )
        except Exception as e:
            logger.warning(f"[ImageGen] ComfyUI SDXL failed ({e}); falling back to diffusers")

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Load SDXL
        loaded = self.manager.load(ModelType.IMAGE_GEN)
        pipe = loaded.model

        # Apply LoRAs if specified
        active_loras = []
        if lora_paths:
            for i, lora_path in enumerate(lora_paths):
                weight = lora_weights[i] if lora_weights and i < len(lora_weights) else 0.7
                adapter_name = f"lora_{i}"
                try:
                    pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                    active_loras.append((adapter_name, weight))
                    logger.info(f"[ImageGen] Loaded LoRA: {lora_path} (weight={weight})")
                except Exception as e:
                    logger.warning(f"[ImageGen] Failed to load LoRA {lora_path}: {e}")

            if active_loras:
                names = [a[0] for a in active_loras]
                weights = [a[1] for a in active_loras]
                pipe.set_adapters(names, adapter_weights=weights)

        # Generate
        t0 = time.time()
        logger.info(f"[ImageGen] Generating {width}x{height}, steps={steps}, cfg={cfg_scale}")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or self._default_negative(),
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
            )

        image: Image.Image = result.images[0]
        elapsed = time.time() - t0

        # Unload LoRAs to prevent bleeding into next generation
        if active_loras:
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

        # Save
        if not output_path:
            output_path = str(
                self.config.paths.assets_dir / "images" / f"img_{seed}.png"
            )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95)

        logger.info(f"[ImageGen] Generated in {elapsed:.1f}s → {output_path}")

        return ImageResult(
            path=output_path,
            width=width,
            height=height,
            seed=seed,
            generation_time=elapsed,
            prompt_used=prompt,
        )

    def generate_thumbnail(
        self,
        prompt: str,
        output_path: str,
        width: int = 1280,
        height: int = 720,
    ) -> ImageResult:
        """Generate a YouTube thumbnail image."""
        return self.generate(
            prompt=prompt,
            negative_prompt="text, watermark, logo, blurry, low quality, human face, hands",
            width=width,
            height=height,
            steps=35,       # higher quality for thumbnail
            cfg_scale=7.5,
            output_path=output_path,
        )

    @staticmethod
    def _default_negative() -> str:
        return (
            "blurry, low quality, worst quality, low resolution, pixelated, "
            "jpeg artifacts, watermark, text, logo, signature, human face, "
            "hands, fingers, deformed, ugly, duplicate, morbid, mutilated, "
            "extra limbs, bad anatomy, bad proportions, disfigured"
        )
