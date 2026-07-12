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

    def _generate_comfyui_zimage(
        self, prompt, width, height, seed, output_path, steps=None,
    ) -> ImageResult:
        """Generate a still via Z-Image-Turbo through ComfyUI. Primary engine:
        far better character fidelity than SDXL+LoRAs (correct clothing, consistent
        faces) at 8-9 steps, cfg 1, and needs no negative prompt or style LoRA."""
        from pathlib import Path as _P
        from app.services.comfyui_client import ComfyUIClient, build_zimage_workflow

        client = ComfyUIClient()
        if not client.wait_ready(30):
            raise RuntimeError("ComfyUI not reachable for Z-Image")
        self.manager.unload()
        client.free_vram()

        ic = self.config.image
        # Render near Z-Image's native ~1MP budget preserving aspect; the video
        # stage downsizes the bigger still (supersampling = cleaner frames).
        gen_w, gen_h = width, height
        min_h = getattr(ic, "zimage_min_height", 0) or 0
        if min_h and height < min_h:
            aspect = width / height
            gen_h = int(round(min_h / 16) * 16)
            gen_w = int(round((gen_h * aspect) / 16) * 16)
            logger.info(f"[ImageGen] still upsized {width}x{height} -> {gen_w}x{gen_h} for Z-Image quality")

        wf = build_zimage_workflow(
            prompt=prompt,
            width=gen_w,
            height=gen_h,
            steps=steps or int(getattr(ic, "zimage_steps", 9)),
            cfg=1.0,
            seed=seed,
            shift=float(getattr(ic, "zimage_shift", 3.0)),
        )
        if not output_path:
            output_path = str(self.config.paths.assets_dir / "images" / f"img_{seed}.png")
        _P(output_path).parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        prompt_id = client.submit(wf)
        history = client.wait_for_completion(prompt_id, timeout=600, poll=1.5)
        final = client.collect_output(history, output_path)
        elapsed = time.time() - t0
        logger.info(f"[ImageGen] Z-Image {gen_w}x{gen_h} in {elapsed:.1f}s -> {final}")
        return ImageResult(
            path=final, width=gen_w, height=gen_h, seed=seed,
            generation_time=elapsed, prompt_used=prompt,
        )

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
        client.free_vram()

        ckpt = _P(str(self.config.image.path)).name
        style_loras = [tuple(x) for x in getattr(self.config.image, "style_loras", []) or []]

        # "Extra effort": render the still at SDXL's native pixel budget (not the
        # small video res), preserving aspect, so faces/eyes/hands resolve. The
        # video stage downsizes the bigger still (supersampling = cleaner frames).
        ic = self.config.image
        gen_w, gen_h = width, height
        min_h = getattr(ic, "min_gen_height", 0) or 0
        if min_h and height < min_h:
            aspect = width / height
            gen_h = min_h
            gen_w = int(round((min_h * aspect) / 8) * 8)
            gen_h = int(round(gen_h / 8) * 8)
            logger.info(f"[ImageGen] still upscaled {width}x{height} -> {gen_w}x{gen_h} for SDXL quality")

        wf = build_sdxl_workflow(
            prompt=prompt, negative_prompt=negative_prompt,
            width=gen_w, height=gen_h, steps=steps, cfg=cfg_scale,
            seed=seed, ckpt_name=ckpt, loras=style_loras or None,
            hires=bool(getattr(ic, "hires", False)),
            hires_scale=float(getattr(ic, "hires_scale", 1.5)),
            hires_denoise=float(getattr(ic, "hires_denoise", 0.45)),
            hires_steps=int(getattr(ic, "hires_steps", 0)),
        )
        if not output_path:
            output_path = str(self.config.paths.assets_dir / "images" / f"img_{seed}.png")
        _P(output_path).parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        prompt_id = client.submit(wf)
        history = client.wait_for_completion(prompt_id, timeout=600, poll=1.5)
        final = client.collect_output(history, output_path)
        elapsed = time.time() - t0
        out_w = int(round(gen_w * (self.config.image.hires_scale if getattr(ic, "hires", False) else 1)))
        out_h = int(round(gen_h * (self.config.image.hires_scale if getattr(ic, "hires", False) else 1)))
        logger.info(f"[ImageGen] ComfyUI SDXL {gen_w}x{gen_h}"
                    f"{f' -> hires {out_w}x{out_h}' if getattr(ic,'hires',False) else ''} in {elapsed:.1f}s -> {final}")
        return ImageResult(
            path=final, width=out_w, height=out_h, seed=seed,
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

        # Preferred: Z-Image-Turbo via ComfyUI (engine="zimage" in config).
        if getattr(self.config.image, "engine", "sdxl") == "zimage":
            try:
                return self._generate_comfyui_zimage(
                    prompt=prompt, width=width, height=height,
                    seed=seed, output_path=output_path,
                )
            except Exception as e:
                logger.warning(f"[ImageGen] Z-Image failed ({e}); falling back to SDXL")

        # SDXL via ComfyUI (same backend as video/music).
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
