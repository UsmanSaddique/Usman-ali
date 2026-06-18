"""
AI Director — Video Generation Service
Generates video clips via ComfyUI API.
Supports LTX GGUF, Wan 2.2, with LoRA support.
"""
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import random

from app.services.model_manager import ModelManager, ModelType
from app.services.comfyui_client import (
    ComfyUIClient,
    detect_family,
    get_defaults_for_model,
    build_ltx_workflow,
    build_wan_workflow,
)

logger = logging.getLogger(__name__)

@dataclass
class VideoResult:
    path: str
    width: int
    height: int
    num_frames: int
    fps: int
    duration: float
    seed: int
    generation_time: float
    prompt_used: str
    model_used: str


class VideoGenService:
    """Generate video clips via ComfyUI API."""

    def __init__(self, model_manager: ModelManager, config):
        self.manager = model_manager
        self.config = config
        self.comfy_client = ComfyUIClient()

    def txt2vid(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: int = -1,
        output_path: Optional[str] = None,
        model_filename: Optional[str] = None,
        loras: Optional[list[tuple[str, float]]] = None,
    ) -> VideoResult:
        return self._generate_comfyui(
            mode="txt2vid",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            output_path=output_path,
            model_filename=model_filename,
            loras=loras,
        )

    def img2vid(
        self,
        prompt: str,
        image_path: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: int = -1,
        output_path: Optional[str] = None,
        model_filename: Optional[str] = None,
        loras: Optional[list[tuple[str, float]]] = None,
    ) -> VideoResult:
        return self._generate_comfyui(
            mode="img2vid",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            output_path=output_path,
            model_filename=model_filename,
            loras=loras,
            image_path=image_path,
        )

    def _generate_comfyui(
        self,
        mode: str,
        prompt: str,
        negative_prompt: str,
        width: Optional[int],
        height: Optional[int],
        num_frames: Optional[int],
        fps: Optional[int],
        steps: Optional[int],
        cfg_scale: Optional[float],
        seed: int,
        output_path: Optional[str],
        model_filename: Optional[str],
        loras: Optional[list[tuple[str, float]]] = None,
        image_path: Optional[str] = None,
    ) -> VideoResult:
        if not self.comfy_client.ping():
            raise RuntimeError("ComfyUI is not reachable at 127.0.0.1:8188")

        if not model_filename:
            model_filename = Path(self.config.video.model_path).name

        defaults = get_defaults_for_model(model_filename)
        width = width or defaults.get("width", 768)
        height = height or defaults.get("height", 512)
        num_frames = num_frames or defaults.get("num_frames", 97)
        fps = fps or defaults.get("fps", 24)
        steps = steps or defaults.get("steps", 8)
        cfg_scale = cfg_scale or defaults.get("cfg", 1.0)

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        prefix = "vid" if mode == "txt2vid" else "vid_i2v"
        if not output_path:
            output_path = str(
                self.config.video.output_dir / f"{prefix}_{seed}.mp4"
            )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        self.manager.unload()

        family = detect_family(model_filename)
        
        logger.info(f"[VideoGen] ComfyUI generation {mode}: {model_filename}")

        if family == "ltx":
            workflow = build_ltx_workflow(
                model_filename=model_filename,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                steps=steps,
                cfg=cfg_scale,
                seed=seed,
                fps=fps,
                loras=loras,
            )
        elif family == "wan":
            workflow = build_wan_workflow(
                model_filename=model_filename,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                steps=steps,
                cfg=cfg_scale,
                seed=seed,
                fps=fps,
                loras=loras,
            )
        else:
            raise ValueError(f"Unknown model family: {family}")

        # Inject image path for img2vid if needed
        if mode == "img2vid" and image_path:
            # Find the load image node if we extended build_*_workflow, 
            # or dynamically add it here.
            # Currently comfyui_client.py doesn't have img2vid native support,
            # so we log a warning and just run txt2vid for now.
            logger.warning(f"img2vid not fully implemented for {family} via ComfyUI, running as txt2vid.")

        t0 = time.time()
        
        prompt_id = self.comfy_client.submit(workflow)
        history = self.comfy_client.wait_for_completion(prompt_id, timeout=3600, poll=2.0)
        
        # Collect output
        final_path = self.comfy_client.collect_output(history, output_path)

        gen_time = time.time() - t0
        logger.info(f"[VideoGen] ComfyUI generation complete in {gen_time:.1f}s -> {final_path}")

        duration = num_frames / fps

        return VideoResult(
            path=final_path,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            duration=duration,
            seed=seed,
            generation_time=gen_time,
            prompt_used=prompt,
            model_used=model_filename,
        )
