"""
AI Director — Video Generation Service
LTX-Video 2.3 Distilled wrapper for txt2vid and img2vid.
Uses DistilledPipeline + SingleGPUModelBuilder (NOT diffusers LTX2Pipeline).
Falls back to subprocess mode if LTX Python modules aren't importable.
"""
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from app.services.model_manager import ModelManager, ModelType

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
    """Generate video clips via LTX-Video 2.3 Distilled."""

    def __init__(self, model_manager: ModelManager, config):
        self.manager = model_manager
        self.config = config

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
    ) -> VideoResult:
        """Generate a video clip from text prompt."""
        import torch

        cfg = self.config.video
        width = width or cfg.default_width
        height = height or cfg.default_height
        num_frames = num_frames or cfg.default_num_frames
        fps = fps or cfg.default_fps
        steps = steps or cfg.default_steps
        cfg_scale = cfg_scale or cfg.default_cfg

        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        if not output_path:
            output_path = str(cfg.output_dir / f"vid_{seed}.mp4")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Load model and check which engine we got
        loaded = self.manager.load(ModelType.VIDEO_GEN)
        engine = loaded.extras.get("engine", "")

        if engine == "wan2.2":
            return self._generate_wan22(
                pipeline=loaded.model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width, height=height,
                num_frames=num_frames, fps=fps,
                steps=steps, cfg_scale=cfg_scale,
                seed=seed, output_path=output_path,
            )
        elif loaded.extras.get("subprocess_mode"):
            return self._generate_subprocess(
                mode="txt2vid",
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width, height=height,
                num_frames=num_frames, fps=fps,
                steps=steps, cfg_scale=cfg_scale,
                seed=seed, output_path=output_path,
            )
        else:
            # Direct LTX pipeline mode
            return self._generate_direct(
                pipeline=loaded.model,
                mode="txt2vid",
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width, height=height,
                num_frames=num_frames, fps=fps,
                steps=steps, cfg_scale=cfg_scale,
                seed=seed, output_path=output_path,
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
    ) -> VideoResult:
        """Generate a video clip from image + text prompt."""
        import torch

        cfg = self.config.video
        width = width or cfg.default_width
        height = height or cfg.default_height
        num_frames = num_frames or cfg.default_num_frames
        fps = fps or cfg.default_fps
        steps = steps or cfg.default_steps
        cfg_scale = cfg_scale or cfg.default_cfg

        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        if not output_path:
            output_path = str(cfg.output_dir / f"vid_i2v_{seed}.mp4")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        loaded = self.manager.load(ModelType.VIDEO_GEN)

        if loaded.extras.get("subprocess_mode"):
            return self._generate_subprocess(
                mode="img2vid",
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width, height=height,
                num_frames=num_frames, fps=fps,
                steps=steps, cfg_scale=cfg_scale,
                seed=seed, output_path=output_path,
                image_path=image_path,
            )

        return self._generate_direct(
            pipeline=loaded.model,
            mode="img2vid",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width, height=height,
            num_frames=num_frames, fps=fps,
            steps=steps, cfg_scale=cfg_scale,
            seed=seed, output_path=output_path,
            image_path=image_path,
        )

    @staticmethod
    def ken_burns(
        image_path: str,
        output_path: str,
        duration: float = 5.0,
        fps: int = 24,
        zoom_start: float = 1.0,
        zoom_end: float = 1.15,
        pan_x: float = 0.0,
        pan_y: float = 0.0,
        target_width: int = 1280,
        target_height: int = 720,
        ffmpeg_bin: str = "ffmpeg",
    ) -> VideoResult:
        """
        Apply Ken Burns (zoom + pan) effect to a still image using FFmpeg.
        No GPU needed — pure CPU via FFmpeg's zoompan filter.
        """
        total_frames = int(duration * fps)

        # FFmpeg zoompan filter
        # zoom: linear interpolation from zoom_start to zoom_end
        # x/y: pan position (center-based)
        zoom_expr = f"if(eq(on,1),{zoom_start},{zoom_start}+({zoom_end}-{zoom_start})*on/{total_frames})"
        x_expr = f"iw/2-(iw/zoom/2)+{pan_x}*on/{total_frames}*iw"
        y_expr = f"ih/2-(ih/zoom/2)+{pan_y}*on/{total_frames}*ih"

        filter_str = (
            f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
            f":d={total_frames}:s={target_width}x{target_height}:fps={fps}"
        )

        cmd = [
            ffmpeg_bin, "-y",
            "-i", image_path,
            "-vf", filter_str,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            output_path
        ]

        t0 = time.time()
        logger.info(f"[VideoGen] Ken Burns: {image_path} → {output_path} ({duration}s)")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg Ken Burns failed: {result.stderr[-500:]}")

        elapsed = time.time() - t0

        return VideoResult(
            path=output_path,
            width=target_width,
            height=target_height,
            num_frames=total_frames,
            fps=fps,
            duration=duration,
            seed=0,
            generation_time=elapsed,
            prompt_used=f"ken_burns:{image_path}",
            model_used="ffmpeg-kenburns",
        )

    # ── Internal: Direct Pipeline ──────────────────────────────────────

    def _generate_direct(
        self, pipeline, mode: str, prompt: str, negative_prompt: str,
        width: int, height: int, num_frames: int, fps: int,
        steps: int, cfg_scale: float, seed: int, output_path: str,
        image_path: Optional[str] = None,
    ) -> VideoResult:
        """Generate using the loaded LTX pipeline directly."""
        import torch

        t0 = time.time()
        logger.info(
            f"[VideoGen] {mode} direct: {width}x{height}, {num_frames}f, "
            f"steps={steps}, seed={seed}"
        )

        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "seed": seed,
            "output_path": output_path,
        }

        if mode == "img2vid" and image_path:
            gen_kwargs["image_path"] = image_path

        with torch.inference_mode():
            # LTX DistilledPipeline interface — adjust to actual API
            pipeline.generate(**gen_kwargs)

        elapsed = time.time() - t0
        logger.info(f"[VideoGen] Generated in {elapsed:.1f}s → {output_path}")

        return VideoResult(
            path=output_path,
            width=width, height=height,
            num_frames=num_frames, fps=fps,
            duration=num_frames / fps,
            seed=seed,
            generation_time=elapsed,
            prompt_used=prompt,
            model_used="ltx-2.3-distilled",
        )

    # ── Internal: Wan2.2 via Diffusers ─────────────────────────────────

    def _generate_wan22(
        self, pipeline, prompt: str, negative_prompt: str,
        width: int, height: int, num_frames: int, fps: int,
        steps: int, cfg_scale: float, seed: int, output_path: str,
    ) -> VideoResult:
        """Generate using Wan2.2 diffusers pipeline."""
        import torch

        # Use Wan2.2 defaults if the LTX defaults were used
        wan_cfg = self.config.video_alt
        width = wan_cfg.default_width
        height = wan_cfg.default_height
        fps = wan_cfg.default_fps
        steps = wan_cfg.default_steps
        cfg_scale = wan_cfg.default_cfg
        # Limit frames to avoid OOM on 16GB (81 frames = ~5s at 16fps)
        num_frames = min(num_frames, wan_cfg.default_num_frames)

        t0 = time.time()
        logger.info(
            f"[VideoGen] Wan2.2 txt2vid: {width}x{height}, {num_frames}f, "
            f"steps={steps}, cfg={cfg_scale}, seed={seed}"
        )

        generator = torch.Generator(device="cuda").manual_seed(seed)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt or "blurry, low quality, distorted",
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
            )

        # Export frames to MP4
        frames = result.frames[0]  # list of PIL images or tensor
        self._frames_to_mp4(frames, output_path, fps)

        elapsed = time.time() - t0
        logger.info(f"[VideoGen] Wan2.2 generated in {elapsed:.1f}s → {output_path}")

        return VideoResult(
            path=output_path,
            width=width, height=height,
            num_frames=num_frames, fps=fps,
            duration=num_frames / fps,
            seed=seed,
            generation_time=elapsed,
            prompt_used=prompt,
            model_used="wan2.2-14B-fp8",
        )

    @staticmethod
    def _frames_to_mp4(frames, output_path: str, fps: int):
        """Convert a list of PIL Images or tensor to MP4 using ffmpeg."""
        import subprocess as sp
        from PIL import Image
        import struct

        # frames could be PIL images or a tensor
        if hasattr(frames, 'shape'):
            # It's a tensor — convert to PIL
            import torch
            if frames.dim() == 4:  # (T, C, H, W) or (T, H, W, C)
                pil_frames = []
                for i in range(frames.shape[0]):
                    f = frames[i]
                    if f.shape[0] in (3, 4):  # C, H, W
                        f = f.permute(1, 2, 0)
                    f = (f.clamp(0, 1) * 255).byte().cpu().numpy()
                    pil_frames.append(Image.fromarray(f))
                frames = pil_frames
        
        if not frames:
            raise ValueError("No frames to encode")

        w, h = frames[0].size
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path,
        ]

        proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
        for frame in frames:
            proc.stdin.write(frame.convert("RGB").tobytes())
        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0:
            stderr = proc.stderr.read().decode()
            raise RuntimeError(f"ffmpeg encoding failed: {stderr[-500:]}")

    # ── Internal: Subprocess Mode ──────────────────────────────────────

    def _generate_subprocess(
        self, mode: str, prompt: str, negative_prompt: str,
        width: int, height: int, num_frames: int, fps: int,
        steps: int, cfg_scale: float, seed: int, output_path: str,
        image_path: Optional[str] = None,
    ) -> VideoResult:
        """
        Generate by calling LTX Desktop's Python as a subprocess.
        Used when LTX modules can't be imported directly.
        """
        cfg = self.config.video

        # First, unload from our model manager (subprocess will use GPU)
        self.manager.unload()

        # Build args for the external script
        args_dict = {
            "mode": mode,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width, "height": height,
            "num_frames": num_frames, "fps": fps,
            "steps": steps, "cfg_scale": cfg_scale,
            "seed": seed,
            "output_path": output_path,
            "model_path": str(cfg.model_path),
            "use_fp8": cfg.use_fp8,
        }
        if image_path:
            args_dict["image_path"] = image_path

        args_json = json.dumps(args_dict)

        cmd = [
            str(cfg.ltx_python),
            str(cfg.ltx_script),
            "--args-json", args_json,
        ]

        t0 = time.time()
        logger.info(f"[VideoGen] {mode} subprocess: {width}x{height}")

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=600,  # 10 min max per clip
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"LTX subprocess failed (code {result.returncode}): "
                f"{result.stderr[-500:]}"
            )

        elapsed = time.time() - t0
        logger.info(f"[VideoGen] Subprocess done in {elapsed:.1f}s")

        return VideoResult(
            path=output_path,
            width=width, height=height,
            num_frames=num_frames, fps=fps,
            duration=num_frames / fps,
            seed=seed,
            generation_time=elapsed,
            prompt_used=prompt,
            model_used="ltx-2.3-distilled-subprocess",
        )
