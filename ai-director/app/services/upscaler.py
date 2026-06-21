"""
AI Director — Upscaler Service
Real-ESRGAN upscaling for images and video clips (frame-by-frame).
"""
import os
import time
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from app.services.model_manager import ModelManager, ModelType

logger = logging.getLogger(__name__)


@dataclass
class UpscaleResult:
    input_path: str
    output_path: str
    input_resolution: tuple[int, int]
    output_resolution: tuple[int, int]
    processing_time: float
    frame_count: int  # 1 for images, N for videos


class UpscalerService:
    """Upscale images and video clips using Real-ESRGAN."""

    def __init__(self, model_manager: ModelManager, config):
        self.manager = model_manager
        self.config = config

    def upscale_image(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        target_width: int = 1920,
        target_height: int = 1080,
    ) -> UpscaleResult:
        """Upscale a single image, then resize to exact target resolution."""
        import numpy as np
        from PIL import Image

        if not output_path:
            p = Path(input_path)
            output_path = str(p.parent / f"{p.stem}_hd{p.suffix}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Load upscaler
        loaded = self.manager.load(ModelType.UPSCALER)
        upsampler = loaded.model

        t0 = time.time()

        # Read image
        img = Image.open(input_path).convert("RGB")
        input_res = (img.width, img.height)
        img_np = np.array(img)

        # Upscale (Real-ESRGAN works with BGR numpy arrays)
        import cv2
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        output_bgr, _ = upsampler.enhance(img_bgr, outscale=self.config.upscale.scale)
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

        # Resize to exact target (upscale might overshoot)
        result_img = Image.fromarray(output_rgb)
        if result_img.width != target_width or result_img.height != target_height:
            result_img = result_img.resize(
                (target_width, target_height), Image.LANCZOS
            )

        result_img.save(output_path, quality=95)
        elapsed = time.time() - t0

        logger.info(
            f"[Upscaler] Image {input_res[0]}x{input_res[1]} → "
            f"{target_width}x{target_height} in {elapsed:.1f}s"
        )

        return UpscaleResult(
            input_path=input_path,
            output_path=output_path,
            input_resolution=input_res,
            output_resolution=(target_width, target_height),
            processing_time=elapsed,
            frame_count=1,
        )

    def upscale_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        target_width: int = 1920,
        target_height: int = 1080,
        ffmpeg_bin: str = "ffmpeg",
    ) -> UpscaleResult:
        """
        Upscale a video clip frame-by-frame:
        1. Extract frames with FFmpeg
        2. Upscale each frame with Real-ESRGAN
        3. Reassemble with FFmpeg (preserving audio if present)
        """
        import numpy as np
        import cv2

        if not output_path:
            p = Path(input_path)
            output_path = str(p.parent / f"{p.stem}_hd{p.suffix}")

        # Create temp dir for frames
        temp_dir = Path(output_path).parent / f"_upscale_temp_{Path(input_path).stem}"
        frames_in = temp_dir / "frames_in"
        frames_out = temp_dir / "frames_out"
        frames_in.mkdir(parents=True, exist_ok=True)
        frames_out.mkdir(parents=True, exist_ok=True)

        t0 = time.time()

        try:
            # Step 1: Extract frames
            logger.info(f"[Upscaler] Extracting frames from {input_path}")
            fps = self._get_video_fps(input_path, ffmpeg_bin)

            extract_cmd = [
                ffmpeg_bin, "-y", "-i", input_path,
                "-qscale:v", "2",
                str(frames_in / "frame_%06d.png")
            ]
            subprocess.run(extract_cmd, capture_output=True, check=True, timeout=120)

            frame_files = sorted(frames_in.glob("frame_*.png"))
            total_frames = len(frame_files)
            logger.info(f"[Upscaler] Extracted {total_frames} frames at {fps} fps")

            if total_frames == 0:
                raise RuntimeError("No frames extracted from video")

            # Get input resolution from first frame
            first_frame = cv2.imread(str(frame_files[0]))
            input_res = (first_frame.shape[1], first_frame.shape[0])

            # Step 2: Upscale each frame — fall back to ffmpeg scaling if the
            # Real-ESRGAN model/weights aren't installed.
            # Preferred: real AI upscale via ComfyUI 4x ESRGAN (crisp detail).
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return self._comfyui_esrgan_upscale(input_path, output_path,
                                                    target_width, target_height, ffmpeg_bin)
            except Exception as e:
                logger.warning(f"[Upscaler] ComfyUI ESRGAN failed ({e}); trying python ESRGAN")
                frames_in.mkdir(parents=True, exist_ok=True); frames_out.mkdir(parents=True, exist_ok=True)
            try:
                loaded = self.manager.load(ModelType.UPSCALER)
                upsampler = loaded.model
            except Exception as e:
                logger.warning(f"[Upscaler] ESRGAN unavailable ({e}); using ffmpeg lanczos upscale")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return self._ffmpeg_upscale(input_path, output_path,
                                            target_width, target_height, ffmpeg_bin)

            for i, frame_path in enumerate(frame_files):
                frame_bgr = cv2.imread(str(frame_path))
                upscaled_bgr, _ = upsampler.enhance(
                    frame_bgr, outscale=self.config.upscale.scale
                )

                # Resize to exact target
                if (upscaled_bgr.shape[1] != target_width or
                        upscaled_bgr.shape[0] != target_height):
                    upscaled_bgr = cv2.resize(
                        upscaled_bgr, (target_width, target_height),
                        interpolation=cv2.INTER_LANCZOS4
                    )

                out_name = f"frame_{i:06d}.png"
                cv2.imwrite(str(frames_out / out_name), upscaled_bgr)

                if (i + 1) % 20 == 0:
                    logger.info(f"[Upscaler] Frame {i+1}/{total_frames}")

            # Step 3: Reassemble video
            logger.info(f"[Upscaler] Reassembling {total_frames} frames")
            assemble_cmd = [
                ffmpeg_bin, "-y",
                "-framerate", str(fps),
                "-i", str(frames_out / "frame_%06d.png"),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                output_path
            ]

            # Copy audio if present
            has_audio = self._video_has_audio(input_path, ffmpeg_bin)
            if has_audio:
                assemble_cmd = [
                    ffmpeg_bin, "-y",
                    "-framerate", str(fps),
                    "-i", str(frames_out / "frame_%06d.png"),
                    "-i", input_path,
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-map", "0:v:0", "-map", "1:a:0?",
                    "-shortest",
                    output_path
                ]

            subprocess.run(assemble_cmd, capture_output=True, check=True, timeout=300)

            elapsed = time.time() - t0
            logger.info(
                f"[Upscaler] Video {input_res[0]}x{input_res[1]} → "
                f"{target_width}x{target_height} ({total_frames} frames) "
                f"in {elapsed:.1f}s"
            )

            return UpscaleResult(
                input_path=input_path,
                output_path=output_path,
                input_resolution=input_res,
                output_resolution=(target_width, target_height),
                processing_time=elapsed,
                frame_count=total_frames,
            )

        finally:
            # Cleanup temp frames
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_ffprobe(ffmpeg_bin: str):
        """Return a usable ffprobe path or None (bundled imageio ffmpeg has none)."""
        import shutil, os
        cand = ffmpeg_bin.replace("ffmpeg", "ffprobe")
        if os.path.isabs(cand) and os.path.exists(cand):
            return cand
        return shutil.which("ffprobe")

    @staticmethod
    def _get_video_fps(video_path: str, ffmpeg_bin: str = "ffmpeg") -> float:
        """Get FPS via ffprobe; fall back to parsing `ffmpeg -i`; default 24."""
        import re
        ffprobe = UpscalerService._resolve_ffprobe(ffmpeg_bin)
        if ffprobe:
            try:
                cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
                       "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if "/" in result.stdout.strip():
                    num, den = result.stdout.strip().split("/")
                    return float(num) / float(den)
            except Exception:
                pass
        try:
            out = subprocess.run([ffmpeg_bin, "-i", video_path],
                                 capture_output=True, text=True, timeout=30).stderr
            m = re.search(r"(\d+(?:\.\d+)?)\s*fps", out)
            if m:
                return float(m.group(1))
        except Exception:
            pass
        return 24.0

    @staticmethod
    def _video_has_audio(video_path: str, ffmpeg_bin: str = "ffmpeg") -> bool:
        """Check for an audio stream via ffprobe; fall back to `ffmpeg -i` parse."""
        ffprobe = UpscalerService._resolve_ffprobe(ffmpeg_bin)
        if ffprobe:
            try:
                cmd = [ffprobe, "-v", "error", "-select_streams", "a",
                       "-show_entries", "stream=index", "-of", "csv=p=0", video_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                return bool(result.stdout.strip())
            except Exception:
                pass
        try:
            out = subprocess.run([ffmpeg_bin, "-i", video_path],
                                 capture_output=True, text=True, timeout=30).stderr
            return "Audio:" in out
        except Exception:
            return False

    def _comfyui_esrgan_upscale(self, input_path, output_path, target_width, target_height,
                                ffmpeg_bin) -> "UpscaleResult":
        """4x ESRGAN (4x-UltraSharp) upscale via ComfyUI then downscale to target —
        genuine detail, not a smeared lanczos resize."""
        import shutil as _sh
        from pathlib import Path as _P
        from app.services.comfyui_client import ComfyUIClient, build_esrgan_video_upscale_workflow
        client = ComfyUIClient()
        if not client.wait_ready(30):
            raise RuntimeError("ComfyUI not reachable for ESRGAN upscale")
        client.free_vram()
        comfy_in = _P(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\input")
        comfy_in.mkdir(parents=True, exist_ok=True)
        name = f"up_{_P(input_path).stem}{_P(input_path).suffix or '.mp4'}"
        _sh.copy2(input_path, comfy_in / name)
        # detect fps from the source so we don't change playback speed
        fps = self._get_video_fps(input_path, ffmpeg_bin)
        t0 = time.time()
        wf = build_esrgan_video_upscale_workflow(
            video_filename=name, target_width=target_width, target_height=target_height, fps=fps)
        pid = client.submit(wf)
        hist = client.wait_for_completion(pid, timeout=1200, poll=2.0)
        _P(output_path).parent.mkdir(parents=True, exist_ok=True)
        client.collect_output(hist, output_path)
        logger.info(f"[Upscaler] ComfyUI 4x ESRGAN -> {target_width}x{target_height} in {time.time()-t0:.0f}s")
        return UpscaleResult(input_path=input_path, output_path=output_path,
                             input_resolution=(0, 0), output_resolution=(target_width, target_height),
                             processing_time=time.time() - t0, frame_count=0)

    def _ffmpeg_upscale(self, input_path, output_path, target_width, target_height,
                        ffmpeg_bin) -> "UpscaleResult":
        """Fallback upscaler using ffmpeg lanczos scaling (no ESRGAN model needed).
        Produces clean target-resolution video; not as sharp as ESRGAN but reliable."""
        t0 = time.time()
        # Lanczos upscale + a two-pass sharpen (unsharp) + light denoise to make
        # the soft LTX/SDXL output look crisp instead of blurry. force_original_
        # aspect_ratio=decrease + pad avoids any stretching.
        # lanczos scale + mild unsharp only. NO hqdn3d — the denoise smeared the
        # output into a "watery/paper" look.
        vf = (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,"
            f"unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.8"
        )
        cmd = [ffmpeg_bin, "-y", "-i", input_path, "-vf", vf,
               "-c:v", "libx264", "-preset", "slow", "-crf", "16",
               "-pix_fmt", "yuv420p", output_path]
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        logger.info(f"[Upscaler] ffmpeg lanczos+sharpen upscale -> {target_width}x{target_height} "
                    f"in {time.time()-t0:.1f}s (ESRGAN weights not installed)")
        return UpscaleResult(
            input_path=input_path, output_path=output_path,
            input_resolution=(0, 0), output_resolution=(target_width, target_height),
            processing_time=time.time() - t0, frame_count=0,
        )
