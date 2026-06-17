"""
AI Director — Assembler Service
FFmpeg-based final video assembly: clips + narration + music → rendered output.
"""
import os
import time
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClipEntry:
    path: str
    duration: float
    transition_in: str = "crossfade"   # crossfade, fade_from_black, cut
    transition_out: str = "crossfade"


@dataclass
class AssemblyResult:
    output_path: str
    total_duration: float
    resolution: str
    file_size_mb: float
    render_time: float


class AssemblerService:
    """Assemble final video from clips, narration, and music."""

    def __init__(self, config):
        self.config = config
        self.ffmpeg = config.paths.ffmpeg_bin

    def assemble(
        self,
        clips: list[ClipEntry],
        output_path: str,
        narration_path: Optional[str] = None,
        music_path: Optional[str] = None,
        music_volume: float = 0.3,       # background music volume (0-1)
        narration_volume: float = 1.0,
        transition_duration: float = 0.5,
        resolution: str = "1080p",
        fps: int = 24,
    ) -> AssemblyResult:
        """
        Assemble all clips into a final video with audio mixing.

        Strategy:
        1. Create a concat file for FFmpeg
        2. Concatenate all clips with crossfade transitions
        3. Mix narration + music audio tracks
        4. Mux video + mixed audio
        """
        t0 = time.time()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        res_map = {
            "1080p": (1920, 1080),
            "2k": (2560, 1440),
            "720p": (1280, 720),
        }
        target_w, target_h = res_map.get(resolution, (1920, 1080))

        if len(clips) == 0:
            raise ValueError("No clips to assemble")

        # Step 1: Concatenate video clips
        logger.info(f"[Assembler] Concatenating {len(clips)} clips")
        concat_path = self._concat_clips(
            clips, target_w, target_h, fps, transition_duration
        )

        # Step 2: Mix audio tracks
        audio_path = None
        if narration_path or music_path:
            logger.info("[Assembler] Mixing audio tracks")
            video_duration = self._get_duration(concat_path)
            audio_path = self._mix_audio(
                narration_path, music_path,
                narration_volume, music_volume,
                video_duration,
            )

        # Step 3: Mux video + audio → final output
        logger.info(f"[Assembler] Final mux → {output_path}")
        self._mux_final(concat_path, audio_path, output_path, fps)

        # Cleanup temp files
        if os.path.exists(concat_path) and concat_path != output_path:
            os.remove(concat_path)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        elapsed = time.time() - t0
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        duration = self._get_duration(output_path)

        logger.info(
            f"[Assembler] Done: {duration:.1f}s, {file_size:.1f}MB, "
            f"rendered in {elapsed:.1f}s"
        )

        return AssemblyResult(
            output_path=output_path,
            total_duration=duration,
            resolution=resolution,
            file_size_mb=file_size,
            render_time=elapsed,
        )

    # ── Internal Steps ─────────────────────────────────────────────────

    def _concat_clips(
        self,
        clips: list[ClipEntry],
        width: int,
        height: int,
        fps: int,
        transition_duration: float,
    ) -> str:
        """Concatenate clips with crossfade transitions via FFmpeg."""
        temp_dir = tempfile.mkdtemp(prefix="aidir_concat_")
        output = os.path.join(temp_dir, "concat.mp4")

        if len(clips) == 1:
            # Single clip — just scale
            cmd = [
                self.ffmpeg, "-y",
                "-i", clips[0].path,
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                       f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-r", str(fps),
                "-pix_fmt", "yuv420p",
                "-an",
                output,
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            return output

        # For multiple clips: use xfade filter chain
        # Build complex filter graph
        inputs = []
        filter_parts = []

        for i, clip in enumerate(clips):
            inputs.extend(["-i", clip.path])

        # Scale all inputs to same resolution
        for i in range(len(clips)):
            filter_parts.append(
                f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,fps={fps}[v{i}]"
            )

        # Chain xfade transitions
        # First pair
        td = transition_duration
        if len(clips) == 2:
            offset = max(0, clips[0].duration - td)
            filter_parts.append(
                f"[v0][v1]xfade=transition=fade:duration={td}:offset={offset}[vout]"
            )
        else:
            # Chain: v0+v1→x0, x0+v2→x1, x1+v3→x2, ...
            offset = max(0, clips[0].duration - td)
            filter_parts.append(
                f"[v0][v1]xfade=transition=fade:duration={td}:offset={offset}[x0]"
            )
            running_offset = offset + clips[1].duration - td
            for i in range(2, len(clips)):
                prev = f"x{i-2}"
                out = f"x{i-1}" if i < len(clips) - 1 else "vout"
                offset = max(0, running_offset)
                filter_parts.append(
                    f"[{prev}][v{i}]xfade=transition=fade:duration={td}:offset={offset}[{out}]"
                )
                running_offset = offset + clips[i].duration - td

        filter_graph = ";\n".join(filter_parts)

        cmd = [
            self.ffmpeg, "-y",
            *inputs,
            "-filter_complex", filter_graph,
            "-map", "[vout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-an",
            output,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            # Fallback: simple concat without transitions
            logger.warning(
                f"[Assembler] xfade failed, falling back to simple concat: "
                f"{result.stderr[-300:]}"
            )
            return self._simple_concat(clips, width, height, fps, temp_dir)

        return output

    def _simple_concat(
        self, clips: list[ClipEntry],
        width: int, height: int, fps: int, temp_dir: str,
    ) -> str:
        """Fallback: simple concat via demuxer (no transitions)."""
        # First normalize all clips to same format
        normalized = []
        for i, clip in enumerate(clips):
            norm_path = os.path.join(temp_dir, f"norm_{i:04d}.mp4")
            cmd = [
                self.ffmpeg, "-y", "-i", clip.path,
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                       f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-r", str(fps),
                "-pix_fmt", "yuv420p", "-an",
                norm_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            normalized.append(norm_path)

        # Write concat list
        list_path = os.path.join(temp_dir, "concat_list.txt")
        with open(list_path, 'w') as f:
            for p in normalized:
                f.write(f"file '{p}'\n")

        output = os.path.join(temp_dir, "concat.mp4")
        cmd = [
            self.ffmpeg, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output,
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        return output

    def _mix_audio(
        self,
        narration_path: Optional[str],
        music_path: Optional[str],
        narration_vol: float,
        music_vol: float,
        target_duration: float,
    ) -> str:
        """Mix narration and music into a single audio track."""
        temp_audio = tempfile.mktemp(suffix=".aac", prefix="aidir_audio_")

        if narration_path and music_path:
            # Mix both tracks
            cmd = [
                self.ffmpeg, "-y",
                "-i", narration_path,
                "-stream_loop", "-1", "-i", music_path,  # loop music to fill duration
                "-filter_complex",
                f"[0:a]volume={narration_vol}[narr];"
                f"[1:a]volume={music_vol}[mus];"
                f"[narr][mus]amix=inputs=2:duration=first:dropout_transition=3[out]",
                "-map", "[out]",
                "-c:a", "aac", "-b:a", "192k",
                "-t", str(target_duration),
                temp_audio,
            ]
        elif narration_path:
            cmd = [
                self.ffmpeg, "-y",
                "-i", narration_path,
                "-c:a", "aac", "-b:a", "192k",
                temp_audio,
            ]
        elif music_path:
            cmd = [
                self.ffmpeg, "-y",
                "-stream_loop", "-1", "-i", music_path,
                "-af", f"volume={music_vol},afade=t=out:st={target_duration-3}:d=3",
                "-c:a", "aac", "-b:a", "192k",
                "-t", str(target_duration),
                temp_audio,
            ]
        else:
            return None

        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        return temp_audio

    def _mux_final(
        self,
        video_path: str,
        audio_path: Optional[str],
        output_path: str,
        fps: int,
    ):
        """Mux video + audio into final output."""
        if audio_path:
            cmd = [
                self.ffmpeg, "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest",
                "-movflags", "+faststart",
                output_path,
            ]
        else:
            cmd = [
                self.ffmpeg, "-y",
                "-i", video_path,
                "-c:v", "copy",
                "-movflags", "+faststart",
                output_path,
            ]

        subprocess.run(cmd, capture_output=True, check=True, timeout=300)

    def _get_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        ffprobe = self.ffmpeg.replace("ffmpeg", "ffprobe")
        cmd = [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        try:
            return float(result.stdout.strip())
        except (ValueError, AttributeError):
            return 0.0
