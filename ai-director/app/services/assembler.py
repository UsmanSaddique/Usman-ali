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
            "1440p": (2560, 1440),
            "4k": (3840, 2160),
            "2160p": (3840, 2160),
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

    # ── Narration mode: duration-exact assembly + 4-bus mix ───────────

    def assemble_narration(
        self,
        blocks: list[dict],       # [{path, duration, offset}] — offset = narration_start
        output_path: str,
        narration_path: str,      # the master VO WAV (timeline master)
        music_path: Optional[str] = None,
        sfx_tracks: Optional[list[dict]] = None,  # [{path, offset, gain_db}]
        resolution: str = "1080p",
        fps: int = 24,
        music_db: float = -21.0,   # music bed level before ducking
        sfx_db: float = -12.0,
        target_lufs: float = -14.0,  # YouTube normalization target
    ) -> AssemblyResult:
        """
        Narration-mode assembly. The VO track is the master clock:
        1. Every clip is trimmed/padded to EXACTLY its planned duration
           (freeze-frame pad via tpad when a clip rendered short), so each
           beat's visual starts precisely at its narration offset — straight
           cuts, zero cumulative drift by construction.
        2. Audio is a 4-bus mix: VO / music / SFX — music sidechain-ducks
           −7dB-ish under the voice, SFX are placed at absolute offsets.
        3. Master is loudness-normalized to −14 LUFS (YouTube), −1.5 dBTP.
        """
        t0 = time.time()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        res_map = {
            "1080p": (1920, 1080), "2k": (2560, 1440), "1440p": (2560, 1440),
            "4k": (3840, 2160), "2160p": (3840, 2160), "720p": (1280, 720),
        }
        w, h = res_map.get(resolution, (1920, 1080))
        if not blocks:
            raise ValueError("No clips to assemble")

        temp_dir = tempfile.mkdtemp(prefix="aidir_narr_")
        
        # Auto-inject transition SFX from local cache at cut points
        import random
        sfx_tracks = sfx_tracks or []
        sfx_lib_dir = self.config.paths.assets_dir / "sfx"
        if sfx_lib_dir.exists():
            cut_time = 0.0
            for i, b in enumerate(blocks[:-1]):
                dur = float(b["duration"])
                cut_time += dur
                # Decide transition type. If next clip is template (diagram/code), use 'pop'.
                # Otherwise, 70% whoosh, 30% impact.
                trans_type = "pop" if blocks[i+1].get("is_template") else random.choices(["whoosh", "impact"], weights=[0.7, 0.3])[0]
                cat_dir = sfx_lib_dir / trans_type
                if cat_dir.exists():
                    files = list(cat_dir.glob("*.mp3")) + list(cat_dir.glob("*.wav"))
                    if files:
                        sfx_file = random.choice(files)
                        # offset so the SFX peak hits right at the cut point
                        offset = max(0, cut_time - 0.5) 
                        sfx_tracks.append({
                            "path": str(sfx_file),
                            "offset": offset,
                            "gain_db": -12.0
                        })

        # 1) Normalize every block to exact duration/format
        norm_paths = []
        for i, b in enumerate(blocks):
            dur = float(b["duration"])
            norm = os.path.join(temp_dir, f"blk_{i:04d}.mp4")
            cmd = [
                self.ffmpeg, "-y", "-i", b["path"],
                "-vf",
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
                f"tpad=stop_mode=clone:stop_duration={dur + 2:.3f}",
                "-t", f"{dur:.3f}",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-an",
                norm,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(f"Block normalize failed for {b['path']}: "
                                   f"{r.stderr[-300:]}")
            norm_paths.append(norm)

        # 2) Straight-cut concat (demuxer, no re-encode)
        list_path = os.path.join(temp_dir, "concat.txt")
        with open(list_path, "w") as f:
            for p in norm_paths:
                f.write(f"file '{p}'\n")
        video_path = os.path.join(temp_dir, "video.mp4")
        subprocess.run([
            self.ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c", "copy", video_path,
        ], capture_output=True, check=True, timeout=600)

        video_dur = self._get_duration(video_path)

        # 3) 4-bus audio mix
        audio_path = os.path.join(temp_dir, "mix.wav")
        try:
            self._mix_narration_audio(
                narration_path, music_path, sfx_tracks or [],
                audio_path, video_dur, music_db, sfx_db, target_lufs)
        except Exception as mix_err:
            logger.warning(f"[Assembler] 4-bus mix failed ({mix_err}); "
                           f"falling back to simple VO+music mix")
            audio_path = self._mix_audio(narration_path, music_path,
                                         1.0, 0.18, video_dur)

        # 4) Mux
        self._mux_final(video_path, audio_path, output_path, fps)

        try:
            import shutil as _sh
            _sh.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

        elapsed = time.time() - t0
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        duration = self._get_duration(output_path)
        logger.info(f"[Assembler] Narration render: {duration:.1f}s, "
                    f"{file_size:.1f}MB in {elapsed:.1f}s")
        return AssemblyResult(
            output_path=output_path, total_duration=duration,
            resolution=resolution, file_size_mb=file_size, render_time=elapsed)

    def _mix_narration_audio(
        self,
        narration_path: str,
        music_path: Optional[str],
        sfx_tracks: list[dict],
        output_path: str,
        duration: float,
        music_db: float,
        sfx_db: float,
        target_lufs: float,
    ):
        """VO / music / SFX buses → sidechain duck → loudnorm master.
        Music+ambience duck under the voice (sidechaincompress ~-7dB feel,
        250ms release); SFX are dropped at absolute offsets via adelay."""
        inputs = ["-i", narration_path]
        n_in = 1
        filters = []

        # VO bus (also feeds the sidechain detector)
        filters.append("[0:a]aresample=48000,pan=stereo|c0=c0|c1=c0[vo];"
                       "[vo]asplit=2[vomix][vokey]")

        mix_srcs = ["[vomix]"]

        if music_path and os.path.exists(music_path):
            inputs += ["-stream_loop", "-1", "-i", music_path]
            music_idx = n_in
            n_in += 1
            filters.append(
                f"[{music_idx}:a]aresample=48000,volume={music_db}dB[musraw];"
                f"[musraw][vokey]sidechaincompress="
                f"threshold=0.02:ratio=6:attack=25:release=250:makeup=1[mus]")
            mix_srcs.append("[mus]")
        else:
            # sidechain key must still terminate somewhere
            filters.append("[vokey]anullsink")

        sfx_labels = []
        for k, sfx in enumerate(sfx_tracks):
            if not sfx.get("path") or not os.path.exists(sfx["path"]):
                continue
            inputs += ["-i", sfx["path"]]
            idx = n_in
            n_in += 1
            delay_ms = max(0, int(float(sfx.get("offset", 0)) * 1000))
            gain = float(sfx.get("gain_db", sfx_db))
            filters.append(
                f"[{idx}:a]aresample=48000,volume={gain}dB,"
                f"adelay={delay_ms}|{delay_ms}[sfx{k}]")
            sfx_labels.append(f"[sfx{k}]")
        if sfx_labels:
            filters.append(
                "".join(sfx_labels) +
                f"amix=inputs={len(sfx_labels)}:normalize=0:duration=longest[sfxbus]")
            mix_srcs.append("[sfxbus]")

        filters.append(
            "".join(mix_srcs) +
            f"amix=inputs={len(mix_srcs)}:normalize=0:duration=first,"
            f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11[out]")

        filter_graph = ";".join(filters)
        graph_file = output_path + ".filter.txt"
        with open(graph_file, "w", encoding="utf-8") as f:
            f.write(filter_graph)

        cmd = [
            self.ffmpeg, "-y", *inputs,
            "-filter_complex_script", graph_file,
            "-map", "[out]",
            "-t", f"{duration:.3f}",
            "-c:a", "pcm_s16le", "-ar", "48000",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        try:
            os.remove(graph_file)
        except Exception:
            pass
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg mix failed: {r.stderr[-400:]}")

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

        # xfade offsets must come from each clip's REAL duration, not the
        # planned scene duration: when a clip renders short (e.g. frame caps),
        # an offset past end-of-stream freezes the last frame until the fade.
        durations = []
        for clip in clips:
            d = self._get_duration(clip.path)
            durations.append(d if d > 0 else clip.duration)

        # Chain xfade transitions
        # First pair
        td = transition_duration
        if len(clips) == 2:
            offset = max(0, durations[0] - td)
            filter_parts.append(
                f"[v0][v1]xfade=transition=fade:duration={td}:offset={offset}[vout]"
            )
        else:
            # Chain: v0+v1→x0, x0+v2→x1, x1+v3→x2, ...
            offset = max(0, durations[0] - td)
            filter_parts.append(
                f"[v0][v1]xfade=transition=fade:duration={td}:offset={offset}[x0]"
            )
            running_offset = offset + durations[1] - td
            for i in range(2, len(clips)):
                prev = f"x{i-2}"
                out = f"x{i-1}" if i < len(clips) - 1 else "vout"
                offset = max(0, running_offset)
                filter_parts.append(
                    f"[{prev}][v{i}]xfade=transition=fade:duration={td}:offset={offset}[{out}]"
                )
                running_offset = offset + durations[i] - td

        filter_graph = ";\n".join(filter_parts)

        # Windows caps a command line at ~32k chars; with 100+ clips the inline
        # graph blows past it (WinError 206), so pass it as a script file.
        filter_script = os.path.join(temp_dir, "filter_complex.txt")
        with open(filter_script, "w", encoding="utf-8") as f:
            f.write(filter_graph)

        cmd = [
            self.ffmpeg, "-y",
            *inputs,
            "-filter_complex_script", filter_script,
            "-map", "[vout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-an",
            output,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
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
        """Get video duration in seconds.
        Prefers ffprobe; falls back to parsing `ffmpeg -i` when ffprobe is not
        installed (the bundled imageio_ffmpeg ships ffmpeg only, no ffprobe)."""
        import shutil, os, re
        ffprobe = self.ffmpeg.replace("ffmpeg", "ffprobe")
        if not (os.path.isabs(ffprobe) and os.path.exists(ffprobe)):
            ffprobe = shutil.which("ffprobe")
        if ffprobe:
            try:
                cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration",
                       "-of", "csv=p=0", video_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                return float(result.stdout.strip())
            except Exception:
                pass
        # Fallback: parse Duration from `ffmpeg -i`
        try:
            out = subprocess.run([self.ffmpeg, "-i", video_path],
                                 capture_output=True, text=True, timeout=30).stderr
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", out)
            if m:
                return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        except Exception:
            pass
        return 0.0
