"""
AI Director — TTS Service
Text-to-Speech via WanGP Omnivoice (local API) + WhisperX forced alignment.
Generates narration audio and word-level timestamps for scene synchronization.
"""
import os
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    word: str
    start: float   # seconds
    end: float     # seconds


@dataclass
class TTSResult:
    audio_path: str
    duration: float
    sample_rate: int
    text: str
    word_timestamps: list[WordTimestamp] = field(default_factory=list)
    generation_time: float = 0.0


@dataclass
class NarrationSegment:
    scene_number: int
    text: str
    audio_path: str
    start_time: float   # offset in the full narration track
    end_time: float
    word_timestamps: list[WordTimestamp] = field(default_factory=list)


class TTSService:
    """
    Generate narration audio using WanGP Omnivoice local API.
    Then run WhisperX forced alignment for word-level timestamps.
    """

    def __init__(self, config):
        self.config = config
        self.api_url = config.tts.api_url
        self.default_voice = config.tts.default_voice
        self.default_speed = config.tts.default_speed

    # ── Single Text → Speech ───────────────────────────────────────────

    def generate(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> TTSResult:
        """
        Generate speech audio for a single text.
        Calls WanGP Omnivoice HTTP API.
        """
        voice = voice or self.default_voice
        speed = speed or self.default_speed

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        logger.info(f"[TTS] Generating: '{text[:60]}...'")

        # Call WanGP Omnivoice API
        try:
            response = httpx.post(
                f"{self.api_url}/api/tts",
                json={
                    "text": text,
                    "voice": voice,
                    "speed": speed,
                    "output_format": "wav",
                },
                timeout=60.0,
            )
            response.raise_for_status()

            # WanGP returns audio bytes directly
            with open(output_path, 'wb') as f:
                f.write(response.content)

        except httpx.HTTPError as e:
            logger.error(f"[TTS] WanGP API error: {e}")
            raise RuntimeError(f"TTS API failed: {e}")

        elapsed = time.time() - t0
        duration = self._get_audio_duration(output_path)

        logger.info(f"[TTS] Generated {duration:.1f}s audio in {elapsed:.1f}s")

        return TTSResult(
            audio_path=output_path,
            duration=duration,
            sample_rate=22050,
            text=text,
            generation_time=elapsed,
        )

    # ── Full Narration (all scenes) ────────────────────────────────────

    def generate_full_narration(
        self,
        scenes: list[dict],
        output_dir: str,
        voice: Optional[str] = None,
        pause_between: float = 0.5,
    ) -> tuple[str, list[NarrationSegment]]:
        """
        Generate narration for all scenes that have narration_text.
        Returns: (combined_audio_path, list of NarrationSegments)

        Each scene gets its own audio file, then they're concatenated
        with silence gaps matching scene timing.
        """
        segments: list[NarrationSegment] = []
        audio_files: list[tuple[str, float]] = []  # (path, pause_after)
        out_dir = Path(output_dir) / "narration"
        out_dir.mkdir(parents=True, exist_ok=True)

        for scene in scenes:
            text = scene.get("narration_text", "").strip()
            if not text:
                continue

            scene_num = scene["scene_number"]
            audio_path = str(out_dir / f"narration_{scene_num:03d}.wav")

            result = self.generate(text, audio_path, voice=voice)

            segment = NarrationSegment(
                scene_number=scene_num,
                text=text,
                audio_path=audio_path,
                start_time=0.0,  # calculated after concat
                end_time=result.duration,
            )
            segments.append(segment)
            audio_files.append((audio_path, pause_between))

        if not audio_files:
            logger.info("[TTS] No narration text in any scene")
            return "", []

        # Concatenate all narration segments with pauses
        combined_path = str(out_dir / "narration_full.wav")
        self._concat_audio_with_pauses(audio_files, combined_path)

        # Run WhisperX alignment on the combined audio
        try:
            timestamps = self.align_audio(combined_path, " ".join(
                s.text for s in segments
            ))
            # Distribute timestamps back to segments
            self._distribute_timestamps(segments, timestamps)
        except Exception as e:
            logger.warning(f"[TTS] WhisperX alignment failed: {e}")

        # Calculate running time offsets
        running_time = 0.0
        for seg in segments:
            seg.start_time = running_time
            seg_duration = self._get_audio_duration(seg.audio_path)
            seg.end_time = running_time + seg_duration
            running_time = seg.end_time + pause_between

        return combined_path, segments

    # ── WhisperX Forced Alignment ──────────────────────────────────────

    def align_audio(
        self,
        audio_path: str,
        transcript: str,
    ) -> list[WordTimestamp]:
        """
        Run WhisperX forced alignment to get word-level timestamps.
        Returns list of WordTimestamp for each word.
        """
        logger.info(f"[TTS] Running WhisperX alignment on {audio_path}")

        try:
            import whisperx

            # Load alignment model (uses GPU if available, but small footprint)
            device = "cuda"
            model = whisperx.load_align_model(language_code="en", device=device)

            # WhisperX needs segments in a specific format
            audio = whisperx.load_audio(audio_path)

            # Create a "transcript" format WhisperX expects
            segments = [{
                "text": transcript,
                "start": 0.0,
                "end": self._get_audio_duration(audio_path),
            }]

            result = whisperx.align(
                segments, model, {"waveform": audio, "sample_rate": 16000},
                device=device,
                return_char_alignments=False,
            )

            timestamps = []
            for seg in result.get("segments", []):
                for word_info in seg.get("words", []):
                    timestamps.append(WordTimestamp(
                        word=word_info.get("word", ""),
                        start=word_info.get("start", 0.0),
                        end=word_info.get("end", 0.0),
                    ))

            logger.info(f"[TTS] Aligned {len(timestamps)} words")
            return timestamps

        except ImportError:
            logger.warning("[TTS] whisperx not installed, skipping alignment")
            return []

    # ── Audio Utilities ────────────────────────────────────────────────

    def _concat_audio_with_pauses(
        self,
        audio_files: list[tuple[str, float]],  # (path, pause_seconds_after)
        output_path: str,
    ):
        """Concatenate audio files with silence gaps using FFmpeg."""
        ffmpeg = self.config.paths.ffmpeg_bin

        # Build complex filter with silence pads
        inputs = []
        filter_parts = []
        concat_inputs = []

        for i, (path, pause) in enumerate(audio_files):
            inputs.extend(["-i", path])
            # Add the audio segment
            concat_inputs.append(f"[{i}:a]")
            # Add silence if not last
            if i < len(audio_files) - 1 and pause > 0:
                sr = 22050  # sample rate
                silence_id = f"s{i}"
                filter_parts.append(
                    f"aevalsrc=0:d={pause}:s={sr}:c=mono[{silence_id}]"
                )
                concat_inputs.append(f"[{silence_id}]")

        n_inputs = len(concat_inputs)
        concat_str = "".join(concat_inputs)
        filter_parts.append(f"{concat_str}concat=n={n_inputs}:v=0:a=1[out]")

        filter_graph = ";".join(filter_parts)

        cmd = [
            ffmpeg, "-y",
            *inputs,
            "-filter_complex", filter_graph,
            "-map", "[out]",
            "-c:a", "pcm_s16le",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            # Fallback: simple concat without pauses
            self._simple_concat_audio(
                [f for f, _ in audio_files], output_path
            )

    def _simple_concat_audio(self, audio_files: list[str], output_path: str):
        """Fallback: simple concatenation via ffmpeg concat demuxer."""
        import tempfile
        ffmpeg = self.config.paths.ffmpeg_bin

        list_file = tempfile.mktemp(suffix=".txt")
        with open(list_file, 'w') as f:
            for path in audio_files:
                f.write(f"file '{path}'\n")

        cmd = [
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        os.remove(list_file)

    def _distribute_timestamps(
        self,
        segments: list[NarrationSegment],
        timestamps: list[WordTimestamp],
    ):
        """Distribute word timestamps back to their respective segments."""
        word_idx = 0
        for seg in segments:
            seg_words = seg.text.split()
            seg.word_timestamps = []
            for _ in seg_words:
                if word_idx < len(timestamps):
                    seg.word_timestamps.append(timestamps[word_idx])
                    word_idx += 1

    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        """Get audio duration using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            audio_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError):
            return 0.0
