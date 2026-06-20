"""
AI Director — Music Generation Service
ACE-Step v1.5 integration for background music generation.
Supports instrumental and vocal generation with style control.
"""
import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from app.services.model_manager import ModelManager, ModelType, LoadedModel

logger = logging.getLogger(__name__)


@dataclass
class MusicResult:
    path: str
    duration: float
    sample_rate: int
    style_prompt: str
    generation_time: float


class MusicGenService:
    """
    Generate background music using ACE-Step v1.5.

    ACE-Step supports:
    - Instrumental generation from style descriptions
    - Vocal generation with lyrics
    - Style transfer from reference audio
    - Tempo and key control

    For our pipeline we primarily use instrumental generation
    with mood/style prompts from the channel profile.
    """

    def __init__(self, model_manager: ModelManager, config):
        self.manager = model_manager
        self.config = config

    def generate(
        self,
        style_prompt: str,
        duration: int = 60,
        lyrics: Optional[str] = None,
        tempo_bpm: Optional[int] = None,
        output_path: Optional[str] = None,
        instrumental: bool = True,
    ) -> MusicResult:
        """
        Generate a music track.

        Args:
            style_prompt: Description of style (e.g., "soft piano lullaby, gentle, dreamy")
            duration: Target duration in seconds
            lyrics: Optional lyrics for vocal tracks (set instrumental=False)
            tempo_bpm: Target BPM (None = model decides)
            output_path: Where to save the audio file
            instrumental: If True, generate instrumental only
        """
        if not output_path:
            output_path = str(
                self.config.paths.assets_dir / "music" / f"music_{int(time.time())}.wav"
            )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        logger.info(f"[MusicGen] Generating {duration}s track: {style_prompt[:80]}")

        # Preferred: generate via ComfyUI ACE-Step (same backend as video)
        try:
            self._generate_comfyui(
                style_prompt=style_prompt, duration=duration, lyrics=lyrics,
                instrumental=instrumental, output_path=output_path,
            )
            elapsed = time.time() - t0
            logger.info(f"[MusicGen] ComfyUI ACE-Step done in {elapsed:.1f}s -> {output_path}")
            return MusicResult(path=output_path, duration=duration,
                               sample_rate=self.config.music.default_sample_rate,
                               style_prompt=style_prompt, generation_time=elapsed)
        except Exception as e:
            logger.warning(f"[MusicGen] ComfyUI ACE-Step unavailable ({e}); trying legacy paths")

        # Try direct Python import first, fall back to subprocess
        try:
            result = self._generate_direct(
                style_prompt=style_prompt,
                duration=duration,
                lyrics=lyrics,
                tempo_bpm=tempo_bpm,
                output_path=output_path,
                instrumental=instrumental,
            )
        except ImportError:
            logger.info("[MusicGen] ACE-Step not importable, using subprocess")
            result = self._generate_subprocess(
                style_prompt=style_prompt,
                duration=duration,
                lyrics=lyrics,
                tempo_bpm=tempo_bpm,
                output_path=output_path,
                instrumental=instrumental,
            )

        elapsed = time.time() - t0
        logger.info(f"[MusicGen] Generated in {elapsed:.1f}s → {output_path}")

        return MusicResult(
            path=output_path,
            duration=duration,
            sample_rate=self.config.music.default_sample_rate,
            style_prompt=style_prompt,
            generation_time=elapsed,
        )

    def generate_for_channel(
        self,
        channel_profile: dict,
        video_duration: float,
        output_path: Optional[str] = None,
    ) -> MusicResult:
        """
        Generate music tailored to a channel's profile.
        Builds the style prompt from the channel's music settings.
        """
        music_cfg = channel_profile.get("music", {})

        style_parts = []
        if music_cfg.get("style"):
            style_parts.append(music_cfg["style"])
        if music_cfg.get("mood"):
            style_parts.append(f"mood: {music_cfg['mood']}")
        if music_cfg.get("instruments"):
            instruments = ", ".join(music_cfg["instruments"])
            style_parts.append(f"instruments: {instruments}")

        style_prompt = ". ".join(style_parts) if style_parts else "gentle background music"

        tempo = music_cfg.get("tempo_bpm")

        # Music should be slightly longer than video for fade-out room
        music_duration = int(video_duration + 5)

        return self.generate(
            style_prompt=style_prompt,
            duration=music_duration,
            tempo_bpm=tempo,
            output_path=output_path,
            instrumental=True,
        )

    # ── ComfyUI ACE-Step Mode (preferred) ──────────────────────────────

    def _generate_comfyui(
        self, style_prompt: str, duration: int, lyrics: Optional[str],
        instrumental: bool, output_path: str,
    ) -> None:
        """Generate music via ComfyUI's ACE-Step nodes."""
        import random
        from app.services.comfyui_client import ComfyUIClient, build_acestep_workflow

        client = ComfyUIClient()
        if not client.wait_ready(30):
            raise RuntimeError("ComfyUI not reachable for ACE-Step music")

        # Free VRAM so the music model has room
        client.free_vram()
        self.manager.unload()

        wf = build_acestep_workflow(
            style_tags=style_prompt,
            lyrics="" if instrumental else (lyrics or ""),
            seconds=float(duration),
            seed=random.randint(0, 2**31 - 1),
        )
        prompt_id = client.submit(wf)
        history = client.wait_for_completion(prompt_id, timeout=600, poll=2.0)
        client.collect_output(history, output_path)

    # ── Direct Python Mode ─────────────────────────────────────────────

    def _generate_direct(
        self,
        style_prompt: str,
        duration: int,
        lyrics: Optional[str],
        tempo_bpm: Optional[int],
        output_path: str,
        instrumental: bool,
    ) -> None:
        """Generate using ACE-Step Python API directly."""
        import torch

        # ACE-Step import (adjust path if needed)
        ace_path = self.config.music.path
        if str(ace_path) not in sys.path:
            sys.path.insert(0, str(ace_path))

        from ace_step.pipeline import ACEStepPipeline

        # Load model through our manager for VRAM tracking
        loaded = self.manager.load(ModelType.MUSIC)
        pipe = loaded.model

        # If not loaded through manager, load directly
        if pipe is None:
            pipe = ACEStepPipeline.from_pretrained(str(ace_path))
            pipe = pipe.to("cuda")

        gen_kwargs = {
            "prompt": style_prompt,
            "duration": duration,
            "instrumental": instrumental,
        }
        if lyrics and not instrumental:
            gen_kwargs["lyrics"] = lyrics
        if tempo_bpm:
            gen_kwargs["tempo"] = tempo_bpm

        with torch.inference_mode():
            audio = pipe(**gen_kwargs)

        # Save audio
        import torchaudio
        if isinstance(audio, torch.Tensor):
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            torchaudio.save(
                output_path, audio.cpu(),
                self.config.music.default_sample_rate
            )
        else:
            # Some ACE-Step versions return numpy array
            import numpy as np
            import soundfile as sf
            if isinstance(audio, np.ndarray):
                sf.write(output_path, audio, self.config.music.default_sample_rate)

    # ── Subprocess Mode ────────────────────────────────────────────────

    def _generate_subprocess(
        self,
        style_prompt: str,
        duration: int,
        lyrics: Optional[str],
        tempo_bpm: Optional[int],
        output_path: str,
        instrumental: bool,
    ) -> None:
        """Generate by calling ACE-Step as a subprocess."""
        # Unload current model to free VRAM for subprocess
        self.manager.unload()

        ace_path = self.config.music.path
        script = ace_path / "generate.py"  # ACE-Step CLI script

        cmd = [
            sys.executable,  # use same Python
            str(script),
            "--prompt", style_prompt,
            "--duration", str(duration),
            "--output", output_path,
        ]
        if instrumental:
            cmd.append("--instrumental")
        if lyrics and not instrumental:
            cmd.extend(["--lyrics", lyrics])
        if tempo_bpm:
            cmd.extend(["--tempo", str(tempo_bpm)])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ACE-Step subprocess failed: {result.stderr[-500:]}")


# ── Model Loader Factory ──────────────────────────────────────────────────

def create_music_loader(config) -> callable:
    """Factory for ACE-Step model loader. Register with ModelManager."""

    def loader() -> LoadedModel:
        import torch

        ace_path = config.music.path
        sys.path.insert(0, str(ace_path))

        try:
            from ace_step.pipeline import ACEStepPipeline
            pipe = ACEStepPipeline.from_pretrained(str(ace_path))
            pipe = pipe.to("cuda")
        except ImportError:
            logger.warning("[MusicGen] ACE-Step not installed, music gen will use subprocess")
            pipe = None

        return LoadedModel(
            model_type=ModelType.MUSIC,
            name=config.music.name,
            model=pipe,
            extras={},
            vram_mb=6000,
        )

    return loader
