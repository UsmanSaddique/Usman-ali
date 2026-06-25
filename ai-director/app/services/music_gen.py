"""
AI Director — Music Generation Service
HeartMuLa 3B (primary) + ACE-Step (fallback) via ComfyUI.

HeartMuLa produces dramatically higher quality music than ACE-Step —
comparable to commercial services like Suno. It runs on 16GB VRAM
with 4-bit quantization via the ComfyUI_FL-HeartMuLa custom node.
"""
import os
import re
import sys
import time
import json
import random
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


def _parse_style_tags(style_prompt: str, channel_profile: dict = None) -> dict:
    """Extract structured fields from style prompt for HeartMuLa TagsBuilder.

    Maps free-form descriptions to HeartMuLa's actual tag vocabulary
    (the model was trained on specific tag formats, not long sentences).
    """
    prompt_lower = style_prompt.lower()
    music_cfg = (channel_profile or {}).get("music", {})

    # Map to HeartMuLa's GENRES list (must match training vocabulary)
    genre_map = {
        "nasheed": "folk",       # closest match in HeartMuLa vocab
        "islamic": "folk",
        "lullaby": "acoustic",
        "nursery": "acoustic",
        "kids": "acoustic",
        "children": "acoustic",
        "cinematic": "cinematic",
        "orchestral": "orchestral",
        "ambient": "ambient",
        "pop": "pop",
        "rock": "rock",
        "jazz": "jazz",
        "classical": "classical",
        "electronic": "electronic",
        "folk": "folk",
        "lo-fi": "lo-fi",
        "indie": "indie",
    }
    genre = ""
    for kw, val in genre_map.items():
        if kw in prompt_lower:
            genre = val
            break
    if not genre:
        genre = "acoustic"

    # Map to HeartMuLa's MOODS list
    mood_map = {
        "peaceful": "peaceful",
        "calm": "calm",
        "gentle": "calm",
        "warm": "uplifting",
        "reverent": "peaceful",
        "happy": "happy",
        "cheerful": "happy",
        "dreamy": "dreamy",
        "playful": "playful",
        "energetic": "energetic",
        "upbeat": "uplifting",
        "sad": "melancholic",
        "epic": "epic",
        "dark": "dark",
        "romantic": "romantic",
        "nostalgic": "nostalgic",
        "mysterious": "mysterious",
    }
    mood = ""
    # Check channel profile mood first
    if music_cfg.get("mood"):
        for kw, val in mood_map.items():
            if kw in music_cfg["mood"].lower():
                mood = val
                break
    if not mood:
        for kw, val in mood_map.items():
            if kw in prompt_lower:
                mood = val
                break
    if not mood:
        mood = "peaceful"

    # Extract instruments
    instrument_keywords = [
        "piano", "guitar", "ukulele", "xylophone", "flute", "violin",
        "drums", "bass", "synthesizer", "bells", "harp", "daf", "tabla",
        "sitar", "oud", "accordion", "trumpet", "cello", "marimba",
        "strings", "organ", "synth",
    ]
    found_instruments = [i for i in instrument_keywords if i in prompt_lower]
    if not found_instruments and music_cfg.get("instruments"):
        found_instruments = [i.strip() for i in music_cfg["instruments"]
                           if isinstance(i, str)]
    instruments = ", ".join(found_instruments) if found_instruments else "piano, strings, flute"

    # Vocal type — HeartMuLa expects: "female vocal", "male vocal", "instrumental", etc.
    vocal_type = "instrumental"
    if "vocal" in prompt_lower and "instrumental" not in prompt_lower:
        vocal_type = "female vocal"
    elif "hum" in prompt_lower:
        vocal_type = "female vocal"
    elif "singing" in prompt_lower:
        vocal_type = "female vocal"

    # Tempo — HeartMuLa expects: "slow", "medium", "fast", "very slow", "very fast"
    tempo = "medium"
    tempo_str = music_cfg.get("tempo", "").lower() if music_cfg else ""
    if not tempo_str:
        tempo_str = prompt_lower
    if "very slow" in tempo_str:
        tempo = "very slow"
    elif "slow" in tempo_str:
        tempo = "slow"
    elif "very fast" in tempo_str:
        tempo = "very fast"
    elif "fast" in tempo_str or "upbeat" in tempo_str:
        tempo = "fast"

    gender = "none" if vocal_type == "instrumental" else "female"

    return {
        "genre": genre,
        "mood": mood,
        "instruments": instruments,
        "vocal_type": vocal_type,
        "tempo": tempo,
        "gender": gender,
    }


class MusicGenService:
    """
    Generate background music — HeartMuLa 3B (primary), ACE-Step (fallback).

    HeartMuLa runs via ComfyUI_FL-HeartMuLa custom nodes with 4-bit
    quantization (~4.9GB VRAM). Falls back to ACE-Step if HeartMuLa
    nodes are not installed.
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
        channel_profile: dict = None,
    ) -> MusicResult:
        if not output_path:
            output_path = str(
                self.config.paths.assets_dir / "music" / f"music_{int(time.time())}.wav"
            )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        logger.info(f"[MusicGen] Generating {duration}s track: {style_prompt[:80]}")

        # 1. Try ACE-Step 1.5 XL Turbo (best quality — commercial grade)
        try:
            self._generate_acestep15xl(
                style_prompt=style_prompt, duration=duration,
                lyrics="" if instrumental else (lyrics or ""),
                instrumental=instrumental, output_path=output_path,
                channel_profile=channel_profile,
            )
            elapsed = time.time() - t0
            logger.info(f"[MusicGen] ACE-Step 1.5 XL done in {elapsed:.1f}s -> {output_path}")
            return MusicResult(path=output_path, duration=duration,
                               sample_rate=48000, style_prompt=style_prompt,
                               generation_time=elapsed)
        except Exception as e:
            logger.warning(f"[MusicGen] ACE-Step 1.5 XL unavailable ({e}); trying HeartMuLa")

        # 2. Fallback: HeartMuLa via ComfyUI
        try:
            self._generate_heartmula(
                style_prompt=style_prompt, duration=duration,
                lyrics="" if instrumental else (lyrics or ""),
                instrumental=instrumental, output_path=output_path,
                channel_profile=channel_profile,
            )
            elapsed = time.time() - t0
            logger.info(f"[MusicGen] HeartMuLa done in {elapsed:.1f}s -> {output_path}")
            return MusicResult(path=output_path, duration=duration,
                               sample_rate=48000, style_prompt=style_prompt,
                               generation_time=elapsed)
        except Exception as e:
            logger.warning(f"[MusicGen] HeartMuLa unavailable ({e}); trying ACE-Step v1 fallback")

        # 3. Last fallback: ACE-Step v1.0 via ComfyUI
        try:
            self._generate_acestep(
                style_prompt=style_prompt, duration=duration,
                lyrics="" if instrumental else (lyrics or ""),
                instrumental=instrumental, output_path=output_path,
            )
            elapsed = time.time() - t0
            logger.info(f"[MusicGen] ACE-Step v1 done in {elapsed:.1f}s -> {output_path}")
            return MusicResult(path=output_path, duration=duration,
                               sample_rate=self.config.music.default_sample_rate,
                               style_prompt=style_prompt, generation_time=elapsed)
        except Exception as e:
            logger.error(f"[MusicGen] All music generation methods failed: {e}")
            raise RuntimeError(f"Music generation failed — no engine available: {e}")

    def generate_for_channel(
        self,
        channel_profile: dict,
        video_duration: float,
        output_path: Optional[str] = None,
    ) -> MusicResult:
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

        lyrics = music_cfg.get("lyrics", "")
        instrumental = not bool(lyrics)

        music_duration = int(video_duration + 5)

        return self.generate(
            style_prompt=style_prompt,
            duration=music_duration,
            lyrics=lyrics,
            output_path=output_path,
            instrumental=instrumental,
            channel_profile=channel_profile,
        )

    # ── ACE-Step 1.5 XL via ComfyUI (primary — commercial grade) ──

    def _generate_acestep15xl(
        self, style_prompt: str, duration: int, lyrics: str,
        instrumental: bool, output_path: str,
        channel_profile: dict = None,
    ) -> None:
        from app.services.comfyui_client import ComfyUIClient, build_acestep15xl_workflow

        client = ComfyUIClient()
        if not client.wait_ready(30):
            raise RuntimeError("ComfyUI not reachable for ACE-Step 1.5 XL")

        client.free_vram()
        self.manager.unload()

        tags = _parse_style_tags(style_prompt, channel_profile)

        # Build clean style tags string for ACE-Step 1.5
        tag_parts = [tags["genre"]]
        if tags["vocal_type"] != "instrumental":
            tag_parts.append(tags["vocal_type"])
        else:
            tag_parts.append("instrumental")
        tag_parts.append(tags["mood"])
        if tags["instruments"]:
            tag_parts.append(tags["instruments"])
        # Add contextual tags
        if "islamic" in style_prompt.lower() or "nasheed" in style_prompt.lower():
            tag_parts.append("middle eastern")
        if "kids" in style_prompt.lower() or "children" in style_prompt.lower():
            tag_parts.append("children's music")
        clean_tags = ", ".join(tag_parts)

        # Map tempo string to approximate BPM
        bpm_map = {"very slow": 60, "slow": 75, "medium": 100, "fast": 130, "very fast": 160}
        bpm = bpm_map.get(tags["tempo"], 100)

        # Detect language for lyrics
        lang = "en"
        music_cfg = (channel_profile or {}).get("music", {})
        if "urdu" in style_prompt.lower() or "urdu" in str(music_cfg).lower():
            lang = "ur"
        elif "hindi" in style_prompt.lower():
            lang = "hi"
        elif "arabic" in style_prompt.lower():
            lang = "ar"

        logger.info(f"[MusicGen] ACE-Step 1.5 XL tags: {clean_tags} | bpm={bpm} | lang={lang}")

        wf = build_acestep15xl_workflow(
            style_tags=clean_tags,
            lyrics=lyrics,
            seconds=float(duration),
            seed=random.randint(0, 2**31 - 1),
            bpm=bpm,
            language=lang,
        )
        prompt_id = client.submit(wf)
        history = client.wait_for_completion(prompt_id, timeout=1800, poll=3.0)
        client.collect_output(history, output_path)

    # ── HeartMuLa via ComfyUI (secondary — good quality) ────────────

    def _generate_heartmula(
        self, style_prompt: str, duration: int, lyrics: str,
        instrumental: bool, output_path: str,
        channel_profile: dict = None,
    ) -> None:
        from app.services.comfyui_client import ComfyUIClient, build_heartmula_workflow

        client = ComfyUIClient()
        if not client.wait_ready(30):
            raise RuntimeError("ComfyUI not reachable for HeartMuLa music")

        client.free_vram()
        self.manager.unload()

        tags = _parse_style_tags(style_prompt, channel_profile)

        # Build clean additional_tags — short keyword tokens, not sentences
        extra_tags = []
        if "islamic" in style_prompt.lower() or "nasheed" in style_prompt.lower():
            extra_tags.append("middle eastern")
        if "kids" in style_prompt.lower() or "children" in style_prompt.lower():
            extra_tags.append("children's music")
        if "cinematic" in style_prompt.lower():
            extra_tags.append("soundtrack")
        if "lullaby" in style_prompt.lower():
            extra_tags.append("lullaby")
        additional = ", ".join(extra_tags) if extra_tags else ""

        logger.info(f"[MusicGen] HeartMuLa tags: genre={tags['genre']}, mood={tags['mood']}, "
                     f"vocal={tags['vocal_type']}, tempo={tags['tempo']}, "
                     f"instruments={tags['instruments']}, extra={additional}")

        wf = build_heartmula_workflow(
            style_tags=additional,
            lyrics=lyrics,
            duration_sec=float(duration),
            seed=random.randint(0, 2**31 - 1),
            genre=tags["genre"],
            mood=tags["mood"],
            instruments=tags["instruments"],
            vocal_type="instrumental" if instrumental else tags["vocal_type"],
            gender="none" if instrumental else tags["gender"],
            tempo=tags["tempo"],
        )
        prompt_id = client.submit(wf)
        history = client.wait_for_completion(prompt_id, timeout=1800, poll=3.0)
        client.collect_output(history, output_path)

    # ── ACE-Step via ComfyUI (fallback) ───────────────────────────

    def _generate_acestep(
        self, style_prompt: str, duration: int, lyrics: str,
        instrumental: bool, output_path: str,
    ) -> None:
        from app.services.comfyui_client import ComfyUIClient, build_acestep_workflow

        client = ComfyUIClient()
        if not client.wait_ready(30):
            raise RuntimeError("ComfyUI not reachable for ACE-Step music")

        client.free_vram()
        self.manager.unload()

        wf = build_acestep_workflow(
            style_tags=style_prompt,
            lyrics="" if instrumental else lyrics,
            seconds=float(duration),
            seed=random.randint(0, 2**31 - 1),
        )
        prompt_id = client.submit(wf)
        history = client.wait_for_completion(prompt_id, timeout=600, poll=2.0)
        client.collect_output(history, output_path)


# ── Model Loader Factory (legacy — kept for ModelManager registry) ────────

def create_music_loader(config) -> callable:
    def loader() -> LoadedModel:
        return LoadedModel(
            model_type=ModelType.MUSIC,
            name="heartmula-3b",
            model=None,
            extras={},
            vram_mb=5000,
        )
    return loader
