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


def _acestep_style_tags(style_prompt: str, condensed: str) -> str:
    """ACE-Step 1.5 was trained on rich free-form captions — collapsing the
    style prompt to HeartMuLa's sparse tag vocabulary throws away most of the
    direction (genre nuance, choir, hooks, rhythm feel) and produces generic
    songs. Pass the full comma-separated style prompt through, and only fall
    back to the condensed tags when no style prompt exists."""
    s = (style_prompt or "").strip().strip(",")
    return s if s else condensed


# ── ACE-Step master prompt (user-mandated format, 2026-07-19) ─────────────
# Every ACE-Step generation is wrapped in this cinematic producer brief.
# [Genre]/[Emotion]/[BPM]/[Key] slots are filled from the channel/style
# brief; the original style string rides in the Genre slot so channel
# identity (nursery-rhyme, nasheed, lullaby...) is never lost.

ACESTEP_MASTER_TEMPLATE = """Create a cinematic, commercially releasable, studio-quality track with flawless production.

Genre:
{genre}

Mood:
{mood}

Tempo:
{bpm} BPM

Key:
{key}

Song Structure:
Intro (8 bars) → Verse → Pre-Chorus → Chorus → Verse 2 → Bridge → Final Chorus → Outro

Production Quality:
- Modern 2026 chart-quality mix.
- Crystal-clear stereo imaging.
- Ultra-clean mastering.
- Wide dynamic range.
- Punchy yet controlled low end.
- Warm analog saturation.
- High-end sparkle without harshness.
- No clipping or distortion.
- Professional loudness suitable for Spotify, Apple Music and YouTube.

Instrumentation:
{instrumentation}

Melody:
- Emotional.
- Memorable.
- Strong hook.
- Catchy motifs.
- Natural progression.
- Dynamic variation.
- Professional songwriting.

Harmony:
- Rich chord extensions.
- Emotional transitions.
- Modern cinematic harmony.
- Layered textures.

Rhythm:
- Humanized timing.
- Groove-focused.
- Natural swing.
- Professional drum programming.

{vocals_block}

Mix:
- Every instrument occupies its own frequency space.
- Excellent instrument separation.
- Clean transient response.
- Balanced frequencies.
- Powerful but controlled bass.
- Wide stereo field.
- Mono-compatible.

Master:
- Radio-ready.
- Streaming-ready.
- Commercial loudness.
- Premium mastering chain.

Creative Direction:
- Build tension gradually.
- Emotional climax in final chorus.
- Cinematic transitions.
- Smooth automation.
- Professional arrangement.
- Avoid repetition.
- Introduce subtle variations every section.
- Add tasteful ear candy throughout.

Reference Style:
Inspired by the production quality of Hans Zimmer, Illenium, Martin Garrix, Alan Walker, and modern Hollywood soundtrack engineering, while remaining completely original and non-derivative.

Negative Prompt:
Avoid muddy mix, distorted bass, clipping, harsh highs, robotic vocals, repetitive melodies, weak transitions, poor mastering, thin sound, off-key notes, timing issues, excessive compression, low-quality synths, amateur arrangement, or generic loops.

Output:
Generate a premium, emotionally engaging, cinematic, high-fidelity production suitable for commercial release and professional streaming platforms."""

_VOCALS_BLOCK = """Vocals:
- Expressive.
- Emotional.
- Natural breathing.
- Perfect pronunciation.
- Studio-quality recording.
- Rich harmonies.
- Layered backing vocals.
- Wide chorus doubles.
- Smooth vocal automation.
- Professional vocal chain with EQ, compression, de-essing, saturation and reverb."""

_INSTRUMENTAL_BLOCK = """Vocals:
- Instrumental only — no vocals, no humming, no voice of any kind."""

_DEFAULT_INSTRUMENTATION = """- Layered cinematic synths.
- Rich atmospheric pads.
- Deep sub bass.
- Tight punchy kick.
- Clean snare.
- Crisp hi-hats.
- Organic percussion.
- Modern FX.
- Reverse impacts.
- Risers.
- Downlifters.
- Textural ambience.
- Live strings and piano where appropriate."""


def _extract_key(style_prompt: str, mood: str) -> str:
    """Explicit key in the brief wins ('D major', 'key of F#m'); otherwise
    pick a sensible default from the mood."""
    m = re.search(r"\b([A-G][#b]?)\s*(major|minor|maj|min)\b",
                  style_prompt or "", re.IGNORECASE)
    if m:
        quality = "major" if m.group(2).lower().startswith("maj") else "minor"
        return f"{m.group(1).upper()} {quality}"
    if mood in ("dark", "melancholic", "mysterious", "sad", "epic"):
        return "A minor"
    return "C major"


def _acestep_master_prompt(style_brief: str, tags: dict, bpm: int,
                           instrumental: bool) -> str:
    """Wrap the channel/style brief in the mandated cinematic master
    template. The full brief goes in the Genre slot verbatim so nothing
    the channel author wrote is thrown away."""
    genre = (style_brief or "").strip().rstrip(".") or tags.get("genre", "cinematic")
    mood = tags.get("mood", "uplifting")
    instruments = tags.get("instruments", "")
    if instruments:
        instrumentation = "\n".join(
            f"- {i.strip().capitalize()}." for i in instruments.split(",") if i.strip())
        instrumentation += "\n- Rich atmospheric pads.\n- Textural ambience."
    else:
        instrumentation = _DEFAULT_INSTRUMENTATION
    vocals_block = _INSTRUMENTAL_BLOCK if instrumental else _VOCALS_BLOCK
    return ACESTEP_MASTER_TEMPLATE.format(
        genre=genre,
        mood=mood,
        bpm=bpm,
        key=_extract_key(style_brief, mood),
        instrumentation=instrumentation,
        vocals_block=vocals_block,
    )


def _explicit_bpm(style_prompt: str, fallback: int) -> int:
    """Honor an explicit 'NNN bpm' in the style prompt over the coarse
    tempo-word buckets (which map 'upbeat' to 130 even when the author
    asked for 100)."""
    m = re.search(r"(\d{2,3})\s*bpm", (style_prompt or "").lower())
    if m:
        bpm = int(m.group(1))
        if 40 <= bpm <= 220:
            return bpm
    return fallback


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
    Generate background music — ACE-Step 1.5 XL SFT (primary), Turbo, HeartMuLa, v1.0 fallbacks.

    Priority: SFT (50-step, highest quality) → Turbo (8-step, fast) → HeartMuLa 3B → ACE-Step v1.0.
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
        engine: str = "auto",
    ) -> MusicResult:
        """`engine` pins a specific generator ("sft"|"turbo"|"heartmula"|"ace1");
        "auto" tries them in quality order with fallback."""
        if not output_path:
            output_path = str(
                self.config.paths.assets_dir / "music" / f"music_{int(time.time())}.wav"
            )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        logger.info(f"[MusicGen] Generating {duration}s track (engine={engine}): {style_prompt[:80]}")

        lyrics_arg = "" if instrumental else (lyrics or "")

        chain = [
            ("sft", "ACE-Step 1.5 XL SFT", lambda: self._generate_acestep15xl_sft(
                style_prompt=style_prompt, duration=duration, lyrics=lyrics_arg,
                instrumental=instrumental, output_path=output_path, channel_profile=channel_profile)),
            ("turbo", "ACE-Step 1.5 XL Turbo", lambda: self._generate_acestep15xl(
                style_prompt=style_prompt, duration=duration, lyrics=lyrics_arg,
                instrumental=instrumental, output_path=output_path, channel_profile=channel_profile)),
            ("heartmula", "HeartMuLa", lambda: self._generate_heartmula(
                style_prompt=style_prompt, duration=duration, lyrics=lyrics_arg,
                instrumental=instrumental, output_path=output_path, channel_profile=channel_profile)),
            ("ace1", "ACE-Step v1", lambda: self._generate_acestep(
                style_prompt=style_prompt, duration=duration, lyrics=lyrics_arg,
                instrumental=instrumental, output_path=output_path)),
        ]
        if engine != "auto":
            chain = [c for c in chain if c[0] == engine]
            if not chain:
                raise ValueError(f"Unknown music engine '{engine}'")

        last_err = None
        for key, label, fn in chain:
            try:
                fn()
                elapsed = time.time() - t0
                logger.info(f"[MusicGen] {label} done in {elapsed:.1f}s -> {output_path}")
                return MusicResult(path=output_path, duration=duration,
                                   sample_rate=48000, style_prompt=style_prompt,
                                   generation_time=elapsed)
            except Exception as e:
                last_err = e
                logger.warning(f"[MusicGen] {label} unavailable ({e})"
                               + ("; trying next" if engine == "auto" else ""))

        raise RuntimeError(f"Music generation failed — engine(s) unavailable: {last_err}")

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

        style_prompt = ". ".join(style_parts) if style_parts else "upbeat background music, strong rhythm, highly enjoyable"

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

    # ── ACE-Step 1.5 XL SFT via ComfyUI (highest quality — 50 steps) ──

    def _generate_acestep15xl_sft(
        self, style_prompt: str, duration: int, lyrics: str,
        instrumental: bool, output_path: str,
        channel_profile: dict = None,
    ) -> None:
        from app.services.comfyui_client import ComfyUIClient, build_acestep15xl_sft_workflow

        sft_dir = self.config.paths.models_dir / "diffusion_models" / "acestep-v15-xl-sft"
        if not (sft_dir / "model.safetensors.index.json").exists():
            raise RuntimeError("ACE-Step 1.5 XL SFT model not downloaded")

        client = ComfyUIClient()
        if not client.wait_ready(30):
            raise RuntimeError("ComfyUI not reachable for ACE-Step 1.5 XL SFT")

        client.free_vram()
        self.manager.unload()

        tags = _parse_style_tags(style_prompt, channel_profile)

        tag_parts = [tags["genre"]]
        if tags["vocal_type"] != "instrumental":
            tag_parts.append(tags["vocal_type"])
        else:
            tag_parts.append("instrumental")
        tag_parts.append(tags["mood"])
        if tags["instruments"]:
            tag_parts.append(tags["instruments"])
        if "islamic" in style_prompt.lower() or "nasheed" in style_prompt.lower():
            tag_parts.append("middle eastern")
        if "kids" in style_prompt.lower() or "children" in style_prompt.lower():
            tag_parts.append("children's music")
        clean_tags = _acestep_style_tags(style_prompt, ", ".join(tag_parts))

        bpm_map = {"very slow": 60, "slow": 75, "medium": 100, "fast": 130, "very fast": 160}
        bpm = _explicit_bpm(style_prompt, bpm_map.get(tags["tempo"], 100))

        lang = "en"
        music_cfg = (channel_profile or {}).get("music", {})
        if "urdu" in style_prompt.lower() or "urdu" in str(music_cfg).lower():
            lang = "ur"
        elif "hindi" in style_prompt.lower():
            lang = "hi"
        elif "arabic" in style_prompt.lower():
            lang = "ar"

        master_prompt = _acestep_master_prompt(clean_tags, tags, bpm, instrumental)
        logger.info(f"[MusicGen] ACE-Step 1.5 XL SFT master brief ({len(master_prompt)} chars): "
                    f"genre='{clean_tags[:80]}' | bpm={bpm} | lang={lang}")

        wf = build_acestep15xl_sft_workflow(
            style_tags=master_prompt,
            lyrics=lyrics,
            seconds=float(duration),
            seed=random.randint(0, 2**31 - 1),
            bpm=bpm,
            language=lang,
        )
        prompt_id = client.submit(wf)
        history = client.wait_for_completion(prompt_id, timeout=1800, poll=3.0)
        client.collect_output(history, output_path)

    # ── ACE-Step 1.5 XL Turbo via ComfyUI (fast — 8 steps) ──

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
        clean_tags = _acestep_style_tags(style_prompt, ", ".join(tag_parts))

        # Map tempo string to approximate BPM
        bpm_map = {"very slow": 60, "slow": 75, "medium": 100, "fast": 130, "very fast": 160}
        bpm = _explicit_bpm(style_prompt, bpm_map.get(tags["tempo"], 100))

        # Detect language for lyrics
        lang = "en"
        music_cfg = (channel_profile or {}).get("music", {})
        if "urdu" in style_prompt.lower() or "urdu" in str(music_cfg).lower():
            lang = "ur"
        elif "hindi" in style_prompt.lower():
            lang = "hi"
        elif "arabic" in style_prompt.lower():
            lang = "ar"

        master_prompt = _acestep_master_prompt(clean_tags, tags, bpm, instrumental)
        logger.info(f"[MusicGen] ACE-Step 1.5 XL master brief ({len(master_prompt)} chars): "
                    f"genre='{clean_tags[:80]}' | bpm={bpm} | lang={lang}")

        wf = build_acestep15xl_workflow(
            style_tags=master_prompt,
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

        tags = _parse_style_tags(style_prompt)
        bpm = _explicit_bpm(style_prompt, 100)
        wf = build_acestep_workflow(
            style_tags=_acestep_master_prompt(style_prompt, tags, bpm, instrumental),
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
