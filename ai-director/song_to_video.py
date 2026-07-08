"""
Song-to-Video Autonomous Pipeline
==================================
Fully offline, overnight-capable pipeline:
  Song (ACE-Step SFT) → Lyrics Parse → Scene Prompts (Qwen LLM) →
  SDXL Stills → LTX Clips → ESRGAN Upscale → FFmpeg Assembly

Run:  python song_to_video.py
Or via ComfyUI embedded python:
  C:\\ComfyUI_windows_portable_nvidia_cu126\\ComfyUI_windows_portable\\python_embeded\\python.exe song_to_video.py
"""
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import random
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, ".")

from app.services.comfyui_client import (
    ComfyUIClient,
    COMFYUI_OUTPUT,
    build_acestep15xl_sft_workflow,
    build_sdxl_workflow,
    build_ltx_img2vid_workflow,
    build_esrgan_video_upscale_workflow,
)
from app.services.lyrics_parser import parse_lyrics, LyricSegment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("s2v")

COMFY_INPUT = Path(
    r"C:\ComfyUI_windows_portable_nvidia_cu126"
    r"\ComfyUI_windows_portable\ComfyUI\input"
)
OUTPUT_ROOT = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\assets_generated\song_videos")


# ── Song Configuration ────────────────────────────────────────────────────

@dataclass
class SongVideoConfig:
    slug: str
    title: str
    channel_slug: str
    lyrics: str
    style_tags: str
    bpm: int
    keyscale: str
    language: str           # "en" or "ur"
    duration_sec: float     # song duration
    song_path: Optional[str] = None  # pre-generated song; None = generate fresh
    use_llm: bool = True    # True = Qwen for prompts, False = template
    max_segments: int = 30  # cap scene count
    steps_music: int = 200  # ACE-Step diffusion steps
    steps_still: int = 30   # SDXL steps
    steps_video: int = 10   # LTX steps (distilled model)
    video_width: int = 832
    video_height: int = 480
    upscale_width: int = 1920
    upscale_height: int = 1080


# ── Checkpoint Manager ────────────────────────────────────────────────────

def load_checkpoint(project_dir: Path) -> dict:
    cp = project_dir / "checkpoint.json"
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(project_dir: Path, data: dict):
    cp = project_dir / "checkpoint.json"
    cp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ── ComfyUI Helpers ───────────────────────────────────────────────────────

client = ComfyUIClient()


def wait_comfy(timeout=600):
    if not client.wait_ready(timeout):
        raise RuntimeError("ComfyUI not reachable")


def free_vram():
    client.free_vram()
    time.sleep(3)


def collect_audio(history: dict, dest: Path) -> bool:
    for _nid, nout in history.get("outputs", {}).items():
        for entry in nout.get("audio", []):
            fn = entry.get("filename", "")
            sub = entry.get("subfolder", "")
            if not fn:
                continue
            src = COMFYUI_OUTPUT / sub / fn if sub else COMFYUI_OUTPUT / fn
            if src.exists():
                shutil.copy2(str(src), str(dest))
                return True
    return False


def collect_image(history: dict, dest: Path) -> bool:
    for _nid, nout in history.get("outputs", {}).items():
        for entry in nout.get("images", []):
            fn = entry.get("filename", "")
            sub = entry.get("subfolder", "")
            if not fn:
                continue
            src = COMFYUI_OUTPUT / sub / fn if sub else COMFYUI_OUTPUT / fn
            if src.exists():
                shutil.copy2(str(src), str(dest))
                return True
    return False


def collect_video(history: dict, dest: Path) -> bool:
    for _nid, nout in history.get("outputs", {}).items():
        for key in ("gifs", "videos", "images"):
            for entry in nout.get(key, []):
                fn = entry.get("filename", "")
                sub = entry.get("subfolder", "")
                if not fn:
                    continue
                src = COMFYUI_OUTPUT / sub / fn if sub else COMFYUI_OUTPUT / fn
                if src.exists():
                    shutil.copy2(str(src), str(dest))
                    return True
    return False


def get_audio_duration(path: str) -> float:
    """Get audio duration via ffprobe or ffmpeg fallback."""
    for tool in ("ffprobe", "ffmpeg"):
        try:
            if tool == "ffprobe":
                cmd = [tool, "-v", "error", "-show_entries", "format=duration",
                       "-of", "csv=p=0", path]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if r.returncode == 0:
                    return float(r.stdout.strip())
            else:
                r = subprocess.run([tool, "-i", path], capture_output=True, text=True, timeout=15)
                m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", r.stderr)
                if m:
                    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        except Exception:
            continue
    return 0.0


# ── Phase 1: Song Generation ─────────────────────────────────────────────

def phase_song(config: SongVideoConfig, project_dir: Path) -> str:
    if config.song_path and Path(config.song_path).exists():
        log.info(f"[Song] Using pre-generated: {config.song_path}")
        dest = project_dir / "song.flac"
        shutil.copy2(config.song_path, str(dest))
        return str(dest)

    log.info(f"[Song] Generating {config.duration_sec}s song, {config.steps_music} steps...")
    free_vram()

    seed = random.randint(0, 2**31 - 1)
    wf = build_acestep15xl_sft_workflow(
        style_tags=config.style_tags,
        lyrics=config.lyrics,
        seconds=config.duration_sec,
        seed=seed,
        steps=config.steps_music,
        bpm=config.bpm,
        language=config.language,
        keyscale=config.keyscale,
    )

    pid = client.submit(wf)
    log.info(f"[Song] prompt_id={pid}, seed={seed}")
    history = client.wait_for_completion(pid, timeout=3600, poll=5.0)
    dest = project_dir / "song.flac"
    if collect_audio(history, dest):
        log.info(f"[Song] OK -> {dest}")
        return str(dest)
    raise RuntimeError("Song generation produced no audio output")


# ── Phase 2: Lyrics Parsing ──────────────────────────────────────────────

def phase_parse(config: SongVideoConfig, song_path: str) -> list[LyricSegment]:
    duration = get_audio_duration(song_path)
    if duration <= 0:
        log.warning(f"[Parse] Could not detect duration, using config: {config.duration_sec}s")
        duration = config.duration_sec

    log.info(f"[Parse] Song duration: {duration:.1f}s, max_segments={config.max_segments}")
    segments = parse_lyrics(
        config.lyrics, duration,
        max_segments=config.max_segments,
        target_segment_sec=5.0,
    )
    log.info(f"[Parse] {len(segments)} segments created")
    for s in segments:
        log.info(f"  [{s.index:2d}] {s.start_sec:6.1f}-{s.end_sec:6.1f}s "
                 f"({s.duration:4.1f}s) [{s.section_type:6s}] "
                 f"{'(instrumental)' if s.is_instrumental else s.text[:60]}")
    return segments


# ── Phase 3: Scene Prompt Generation ──────────────────────────────────────

SONG_SCENE_SYSTEM_PROMPT = """You are an expert AI video director creating visuals for an Islamic children's music video.

You will receive a song's lyrics broken into timed segments. For EACH segment, generate a visual scene prompt that ILLUSTRATES the lyric content. The visuals should tell a gentle story with moral values matching the song's message.

## Visual Rules
1. Write ALL prompts in ENGLISH (the video model is English-trained) regardless of song language.
2. Every prompt MUST start with the shot type: "wide establishing shot,", "medium shot,", "medium wide shot,", "medium close-up,"
3. Alternate shot types — NEVER use the same type for consecutive scenes. Bias toward wider shots (~60%).
4. 45-85 words per prompt. Rich, cinematic descriptions: lighting, textures, depth, atmosphere.
5. NO readable text, NO human faces in close-up, NO copyrighted characters.
6. Each scene must show a gentle ACTION for animation (reaching, walking, looking, wind blowing).
7. Maintain VISUAL CONTINUITY: same art style, palette, and character descriptors across all scenes.
8. For instrumental/intro/outro segments: use beautiful environment shots (gardens, sky, mosque, nature).

## Character Consistency
{character_bible}

## Art Style
{art_style}

## Color Palette
{color_palette}

## Negative Prompt (use for ALL scenes)
{negative_prompt}

## Output Format
Return ONLY valid JSON — an array of objects, one per segment:
[
  {{
    "segment_index": 0,
    "prompt": "wide establishing shot, ... detailed English visual prompt ...",
    "camera_motion": "static"
  }},
  ...
]

camera_motion options: static, zoom_in, zoom_out, pan_left, pan_right, tilt_up
"""

TEMPLATE_ACTIONS = {
    "intro": [
        "golden sunrise breaking over a peaceful {setting}, warm volumetric light rays, floating dust motes",
        "gentle morning light illuminating a beautiful {setting}, soft mist, serene atmosphere",
    ],
    "hook": [
        "{character} standing joyfully in a sunlit {setting}, arms gently open, welcoming smile",
        "{character} looking up with wonder at the sky in a bright {setting}, gentle breeze",
    ],
    "verse": [
        "{character} walking gently through a beautiful {setting}, soft golden hour light",
        "{character} sitting peacefully in a cozy {setting}, reading a small book",
        "{character} reaching out to help a small bird in a sunny {setting}",
        "{character} kneeling on a soft prayer rug in a warm {setting}, hands raised in dua",
        "{character} sharing food with friends in a cheerful {setting}, warm smiles",
        "{character} looking at flowers blooming in a colorful {setting}, gentle wonder",
        "{character} walking hand in hand with a friend in a peaceful {setting}",
        "{character} sitting under a large tree in a serene {setting}, dappled sunlight",
    ],
    "chorus": [
        "{character} joyfully singing with arms raised in a bright {setting}, sparkles floating",
        "{character} clapping hands happily in a sunlit {setting}, warm celebration",
        "{character} dancing gently in a beautiful {setting}, soft particles floating around",
    ],
    "bridge": [
        "{character} looking thoughtfully at the stars in a calm {setting}, moonlit atmosphere",
        "{character} sitting quietly by a gentle stream in a peaceful {setting}, reflective mood",
    ],
    "outro": [
        "beautiful sunset over a peaceful {setting}, warm orange and pink sky, silhouette of a mosque",
        "soft twilight over a serene {setting}, first stars appearing, gentle peace",
    ],
}

SETTINGS = [
    "garden with blooming flowers and soft grass",
    "cozy sunlit room with arched windows and cushions",
    "mosque courtyard with ornate fountain and gentle mist",
    "meadow with wildflowers under a blue sky with soft clouds",
    "ancient stone pathway lined with lanterns and greenery",
    "rooftop terrace overlooking a warm sunset cityscape",
    "quiet library corner with wooden shelves and golden light",
    "peaceful riverbank with stepping stones and willow trees",
]

SHOT_TYPES = ["wide establishing shot", "medium wide shot", "medium shot",
              "wide shot", "medium close-up", "medium wide shot"]


def phase_prompts_template(
    segments: list[LyricSegment],
    config: SongVideoConfig,
    channel_profile: dict,
) -> list[dict]:
    """Generate scene prompts using templates (no LLM needed)."""
    log.info("[Prompts] Using template mode (no LLM)")

    char_bible = channel_profile.get("character_bible", [])
    art_style = channel_profile.get("art_style_phrase", "soft 3D Pixar-style cartoon render")
    palette = channel_profile.get("color_palette", "bright cheerful pastels")
    neg = channel_profile.get("negative_prompt_additions",
                              "photorealistic, realistic skin, text, watermark, scary, deformed, extra limbs, blurry")

    characters = char_bible if char_bible else ["a cute cartoon child in modest colorful clothing"]
    seed_hash = int(hashlib.md5(config.slug.encode()).hexdigest()[:8], 16)

    prompts = []
    for seg in segments:
        rng = random.Random(seed_hash + seg.index)

        sec = seg.section_type
        templates = TEMPLATE_ACTIONS.get(sec, TEMPLATE_ACTIONS["verse"])
        template = rng.choice(templates)

        character = characters[seg.index % len(characters)]
        # Extract just the name and description from character bible entry
        char_desc = character.split(":")[1].strip() if ":" in character else character
        setting = rng.choice(SETTINGS)

        action = template.format(character=char_desc, setting=setting)
        shot = SHOT_TYPES[seg.index % len(SHOT_TYPES)]
        motions = ["static", "zoom_in", "pan_left", "pan_right", "static", "zoom_in"]
        cam = motions[seg.index % len(motions)]

        prompt = (
            f"{shot}, {action}, "
            f"{art_style}, {palette}, "
            f"highly detailed, cinematic, soft volumetric lighting, depth of field, "
            f"beautifully rendered, warm rim light, gentle bokeh background"
        )

        prompts.append({
            "segment_index": seg.index,
            "prompt": prompt,
            "negative_prompt": neg,
            "camera_motion": cam,
        })

    return prompts


def phase_prompts_llm(
    segments: list[LyricSegment],
    config: SongVideoConfig,
    channel_profile: dict,
) -> list[dict]:
    """Generate scene prompts using local Qwen LLM."""
    log.info("[Prompts] Using LLM mode (Qwen 27B)")

    char_bible = channel_profile.get("character_bible", [])
    art_style = channel_profile.get("art_style_phrase", "soft 3D Pixar-style cartoon render")
    palette = channel_profile.get("color_palette", "bright cheerful pastels")
    neg = channel_profile.get("negative_prompt_additions",
                              "photorealistic, realistic skin, text, watermark, scary, deformed")

    system = SONG_SCENE_SYSTEM_PROMPT.format(
        character_bible="\n".join(f"- {c}" for c in char_bible) if char_bible else "(no characters defined)",
        art_style=art_style,
        color_palette=palette,
        negative_prompt=neg,
    )

    seg_desc = []
    for s in segments:
        label = "(instrumental)" if s.is_instrumental else s.text.replace("\n", " / ")
        seg_desc.append(f"Segment {s.index}: [{s.section_type}] {s.duration:.1f}s — {label}")

    user_msg = (
        f"Song: \"{config.title}\" ({config.language})\n"
        f"Total segments: {len(segments)}\n\n"
        + "\n".join(seg_desc)
        + "\n\nGenerate a visual scene prompt for EACH segment. Return JSON array."
    )

    free_vram()
    time.sleep(5)

    from app.config import settings
    from app.services.model_manager import ModelManager, ModelType, register_all_loaders

    manager = ModelManager()
    register_all_loaders(manager, settings)

    try:
        loaded = manager.load(ModelType.LLM)
        llm = loaded.model

        log.info("[Prompts] LLM loaded, generating prompts...")
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=8192,
            response_format={"type": "json_object"},
            stream=False,
        )

        raw = response["choices"][0]["message"].get("content", "") or ""
        log.info(f"[Prompts] LLM response: {len(raw)} chars")

        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("scenes") or data.get("prompts") or data.get("segments") or []
        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data)}")

        # Ensure negative_prompt is present
        for p in data:
            if "negative_prompt" not in p:
                p["negative_prompt"] = neg

        log.info(f"[Prompts] Got {len(data)} prompts from LLM")
        return data

    finally:
        manager.unload()
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        time.sleep(3)


# ── Phase 4: SDXL Still Generation ────────────────────────────────────────

def phase_stills(
    prompts: list[dict],
    segments: list[LyricSegment],
    config: SongVideoConfig,
    channel_profile: dict,
    project_dir: Path,
) -> list[Optional[str]]:
    """Generate SDXL stills for each segment."""
    log.info(f"[Stills] Generating {len(prompts)} SDXL stills")
    free_vram()

    images_dir = project_dir / "images"
    images_dir.mkdir(exist_ok=True)

    ckpt = "sd_xl_base_1.0.safetensors"
    style_loras = channel_profile.get("generation", {}).get("default_loras", [])
    # Also check config for style_loras (like samaritan-3d, yusuf)
    lora_pairs = None
    try:
        from app.config import settings
        sl = getattr(settings.image, "style_loras", None)
        if sl:
            lora_pairs = [tuple(x) for x in sl]
    except Exception:
        pass

    seed_base = int(hashlib.md5(config.slug.encode()).hexdigest()[:7], 16)
    results = []

    for i, p in enumerate(prompts):
        dest = images_dir / f"scene_{i:03d}.png"
        if dest.exists():
            log.info(f"  [{i:2d}] exists, skipping")
            results.append(str(dest))
            continue

        prompt_text = p.get("prompt", "")
        neg_text = p.get("negative_prompt", "")

        wf = build_sdxl_workflow(
            prompt=prompt_text,
            negative_prompt=neg_text,
            width=config.video_width,
            height=config.video_height,
            steps=config.steps_still,
            cfg=7.0,
            seed=seed_base + i,
            ckpt_name=ckpt,
            loras=lora_pairs,
            hires=True,
            hires_scale=1.5,
            hires_denoise=0.45,
            hires_steps=18,
        )

        try:
            pid = client.submit(wf)
            history = client.wait_for_completion(pid, timeout=300, poll=1.5)
            if collect_image(history, dest):
                log.info(f"  [{i:2d}] OK -> {dest.name}")
                results.append(str(dest))
            else:
                log.warning(f"  [{i:2d}] no image output")
                results.append(None)
        except Exception as e:
            log.error(f"  [{i:2d}] FAILED: {e}")
            results.append(None)

    ok = sum(1 for r in results if r)
    log.info(f"[Stills] {ok}/{len(prompts)} stills generated")
    return results


# ── Phase 5: LTX Video Clip Generation ───────────────────────────────────

def phase_clips(
    prompts: list[dict],
    image_paths: list[Optional[str]],
    segments: list[LyricSegment],
    config: SongVideoConfig,
    project_dir: Path,
) -> list[Optional[str]]:
    """Generate LTX img2vid clips from SDXL stills."""
    log.info(f"[Clips] Generating {len(prompts)} LTX video clips")
    free_vram()

    clips_dir = project_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)

    model = "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf"
    seed_base = int(hashlib.md5(config.slug.encode()).hexdigest()[:7], 16)
    results = []

    for i, (p, img_path) in enumerate(zip(prompts, image_paths)):
        dest = clips_dir / f"scene_{i:03d}.mp4"
        if dest.exists():
            log.info(f"  [{i:2d}] exists, skipping")
            results.append(str(dest))
            continue

        if not img_path or not Path(img_path).exists():
            log.warning(f"  [{i:2d}] no still image, skipping")
            results.append(None)
            continue

        seg = segments[i] if i < len(segments) else segments[-1]
        num_frames = max(int(seg.duration * 24), 49)  # at least ~2s
        num_frames = min(num_frames, 161)  # cap at ~6.7s to avoid OOM

        # Copy still to ComfyUI input
        img_name = f"s2v_{config.slug}_{i:03d}.png"
        shutil.copy2(img_path, str(COMFY_INPUT / img_name))

        prompt_text = p.get("prompt", "")
        neg_text = p.get("negative_prompt", "")

        # Add motion cues
        motion_prompt = (
            f"{prompt_text}, the subject moving and acting with lively dynamic motion, "
            f"gentle natural movement, subtle camera push-in"
        )
        motion_neg = f"{neg_text}, static, still image, frozen, motionless, no movement"

        wf = build_ltx_img2vid_workflow(
            model_filename=model,
            image_filename=img_name,
            prompt=motion_prompt,
            negative_prompt=motion_neg,
            width=config.video_width,
            height=config.video_height,
            num_frames=num_frames,
            steps=config.steps_video,
            cfg=1.0,
            seed=seed_base + i,
            fps=24,
            strength=0.6,
        )

        try:
            pid = client.submit(wf)
            history = client.wait_for_completion(pid, timeout=600, poll=2.0)
            if collect_video(history, dest):
                log.info(f"  [{i:2d}] OK ({num_frames}f) -> {dest.name}")
                results.append(str(dest))
            else:
                log.warning(f"  [{i:2d}] no video output")
                results.append(None)
        except Exception as e:
            log.error(f"  [{i:2d}] FAILED: {e}")
            results.append(None)

    ok = sum(1 for r in results if r)
    log.info(f"[Clips] {ok}/{len(prompts)} clips generated")
    return results


# ── Phase 6: ESRGAN Upscale ──────────────────────────────────────────────

def phase_upscale(
    clip_paths: list[Optional[str]],
    config: SongVideoConfig,
    project_dir: Path,
) -> list[Optional[str]]:
    """Upscale clips via ESRGAN through ComfyUI."""
    log.info(f"[Upscale] Upscaling {len(clip_paths)} clips to {config.upscale_width}x{config.upscale_height}")
    free_vram()

    hd_dir = project_dir / "clips_hd"
    hd_dir.mkdir(exist_ok=True)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)

    results = []
    for i, cp in enumerate(clip_paths):
        dest = hd_dir / f"scene_{i:03d}_hd.mp4"
        if dest.exists():
            log.info(f"  [{i:2d}] exists, skipping")
            results.append(str(dest))
            continue

        if not cp or not Path(cp).exists():
            results.append(None)
            continue

        # Copy clip to ComfyUI input
        vid_name = f"s2v_up_{config.slug}_{i:03d}.mp4"
        shutil.copy2(cp, str(COMFY_INPUT / vid_name))

        wf = build_esrgan_video_upscale_workflow(
            video_filename=vid_name,
            target_width=config.upscale_width,
            target_height=config.upscale_height,
            fps=24.0,
            model_name="4x-UltraSharp.pth",
        )

        try:
            pid = client.submit(wf)
            history = client.wait_for_completion(pid, timeout=600, poll=2.0)
            if collect_video(history, dest):
                log.info(f"  [{i:2d}] OK -> {dest.name}")
                results.append(str(dest))
            else:
                log.warning(f"  [{i:2d}] no upscaled output, using raw")
                results.append(cp)
        except Exception as e:
            log.error(f"  [{i:2d}] upscale failed ({e}), using raw")
            results.append(cp)

    ok = sum(1 for r in results if r)
    log.info(f"[Upscale] {ok}/{len(clip_paths)} clips upscaled")
    return results


# ── Phase 7: Final Assembly ──────────────────────────────────────────────

def phase_assemble(
    final_clips: list[Optional[str]],
    segments: list[LyricSegment],
    song_path: str,
    config: SongVideoConfig,
    project_dir: Path,
) -> str:
    """Assemble clips + song audio into final MP4."""
    log.info("[Assemble] Building final video")

    from app.services.assembler import AssemblerService, ClipEntry
    from app.config import settings

    assembler = AssemblerService(settings)

    clips = []
    for i, (cp, seg) in enumerate(zip(final_clips, segments)):
        if cp and Path(cp).exists():
            clips.append(ClipEntry(
                path=cp,
                duration=seg.duration,
                transition_in="crossfade",
                transition_out="crossfade",
            ))

    if not clips:
        raise RuntimeError("No clips available for assembly")

    # First clip fades from black
    clips[0].transition_in = "fade_from_black"

    output = str(project_dir / "final_render.mp4")

    result = assembler.assemble(
        clips=clips,
        output_path=output,
        narration_path=None,
        music_path=song_path,
        music_volume=1.0,       # Song is the PRIMARY audio, full volume
        transition_duration=0.3,
        resolution="1080p",
    )

    log.info(f"[Assemble] DONE: {result.total_duration:.1f}s, "
             f"{result.file_size_mb:.1f}MB -> {output}")
    return output


# ── Master Pipeline ──────────────────────────────────────────────────────

def run_pipeline(config: SongVideoConfig):
    """Run the full song-to-video pipeline with checkpointing."""
    project_dir = OUTPUT_ROOT / config.slug
    project_dir.mkdir(parents=True, exist_ok=True)

    # Save lyrics
    (project_dir / "lyrics.txt").write_text(config.lyrics, encoding="utf-8")

    cp = load_checkpoint(project_dir)
    t_start = time.time()

    log.info("=" * 70)
    log.info(f"  Song-to-Video: {config.title}")
    log.info(f"  Slug: {config.slug} | Lang: {config.language} | BPM: {config.bpm}")
    log.info(f"  Duration: {config.duration_sec}s | Max segments: {config.max_segments}")
    log.info(f"  LLM prompts: {config.use_llm} | Output: {project_dir}")
    log.info("=" * 70)

    wait_comfy()

    # Load channel profile
    try:
        import yaml
        profile_path = Path("channels") / f"{config.channel_slug}.yaml"
        if profile_path.exists():
            channel_profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        else:
            channel_profile = {}
    except Exception:
        channel_profile = {}

    # Phase 1: Song
    if cp.get("phase_song"):
        song_path = cp["song_path"]
        log.info(f"[Song] Resuming — already done: {song_path}")
    else:
        song_path = phase_song(config, project_dir)
        cp["phase_song"] = True
        cp["song_path"] = song_path
        save_checkpoint(project_dir, cp)

    # Phase 2: Parse
    if cp.get("phase_parse"):
        segments_data = cp["segments"]
        segments = [LyricSegment(**s) for s in segments_data]
        log.info(f"[Parse] Resuming — {len(segments)} segments")
    else:
        segments = phase_parse(config, song_path)
        cp["phase_parse"] = True
        cp["segments"] = [asdict(s) for s in segments]
        save_checkpoint(project_dir, cp)

    # Phase 3: Prompts
    if cp.get("phase_prompts"):
        prompts = cp["prompts"]
        log.info(f"[Prompts] Resuming — {len(prompts)} prompts")
    else:
        if config.use_llm:
            try:
                prompts = phase_prompts_llm(segments, config, channel_profile)
            except Exception as e:
                log.warning(f"[Prompts] LLM failed ({e}), falling back to template")
                prompts = phase_prompts_template(segments, config, channel_profile)
        else:
            prompts = phase_prompts_template(segments, config, channel_profile)

        # Ensure we have exactly as many prompts as segments
        while len(prompts) < len(segments):
            prompts.append(prompts[-1] if prompts else {"prompt": "beautiful scene", "negative_prompt": "", "camera_motion": "static"})
        prompts = prompts[:len(segments)]

        cp["phase_prompts"] = True
        cp["prompts"] = prompts
        save_checkpoint(project_dir, cp)

    # Phase 4: Stills
    if cp.get("phase_stills"):
        image_paths = cp["image_paths"]
        log.info(f"[Stills] Resuming — {sum(1 for p in image_paths if p)} stills")
    else:
        image_paths = phase_stills(prompts, segments, config, channel_profile, project_dir)
        cp["phase_stills"] = True
        cp["image_paths"] = image_paths
        save_checkpoint(project_dir, cp)

    # Phase 5: Clips
    if cp.get("phase_clips"):
        clip_paths = cp["clip_paths"]
        log.info(f"[Clips] Resuming — {sum(1 for p in clip_paths if p)} clips")
    else:
        clip_paths = phase_clips(prompts, image_paths, segments, config, project_dir)
        cp["phase_clips"] = True
        cp["clip_paths"] = clip_paths
        save_checkpoint(project_dir, cp)

    # Phase 6: Upscale
    if cp.get("phase_upscale"):
        upscaled = cp["upscaled_paths"]
        log.info(f"[Upscale] Resuming — {sum(1 for p in upscaled if p)} upscaled")
    else:
        upscaled = phase_upscale(clip_paths, config, project_dir)
        cp["phase_upscale"] = True
        cp["upscaled_paths"] = upscaled
        save_checkpoint(project_dir, cp)

    # Phase 7: Assemble
    final_clips = upscaled if upscaled else clip_paths
    output = phase_assemble(final_clips, segments, song_path, config, project_dir)

    cp["phase_assemble"] = True
    cp["output"] = output
    save_checkpoint(project_dir, cp)

    elapsed = time.time() - t_start
    log.info(f"\n{'=' * 70}")
    log.info(f"  COMPLETE: {config.title}")
    log.info(f"  Output: {output}")
    log.info(f"  Time: {elapsed/60:.1f} minutes")
    log.info(f"{'=' * 70}\n")

    return output


# ── Song Configurations ──────────────────────────────────────────────────

PREMIUM_DIR = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\assets_generated\music\islamic_kids_bilingual")

SONGS = [
    SongVideoConfig(
        slug="en_bismillah_twinkle",
        title="Bismillah Song for Kids — Say Bismillah Every Day!",
        channel_slug="little-muslim-nation",
        language="en",
        bpm=80,
        keyscale="C major",
        duration_sec=270.0,
        max_segments=30,
        song_path=str(PREMIUM_DIR / "en_01_bismillah_twinkle.flac"),
        style_tags=(
            "children's song, kids music, female lead vocal, "
            "children choir harmony, ukulele, glockenspiel, soft piano, "
            "gentle hand claps, warm strings, happy, uplifting, sing-along, "
            "twinkle twinkle melody style, lullaby, professional studio mix"
        ),
        lyrics="""[intro]
(gentle music box melody, soft piano)

[hook]
Bismillah, Bismillah, before I start my day,
Bismillah, Bismillah, the beautiful way.

[verse]
Bismillah before I eat my food,
Bismillah it puts me in the mood.
In the name of Allah kind and true,
Every single moment starts brand new.
When I open up a book to read,
Bismillah is all I really need.

[chorus]
Bismillah, Bismillah, say it with a smile,
Bismillah, Bismillah, every little while.
Bismillah, Bismillah, sweet and soft and true,
In the name of Allah, He takes care of you.
La la la la la la, He takes care of you.

[verse]
Bismillah when I wake up from my sleep,
Bismillah before I count my sheep.
When I tie my shoes and comb my hair,
Bismillah reminds me Allah's there.
Walking to the masjid hand in hand,
Bismillah across this beautiful land.

[chorus]
Bismillah, Bismillah, say it with a smile,
Bismillah, Bismillah, every little while.
Bismillah, Bismillah, sweet and soft and true,
In the name of Allah, He takes care of you.
La la la la la la, He takes care of you.

[bridge]
Every step I take, every breath I breathe,
Allah watches over you and me.
So whisper Bismillah, let your heart be light,
From the morning sun until the night.

[chorus]
Bismillah, Bismillah, say it with a smile,
Bismillah, Bismillah, every little while.
Bismillah, Bismillah, sweet and soft and true,
In the name of Allah, He takes care of you.

[outro]
Bismillah, Bismillah,
In the name of Allah.
(soft fade, music box)""",
    ),
    SongVideoConfig(
        slug="ur_bismillah_chanda",
        title="Bismillah Kaho — Urdu Islamic Song for Kids",
        channel_slug="little-muslim-nation",
        language="ur",
        bpm=84,
        keyscale="G major",
        duration_sec=270.0,
        max_segments=30,
        song_path=str(PREMIUM_DIR / "ur_02_bismillah_chanda_mama.flac"),
        style_tags=(
            "urdu nasheed, islamic kids song, female lead vocal, "
            "children choir harmony, soft duff frame drum, ney flute, "
            "qanun, soft piano, warm strings, south asian melodic style, "
            "chanda mama melody style, gentle lullaby, professional studio mix"
        ),
        lyrics="""[intro]
(soft sitar plucks, gentle piano melody)

[hook]
Bismillah kaho, barkat pao,
Bismillah kaho, khushiyan lao.
Har kaam mein Bismillah kaho,
Allah ka naam sab se pehle lao!

[verse]
Subah uthoon to Bismillah kahoon,
Nashta karoon to Bismillah kahoon.
School ki taraf nikalta hoon jab,
Bismillah kehta ja raha hoon tab.
Kitab kholoon, qalam uthaaon,
Bismillah keh ke likhna shuru karaon.

[chorus]
Bismillah, Bismillah, pyara pyara naam,
Bismillah, Bismillah, har roz har shaam.
Bismillah kehne se barkat aati hai,
Allah ki rehmat ghar mein samati hai.
Bismillah, Bismillah, la la la la la,
Bismillah, Bismillah, MashaAllah!

[verse]
Khana pakaye ammi jab bhi,
Bismillah kehti pehle sabhi.
Abbu jab gaari chalate hain,
Bismillah keh ke nikaltay hain.
Neend aaye to bistar pe jaoon,
Bismillah keh ke aankhein band karaon.

[chorus]
Bismillah, Bismillah, pyara pyara naam,
Bismillah, Bismillah, har roz har shaam.
Bismillah kehne se barkat aati hai,
Allah ki rehmat ghar mein samati hai.
Bismillah, Bismillah, la la la la la,
Bismillah, Bismillah, MashaAllah!

[bridge]
Chote haath jod ke dua karo,
Bismillah keh ke har kaam shuru karo.
Allah dekh raha hai tum ko pyaar se,
Bismillah kaho apne dil ke taar se.

[chorus]
Bismillah, Bismillah, pyara pyara naam,
Bismillah, Bismillah, har roz har shaam.
Bismillah kehne se barkat aati hai,
Allah ki rehmat ghar mein samati hai.

[outro]
Bismillah, Bismillah,
Allah ka pyara naam.
(gentle piano outro)""",
    ),
]


# ── 1-Min Quick Test Config ──────────────────────────────────────────────

TEST_SONG = SongVideoConfig(
    slug="test_dua_poem",
    title="Dua Ki Taqat — Chhoti Dua Badi Barkat (1-min test)",
    channel_slug="little-muslim-nation",
    language="ur",
    bpm=78,
    keyscale="D minor",
    duration_sec=60.0,
    max_segments=12,
    steps_music=50,       # fast for test (50 instead of 200)
    steps_still=25,       # fast for test
    steps_video=8,        # distilled model, 8 is fine
    use_llm=False,        # template mode — no Qwen needed, faster
    style_tags=(
        "urdu nasheed, islamic kids poem, soft female vocal, "
        "children choir harmony, duff frame drum, ney flute, "
        "warm piano, gentle strings, south asian melodic style, "
        "lullaby, devotional, professional studio mix"
    ),
    lyrics="""[intro]
(soft piano and ney flute, gentle melody)

[hook]
Dua karo, dua karo,
Allah se dil ki baat karo.
Haath uthao, aankhein band karo,
Pyaar se apni dua karo!

[verse]
Subah ki dua se din shuru,
Raat ki dua se neend ka guru.
Khana khao to Bismillah kaho,
Har qadam pe Allah ko yaad karo.

[chorus]
Dua ki taqat badi azeem,
Allah sunay har dua nazeem.
Chhoti si dua, badi barkat,
Allah deta hai behtareen!

[outro]
Dua karo, dua karo,
Allah se apni baat karo.
(gentle piano fade out)""",
)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    is_test = "--test" in sys.argv

    if is_test:
        song_list = [TEST_SONG]
        label = "TEST MODE (1 song, 1-min)"
    else:
        song_list = SONGS
        label = f"BATCH MODE ({len(SONGS)} songs)"

    print("\n" + "=" * 70)
    print(f"  SONG-TO-VIDEO Autonomous Pipeline — {label}")
    print(f"  Output: {OUTPUT_ROOT}")
    print("=" * 70 + "\n")

    overall_t0 = time.time()
    results = []

    for i, config in enumerate(song_list, 1):
        print(f"\n{'-' * 70}")
        print(f"  [{i}/{len(song_list)}] {config.title}")
        print(f"{'-' * 70}")

        try:
            output = run_pipeline(config)
            results.append((config.slug, "OK", output))
        except Exception as e:
            log.error(f"Pipeline failed for {config.slug}: {e}", exc_info=True)
            results.append((config.slug, f"FAIL: {e}", ""))

    total = time.time() - overall_t0
    print(f"\n{'=' * 70}")
    print(f"  {'TEST' if is_test else 'BATCH'} SUMMARY ({total/60:.1f} min total)")
    print(f"{'=' * 70}")
    for slug, status, path in results:
        print(f"  {slug:35s} {status}")
    ok = sum(1 for _, s, _ in results if s == "OK")
    print(f"\n  {ok}/{len(song_list)} succeeded")
    print(f"  Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
