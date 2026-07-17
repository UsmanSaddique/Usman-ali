"""
AI Director — QA Service
Automation-readiness gates for hands-off (overnight) production runs.

Three layers, called by the pipeline:
  1. preflight()      — before anything runs: engines, models, ffmpeg, disk, DB.
  2. lint_script()    — after script gen: creative rules (framing, durations,
                        negatives, scene types) with conservative auto-fixes.
  3. check_clip() / check_audio() / check_final() — after each artifact: the
                        file actually plays, has the expected duration/streams,
                        and clips contain real motion (not a frozen frame).

A per-project qa_report.json is written by the pipeline so a failed overnight
run can be diagnosed in one look.
"""
import json
import re
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

SHOT_TYPES = ["wide shot", "wide establishing shot", "medium wide shot",
              "full body shot", "medium shot", "medium close-up"]
# rotation used when auto-fixing framing (bias to wide per channel directive)
SHOT_ROTATION = ["wide shot", "medium wide shot", "wide establishing shot",
                 "medium shot", "full body shot", "medium wide shot"]


# ── Report Types ───────────────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    critical: bool = True


@dataclass
class PreflightReport:
    checks: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.critical)

    def summary(self) -> str:
        bad = [c for c in self.checks if not c.ok]
        if not bad:
            return "all checks passed"
        return "; ".join(f"{c.name}: {c.detail}" for c in bad)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks]}


@dataclass
class MediaQA:
    path: str
    ok: bool
    duration: float = 0.0
    width: int = 0
    height: int = 0
    has_audio: bool = False
    motion_score: float = -1.0   # -1 = not measured
    issues: list = field(default_factory=list)


# ── Preflight ──────────────────────────────────────────────────────────────

def preflight(config) -> PreflightReport:
    """Verify every engine/asset a full-auto run needs BEFORE burning GPU time."""
    r = PreflightReport()
    models = config.paths.models_dir

    # 1. ComfyUI reachable (video, stills, music, upscale all go through it)
    try:
        from app.services.comfyui_client import ComfyUIClient
        up = ComfyUIClient().ping()
        r.checks.append(Check("comfyui", up, "" if up else "not reachable at 127.0.0.1:8188 — start ComfyUI"))
    except Exception as e:
        r.checks.append(Check("comfyui", False, str(e)))

    # 2. Video model
    vid = models / "diffusion_models" / Path(str(config.video.model_path)).name
    r.checks.append(Check("video_model", vid.exists(), "" if vid.exists() else f"missing {vid.name}"))

    # 3. Still-image engine (Z-Image primary, SDXL fallback — either passes)
    zim = models / "diffusion_models" / getattr(config.image, "zimage_unet", "z_image_turbo_bf16.safetensors")
    sdxl = Path(str(config.image.path))
    img_ok = zim.exists() or sdxl.exists()
    r.checks.append(Check("image_model", img_ok,
                          "" if img_ok else "neither Z-Image nor SDXL weights found"))

    # 4. Music engine (any ACE-Step variant; non-critical — render can go silent)
    music_files = ["acestep_v1.5_xl_sft_bf16.safetensors",
                   "acestep_v1.5_xl_turbo_bf16.safetensors"]
    music_ok = any((models / "diffusion_models" / f).exists() for f in music_files) \
        or (models / "checkpoints" / "ace_step_v1_3.5b.safetensors").exists()
    r.checks.append(Check("music_model", music_ok,
                          "" if music_ok else "no ACE-Step weights found", critical=False))

    # 5. Upscaler (non-critical — pipeline falls back to raw clips)
    up_model = config.upscale.anime_model_path if getattr(config.upscale, "use_anime_model", False) \
        else config.upscale.model_path
    r.checks.append(Check("upscale_model", Path(str(up_model)).exists(),
                          "" if Path(str(up_model)).exists() else f"missing {Path(str(up_model)).name}",
                          critical=False))

    # 6. Director LLM (critical for script generation)
    llm = Path(str(config.llm.model_path)) if config.llm.model_path else None
    llm_ok = bool(llm and llm.exists())
    r.checks.append(Check("director_llm", llm_ok, "" if llm_ok else "Qwen GGUF missing — script gen will fail"))

    # 7. ffmpeg runs
    try:
        p = subprocess.run([config.paths.ffmpeg_bin, "-version"],
                           capture_output=True, timeout=10)
        r.checks.append(Check("ffmpeg", p.returncode == 0, "" if p.returncode == 0 else "ffmpeg -version failed"))
    except Exception as e:
        r.checks.append(Check("ffmpeg", False, str(e)))

    # 8. Disk space for renders (10GB floor on the assets drive)
    try:
        free_gb = shutil.disk_usage(str(config.paths.assets_dir)).free / (1024 ** 3)
        r.checks.append(Check("disk_space", free_gb >= 10.0, f"{free_gb:.1f}GB free"))
    except Exception as e:
        r.checks.append(Check("disk_space", False, str(e)))

    if not r.ok:
        logger.warning(f"[QA] Preflight FAILED: {r.summary()}")
    else:
        logger.info("[QA] Preflight passed")
    return r


# ── Creative Script Lint (auto-fixing) ─────────────────────────────────────

def lint_script(scenes: list, channel_profile: dict, target_duration: float,
                clip_range: tuple = (3.0, 8.0),
                video_clip_cap: float = None) -> list[str]:
    """Enforce the channel's creative rules on DB Scene rows, IN PLACE.

    Conservative auto-fixes (a senior creative pass, not a rewrite):
      - every prompt leads with a shot type; wide-biased rotation fills gaps
      - no two consecutive scenes share the exact same shot type
      - channel negative_prompt_additions are always present
      - still_ratio==0 channels force scene_type img2vid (real motion)
      - durations clamped to clip_range then scaled so the sum == target
    Returns a list of human-readable fix notes for the QA report.
    """
    profile = channel_profile or {}
    notes = []
    neg_additions = profile.get("negative_prompt_additions", "")
    force_img2vid = float(profile.get("still_ratio", 1.0) or 0.0) == 0.0

    prev_shot = None
    for i, scene in enumerate(scenes):
        prompt = (scene.prompt or "").strip()

        # framing: must lead with a shot type
        lead = next((s for s in SHOT_TYPES if prompt.lower().startswith(s)), None)
        if not lead:
            inline = next((s for s in SHOT_TYPES if s in prompt.lower()), None)
            lead = inline or SHOT_ROTATION[i % len(SHOT_ROTATION)]
            if not inline:
                prompt = f"{lead}, {prompt}"
                notes.append(f"scene {scene.scene_number}: prepended '{lead}' (framing rule)")

        # variety: never repeat the previous shot type verbatim
        if lead == prev_shot:
            alt = next(s for s in SHOT_ROTATION if s != lead)
            prompt = re.sub(re.escape(lead), alt, prompt, count=1, flags=re.IGNORECASE)
            notes.append(f"scene {scene.scene_number}: shot '{lead}' repeated — swapped to '{alt}'")
            lead = alt
        prev_shot = lead

        # negatives: channel additions always applied
        neg = (scene.negative_prompt or "").strip()
        if neg_additions and neg_additions.split(",")[0].strip().lower() not in neg.lower():
            scene.negative_prompt = f"{neg}, {neg_additions}" if neg else neg_additions
            notes.append(f"scene {scene.scene_number}: appended channel negative prompt")

        # scene type: channels with still_ratio 0 want real motion everywhere
        if force_img2vid and getattr(scene.scene_type, "value", str(scene.scene_type)) == "still_pan":
            from app.database import SceneType
            scene.scene_type = SceneType.IMG2VID
            notes.append(f"scene {scene.scene_number}: still_pan -> img2vid (channel wants real motion)")

        scene.prompt = prompt

    # durations: clamp, then scale the total to hit the target exactly.
    # video_clip_cap = the PHYSICAL per-clip ceiling (max_num_frames / fps —
    # e.g. 121f @ 24fps = 5.04s for LTX). A txt2vid/img2vid scene assigned
    # more than that silently renders at the cap, so the final video comes
    # out shorter than target. Only still_pan (ffmpeg, any length) is exempt.
    def _ceiling(scene) -> float:
        stype = getattr(scene.scene_type, "value", str(scene.scene_type))
        if video_clip_cap and stype != "still_pan":
            return float(video_clip_cap)
        return float("inf")

    lo, hi = clip_range
    for scene in scenes:
        d = float(scene.duration or lo)
        clamped = max(lo, min(hi, _ceiling(scene), d))
        if abs(clamped - d) > 0.01:
            notes.append(f"scene {scene.scene_number}: duration {d:.1f}s clamped to {clamped:.1f}s")
        scene.duration = clamped

    # Scale up toward the target, but never past a scene's physical ceiling
    # (water-filling: capped scenes stay put, the rest absorb the remainder).
    for _ in range(3):
        total = sum(float(s.duration) for s in scenes)
        if total <= 0 or abs(total - target_duration) / max(target_duration, 1) <= 0.05:
            break
        scale = target_duration / total
        for scene in scenes:
            scene.duration = round(min(float(scene.duration) * scale, _ceiling(scene)), 2)
        notes.append(f"durations scaled x{scale:.2f} ({total:.1f}s -> {target_duration:.0f}s target)")

    total = sum(float(s.duration) for s in scenes)
    if target_duration > 0 and (target_duration - total) / target_duration > 0.05:
        notes.append(
            f"WARNING: {len(scenes)} scenes at the {video_clip_cap:.2f}s/clip ceiling "
            f"reach only {total:.0f}s of the {target_duration:.0f}s target — "
            f"add more scenes (need ~{int(target_duration / video_clip_cap) + 1})")

    for n in notes:
        logger.info(f"[QA] lint: {n}")
    return notes


# ── Artifact Checks ────────────────────────────────────────────────────────

def _probe(ffmpeg_bin: str, path: str) -> dict:
    """Duration / resolution / audio via `ffmpeg -i` stderr parse (no ffprobe dependency)."""
    info = {"duration": 0.0, "width": 0, "height": 0, "has_audio": False}
    try:
        p = subprocess.run([ffmpeg_bin, "-i", path], capture_output=True, text=True, timeout=30)
        err = p.stderr or ""
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", err)
        if m:
            info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", err)
        if m:
            info["width"], info["height"] = int(m.group(1)), int(m.group(2))
        info["has_audio"] = "Audio:" in err
    except Exception as e:
        logger.warning(f"[QA] probe failed for {path}: {e}")
    return info


def _motion_score(ffmpeg_bin: str, path: str, duration: float) -> float:
    """Mean absolute pixel difference between an early and a late frame.
    < ~3.0 means the 'video' is essentially a frozen still (a known LTX failure
    mode) — flag it so the scene retries instead of shipping dead footage."""
    try:
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            frames = []
            for tag, ts in (("a", max(0.1, duration * 0.15)), ("b", duration * 0.75)):
                out = str(Path(td) / f"{tag}.png")
                subprocess.run(
                    [ffmpeg_bin, "-y", "-ss", f"{ts:.2f}", "-i", path,
                     "-frames:v", "1", out],
                    capture_output=True, timeout=60,
                )
                frames.append(np.asarray(Image.open(out).convert("L"), dtype=np.float32))
            if frames[0].shape != frames[1].shape:
                return -1.0
            return float(np.abs(frames[0] - frames[1]).mean())
    except Exception as e:
        logger.debug(f"[QA] motion check skipped ({e})")
        return -1.0


def check_clip(ffmpeg_bin: str, path: str, expected_duration: float,
               motion_threshold: float = 3.0) -> MediaQA:
    """A generated clip must exist, decode, roughly match its scene duration,
    and contain actual motion."""
    qa = MediaQA(path=path, ok=True)
    p = Path(path)
    if not p.exists() or p.stat().st_size < 10_000:
        qa.ok = False
        qa.issues.append("file missing or truncated")
        return qa

    info = _probe(ffmpeg_bin, path)
    qa.duration, qa.width, qa.height = info["duration"], info["width"], info["height"]
    qa.has_audio = info["has_audio"]

    if qa.duration <= 0:
        qa.ok = False
        qa.issues.append("unreadable / zero duration")
        return qa
    if expected_duration > 0 and abs(qa.duration - expected_duration) / expected_duration > 0.35:
        qa.issues.append(f"duration {qa.duration:.1f}s vs expected {expected_duration:.1f}s")
        qa.ok = False

    qa.motion_score = _motion_score(ffmpeg_bin, path, qa.duration)
    if 0 <= qa.motion_score < motion_threshold:
        qa.issues.append(f"static clip (motion {qa.motion_score:.1f} < {motion_threshold})")
        qa.ok = False
    return qa


def check_audio(ffmpeg_bin: str, path: str, min_duration: float) -> MediaQA:
    qa = MediaQA(path=path, ok=True)
    p = Path(path)
    if not p.exists() or p.stat().st_size < 5_000:
        qa.ok = False
        qa.issues.append("file missing or truncated")
        return qa
    info = _probe(ffmpeg_bin, path)
    qa.duration = info["duration"]
    qa.has_audio = True
    if qa.duration < min_duration * 0.8:
        qa.ok = False
        qa.issues.append(f"audio {qa.duration:.1f}s < required {min_duration:.1f}s")
    return qa


def check_final(ffmpeg_bin: str, path: str, target_duration: float,
                expect_audio: bool = True) -> MediaQA:
    """Final render: plays, ~target duration, 16:9 HD, and has an audio track."""
    qa = MediaQA(path=path, ok=True)
    p = Path(path)
    if not p.exists() or p.stat().st_size < 100_000:
        qa.ok = False
        qa.issues.append("final render missing or truncated")
        return qa
    info = _probe(ffmpeg_bin, path)
    qa.duration, qa.width, qa.height = info["duration"], info["width"], info["height"]
    qa.has_audio = info["has_audio"]

    if target_duration > 0 and abs(qa.duration - target_duration) / target_duration > 0.25:
        qa.issues.append(f"duration {qa.duration:.1f}s vs target {target_duration:.0f}s")
        qa.ok = False
    if qa.height < 720:
        qa.issues.append(f"resolution {qa.width}x{qa.height} below HD")
        qa.ok = False
    if expect_audio and not qa.has_audio:
        qa.issues.append("no audio track in final render")
        qa.ok = False
    return qa


# ── Report Writer ──────────────────────────────────────────────────────────

def write_report(project_dir: Path, report: dict):
    """Persist qa_report.json next to the render so overnight failures are
    diagnosable at a glance."""
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        out = project_dir / "qa_report.json"
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info(f"[QA] report -> {out}")
    except Exception as e:
        logger.warning(f"[QA] could not write report: {e}")
