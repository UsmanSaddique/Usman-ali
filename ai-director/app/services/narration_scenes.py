"""
AI Director — Narration Beat Planner
Maps the aligned narration master (exact beat timings) onto renderable Scene
rows. The narration WAV is the TIMELINE MASTER: every scene carries
(narration_start, narration_end) and the narration assembler trims each beat's
video block to exactly that window (straight cuts — cumulative drift is
impossible by construction).

Long beats are split into sub-clips of at most the model's clip ceiling
(5.04s on LTX); all sub-clips of a beat share its visual prompt with
varied camera motion so the shot doesn't feel frozen.
"""
import math
import logging

logger = logging.getLogger(__name__)

# visual_type → (SceneType value, camera motions cycled across sub-clips)
VISUAL_MAP = {
    "broll":      ("img2vid",  ["slow push-in", "slow pan right", "slow pan left", "static"]),
    "still":      ("still_pan", ["ken burns in", "ken burns out", "pan right"]),
    # Template-rendered types: sharp text, CPU-only, zero VRAM.
    # The template renderer produces a ready MP4 at native 1920x1080;
    # camera_motion is ignored (the template has its own CSS animations).
    "diagram":    ("template", ["static"]),
    "code":       ("template", ["static"]),
    "map":        ("template", ["static"]),
    "title_card": ("template", ["static"]),
}

NEGATIVE_DEFAULT = (
    "text, watermark, logo, subtitles, captions, low quality, blurry, "
    "deformed, disfigured, extra limbs, warped face, jpeg artifacts"
)

# Retention pacing: hard ceiling on how long ONE visual may hold the screen.
MAX_SHOT_SECONDS = {"documentary": 12.0, "explainer": 8.0, "tutorial": 8.0}


def flatten_beats(script: dict) -> list[dict]:
    """narration_script JSON → flat beat list for TTS + planning.
    Marks the last beat of each chapter with chapter_break=True (longer pause)."""
    flat = []
    for ci, ch in enumerate(script.get("chapters", []), start=1):
        beats = ch.get("beats", [])
        for bi, b in enumerate(beats):
            flat.append({
                "text": b.get("narration_text", ""),
                "chapter_index": ci,
                "chapter_title": ch.get("title", f"Chapter {ci}"),
                "chapter_break": bi == len(beats) - 1,
                "visual_type": b.get("visual_type", "broll"),
                "visual_prompt": b.get("visual_prompt", ""),
                "sfx_prompt": b.get("sfx_prompt", ""),
                "mood": b.get("mood", "neutral"),
            })
    return flat


def _style_prompt(beat: dict, profile: dict, style: str) -> str:
    """Beat visual prompt + channel/style boosters, model-renderable."""
    base = (beat.get("visual_prompt") or "").strip()
    if not base:
        base = f"wide establishing shot, {beat.get('text', '')[:120]}"
    vt = beat.get("visual_type", "broll")
    art = (profile or {}).get("art_style_phrase", "")

    if vt in ("diagram", "code", "map", "title_card"):
        # rendered as a clean still until the template renderer exists —
        # abstract/visual metaphor, NEVER ask the model for readable text
        return (f"clean minimalist {vt.replace('_', ' ')} concept, {base}, "
                f"abstract geometric shapes, no readable text, flat design, "
                f"dark background, subtle glow, high contrast, professional "
                f"motion-graphics style frame")

    boosters = ("cinematic, highly detailed, volumetric lighting, "
                "depth of field, film grain, professional color grade")
    if style == "documentary":
        boosters = ("cinematic documentary photography, dramatic natural light, "
                    "atmospheric haze, shallow depth of field, film grain, "
                    "epic composition")
    parts = [base]
    if art:
        parts.append(art)
    parts.append(boosters)
    return ", ".join(parts)


def plan_scenes(
    script: dict,
    timing_beats: list[dict],   # from narration_timing.json: beat_index/start/end
    master_duration: float,
    profile: dict,
    max_clip_sec: float,
    assets_dir: "Path" = None,
) -> list[dict]:
    """Beat timings → scene dicts (one per sub-clip), duration-exact.

    Each beat owns the wall-clock window [start, next_beat.start) — the pause
    after a beat belongs to its visual (the shot holds through the breath).
    Returns dicts ready to insert as Scene rows.
    """
    flat = flatten_beats(script)
    style = script.get("style", "explainer")
    max_shot = MAX_SHOT_SECONDS.get(style, 8.0)

    scenes = []
    scene_num = 0
    for i, tb in enumerate(timing_beats):
        # timing beats and flattened script beats are index-aligned (TTS skips
        # empty texts in both places)
        beat = flat[tb["beat_index"] - 1] if tb["beat_index"] - 1 < len(flat) else {}
        window_start = float(tb["start"])
        window_end = float(timing_beats[i + 1]["start"]) if i + 1 < len(timing_beats) \
            else float(master_duration)
        window = max(0.5, window_end - window_start)

        vt = beat.get("visual_type", "broll")
        scene_type, motions = VISUAL_MAP.get(vt, VISUAL_MAP["broll"])

        # Check for user-supplied asset markers in the prompt e.g. "[asset:chart.png]"
        asset_path = None
        raw_prompt = beat.get("visual_prompt") or beat.get("prompt") or ""
        import re, os
        asset_match = re.search(r"\[asset:([^\]]+)\]", raw_prompt)
        if asset_match and assets_dir:
            cand = assets_dir / asset_match.group(1).strip()
            if cand.exists():
                scene_type = "user_asset"
                asset_path = str(cand)

        # split the window into sub-clips ≤ min(model cap, retention ceiling)
        ceil_sec = min(max_clip_sec, max_shot)
        n_sub = max(1, math.ceil(window / ceil_sec - 1e-6))
        
        # User assets shouldn't be split into sub-clips; play the asset for the whole beat window.
        if scene_type == "user_asset":
            n_sub = 1
            
        sub_len = window / n_sub

        prompt = _style_prompt(beat, profile, style)
        for k in range(n_sub):
            scene_num += 1
            sub_start = window_start + k * sub_len
            
            notes = {
                "beat_index": tb["beat_index"],
                "chapter_index": tb.get("chapter_index", 1),
                "chapter_title": beat.get("chapter_title", ""),
                "mood": beat.get("mood", "neutral"),
                "sub_clip": f"{k + 1}/{n_sub}",
                "transition_in": "cut", "transition_out": "cut",
            }
            if scene_type == "user_asset":
                notes["user_asset_path"] = asset_path
                
            scenes.append({
                "scene_number": scene_num,
                "scene_type": scene_type,
                "prompt": prompt,
                "negative_prompt": NEGATIVE_DEFAULT,
                "duration": round(sub_len, 3),
                "camera_motion": motions[k % len(motions)],
                "narration_start": round(sub_start, 3),
                "narration_end": round(min(sub_start + sub_len, window_end), 3),
                "visual_type": vt,
                "sfx_prompt": beat.get("sfx_prompt", ""),
                "narration_text": beat.get("text", "") if k == 0 else "",
                "director_notes": notes,
            })

    logger.info(f"[NarrationScenes] Planned {len(scenes)} scenes from "
                f"{len(timing_beats)} beats over {master_duration:.1f}s")
    return scenes


def chapter_markers(script: dict, timing_beats: list[dict]) -> list[tuple[float, str]]:
    """(start_seconds, chapter_title) list for YouTube description chapters."""
    markers = []
    seen = set()
    flat = flatten_beats(script)
    for tb in timing_beats:
        idx = tb["beat_index"] - 1
        if idx >= len(flat):
            continue
        ci = flat[idx]["chapter_index"]
        if ci not in seen:
            seen.add(ci)
            markers.append((float(tb["start"]), flat[idx]["chapter_title"]))
    return markers
