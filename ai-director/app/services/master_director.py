"""
AI Director — Master Director Guidance
Acts as a "master director" over the whole video: every scene gets a
deterministic cinematography guidance block (shot, camera move, lighting,
mood, composition, continuity note) that follows a dramatic arc across the
video (opening → build → peak → resolve → finale).

The guidance is:
  * appended to the scene's generation prompt (both the clip-by-clip engine
    and the LTX Director engine),
  * saved per scene in scene.director_notes["director_guidance"], and
  * saved per project to projects/<id>/director_guidance.json.

Deterministic on (seed_key, scene index) using pure modular arithmetic — the
C# port (AiDirector.Application.Directing.MasterDirector) implements the SAME
algorithm so both backends produce identical guidance for a project.
"""
import json
import hashlib
from pathlib import Path

# selection tables — index math only (no random.Random) so the C# port can
# reproduce the exact same picks
SHOTS = {
    "opening": ["wide establishing shot", "slow aerial establishing shot",
                "wide master shot"],
    "build": ["medium shot", "medium wide tracking shot",
              "over-the-shoulder medium shot"],
    "peak": ["dynamic medium close-up", "hero low-angle medium shot",
             "sweeping circular medium shot"],
    "resolve": ["medium close-up", "gentle wide shot", "profile medium shot"],
    "finale": ["slow pull-back wide shot", "closing crane-up wide shot",
               "fading wide twilight shot"],
}

CAMERAS = {
    "opening": ["slow push-in", "gentle crane down", "drifting lateral glide"],
    "build": ["smooth tracking follow", "slow arc around the subject",
              "steady glide forward"],
    "peak": ["energetic slow orbit", "rising crane-up", "confident push-in"],
    "resolve": ["static camera with subtle drift", "slow pan", "soft push-in"],
    "finale": ["slow pull-back", "tilt up toward the sky", "locked-off wide"],
}

LIGHTING = [
    "warm golden-hour key light with a soft rim",
    "soft diffused daylight with gentle bounce fill",
    "bright airy high-key lighting",
    "amber sunset backlight with long soft shadows",
    "cool serene twilight key with warm practical accents",
]

MOODS = {
    "intro": "calm anticipation",
    "hook": "wonder and delight",
    "verse": "warm storytelling",
    "chorus": "joyful celebration",
    "bridge": "quiet reflection",
    "outro": "peaceful resolution",
}

COMPOSITIONS = [
    "subject on the left third with leading room",
    "subject centered with symmetrical framing",
    "subject on the right third with leading room",
    "foreground framing with soft depth of field",
]

CONTINUITY = ("keep the same character design, outfit, color palette and "
              "setting style as the previous shot")


def _slot_hash(seed_key: str, salt: int) -> int:
    """Independent hash per (project, selection slot). A single shared seed
    reduced mod the small table sizes correlates every pick across projects
    (all tables share divisors of 60) — hashing the salt in decorrelates them."""
    key = f"{seed_key}|{salt}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def _phase(index: int, total: int, section_type: str) -> str:
    """Dramatic arc position of this scene."""
    if section_type in ("chorus", "hook"):
        return "peak"
    if index <= 0:
        return "opening"
    if total > 1 and index >= total - 1:
        return "finale"
    if total > 1 and index / (total - 1) < 0.5:
        return "build"
    return "resolve"


def _pick(items: list, seed_key: str, index: int, salt: int) -> str:
    return items[(_slot_hash(seed_key, salt) + index * 31) % len(items)]


def guidance_for(index: int, total: int, section_type: str = "verse",
                 seed_key: str = "") -> dict:
    """Master-director guidance for scene `index` (0-based) of `total`.
    Deterministic for a given (seed_key, index) so regeneration and resume
    always see the same storyboard."""
    section = (section_type or "verse").lower()
    phase = _phase(index, max(total, 1), section)
    shot = _pick(SHOTS[phase], seed_key, index, 1)
    camera = _pick(CAMERAS[phase], seed_key, index, 2)
    lighting = _pick(LIGHTING, seed_key, index, 3)
    mood = MOODS.get(section, MOODS["verse"])
    composition = _pick(COMPOSITIONS, seed_key, index, 4)
    cue = (f"{shot}, camera: {camera}, {lighting}, {mood} mood, {composition}")
    return {
        "scene": index + 1,
        "phase": phase,
        "shot": shot,
        "camera": camera,
        "lighting": lighting,
        "mood": mood,
        "composition": composition,
        "continuity": CONTINUITY,
        "prompt_cue": cue,
    }


def apply_cue(prompt: str, guidance: dict) -> str:
    """Append the director cue to a generation prompt (idempotent — safe to
    call again on resume/retry without doubling the cue)."""
    g = guidance or {}
    cue = g.get("prompt_cue") or ""
    prompt = prompt or ""
    # already applied — either the whole cue, or the guidance was inlined at
    # scene-planning time (lyric_scenes embeds "camera: <move>" in the prompt)
    marker = f"camera: {g.get('camera', '')}" if g.get("camera") else cue
    if not cue or cue in prompt or (marker and marker in prompt):
        return prompt
    return f"{prompt}, {cue}"


def save_plan(project_dir, entries: list) -> str:
    """Persist the per-scene guidance to projects/<id>/director_guidance.json
    so the storyboard survives restarts and is visible next to the renders."""
    pdir = Path(project_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    out = pdir / "director_guidance.json"
    tmp = pdir / "director_guidance.json.tmp"
    tmp.write_text(json.dumps({"scenes": entries}, indent=2,
                              ensure_ascii=False), encoding="utf-8")
    import os
    os.replace(tmp, out)
    return str(out)
