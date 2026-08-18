"""
AI Director — Content Archetypes

An *archetype* is a reusable recipe (archetypes/<id>.yaml) that wires the pipeline
for a class of niches (kids poem, AI dreamscape, reddit story, edu facts, ...).

A Channel picks an archetype; a Project may override it. This module loads the
YAML registry and resolves the effective recipe for a project by merging:

    project.content_archetype  >  channel.content_archetype  >  archetype defaults

The tier drives the default human-in-the-loop (HITL) policy, so a niche only has
to declare its tier to get the right gate strictness. See CHANNEL_ARCHETYPES_PLAN.md.

Design rule: this layer is ADDITIVE. A project/channel with content_archetype=None
resolves to the legacy behavior (project_type carries song/narration as before),
so existing channels keep working unchanged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ── Enumerated string values (kept as plain str for YAML/JSON friendliness) ──

AUDIO_MODES = {"song", "narration", "ambient"}
VIDEO_ENGINES = {"clips", "ltx_director"}
VISUAL_MODES = {"character_panels", "surreal_loop", "voice_over_bg", "reddit_broll"}
SOURCES = {"llm", "scrape", "mixed"}
SCENE_PLANNERS = {"lyric_scenes", "narration_scenes", "freeform", "qa_pairs"}
SCRIPT_REVIEW = {"optional", "required", "blocked"}
SAFETY_GATE = {"standard", "strict", "blocked"}


# ── Tier → default HITL policy ───────────────────────────────────────────────
# An archetype may override either field in its `hitl:` block.

_TIER_HITL_DEFAULTS = {
    1: {"script_review": "optional", "safety_gate": "standard"},
    2: {"script_review": "required", "safety_gate": "strict"},
    3: {"script_review": "blocked", "safety_gate": "blocked"},
}


@dataclass
class ResolvedRecipe:
    """The effective pipeline wiring for a project after merge."""
    archetype_id: Optional[str]     # None => legacy (no archetype)
    tier: int = 1
    label: str = ""
    enabled: bool = True

    audio_mode: str = "song"        # song | narration | ambient
    video_engine: str = "clips"     # clips | ltx_director
    orientation: str = "landscape"  # landscape(16:9) | vertical(9:16 shorts/reels) | square(1:1)
    # Per-clip pacing. clip_seconds = desired length of ONE clip; clip_fps lets
    # slow niches (ASMR) trade frame-rate for LONGER clips within the fixed
    # 121-frame VRAM cap (121f @ 24fps = 5s, @ 12fps = ~10s). None = model default.
    clip_seconds: Optional[float] = None
    clip_fps: Optional[int] = None
    visual_mode: str = "character_panels"
    character_consistency: bool = True
    scene_planner: str = "lyric_scenes"
    source: str = "llm"             # llm | scrape | mixed

    script_review: str = "optional"  # optional | required | blocked
    safety_gate: str = "standard"    # standard | strict | blocked

    seo_profile: str = "default"
    ip_denylist: list = field(default_factory=list)  # copyrighted-IP terms → block (strict gate)

    # The internal pipeline lane. audio_mode maps to this so legacy code that
    # branches on project_type keeps working:
    #   song              -> "song"
    #   narration|ambient -> "narration"   (ambient still uses the narration
    #                        timeline machinery but skips TTS; the audio_mode
    #                        field is what the ambient branch checks)
    @property
    def project_type(self) -> str:
        return "song" if self.audio_mode == "song" else "narration"

    @property
    def is_blocked(self) -> bool:
        """Tier-3 / disabled archetypes must refuse at start-generation."""
        return (not self.enabled) or self.tier == 3 \
            or self.script_review == "blocked" or self.safety_gate == "blocked"

    def block_reason(self) -> str:
        return (
            f"Archetype '{self.archetype_id or 'unknown'}' ({self.label or 'Tier 3'}) is "
            "not automatable: this niche relies on real-world footage, exact physics, "
            "copyrighted material, or human authenticity that local AI generation "
            "cannot produce credibly. Enable it only with explicit human production."
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["project_type"] = self.project_type
        d["is_blocked"] = self.is_blocked
        return d


# ── Registry loading ─────────────────────────────────────────────────────────

_CACHE: dict[str, dict] = {}
_LOADED = False


def _archetypes_dir() -> Path:
    from app.config import settings
    return settings.paths.archetypes_dir


def load_archetypes(force: bool = False) -> dict[str, dict]:
    """Load every archetypes/*.yaml into a {id: raw_dict} map (cached)."""
    global _LOADED
    if _LOADED and not force:
        return _CACHE

    _CACHE.clear()
    d = _archetypes_dir()
    if d.exists():
        for path in sorted(d.glob("*.yaml")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                aid = raw.get("id") or path.stem
                raw["id"] = aid
                _CACHE[aid] = raw
            except Exception as e:  # a broken YAML must not take the app down
                logger.error(f"[archetypes] failed to load {path.name}: {e}")
    else:
        logger.warning(f"[archetypes] dir not found: {d}")

    _LOADED = True
    logger.info(f"[archetypes] loaded {len(_CACHE)}: {sorted(_CACHE)}")
    return _CACHE


def get_archetype(archetype_id: Optional[str]) -> Optional[dict]:
    if not archetype_id:
        return None
    return load_archetypes().get(archetype_id)


def _coerce_recipe(raw: dict, orientation: str = "landscape") -> ResolvedRecipe:
    """Build a ResolvedRecipe from a raw archetype dict, applying tier HITL
    defaults for any hitl field the YAML does not set. `orientation` is resolved
    by the caller (project/channel override the archetype default)."""
    tier = int(raw.get("tier", 1))
    hitl_defaults = _TIER_HITL_DEFAULTS.get(tier, _TIER_HITL_DEFAULTS[1])
    hitl = {**hitl_defaults, **(raw.get("hitl") or {})}

    return ResolvedRecipe(
        archetype_id=raw.get("id"),
        tier=tier,
        label=raw.get("label", ""),
        enabled=bool(raw.get("enabled", True)),
        audio_mode=raw.get("audio_mode", "song"),
        video_engine=raw.get("video_engine", "clips"),
        orientation=orientation,
        clip_seconds=(float(raw["clip_seconds"]) if raw.get("clip_seconds") else None),
        clip_fps=(int(raw["clip_fps"]) if raw.get("clip_fps") else None),
        visual_mode=raw.get("visual_mode", "character_panels"),
        character_consistency=bool(raw.get("character_consistency", True)),
        scene_planner=raw.get("scene_planner", "lyric_scenes"),
        source=raw.get("source", "llm"),
        script_review=hitl.get("script_review", "optional"),
        safety_gate=hitl.get("safety_gate", "standard"),
        seo_profile=raw.get("seo_profile", "default"),
        ip_denylist=list(raw.get("ip_denylist") or []),
    )


def _legacy_recipe(project, orientation: str = "landscape") -> ResolvedRecipe:
    """No archetype set anywhere → reconstruct a recipe from the project's
    existing fields so behavior is identical to before this layer existed."""
    ptype = getattr(project, "project_type", "song") or "song"
    audio = "song" if ptype == "song" else "narration"
    return ResolvedRecipe(
        archetype_id=None,
        tier=1,
        label="(legacy)",
        enabled=True,
        audio_mode=audio,
        video_engine=getattr(project, "video_engine", "clips") or "clips",
        orientation=orientation,
        visual_mode="character_panels" if audio == "song" else "voice_over_bg",
        character_consistency=(audio == "song"),
        scene_planner="lyric_scenes" if audio == "song" else "narration_scenes",
        source="llm",
    )


def resolve(project, channel=None, channel_profile=None) -> ResolvedRecipe:
    """Resolve the effective recipe for a project.

    Precedence (most specific wins):
        project.content_archetype (DB)
        > channel.content_archetype (DB row)
        > channel_profile["content_archetype"] (channels/<slug>.yaml)
        > legacy (reconstruct from project_type / video_engine)

    `channel` (DB object) and `channel_profile` (parsed YAML dict) are both
    optional. Channels are configured via YAML, so channel_profile is the usual
    place the archetype id lives.
    """
    aid = getattr(project, "content_archetype", None)
    if not aid and channel is not None:
        aid = getattr(channel, "content_archetype", None)
    if not aid and channel_profile:
        aid = channel_profile.get("content_archetype")

    raw = get_archetype(aid)

    # Orientation precedence: project > channel (DB) > channel YAML > archetype
    # default > "landscape". This lets a channel/project be "shorts" (vertical)
    # regardless of archetype, and an archetype declare a sensible default.
    orientation = (
        _norm_orient(getattr(project, "orientation", None))
        or (_norm_orient(getattr(channel, "orientation", None)) if channel is not None else None)
        or (_norm_orient((channel_profile or {}).get("orientation")))
        or (_norm_orient((raw or {}).get("orientation")))
        or "landscape"
    )

    if raw is None:
        if aid:
            logger.warning(
                f"[archetypes] project references unknown archetype '{aid}' — "
                "falling back to legacy recipe")
        return _legacy_recipe(project, orientation)

    # The archetype fully defines the wiring (including video_engine). Legacy
    # projects with no archetype keep their per-project video_engine via
    # _legacy_recipe above; once a project/channel opts into an archetype, the
    # archetype's engine is authoritative.
    return _coerce_recipe(raw, orientation)


def _norm_orient(value):
    """Return a canonical orientation for a raw value, or None if unset.
    Kept local (not importing orientation.py) so this pure module stays
    dependency-free; mirrors orientation.normalize()'s alias table for the
    common labels."""
    if not value:
        return None
    v = str(value).strip().lower()
    if v in ("vertical", "portrait", "9:16", "short", "shorts", "reel", "reels",
             "tiktok", "story", "stories"):
        return "vertical"
    if v in ("square", "1:1"):
        return "square"
    if v in ("landscape", "wide", "16:9", "horizontal", "long", "longform",
             "long_form", "youtube"):
        return "landscape"
    return "landscape"
