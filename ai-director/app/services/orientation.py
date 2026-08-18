"""
AI Director — Output Orientation (long-form vs Shorts/Reels)

A single place that turns a human-friendly orientation label ("shorts", "reel",
"vertical", "landscape", "square") into the concrete pixel dimensions the pipeline
renders at, at every stage:

    base    — the per-clip img2vid/txt2vid render size (VRAM-critical)
    premium — the higher-quality "hook" render size for the opening scenes
    target  — the final upscale canvas for a given resolution label (1080p/2k/4k)

Why this exists: shorts/reels are 9:16 vertical, long-form is 16:9. Everything
downstream (stills, clips, premium opening, upscale, assembler canvas) must agree
on the aspect ratio or the video ends up pillar-boxed. Resolving it here keeps the
VRAM-safe pixel budget identical across orientations — vertical just swaps the
axes of the benched landscape sizes, so peak VRAM is unchanged on the 16GB card.

Design rule: ADDITIVE. Anything that does not set an orientation resolves to
"landscape", i.e. the exact legacy 832x480 / 960x544 / 1920x1080 behavior.
"""
from __future__ import annotations

LANDSCAPE = "landscape"
VERTICAL = "vertical"
SQUARE = "square"

# Human-typed synonyms → canonical orientation. "shorts"/"reel"/"reels" all mean
# the 9:16 vertical short-form format.
_ALIASES = {
    "landscape": LANDSCAPE, "wide": LANDSCAPE, "16:9": LANDSCAPE,
    "horizontal": LANDSCAPE, "long": LANDSCAPE, "longform": LANDSCAPE,
    "long_form": LANDSCAPE, "youtube": LANDSCAPE,
    "vertical": VERTICAL, "portrait": VERTICAL, "9:16": VERTICAL,
    "short": VERTICAL, "shorts": VERTICAL, "reel": VERTICAL, "reels": VERTICAL,
    "tiktok": VERTICAL, "story": VERTICAL, "stories": VERTICAL,
    "square": SQUARE, "1:1": SQUARE,
}

# Per-orientation render sizes. base/premium mirror the benched landscape budgets
# (832x480 base, 960x544 premium — VRAM-safe on 16GB with the Gemma TE resident);
# vertical/square keep the SAME pixel count, only the axes differ. All values are
# multiples of 32 (LTX latent requirement) and even.
_PRESETS = {
    LANDSCAPE: {"base": (832, 480), "premium": (960, 544)},
    VERTICAL:  {"base": (480, 832), "premium": (544, 960)},
    SQUARE:    {"base": (640, 640), "premium": (704, 704)},
}

# Final-render short-edge (in px) for each resolution label. "1080p" == 1080 on
# the short edge (1920x1080 landscape, 1080x1920 vertical, 1080x1080 square).
_SHORT_EDGE = {
    "720p": 720, "1080p": 1080, "1440p": 1440, "2k": 1440, "4k": 2160, "2160p": 2160,
}

# The reference still for the LTX Director engine and img2vid seeds. Landscape
# keeps its legacy 1280x720; the others match orientation so composition survives
# the director's aspect-preserving crop.
_STILL = {
    LANDSCAPE: (1280, 720),
    VERTICAL:  (720, 1280),
    SQUARE:    (960, 960),
}


def normalize(orientation) -> str:
    """Map any label/synonym (case-insensitive) to a canonical orientation.
    Unknown / empty → 'landscape' (legacy behavior)."""
    if not orientation:
        return LANDSCAPE
    return _ALIASES.get(str(orientation).strip().lower(), LANDSCAPE)


def is_vertical(orientation) -> bool:
    return normalize(orientation) == VERTICAL


def base_dims(orientation) -> tuple[int, int]:
    """Per-clip render size (VRAM-critical). Returns (width, height)."""
    return _PRESETS[normalize(orientation)]["base"]


def premium_dims(orientation) -> tuple[int, int]:
    """Higher-quality opening/hook render size. Returns (width, height)."""
    return _PRESETS[normalize(orientation)]["premium"]


def still_dims(orientation) -> tuple[int, int]:
    """Reference-still size for LTX Director / img2vid seeds."""
    return _STILL[normalize(orientation)]


def target_dims(orientation, resolution: str = "1080p") -> tuple[int, int]:
    """Final upscale canvas for a resolution label, oriented correctly.
    e.g. ('vertical', '1080p') -> (1080, 1920). Returns (width, height)."""
    o = normalize(orientation)
    short = _SHORT_EDGE.get((resolution or "1080p").lower(), 1080)
    long = _even(short * 16 / 9)
    if o == VERTICAL:
        return (short, long)
    if o == SQUARE:
        return (short, short)
    return (long, short)


def _even(x) -> int:
    v = int(round(x))
    return v if v % 2 == 0 else v + 1
