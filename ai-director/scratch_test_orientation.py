"""Smoke test: orientation presets + archetype orientation resolution."""
from types import SimpleNamespace
from app.services import orientation as O
from app.services import archetypes as A

def check(label, cond):
    print(("PASS" if cond else "FAIL"), "-", label)
    assert cond, label

# ── orientation presets ──
check("shorts->vertical", O.normalize("shorts") == "vertical")
check("reel->vertical", O.normalize("reel") == "vertical")
check("16:9->landscape", O.normalize("16:9") == "landscape")
check("None->landscape", O.normalize(None) == "landscape")
check("vertical base 480x832", O.base_dims("vertical") == (480, 832))
check("landscape base 832x480", O.base_dims("landscape") == (832, 480))
check("vertical premium 544x960", O.premium_dims("vertical") == (544, 960))
check("vertical 1080p final 1080x1920", O.target_dims("vertical", "1080p") == (1080, 1920))
check("landscape 1080p final 1920x1080", O.target_dims("landscape", "1080p") == (1920, 1080))
check("square 1080p 1080x1080", O.target_dims("square", "1080p") == (1080, 1080))
check("vertical 4k 2160x3840", O.target_dims("vertical", "4k") == (2160, 3840))
check("vertical still 720x1280", O.still_dims("vertical") == (720, 1280))
# all base/premium dims divisible by 32 (LTX latent requirement)
for o in ("landscape", "vertical", "square"):
    for w, h in (O.base_dims(o), O.premium_dims(o)):
        check(f"{o} {w}x{h} div32", w % 32 == 0 and h % 32 == 0)

# ── archetype loads ──
regs = A.load_archetypes(force=True)
check("asmr_ambient archetype loaded", "asmr_ambient" in regs)
r = A._coerce_recipe(regs["asmr_ambient"], "vertical")
check("asmr ambient audio", r.audio_mode == "ambient")
check("asmr vertical", r.orientation == "vertical")
check("asmr not blocked (Tier1)", not r.is_blocked)
check("asmr no character consistency", r.character_consistency is False)

# ── resolution precedence ──
# project vertical override beats channel landscape
proj = SimpleNamespace(content_archetype="asmr_ambient", orientation="shorts",
                       project_type="song", video_engine="clips")
chan = SimpleNamespace(content_archetype=None, orientation="landscape")
rec = A.resolve(proj, chan, None)
check("project orientation wins", rec.orientation == "vertical")

# channel orientation inherited when project unset
proj2 = SimpleNamespace(content_archetype=None, orientation=None,
                        project_type="song", video_engine="clips")
chan2 = SimpleNamespace(content_archetype="asmr_ambient", orientation="vertical")
rec2 = A.resolve(proj2, chan2, None)
check("channel orientation inherited", rec2.orientation == "vertical")
check("channel archetype inherited", rec2.archetype_id == "asmr_ambient")

# legacy project (no archetype, no orientation) stays landscape
proj3 = SimpleNamespace(content_archetype=None, orientation=None,
                        project_type="song", video_engine="clips")
rec3 = A.resolve(proj3, None, None)
check("legacy landscape default", rec3.orientation == "landscape")
check("legacy archetype None", rec3.archetype_id is None)

# authenticity_trap still blocks (real ASMR/cutting refused)
trap = A._coerce_recipe(regs["authenticity_trap"]) if "authenticity_trap" in regs else None
if trap is not None:
    check("authenticity_trap blocked", trap.is_blocked)

print("\nALL ORIENTATION + ARCHETYPE CHECKS PASSED")
