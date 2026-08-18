"""Phase 0 smoke test — archetype loading + resolution. No GPU, no DB writes."""
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.services import archetypes as A


def main():
    reg = A.load_archetypes(force=True)
    print(f"loaded {len(reg)} archetypes: {sorted(reg)}")
    assert reg, "no archetypes loaded"
    for want in ["kids_poem", "ai_dreamscape", "funny_ai_qa", "reddit_story",
                 "edu_facts", "faith_kids", "authenticity_trap"]:
        assert want in reg, f"missing {want}"

    # 1. legacy project (no archetype anywhere) -> song lane, not blocked
    p_legacy = SimpleNamespace(content_archetype=None, project_type="song",
                               video_engine="clips")
    r = A.resolve(p_legacy, channel=None)
    print("legacy:", r.archetype_id, r.project_type, r.audio_mode, "blocked=", r.is_blocked)
    assert r.archetype_id is None and r.project_type == "song" and not r.is_blocked

    # 2. channel-level archetype inherited by project
    p = SimpleNamespace(content_archetype=None, project_type="song", video_engine="clips")
    ch = SimpleNamespace(content_archetype="ai_dreamscape")
    r = A.resolve(p, channel=ch)
    print("inherited:", r.archetype_id, r.audio_mode, r.video_engine, r.visual_mode)
    assert r.archetype_id == "ai_dreamscape" and r.audio_mode == "ambient"
    assert r.video_engine == "ltx_director" and r.project_type == "narration"

    # 3. project override beats channel
    p2 = SimpleNamespace(content_archetype="kids_poem", project_type="song", video_engine="clips")
    r = A.resolve(p2, channel=ch)
    print("override:", r.archetype_id, r.audio_mode)
    assert r.archetype_id == "kids_poem" and r.audio_mode == "song"

    # 4. Tier-2 defaults: required review + strict gate
    p3 = SimpleNamespace(content_archetype="edu_facts", project_type="narration", video_engine="clips")
    r = A.resolve(p3)
    print("edu_facts:", r.tier, r.script_review, r.safety_gate, "blocked=", r.is_blocked)
    assert r.tier == 2 and r.script_review == "required" and r.safety_gate == "strict"
    assert not r.is_blocked

    # 5. Tier-3 trap: blocked
    p4 = SimpleNamespace(content_archetype="authenticity_trap", project_type="narration", video_engine="clips")
    r = A.resolve(p4)
    print("trap:", r.tier, r.enabled, "blocked=", r.is_blocked)
    assert r.is_blocked and not r.enabled
    print("  reason:", r.block_reason()[:80], "...")

    # 6. unknown archetype -> legacy fallback, not a crash
    p5 = SimpleNamespace(content_archetype="does_not_exist", project_type="narration", video_engine="clips")
    r = A.resolve(p5)
    print("unknown->legacy:", r.archetype_id, r.project_type)
    assert r.archetype_id is None and r.project_type == "narration"

    print("\nALL PHASE 0 CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
