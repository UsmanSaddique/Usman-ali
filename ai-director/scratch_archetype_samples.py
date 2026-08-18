"""Generate a STORYBOARD SAMPLE for every content archetype using the REAL
planners + gates (no GPU, no LLM, no ComfyUI). Emits JSON to stdout.

For each archetype it runs the actual pipeline decisions:
  - resolve() -> routing (audio/visual/engine/source) + tier + HITL gates
  - the real scene planner for that lane (lyric_scenes / narration_scenes /
    ambient planner)  on a seed idea
  - the real IP deny-list scan (shows faith_kids blocking copyrighted IP)
This is what the archetype layer contributes: one pipeline, 7 behaviors, config-only.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from types import SimpleNamespace
from app.services import archetypes as A
from app.services import lyric_scenes, narration_scenes
from app.services.lyrics_parser import parse_lyrics
from app.services.yt_safety import SafetyGateService

svc = SafetyGateService.__new__(SafetyGateService)  # rules-only, no models


def _profile(**kw):
    base = {
        "art_style_phrase": "cinematic, highly detailed, volumetric lighting, depth of field",
        "negative_prompt_additions": "text, watermark, blurry, deformed, extra limbs",
    }
    base.update(kw)
    return base


def song_story(lyrics, profile, seed):
    segs = parse_lyrics(lyrics, 30.0)
    rows = lyric_scenes.build_prompts(segs, profile, seed)
    return [{"n": i + 1, "prompt": r["prompt"][:240],
             "motion": r.get("camera_motion", "")} for i, r in enumerate(rows[:5])]


def beat_story(beats, profile, style):
    out = []
    for i, b in enumerate(beats[:5]):
        out.append({"n": i + 1,
                    "prompt": narration_scenes._style_prompt(b, profile, style)[:240],
                    "narration": b.get("text", "")[:90]})
    return out


def ambient_story(beats, profile):
    # mirrors _plan_ambient_scenes prompt construction
    out = []
    for i, b in enumerate(beats[:5]):
        out.append({"n": i + 1,
                    "prompt": narration_scenes._style_prompt(b, profile, "cinematic")[:240],
                    "duration": "6.0s (fixed loop)", "narration": "(none — ambient bed)"})
    return out


SAMPLES = []


def add(aid, idea, storyboard, extra=None, denylist_field=None):
    r = A.resolve(SimpleNamespace(content_archetype=aid, project_type="song",
                                  video_engine="clips"))
    gate = {"verdict": "pass (rules)", "ip_block": []}
    if denylist_field and r.ip_denylist:
        hits = svc._scan_ip_denylist(denylist_field, r.ip_denylist)
        gate["ip_block"] = [h.detail for h in hits]
        if hits:
            gate["verdict"] = "BLOCK (copyright)"
    SAMPLES.append({
        "id": aid, "label": r.label, "tier": r.tier, "idea": idea,
        "routing": {"audio_mode": r.audio_mode, "video_engine": r.video_engine,
                    "visual_mode": r.visual_mode, "source": r.source,
                    "character_consistency": r.character_consistency,
                    "scene_planner": r.scene_planner},
        "gates": {"script_review": r.script_review, "safety_gate": r.safety_gate,
                  "is_blocked": r.is_blocked, "runtime": gate},
        "storyboard": storyboard, "extra": extra or {},
    })


# ── 1. kids_poem (Tier-1, song) ──────────────────────────────────────────────
kids_profile = _profile(
    art_style_phrase="adorable soft 3D cartoon render, plush baby animals, huge glossy eyes, pastel colors",
    character_bible=["hero: a chubby fluffy yellow baby duckling, oversized round head, tiny orange beak"],
    color_palette="butter yellow, sky blue, mint green")
kids_lyrics = ("[verse]\nLittle duckling, soft and small\nQuack quack quack, he waddles tall\n"
               "[chorus]\nSplash splash splash in the pond so blue\nLittle duckling, I love you\n"
               "[verse]\nFlappy wings and tiny feet\nQuack quack quack, so very sweet")
add("kids_poem", "An animal poem: a baby duckling learning to splash and quack",
    song_story(kids_lyrics, kids_profile, "kids-demo"))

# ── 2. ai_dreamscape (Tier-1, ambient, ltx_director) ────────────────────────
dream_profile = _profile(
    art_style_phrase="surreal dreamscape, impossible physics, iridescent, hyper-detailed, cinematic volumetric light")
dream_beats = [
    {"text": "", "visual_prompt": "an endless ocean made of liquid mercury under three purple moons", "visual_type": "broll"},
    {"text": "", "visual_prompt": "a glass whale drifting through a sky of floating candle flames", "visual_type": "broll"},
    {"text": "", "visual_prompt": "crystalline flowers blooming in reverse into geometric fractals", "visual_type": "broll"},
    {"text": "", "visual_prompt": "a city of soft clouds folding into origami skyscrapers", "visual_type": "broll"},
    {"text": "", "visual_prompt": "rivers of light pouring upward into a mirrored infinity", "visual_type": "broll"},
]
add("ai_dreamscape", "Satisfying surreal dreamscape loops set to an ambient bed",
    ambient_story(dream_beats, dream_profile),
    extra={"music_style": "dreamy ambient soundscape, slow synth pads, no drums, no vocals, instrumental",
           "note": "no voice at all — music bed is the master audio"})

# ── 3. funny_ai_qa (Tier-1, narration) ──────────────────────────────────────
qa_profile = _profile(art_style_phrase="surreal, glitchy, dreamlike, impossible, comedic")
qa_beats = [
    {"text": "What happens if you microwave the ocean?", "visual_prompt": "a giant microwave containing an entire glowing ocean, fish in sunglasses", "visual_type": "broll"},
    {"text": "The AI says: it becomes soup for the moon.", "visual_prompt": "the moon slurping ocean-soup through a bendy straw the size of a mountain", "visual_type": "broll"},
    {"text": "Can penguins do taxes?", "visual_prompt": "a boardroom of penguins buried in glowing spreadsheets made of ice", "visual_type": "broll"},
]
add("funny_ai_qa", "Asking an AI ridiculous questions; it answers absurdly",
    beat_story(qa_beats, qa_profile, "explainer"),
    extra={"planner_directive": "qa_pairs — Q&A beats + deliberately surreal visuals"})

# ── 4. reddit_story (Tier-1, scrape -> narration over bg-loop) ──────────────
reddit_profile = _profile(art_style_phrase="soft cinematic ambient background, calm muted colors, slow drifting motion")
reddit_beats = [
    {"text": "So this happened to me last Tuesday at the office...", "visual_prompt": "a cozy dim office at dusk, warm lamp glow, calm drifting dust motes", "visual_type": "broll"},
    {"text": "I hit reply-all to four hundred people by accident.", "visual_prompt": "abstract cascade of soft glowing envelopes drifting endlessly", "visual_type": "broll"},
    {"text": "What happened next changed my whole week.", "visual_prompt": "a quiet rain-streaked window, soft bokeh city lights beyond", "visual_type": "broll"},
]
add("reddit_story", "A scraped Reddit thread retold as narration over a looping background",
    beat_story(reddit_beats, reddit_profile, "documentary"),
    extra={"source": "r/tifu (public JSON, SFW only, no login)",
           "visual_mode": "reddit_broll — one looping bg clip under the whole voiceover"})

# ── 5. edu_facts (Tier-2, narration, REQUIRED review) ───────────────────────
edu_profile = _profile(art_style_phrase="clean documentary, dramatic natural light, atmospheric, cinematic")
edu_beats = [
    {"text": "In its first month, a baby's stomach grows from the size of a cherry to an egg.", "visual_prompt": "macro cinematic shot of a cherry and an egg side by side on soft linen, warm light", "visual_type": "broll"},
    {"text": "Newborns can only see about 8 to 12 inches in front of them.", "visual_prompt": "soft-focus close-up of a gentle face at arm's length, dreamy shallow depth", "visual_type": "broll"},
    {"text": "A baby's sense of smell guides them to their mother within days.", "visual_prompt": "warm intimate nursery scene, golden hour glow, tender atmosphere", "visual_type": "broll"},
]
add("edu_facts", "Baby-development facts (health) — must be human-verified first",
    beat_story(edu_beats, edu_profile, "documentary"),
    extra={"gate_note": "Tier-2: pipeline HARD-STOPS after script for human fact-check (POST /approve-script)"})

# ── 6. faith_kids (Tier-2, song, strict gate + IP block DEMO) ───────────────
faith_profile = _profile(
    art_style_phrase="warm 3D cartoon, gentle, wholesome, soft light",
    character_bible=["hero: a kind cartoon child in modest colorful clothes"])
faith_lyrics = ("[verse]\nSay Bismillah before you eat\nBe kind and gentle, oh so sweet\n"
                "[chorus]\nShare and care and always pray\nBe a good friend every day")
# DEMO the IP deny-list: a title that infringes copyrighted IP is BLOCKED
faith_denylist_field = {"title": "Chhota Bheem Learns to Pray", "lyrics": faith_lyrics}
add("faith_kids", "Wholesome faith/values song for kids (strict gate + IP protection)",
    song_story(faith_lyrics, faith_profile, "faith-demo"),
    extra={"gate_note": "Tier-2: required human review + strict gate + copyrighted-IP deny-list"},
    denylist_field=faith_denylist_field)

# ── 7. authenticity_trap (Tier-3, REFUSED) ──────────────────────────────────
r = A.resolve(SimpleNamespace(content_archetype="authenticity_trap", project_type="song", video_engine="clips"))
SAMPLES.append({
    "id": "authenticity_trap", "label": r.label, "tier": 3,
    "idea": "Satisfying real pets / luxury cars / sports moments / ASMR — DO NOT AUTOMATE",
    "routing": {"audio_mode": r.audio_mode, "video_engine": r.video_engine,
                "visual_mode": r.visual_mode, "source": r.source,
                "character_consistency": r.character_consistency, "scene_planner": r.scene_planner},
    "gates": {"script_review": r.script_review, "safety_gate": r.safety_gate,
              "is_blocked": r.is_blocked, "runtime": {"verdict": "REFUSED at start-generation"}},
    "storyboard": [], "extra": {"refusal": r.block_reason()},
})

_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archetype_samples.json")
with open(_out, "w", encoding="utf-8") as _f:
    json.dump({"samples": SAMPLES}, _f, ensure_ascii=False, indent=2)
print("wrote", _out, "with", len(SAMPLES), "samples")
