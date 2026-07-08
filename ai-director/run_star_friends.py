"""
Generate a 10s test video: "The Star Friends"
Hand-crafted scenes with director-style varied framing + LTX quality LoRAs.
"""
import sys, os, time, json, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

from app.config import settings
from app.database import (
    get_session, Channel, Project, Scene, ProjectStatus,
    SceneType, SceneStatus,
)
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.pipeline import PipelineOrchestrator

# ── Characters (exact descriptors — repeat verbatim in every prompt) ───────
LUNA = (
    "cute 3D Pixar-style cartoon girl Luna, 4 years old, soft lavender hijab, "
    "big bright violet eyes, dimpled rosy cheeks, soft purple dress with golden star patterns, "
    "sweet gentle smile, rounded chubby features, smooth subsurface skin shading, glossy highlights"
)
ZAIN = (
    "cute 3D Pixar-style cartoon boy Zain, 5 years old, white embroidered kufi cap, "
    "short black hair, big warm hazel eyes, olive-green kurta with white trim, "
    "brave kind expression, rounded chubby features, smooth subsurface skin shading, glossy highlights"
)
ART = (
    "polished 3D animated-movie cartoon style, soft cinematic daylight, warm golden hour lighting, "
    "soft depth of field, bokeh background, beautifully rendered, children's animation movie quality, "
    "highly detailed, cinematic composition, soft volumetric lighting"
)
NEG = (
    "photorealistic, realistic skin, real human photo, uncanny, text, watermark, scary, dark, "
    "harsh lighting, deformed, extra limbs, extra fingers, blurry, low quality, immodest clothing, "
    "extreme close-up, macro, face only, head only, cropped frame"
)

# ── LTX Video LoRAs for quality boost ─────────────────────────────────────
# NOTE: Crisp_Enhance (672MB) is too large for 16GB VRAM with the 22B model —
# causes severe CPU offload spill (50s -> 370s). HDR alone (312MB) fits fine.
LTX_LORAS = [
    ("ltx-2.3-22b-ic-lora-hdr-0.9.safetensors", 0.6),    # richer HDR lighting
]

# ── Director-crafted scenes (varied framing like a real filmmaker) ─────────
SCENES = [
    {
        # Scene 1: WIDE ESTABLISHING — open with the world, set the scene
        "scene_number": 1,
        "scene_type": "img2vid",
        "prompt": (
            f"wide establishing shot, full body head to toe, "
            f"{LUNA} and {ZAIN} walking side by side down a sun-dappled cobblestone garden path, "
            f"lush colorful flower beds on both sides, tall green hedges receding into the distance, "
            f"a tiny lost calico kitten sitting alone under a pink rose bush in the foreground, "
            f"the children spot it and pause mid-step with surprised expressions, "
            f"butterflies floating in the warm air, dappled sunlight through overhead tree branches, {ART}"
        ),
        "negative_prompt": NEG,
        "duration": 5.0,
        "camera_motion": "zoom_in",
        "narration_text": "One sunny morning, Luna and Zain found a tiny lost kitten in the garden.",
        "director_notes": {"transition_in": "fade_from_black", "transition_out": "crossfade", "importance": "high"},
        "lora_ids": [l[0] for l in LTX_LORAS],
        "lora_weights": [l[1] for l in LTX_LORAS],
    },
    {
        # Scene 2: MEDIUM SHOT — emotional beat, character interaction, warmth
        "scene_number": 2,
        "scene_type": "img2vid",
        "prompt": (
            f"medium shot, waist up with garden visible behind, "
            f"{LUNA} kneeling on the cobblestones cradling the tiny calico kitten gently against her chest, "
            f"the kitten nuzzles into her lavender hijab, {ZAIN} standing beside her with one hand on her shoulder, "
            f"looking down at the kitten with a warm proud smile, "
            f"soft golden sunset light wrapping around them, pink and white roses blurred in the background, "
            f"warm emotional moment, tender and gentle, {ART}"
        ),
        "negative_prompt": NEG,
        "duration": 5.0,
        "camera_motion": "static",
        "narration_text": "Be kind to every creature, for that is part of our faith. Kindness always finds its way home.",
        "director_notes": {"transition_in": "crossfade", "transition_out": "fade_to_black", "importance": "high"},
        "lora_ids": [l[0] for l in LTX_LORAS],
        "lora_weights": [l[1] for l in LTX_LORAS],
    },
]

# ── Setup ──────────────────────────────────────────────────────────────────
CHANNEL = "little-muslim-nation"
mm = ModelManager(); register_all_loaders(mm, settings)
orch = PipelineOrchestrator(settings, mm)

def on_prog(p):
    print(f"  [{p.phase.value}] {p.message}", flush=True)
orch.on_progress(on_prog)

# ── Create project + scenes in DB ──────────────────────────────────────────
s = get_session()
ch = s.query(Channel).filter(Channel.slug == CHANNEL).first()
proj = Project(
    title="The Star Friends — Luna and Zain find a lost kitten",
    channel_id=ch.id,
    duration_target=10,
    num_scenes_target=2,
    context="Kindness to animals story with new characters Luna and Zain. 10s test.",
    status=ProjectStatus.APPROVED,
    video_model="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
    default_lora_ids=[l[0] for l in LTX_LORAS],
    default_lora_weights=[l[1] for l in LTX_LORAS],
)
proj.script_raw = json.dumps({
    "music_style": "soft gentle lullaby, light piano, music box melody, warm and peaceful",
    "music_mood": "warm, gentle, peaceful",
    "thumbnail_prompt": f"wide shot, {LUNA} and {ZAIN} holding a tiny calico kitten in a sunny garden, {ART}",
})
proj.total_scenes = len(SCENES)
s.add(proj); s.commit()
pid = proj.id

for sc in SCENES:
    scene = Scene(
        project_id=pid,
        scene_number=sc["scene_number"],
        scene_type=SceneType(sc["scene_type"]),
        prompt=sc["prompt"],
        negative_prompt=sc["negative_prompt"],
        duration=sc["duration"],
        camera_motion=sc["camera_motion"],
        narration_text=sc["narration_text"],
        director_notes=sc["director_notes"],
        lora_ids=sc.get("lora_ids"),
        lora_weights=sc.get("lora_weights"),
        status=SceneStatus.PENDING,
    )
    s.add(scene)
s.commit()
s.close()

print(f"Project {pid[:8]} created — 2 scenes, 10s target")
print(f"Characters: Luna (lavender hijab) + Zain (kufi cap)")
print(f"Framing: Scene 1 = wide establishing | Scene 2 = medium emotional")
print(f"LTX LoRAs: HDR (0.7) + Crisp Enhance (0.5)")
print(f"Quality: SDXL 36 steps + hires 18 steps | LTX 12 steps")
print(f"Starting pipeline...\n")

# ── Run ────────────────────────────────────────────────────────────────────
t0 = time.time()
orch.run_full_auto(pid)
elapsed = time.time() - t0
print(f"\nrun_full_auto finished in {elapsed:.0f}s ({elapsed/60:.1f}m)")

# ── QA ─────────────────────────────────────────────────────────────────────
proj_dir = settings.paths.projects_dir / pid
final = str(proj_dir / "final_render.mp4")
print("=" * 60)
if os.path.exists(final):
    ff = settings.paths.ffmpeg_bin
    out = subprocess.run([ff, "-i", final], capture_output=True, text=True).stderr
    dur = 0.0; w=h=0; audio = "Audio:" in out
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", out)
    if m: dur = int(m.group(1))*3600+int(m.group(2))*60+float(m.group(3))
    mv = re.search(r"Video:.*?(\d{3,5})x(\d{3,5})", out)
    if mv: w,h = int(mv.group(1)), int(mv.group(2))
    mb = os.path.getsize(final)/(1024*1024)
    print(f"[OK] FINAL VIDEO: {final}")
    print(f"     {mb:.1f}MB | {w}x{h} | {dur:.1f}s | audio={'YES' if audio else 'NO'}")
    imgs = list((proj_dir/'images').glob('*.png')) if (proj_dir/'images').exists() else []
    print(f"     SDXL stills: {len(imgs)}")
    print(f"     narration: {(proj_dir/'narration').exists()}  music: {os.path.exists(proj_dir/'music.wav')}")
else:
    print(f"[XX] no final video at {final}")
    s = get_session(); p = s.get(Project, pid)
    print(f"     status: {p.status} | error: {(p.error_log or '')[:300]}"); s.close()
print("=" * 60)
print(f"PROJECT_ID={pid}")
