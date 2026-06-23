"""
20s cute Islamic poem video — CHARACTER BIBLE + LTX text-to-video approach.
No per-scene SDXL still (that caused VRAM-thrash slowdown + character drift).
Consistency comes from: a fixed STYLE block + a fixed CHARACTER block repeated in
every prompt + one locked seed for all scenes. Each scene only changes the ACTION
+ LOCATION + TIME/WEATHER. Upscales to 1080p with the anime ESRGAN (clean edges).
"""
import sys, os, time, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from app.config import settings
from app.database import (get_session, Channel, Project, Scene, SceneType,
                          SceneStatus, ProjectStatus)
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.pipeline import PipelineOrchestrator

mm = ModelManager(); register_all_loaders(mm, settings)
orch = PipelineOrchestrator(settings, mm)
_t = time.time()
orch.on_progress(lambda p: print(f"{time.time()-_t:6.0f}s [{p.phase.value}] {p.message}", flush=True))

# ---- FIXED blocks (repeated every scene => consistent look + character) ----
STYLE = ("3D Pixar-style animated cartoon, high quality 3D render, soft cinematic studio lighting, "
         "shallow depth of field, warm vibrant colors, clean smooth surfaces, eye-level medium close-up shot")
CHAR = ("Yusuf a cute 5 year old Muslim boy, chubby round face, warm light-brown skin, big round dark-brown eyes, "
        "thick curly dark-brown hair, white knit prayer cap, light cream cotton kurta, rosy cheeks, gentle happy smile")
# ---- per-scene: ACTION + LOCATION + TIME/WEATHER only ----
ACTIONS = [
    "raising both cupped hands together making a peaceful dua, eyes gently closed, in a cozy warm room, soft daylight",
    "happily reading an open Quran on a small wooden stand, in a bright room, warm morning sunlight from a window",
    "saying bismillah with hands together before a small plate of fresh fruit, at a tidy little wooden table, daytime",
    "looking up in wonder at a sky full of glowing stars, hands on cheeks, standing on a balcony, calm clear night",
    "standing peacefully with hands together in front of a beautiful domed mosque, warm golden sunset light",
]
NEG = ("photorealistic, realistic human, live action, text, watermark, two boys, extra people, "
       "deformed face, distorted face, asymmetric eyes, melted eyes, extra fingers, fused fingers, "
       "mutated hands, malformed hands, blurry, soft focus, grainy, lowres, oil painting, sketch")
SCENES = [f"{STYLE}. {CHAR}. {a}." for a in ACTIONS]

s = get_session()
ch = s.query(Channel).filter(Channel.slug == "little-muslim-nation").first()
proj = Project(title="Yusuf's Little Islamic Poem (t2v)", channel_id=ch.id, duration_target=20,
               num_scenes_target=5, context="20s Islamic poem, character-bible txt2vid",
               status=ProjectStatus.APPROVED,
               video_model="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf", total_scenes=5)
s.add(proj); s.flush(); pid = proj.id
for i, pr in enumerate(SCENES, 1):
    s.add(Scene(project_id=pid, scene_number=i, scene_type=SceneType.TXT2VID,
                prompt=pr, negative_prompt=NEG, duration=4.0, status=SceneStatus.PENDING))
s.commit(); s.close()
print(f"Project {pid[:8]} | 5 TXT2VID scenes (character bible, locked seed) | 1080p\n")

proj_dir = settings.paths.projects_dir / pid
song = str(proj_dir / "song.mp3")
# Cocomelon-style: simple bouncy nursery-rhyme melody, a STRONG repeated hook the
# kid can sing along to, AABB rhymes, claps. Hook repeats so it sticks.
LYRICS = (
    "[chorus]\n"
    "Bismillah, Bismillah, before we eat and play!\n"
    "Bismillah, Bismillah, we say it every day!\n"
    "[verse]\n"
    "Alhamdulillah for the sun so bright,\n"
    "Alhamdulillah for the stars at night,\n"
    "Be kind and share and say your dua,\n"
    "Allah loves you, yes He does, hurray!\n"
    "[chorus]\n"
    "Bismillah, Bismillah, before we eat and play!\n"
    "Bismillah, Bismillah, we say it every day!\n"
)
try:
    from app.services.music_gen import MusicGenService
    MusicGenService(mm, settings).generate(
        style_prompt=("cocomelon style nursery rhyme for kids, cheerful children's choir vocals, "
                      "bright bouncy upbeat melody, simple catchy repetitive sing-along hook, "
                      "claps and xylophone and ukulele, playful, major key, wholesome, kids pop"),
        duration=22, lyrics=LYRICS, instrumental=False, output_path=song)
    print("[song] ok")
except Exception as e:
    print("song:", e); song = None

# free the ACE-Step music model from VRAM so LTX has the full 16GB
try:
    from app.services.comfyui_client import ComfyUIClient
    ComfyUIClient().free_vram(); print("[vram] freed before generation")
except Exception as e:
    print("free_vram:", e)

# 1. generate ALL scenes via LTX txt2vid (one model load, no SDXL swap).
# 832x480 (16:9) keeps LTX-22B + Gemma encoder resident in 16GB -> stays FAST
# (~1 min/scene). Above ~768x512 it spills to system RAM and crawls (49s/step).
g0 = time.time()
orch.start_generation(pid, width=832, height=480)
gen_t = time.time() - g0

# 2. upscale -> 1080p (anime ESRGAN, clean cartoon edges)
TW, TH = 1920, 1080
u0 = time.time()
print(f"[upscale] -> {TW}x{TH} (anime ESRGAN) ...")
s2 = get_session()
scenes = s2.query(Scene).filter(Scene.project_id == pid,
                                Scene.status.in_([SceneStatus.GENERATED, SceneStatus.APPROVED])
                                ).order_by(Scene.scene_number).all()
for sc in scenes:
    gen = sc.active_generation
    if not gen or not gen.output_path:
        continue
    try:
        r = orch.upscaler.upscale_video(input_path=gen.output_path, target_width=TW,
                                        target_height=TH, ffmpeg_bin=settings.paths.ffmpeg_bin)
        gen.upscaled_path = r.output_path
        print(f"    scene {sc.scene_number} ok")
    except Exception as e:
        print(f"    scene {sc.scene_number} upscale err: {type(e).__name__}: {e}")
        gen.upscaled_path = gen.output_path
    s2.commit()
s2.close()
ups_t = time.time() - u0

# 3. render @1080p
r0 = time.time()
try: orch.render(pid, narration_path=None, music_path=song, resolution="1080p")
except Exception as e: print("render:", e)
rnd_t = time.time() - r0

final = str(proj_dir / "final_render.mp4")
print("=" * 60)
if os.path.exists(final):
    ff = settings.paths.ffmpeg_bin
    out = subprocess.run([ff, "-i", final], capture_output=True, text=True).stderr
    dur = 0.0; w = h = 0; audio = "Audio:" in out
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", out)
    if m: dur = int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3))
    mv = re.search(r"Video:.*?(\d{3,5})x(\d{3,5})", out)
    if mv: w, h = int(mv.group(1)), int(mv.group(2))
    print(f"[OK] FINAL: {final}")
    print(f"     {os.path.getsize(final)/1048576:.1f}MB | {w}x{h} | {dur:.1f}s | audio={'YES' if audio else 'no'}")
else:
    print(f"[XX] no final at {final}")
print(f"STATS | gen={gen_t:.0f}s upscale={ups_t:.0f}s render={rnd_t:.0f}s total={time.time()-_t:.0f}s")
print("=" * 60); print(f"PROJECT_ID={pid}")
