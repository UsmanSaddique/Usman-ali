"""
20s cute Islamic poem (nasheed) video with the Yusuf character.
Uses the fixed HQ pipeline (Samaritan 3D-cartoon style + hires stills + tight
framing). Upscales to 1080p (not 4K).
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

ID = ("a cute 3D pixar style cartoon muslim boy, white knit prayer cap, brown curly hair, "
      "big expressive brown eyes, light kurta")
FRAME = ("medium close-up shot from the waist up, face large and clearly visible, "
         "highly detailed symmetric face, sharp clean eyes, detailed hands, smooth clean 3D render, "
         "soft warm lighting, vibrant colors, high detail")
SCENES = [
    f"{ID}, raising both cupped hands together making a peaceful dua, gentle smile, {FRAME}, in a cozy warm room",
    f"{ID}, happily reading an open Quran on a small wooden stand, {FRAME}, soft sunlight from a window",
    f"{ID}, saying bismillah with hands together before a plate of fruit, smiling, {FRAME}, at a tidy little table",
    f"{ID}, looking up in wonder at a starry night sky, hands on cheeks, {FRAME}, on a balcony at night",
    f"{ID}, standing peacefully with hands together in front of a beautiful mosque, {FRAME}, warm golden sunset",
]
NEG = ("photorealistic, realistic skin, text, watermark, two boys, extra people, "
       "deformed face, distorted face, asymmetric eyes, crossed eyes, melted eyes, "
       "extra fingers, fused fingers, missing fingers, mutated hands, malformed hands, "
       "blurry, soft focus, grainy, lowres, low detail, oil painting, sketch")

s = get_session()
ch = s.query(Channel).filter(Channel.slug == "little-muslim-nation").first()
proj = Project(title="Yusuf's Little Islamic Poem", channel_id=ch.id, duration_target=20,
               num_scenes_target=5, context="20s cute Islamic poem nasheed with Yusuf",
               status=ProjectStatus.APPROVED,
               video_model="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf", total_scenes=5)
s.add(proj); s.flush(); pid = proj.id
for i, pr in enumerate(SCENES, 1):
    s.add(Scene(project_id=pid, scene_number=i, scene_type=SceneType.IMG2VID,
                prompt=pr, negative_prompt=NEG, duration=4.0, status=SceneStatus.PENDING))
s.commit(); s.close()
print(f"Project {pid[:8]} | 5 Yusuf Islamic-poem scenes (20s) | 1080p\n")

# gentle nasheed-style sung poem (wholesome, kid-friendly)
proj_dir = settings.paths.projects_dir / pid
song = str(proj_dir / "song.mp3")
LYRICS = (
    "[verse]\n"
    "Bismillah I always say, before I eat and before I play\n"
    "Alhamdulillah every day, for all the gifts that come my way\n"
    "[verse]\n"
    "Allah made the moon and stars, the gentle breeze, the world so far\n"
    "I'll be kind and I will share, and say my dua with loving care\n"
)
try:
    from app.services.music_gen import MusicGenService
    MusicGenService(mm, settings).generate(
        style_prompt="gentle wholesome children nasheed, soft warm vocals, light hand percussion, calm, sweet, islamic kids song",
        duration=22, lyrics=LYRICS, instrumental=False, output_path=song)
    print("[song] ok")
except Exception as e:
    print("song:", e); song = None

# 1. generate (Samaritan 3D still -> LTX img2vid) @1024x576
orch.start_generation(pid, width=1024, height=576)

# 2. upscale -> 1080p (real 4x-UltraSharp ESRGAN via ComfyUI, downscaled to 1080p)
TW, TH = 1920, 1080
print(f"[upscale] -> {TW}x{TH} (4x-UltraSharp ESRGAN) ...")
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

# 3. render @1080p with song
try: orch.render(pid, narration_path=None, music_path=song, resolution="1080p")
except Exception as e: print("render:", e)

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
print("=" * 60); print(f"PROJECT_ID={pid}")
