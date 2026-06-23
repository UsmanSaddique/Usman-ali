"""
30s/40s kids Islamic video — dynamic LLM prompt generation.
Uses DirectorService to generate proper, descriptive scenes based on context.
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
from app.services.director import DirectorService

mm = ModelManager(); register_all_loaders(mm, settings)
orch = PipelineOrchestrator(settings, mm)
director = DirectorService(mm, settings)

_t = time.time()
orch.on_progress(lambda p: print(f"{time.time()-_t:6.0f}s [{p.phase.value}] {p.message}", flush=True))

s = get_session()
ch = s.query(Channel).filter(Channel.slug == "little-muslim-nation").first()

print("1. Generating Dynamic LLM Script via AI Director...")
script = director.generate_script(
    title="Yusuf and Amina: Wheels on the Bus (Islamic Version)",
    duration=60,
    context="A cute, magical 3D Pixar-style animated Islamic version of Wheels on the Bus. Yusuf (a cute 5 year old Muslim boy) and Amina (a cute 4 year old Muslim girl) take a joyful ride on a bright yellow school bus and arrive at a beautiful mosque. Write incredibly rich, descriptive visual prompts for each scene that maintain the character descriptions exactly and show them performing distinct actions. Include dynamic camera motions like 'pan_right' or 'zoom_in'.",
    channel_slug=ch.slug,
    num_scenes=12
)
director.manager.unload()

proj = Project(title=script.title, channel_id=ch.id, duration_target=60,
               num_scenes_target=len(script.scenes), context="60s kids islamic wheels on the bus, dynamic LLM generation",
               status=ProjectStatus.APPROVED,
               video_model="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf", total_scenes=len(script.scenes))
s.add(proj); s.flush(); pid = proj.id

for sc_plan in script.scenes:
    s.add(Scene(
        project_id=pid,
        scene_number=sc_plan.scene_number,
        scene_type=SceneType.TXT2VID,
        prompt=sc_plan.prompt,
        negative_prompt=sc_plan.negative_prompt,
        duration=5.0,
        camera_motion=sc_plan.camera_motion or "zoom_in",
        status=SceneStatus.PENDING
    ))
s.commit(); s.close()

print(f"Project {pid[:8]} | {len(script.scenes)} TXT2VID scenes (~60s) | AI Generated Script | 1080p\n")

proj_dir = settings.paths.projects_dir / pid
song = str(proj_dir / "song.mp3")
LYRICS = (
    "[chorus]\n"
    "The wheels on the bus go round and round,\n"
    "Round and round, round and round!\n"
    "The wheels on the bus go round and round,\n"
    "All through the town!\n"
    "[verse]\n"
    "We say Bismillah when we take our seat,\n"
    "Take our seat, take our seat,\n"
    "We say Bismillah when we take our seat,\n"
    "Before we eat and play!\n"
    "[chorus]\n"
    "The wheels on the bus go round and round,\n"
    "Round and round, round and round,\n"
    "Alhamdulillah for the friends we found,\n"
    "All through the town!\n"
)
try:
    from app.services.music_gen import MusicGenService
    music_prompt = script.music_style + ", cocomelon style nursery rhyme for kids, cheerful children's choir vocals, bright bouncy upbeat melody, simple catchy repetitive sing-along hook, claps and xylophone and ukulele, major key, wholesome, kids pop"
    MusicGenService(mm, settings).generate(
        style_prompt=music_prompt,
        duration=60, lyrics=LYRICS, instrumental=False, output_path=song)
    print("[song] ok")
except Exception as e:
    print("song:", e); song = None

# free VRAM so LTX has the full 16GB
try:
    from app.services.comfyui_client import ComfyUIClient
    ComfyUIClient().free_vram(); print("[vram] freed")
except Exception as e: print("free_vram:", e)

# generate (LTX txt2vid, 832x480)
g0 = time.time()
orch.start_generation(pid, width=832, height=480, batch=True)
gen_t = time.time() - g0

# upscale -> 1080p (anime ESRGAN clean cartoon edges)
TW, TH = 1920, 1080
u0 = time.time()
print(f"[upscale] -> {TW}x{TH} (anime ESRGAN) ...")
s2 = get_session()
scenes = s2.query(Scene).filter(Scene.project_id == pid,
                                Scene.status.in_([SceneStatus.GENERATED, SceneStatus.APPROVED])
                                ).order_by(Scene.scene_number).all()
for sc in scenes:
    gen = sc.active_generation
    if not gen or not gen.output_path: continue
    try:
        r = orch.upscaler.upscale_video(input_path=gen.output_path, target_width=TW,
                                        target_height=TH, ffmpeg_bin=settings.paths.ffmpeg_bin)
        gen.upscaled_path = r.output_path; print(f"    scene {sc.scene_number} ok")
    except Exception as e:
        print(f"    scene {sc.scene_number} upscale err: {type(e).__name__}: {e}")
        gen.upscaled_path = gen.output_path
    s2.commit()
s2.close()
ups_t = time.time() - u0

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
print(f"STATS | gen={gen_t:.0f}s upscale={ups_t:.0f}s render={rnd_t:.0f}s total={time.time()-_t:.0f}s")
print("=" * 60); print(f"PROJECT_ID={pid}")
