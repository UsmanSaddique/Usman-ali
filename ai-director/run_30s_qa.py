"""
2-min Dada Abu Quran Story (Old MacDonald Islamic Version).
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
    title="Dada Abu's Quran Story (Old MacDonald Islamic Version)",
    duration=120,
    context=(
        "A cute, magical 3D Pixar-style animated Islamic kids video inspired by 'Old MacDonald Had a Farm' "
        "but reimagined as 'Old Dada Abu Had a Quran Pak'. "
        "Dada Abu is a lovable old Muslim grandfather — round face, long fluffy white beard, warm brown skin, "
        "kind crinkly eyes behind small round glasses, white prayer cap (topi), long flowing white cotton kurta, "
        "always holding a beautiful golden Quran. He lives on a peaceful little Islamic homestead/garden. "
        "Each verse introduces a different cute 3D animal friend (a fluffy white lamb, a soft brown camel, a colorful parrot, "
        "a little orange cat, a gentle donkey) and Dada Abu teaches them a small Islamic lesson. "
        "CRITICAL: Dada Abu should not be alone. In every scene, Dada Abu must be interacting with Yusuf (a cute 5-year-old Muslim boy in a white topi and blue kurta) and Amina (a cute 4-year-old Muslim girl in a pink hijab and yellow dress). They are all laughing and playing with the animals together. "
        "The structure follows Old MacDonald exactly: 'Old Dada Abu had a Quran, Alhamdulillah! "
        "And with that Quran he taught a [animal], Alhamdulillah! With a [lesson] here and a [lesson] there...' "
        "Write incredibly rich, detailed 3D Pixar-style visual prompts. Each scene must describe Dada Abu, Yusuf, Amina, and the exact animal. "
        "CRITICAL: Frame the shots as wide shots or full-body medium shots so the characters are NOT too zoomed in. "
        "Use dynamic camera motions like zoom_out, pan_right, pan_left. "
        "Include peaceful outdoor garden/farm settings with golden hour lighting, soft bokeh, and warm saturated colors."
    ),
    channel_slug=ch.slug,
    num_scenes=20
)
director.manager.unload()

proj = Project(title=script.title, channel_id=ch.id, duration_target=120,
               num_scenes_target=len(script.scenes), context="2min Dada Abu Quran Story, Old MacDonald Islamic, 20x6s, dynamic LLM",
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
        duration=6.0,
        camera_motion=sc_plan.camera_motion or "zoom_out",
        status=SceneStatus.PENDING
    ))
s.commit(); s.close()

print(f"Project {pid[:8]} | {len(script.scenes)} TXT2VID scenes (~2min, 20x6s) | AI Generated Script | 1080p\n")

proj_dir = settings.paths.projects_dir / pid
song = str(proj_dir / "song.mp3")
LYRICS = (
    "[chorus]\n"
    "Old Dada Abu had a farm, Alhamdulillah!\n"
    "And on that farm he had a lamb, Alhamdulillah!\n"
    "With a baa baa here and a baa baa there,\n"
    "Saying Bismillah everywhere,\n"
    "Old Dada Abu had a farm, Alhamdulillah!\n"
    "[verse]\n"
    "Old Dada Abu had a farm, Alhamdulillah!\n"
    "And on that farm he had a cat, Alhamdulillah!\n"
    "With a meow meow here and a meow meow there,\n"
    "Saying Salam everywhere,\n"
    "Old Dada Abu had a farm, Alhamdulillah!\n"
    "[verse]\n"
    "Old Dada Abu had a farm, Alhamdulillah!\n"
    "And on that farm he had a camel, Alhamdulillah!\n"
    "With a walk walk here and a walk walk there,\n"
    "Saying Subhanallah everywhere,\n"
    "Old Dada Abu had a farm, Alhamdulillah!\n"
    "[verse]\n"
    "Old Dada Abu had a farm, Alhamdulillah!\n"
    "And on that farm he had a parrot, Alhamdulillah!\n"
    "With a tweet tweet here and a tweet tweet there,\n"
    "Saying Alhamdulillah everywhere,\n"
    "Old Dada Abu had a farm, Alhamdulillah!\n"
    "[chorus]\n"
    "Old Dada Abu had a farm, Alhamdulillah!\n"
    "And on that farm he had a donkey, Alhamdulillah!\n"
    "With a hee haw here and a hee haw there,\n"
    "Making Dua everywhere,\n"
    "Old Dada Abu had a farm, Alhamdulillah!\n"
)
try:
    from app.services.music_gen import MusicGenService
    music_prompt = script.music_style + ", simple acoustic children's song, solo sweet female voice, one acoustic guitar, high quality studio recording, perfect clear vocals, catchy nursery rhyme, kids music"
    MusicGenService(mm, settings).generate(
        style_prompt=music_prompt,
        duration=120, lyrics=LYRICS, instrumental=False, output_path=song)
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
