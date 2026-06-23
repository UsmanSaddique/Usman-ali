"""
~12s dance-song video with the TRAINED Yusuf character (LoRA) — 3 img2vid
dancing scenes + an upbeat song. No LLM (manual scenes).
Uses config.image.style_loras (pixar 0.6 + yusuf 0.5).
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

ID = "a cute 3D pixar cartoon muslim boy, white knit prayer cap, brown curly hair, big brown eyes, light kurta"
SCENES = [
    f"{ID}, dancing happily and clapping hands in a colorful party room with balloons, energetic bouncing motion",
    f"{ID}, spinning and jumping with joy on a bright stage with sparkling lights, lively dance moves",
    f"{ID}, doing a fun arms-up dance in a sunny garden with confetti falling, cheerful hopping motion",
]
NEG = "photorealistic, realistic skin, text, watermark, deformed, extra fingers, blurry, grainy, two boys"

s = get_session()
ch = s.query(Channel).filter(Channel.slug == "little-muslim-nation").first()
proj = Project(title="Yusuf's Dance Party", channel_id=ch.id, duration_target=12,
               num_scenes_target=3, context="12s dance song with Yusuf", status=ProjectStatus.APPROVED,
               video_model="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf", total_scenes=3)
s.add(proj); s.flush(); pid = proj.id
for i, pr in enumerate(SCENES, 1):
    s.add(Scene(project_id=pid, scene_number=i, scene_type=SceneType.IMG2VID,
                prompt=pr, negative_prompt=NEG, duration=4.0, status=SceneStatus.PENDING))
s.commit(); s.close()
print(f"Project {pid[:8]} | 3 Yusuf dancing img2vid scenes (12s) | LoRA cast\n")

# upbeat dance song (wholesome, kid-friendly nasheed-style but danceable)
proj_dir = settings.paths.projects_dir / pid
song = str(proj_dir / "song.mp3")
try:
    from app.services.music_gen import MusicGenService
    MusicGenService(mm, settings).generate(
        style_prompt="upbeat happy kids dance song, catchy rhythm, light percussion and claps, joyful, energetic, wholesome",
        duration=14, lyrics="[verse]\nClap your hands and move your feet\nDance along to the happy beat\n",
        instrumental=False, output_path=song)
    print("[song] ok")
except Exception as e:
    print("song:", e); song = None

# generate (img2vid @ 1024x576 16:9) -> upscale -> render with song
orch.start_generation(pid, width=1024, height=576)
try: orch.start_upscale(pid)
except Exception as e: print("upscale:", e)
try: orch.render(pid, narration_path=None, music_path=song)
except Exception as e: print("render:", e)

final = str(proj_dir / "final_render.mp4")
print("=" * 60)
if os.path.exists(final):
    ff = settings.paths.ffmpeg_bin
    out = subprocess.run([ff, "-i", final], capture_output=True, text=True).stderr
    dur=0; w=h=0; audio="Audio:" in out
    m=re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", out)
    if m: dur=int(m.group(1))*3600+int(m.group(2))*60+float(m.group(3))
    mv=re.search(r"Video:.*?(\d{3,5})x(\d{3,5})", out)
    if mv: w,h=int(mv.group(1)),int(mv.group(2))
    print(f"[OK] FINAL: {final}")
    print(f"     {os.path.getsize(final)/1048576:.1f}MB | {w}x{h} | {dur:.1f}s | song={'YES' if audio else 'no'}")
else:
    print(f"[XX] no final at {final}")
print("=" * 60); print(f"PROJECT_ID={pid}")
