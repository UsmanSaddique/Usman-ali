"""
20s video with the TRAINED Yusuf character (LoRA) — 5 img2vid scenes + song.
No LLM (manual scenes). Uses config.image.style_loras (pixar 0.6 + yusuf 0.5).
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
    f"{ID}, waving hello happily in a sunny flower garden, gentle motion",
    f"{ID}, riding a small bicycle along a pastel old-town street, moving forward",
    f"{ID}, sitting under a tree reading a book, pages turning, leaves swaying",
    f"{ID}, playing joyfully with a colorful ball in a green park, bouncing motion",
    f"{ID}, sitting peacefully with hands together in front of a beautiful mosque at sunset",
]
NEG = "photorealistic, realistic skin, text, watermark, deformed, extra fingers, blurry, grainy, two boys"

s = get_session()
ch = s.query(Channel).filter(Channel.slug == "little-muslim-nation").first()
proj = Project(title="Yusuf's Happy Day", channel_id=ch.id, duration_target=20,
               num_scenes_target=5, context="20s cartoon with Yusuf", status=ProjectStatus.APPROVED,
               video_model="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf", total_scenes=5)
s.add(proj); s.flush(); pid = proj.id
for i, pr in enumerate(SCENES, 1):
    s.add(Scene(project_id=pid, scene_number=i, scene_type=SceneType.IMG2VID,
                prompt=pr, negative_prompt=NEG, duration=4.0, status=SceneStatus.PENDING))
s.commit(); s.close()
print(f"Project {pid[:8]} | 5 Yusuf img2vid scenes (20s) | LoRA cast\n")

# song (gentle, instrumental so it suits kids/islamic tone)
proj_dir = settings.paths.projects_dir / pid
song = str(proj_dir / "song.mp3")
try:
    from app.services.music_gen import MusicGenService
    MusicGenService(mm, settings).generate(
        style_prompt="gentle warm children's nasheed-style song, soft vocals, light percussion, peaceful, wholesome",
        duration=24, lyrics="[verse]\nA happy day, we smile and play\nKindness lights our way\n",
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
