"""
Regenerate an already-scripted project with the consistency + audio fixes,
SKIPPING the slow LLM. All scenes -> SDXL stills + Ken Burns (locked seed),
English voiceover (MMS-TTS), instrumental music (ACE-Step), 1080p.
"""
import sys, os, time, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from app.config import settings
from app.database import get_session, Project, Scene, SceneType, SceneStatus
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.pipeline import PipelineOrchestrator

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "ec26a326"
mm = ModelManager(); register_all_loaders(mm, settings)
orch = PipelineOrchestrator(settings, mm)
orch.on_progress(lambda p: print(f"  [{p.phase.value}] {p.message}", flush=True))

s = get_session()
p = s.query(Project).filter(Project.id.like(PREFIX + "%")).first()
pid = p.id
# force ALL scenes -> still_pan (SDXL + Ken Burns), reset to PENDING
for sc in s.query(Scene).filter(Scene.project_id == pid).all():
    sc.scene_type = SceneType.STILL_PAN
    sc.status = SceneStatus.PENDING
    sc.retry_count = 0
    if sc.duration > 8: sc.duration = 8.0
s.commit()
print(f"Project {pid[:8]}: all scenes -> still_pan, reset.\n")
s.close()

# 1) Narration (English MMS-TTS)
t0 = time.time()
narr = orch.generate_tts(pid)
print(f"[narration] {'OK '+str(narr) if narr else 'none'} ({time.time()-t0:.0f}s)")

# 2) Generate stills + Ken Burns (SDXL via ComfyUI, locked seed)
t0 = time.time()
orch.start_generation(pid)   # still_pan -> SDXL image + ken burns
print(f"[generation] {time.time()-t0:.0f}s")

# 3) Upscale -> 1080p
try: orch.start_upscale(pid)
except Exception as e: print("upscale:", e)

# 4) Instrumental music (ACE-Step)
t0 = time.time()
music = None
try:
    music = orch.generate_music(pid)
    print(f"[music] {music} ({time.time()-t0:.0f}s)")
except Exception as e: print("music:", e)

# 5) Render
try: orch.render(pid, narration_path=narr, music_path=music)
except Exception as e: print("render:", e)

# QA
pdir = settings.paths.projects_dir / pid
final = str(pdir / "final_render.mp4")
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
    print(f"     {os.path.getsize(final)/1048576:.1f}MB | {w}x{h} | {dur:.1f}s | audio={'YES' if audio else 'no'} | 1080p={'YES' if h>=1000 else 'no'}")
    print(f"     SDXL stills: {len(list((pdir/'images').glob('*.png')))}")
else:
    print(f"[XX] no final at {final}")
print("=" * 60)
