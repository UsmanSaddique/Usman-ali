"""
FULL one-click run: fresh English project -> run_full_auto
(script -> SDXL consistent stills + Ken Burns / LTX motion -> upscale 1080p
 -> instrumental music (ACE-Step) -> English voiceover (MMS-TTS) -> assemble).
Then QA the final video.
"""
import sys, os, time, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

from app.config import settings
from app.database import get_session, Channel, Project, ProjectStatus
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.pipeline import PipelineOrchestrator

CHANNEL = "urdu-moral-stories"  # now English narration
mm = ModelManager(); register_all_loaders(mm, settings)
orch = PipelineOrchestrator(settings, mm)

# progress logging
def on_prog(p):
    print(f"  [{p.phase.value}] {p.message}", flush=True)
orch.on_progress(on_prog)

s = get_session()
ch = s.query(Channel).filter(Channel.slug == CHANNEL).first()
proj = Project(
    title="The kind little rabbit who shared his carrots",
    channel_id=ch.id, duration_target=40, num_scenes_target=5,
    context="A warm 40-second moral story for young kids about kindness and sharing.",
    status=ProjectStatus.DRAFT,
    video_model="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
)
s.add(proj); s.commit()
pid = proj.id
s.close()
print(f"Project {pid[:8]} created on {CHANNEL}\nStarting full auto run...\n")

t0 = time.time()
orch.run_full_auto(pid)   # does everything, catches errors internally
print(f"\nrun_full_auto finished in {time.time()-t0:.0f}s")

# QA
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
    print(f"     {mb:.1f}MB | {w}x{h} | {dur:.1f}s | audio={'YES' if audio else 'NO'} | 1080p={'YES' if h>=1000 else 'no'}")
    # report what's in the project
    imgs = list((proj_dir/'images').glob('*.png')) if (proj_dir/'images').exists() else []
    print(f"     SDXL stills generated: {len(imgs)}")
    print(f"     narration: {(proj_dir/'narration').exists()}  music: {os.path.exists(proj_dir/'music.wav')}")
else:
    print(f"[XX] no final video at {final}")
    # show project error
    s = get_session(); p = s.query(Project).get(pid)
    print(f"     project status: {p.status} | error: {(p.error_log or '')[:200]}"); s.close()
print("=" * 60)
print(f"PROJECT_ID={pid}")
