"""Finish E2E: upscale (ffmpeg fallback) + render on already-generated clips."""
import sys, os, time, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from app.config import settings
from app.database import get_session, Project, Scene, SceneStatus
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.pipeline import PipelineOrchestrator

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "b6575e5c"
mm = ModelManager(); register_all_loaders(mm, settings)
orch = PipelineOrchestrator(settings, mm)

s = get_session()
proj = s.query(Project).filter(Project.id.like(PREFIX + "%")).first()
pid = proj.id
gen = s.query(Scene).filter(Scene.project_id == pid, Scene.status == SceneStatus.GENERATED).count()
print(f"Project {pid[:8]} | {gen} generated scenes")
s.close()

print("[2] upscale -> 1080p (ffmpeg lanczos fallback) ...")
t0 = time.time()
try:
    orch.start_upscale(pid); print(f"    done {time.time()-t0:.0f}s")
except Exception as e:
    print(f"    upscale err: {e}")

proj_dir = settings.paths.projects_dir / pid
song = str(proj_dir / "my_custom_song.wav")
if not os.path.exists(song):
    subprocess.run([settings.paths.ffmpeg_bin, "-y", "-f", "lavfi",
                    "-i", "sine=frequency=330:duration=45", "-af", "volume=0.2", song],
                   capture_output=True)

print("[3] render ...")
t0 = time.time()
try:
    orch.render(pid, music_path=song); print(f"    done {time.time()-t0:.0f}s")
except Exception as e:
    print(f"    render err: {type(e).__name__}: {e}")

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
    mb = os.path.getsize(final)/(1024*1024)
    print(f"[OK] FINAL: {final}")
    print(f"     {mb:.1f}MB | {w}x{h} | {dur:.1f}s | audio={'yes' if audio else 'NO'} | 1080p={'YES' if h>=1000 else 'no'}")
else:
    print(f"[XX] no final at {final}")
print("=" * 60)
