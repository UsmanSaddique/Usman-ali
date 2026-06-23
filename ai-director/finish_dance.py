"""Finish the Yusuf's Dance Party project: upscale generated clips + render with its song.mp3."""
import sys, os, time, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from app.config import settings
from app.database import get_session, Project, Scene, SceneStatus
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.pipeline import PipelineOrchestrator

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "3a26ea5d"
mm = ModelManager(); register_all_loaders(mm, settings)
orch = PipelineOrchestrator(settings, mm)
_t = time.time()
orch.on_progress(lambda p: print(f"{time.time()-_t:6.0f}s [{p.phase.value}] {p.message}", flush=True))

s = get_session()
proj = s.query(Project).filter(Project.id.like(PREFIX + "%")).first()
pid = proj.id
gen = s.query(Scene).filter(Scene.project_id == pid, Scene.status == SceneStatus.GENERATED).count()
print(f"Project {pid[:8]} | {proj.title} | {gen} generated scenes\n")
s.close()

proj_dir = settings.paths.projects_dir / pid
song = str(proj_dir / "song.mp3")
song = song if os.path.exists(song) else None
print(f"[song] {'using ' + song if song else 'none found'}")

TW, TH = 3840, 2160  # true 4K
print(f"[upscale] -> {TW}x{TH} (real 4x-UltraSharp ESRGAN via ComfyUI) ...")
t0 = time.time()
s2 = get_session()
scenes = s2.query(Scene).filter(Scene.project_id == pid,
                                Scene.status.in_([SceneStatus.GENERATED, SceneStatus.APPROVED])
                                ).order_by(Scene.scene_number).all()
for sc in scenes:
    gen = sc.active_generation
    if not gen or not gen.output_path:
        continue
    try:
        r = orch.upscaler.upscale_video(input_path=gen.output_path,
                                        target_width=TW, target_height=TH,
                                        ffmpeg_bin=settings.paths.ffmpeg_bin)
        gen.upscaled_path = r.output_path
        print(f"    scene {sc.scene_number}: {r.output_path}")
    except Exception as e:
        print(f"    scene {sc.scene_number} upscale err: {type(e).__name__}: {e}")
        gen.upscaled_path = gen.output_path
    s2.commit()
s2.close()
print(f"    done {time.time()-t0:.0f}s")

print("[render] ...")
t0 = time.time()
try: orch.render(pid, narration_path=None, music_path=song); print(f"    done {time.time()-t0:.0f}s")
except Exception as e: print(f"    render err: {type(e).__name__}: {e}")

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
