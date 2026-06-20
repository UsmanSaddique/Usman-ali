"""
Resume E2E on an already-scripted project: clips @720p -> upscale 1080p -> render.
No LLM load (script already exists) => no single-GPU VRAM contention with ComfyUI.

Usage: python_embeded\\python.exe resume_e2e.py <project_id_prefix>
"""
import sys, os, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

from app.config import settings
from app.database import get_session, Project, Scene, SceneStatus, Generation, GenerationStatus
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.pipeline import PipelineOrchestrator
from app.services.comfyui_client import ComfyUIClient

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "b6575e5c"
BASE_W, BASE_H = 768, 512   # VRAM-safe for LTX-22B on 16GB; 4x upscaler takes it to 1080p
CLIP_SECONDS = 5.0

def ffinfo(path):
    ff = settings.paths.ffmpeg_bin
    out = subprocess.run([ff, "-i", path], capture_output=True, text=True).stderr
    import re
    dur = 0.0; w = h = 0; audio = False
    for ln in out.splitlines():
        if "Duration:" in ln:
            t = ln.split("Duration:")[1].split(",")[0].strip().split(":")
            dur = int(t[0])*3600 + int(t[1])*60 + float(t[2])
        if "Video:" in ln:
            m = re.search(r"(\d{3,5})x(\d{3,5})", ln)
            if m: w, h = int(m.group(1)), int(m.group(2))
        if "Audio:" in ln: audio = True
    return dur, w, h, audio

if not ComfyUIClient().wait_ready(30):
    print("[XX] ComfyUI not reachable — start it first."); sys.exit(2)

mm = ModelManager(); register_all_loaders(mm, settings)
orch = PipelineOrchestrator(settings, mm)

s = get_session()
proj = s.query(Project).filter(Project.id.like(PREFIX + "%")).first()
if not proj:
    print(f"[XX] no project matching {PREFIX}"); sys.exit(1)
pid = proj.id
scenes = s.query(Scene).filter(Scene.project_id == pid).order_by(Scene.scene_number).all()
print(f"Project: {proj.title[:50]} | {len(scenes)} scenes")
# reset scenes -> PENDING, cap durations LTX-safe, clear old failed gens
for sc in scenes:
    sc.status = SceneStatus.PENDING
    sc.retry_count = 0
    if sc.duration > CLIP_SECONDS:
        sc.duration = CLIP_SECONDS
s.commit(); s.close()

# ── generate @720p ──
t0 = time.time()
print(f"\n[1] Generating {len(scenes)} clips @ {BASE_W}x{BASE_H} ...")
try:
    orch.start_generation(pid, width=BASE_W, height=BASE_H)
except Exception as e:
    print(f"   generation raised: {type(e).__name__}: {e}")
s = get_session()
gen_ok = s.query(Scene).filter(Scene.project_id == pid, Scene.status == SceneStatus.GENERATED).count()
s.close()
print(f"[1] {gen_ok}/{len(scenes)} clips generated in {time.time()-t0:.0f}s")

# ── upscale -> 1080p ──
print("\n[2] Upscaling -> 1080p ...")
t0 = time.time()
try:
    orch.start_upscale(pid)
    print(f"[2] upscale done in {time.time()-t0:.0f}s")
except Exception as e:
    print(f"[2] upscale failed: {e}")

# ── custom song ──
proj_dir = settings.paths.projects_dir / pid
song = str(proj_dir / "my_custom_song.wav")
if not os.path.exists(song):
    subprocess.run([settings.paths.ffmpeg_bin, "-y", "-f", "lavfi",
                    "-i", "sine=frequency=330:duration=45", "-af", "volume=0.2", song],
                   capture_output=True)

# ── render ──
print("\n[3] Rendering final video (with custom song) ...")
t0 = time.time()
try:
    orch.render(pid, music_path=song)
    print(f"[3] render done in {time.time()-t0:.0f}s")
except Exception as e:
    print(f"[3] render failed: {type(e).__name__}: {e}")

# ── QA ──
final = str(proj_dir / "final_render.mp4")
print("\n" + "=" * 60)
if os.path.exists(final):
    dur, w, h, audio = ffinfo(final)
    mb = os.path.getsize(final) / (1024*1024)
    print(f"[OK] FINAL VIDEO: {final}")
    print(f"     {mb:.1f}MB | {w}x{h} | {dur:.1f}s | audio={'yes' if audio else 'NO'}")
    print(f"     resolution 1080p: {'YES' if h >= 1000 else 'NO ('+str(h)+'p)'}")
else:
    print(f"[XX] no final video at {final}")
print("=" * 60)
