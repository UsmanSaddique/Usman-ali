"""
NEW song video (different story), addressing latest feedback:
  - different story (Greedy Monkey, not the rabbit), 8 scenes (more clips)
  - NATIVE 16:9 base 1024x576 -> 1920x1080 (TV size, no ultra-wide stretch / no letterbox)
  - LLM-enhanced cinematic prompts
  - img2vid (real motion) for every scene
  - generated SONG audio + sharpened upscale (crisper, less blur)
"""
import sys, os, time, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from app.config import settings
from app.database import get_session, Project, Scene, SceneType, SceneStatus
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.pipeline import PipelineOrchestrator
from app.services.director import DirectorService

SRC = "d622554a"          # Greedy Monkey story
N_SCENES = 8
BASE_W, BASE_H = 1024, 576   # true 16:9 -> clean 1920x1080, no stretch/letterbox

mm = ModelManager(); register_all_loaders(mm, settings)
orch = PipelineOrchestrator(settings, mm)
director = DirectorService(mm, settings)
_t = time.time()
orch.on_progress(lambda p: print(f"{time.time()-_t:6.0f}s [{p.phase.value}] {p.message}", flush=True))

s = get_session()
p = s.query(Project).filter(Project.id.like(SRC + "%")).first()
pid = p.id; slug = p.channel.slug
scenes = s.query(Scene).filter(Scene.project_id == pid).order_by(Scene.scene_number).all()[:N_SCENES]
keep = {sc.id for sc in scenes}
s.query(Scene).filter(Scene.project_id == pid, ~Scene.id.in_(keep)).delete(synchronize_session=False)
to_enh = [{"scene_number": sc.scene_number, "prompt": sc.prompt, "negative_prompt": sc.negative_prompt} for sc in scenes]
s.commit(); s.close()

# 1) Enhance prompts (cinematic) via LLM
print(f"Enhancing {len(to_enh)} prompts...")
enh = {e.get("scene_number"): e for e in director.enhance_prompts(to_enh, slug)}
director.manager.unload()

s = get_session()
for sc in s.query(Scene).filter(Scene.project_id == pid).order_by(Scene.scene_number).all():
    e = enh.get(sc.scene_number)
    if e and e.get("prompt"): sc.prompt = e["prompt"]
    if e and e.get("negative_prompt"): sc.negative_prompt = e["negative_prompt"]
    sc.scene_type = SceneType.IMG2VID
    sc.status = SceneStatus.PENDING
    sc.retry_count = 0
    sc.duration = 4.0
s.commit(); s.close()
print(f"Project {pid[:8]}: {N_SCENES} img2vid scenes @ {BASE_W}x{BASE_H} (16:9), enhanced\n")

# 2) Song (upbeat, vocal)
proj_dir = settings.paths.projects_dir / pid
song = str(proj_dir / "song.mp3")
try:
    from app.services.music_gen import MusicGenService
    MusicGenService(mm, settings).generate(
        style_prompt="upbeat playful kids adventure song, acoustic, bright, catchy, light vocals",
        duration=36, lyrics="[verse]\nUp the tree the mango grows\nShare the sweetness, everyone knows\n",
        instrumental=False, output_path=song)
    print("[song] ok")
except Exception as e:
    print("song failed:", e); song = None

# 3) Generate motion clips at native 16:9, then sharpened upscale
orch.start_generation(pid, width=BASE_W, height=BASE_H)
try: orch.start_upscale(pid)
except Exception as e: print("upscale:", e)
try: orch.render(pid, narration_path=None, music_path=song)
except Exception as e: print("render:", e)

# QA
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
    print(f"     {os.path.getsize(final)/1048576:.1f}MB | {w}x{h} ({'16:9' if abs(w/h-16/9)<0.02 else 'other'}) | {dur:.1f}s | song={'YES' if audio else 'no'}")
else:
    print(f"[XX] no final at {final}")
print("=" * 60); print(f"PROJECT_ID={pid}")
