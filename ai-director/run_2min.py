"""
2-minute director-grade video on the reference-channel niche (kids moral story).
Strong hook + upbeat intro feel, 15 consistent SDXL-still scenes, voiceover + music.
Writes phase timings to phases.log for the perf report.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from app.config import settings
from app.database import get_session, Channel, Project, Scene, SceneType, SceneStatus, ProjectStatus
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.pipeline import PipelineOrchestrator

CHANNEL = "urdu-moral-stories"
mm = ModelManager(); register_all_loaders(mm, settings)
orch = PipelineOrchestrator(settings, mm)

phase_log = open("phases.log", "w")
_start = time.time()
def on_prog(p):
    line = f"{time.time()-_start:7.0f}s  [{p.phase.value}] {p.message}"
    print(line, flush=True); phase_log.write(line + "\n"); phase_log.flush()
orch.on_progress(on_prog)

s = get_session()
ch = s.query(Channel).filter(Channel.slug == CHANNEL).first()
proj = Project(
    title="The Greedy Monkey and the Magic Mango Tree",
    channel_id=ch.id, duration_target=120, num_scenes_target=12,
    context=("A 2-minute moral story for kids. OPEN WITH A STRONG HOOK in scene 1 "
             "(a mysterious/exciting moment + a narration question that makes kids HAVE to keep watching), "
             "upbeat magical intro feel. Build curiosity, a clear lesson about greed vs sharing, "
             "and a warm satisfying ending. Keep one consistent animal hero throughout."),
    status=ProjectStatus.DRAFT,
    video_model="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
)
s.add(proj); s.commit(); pid = proj.id; s.close()
print(f"Project {pid[:8]} | 2-min video | {CHANNEL}\n")

t0 = time.time()
orch.run_full_auto(pid)
phase_log.write(f"{time.time()-_start:7.0f}s  [TOTAL] run_full_auto {time.time()-t0:.0f}s\n")
phase_log.close()
print(f"\nrun_full_auto: {time.time()-t0:.0f}s")
print(f"PROJECT_ID={pid}")
