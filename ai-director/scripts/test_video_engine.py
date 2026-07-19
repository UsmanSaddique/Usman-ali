"""QA: per-project video_engine config + LTX Director multi-director workflow.

Offline-ish checks (no GPU run):
  1. DB column + pipeline routing helper
  2. build_workflow: segments split across the two LTXDirector nodes,
     each segment carries its reference image, lengths/frames consistent
Needs: app server DB reachable, ComfyUI on :8188 (for /object_info).
Run:  python scripts/test_video_engine.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

FAILS = []

def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)

# ── 1. DB + routing ────────────────────────────────────────────────────────
print("1) DB column + engine routing")
from app.database import init_db, get_session, Project
init_db()
s = get_session()
p = s.query(Project).filter(Project.title == "ENGINE QA TEST").first()
check("video_engine column readable", p is not None and hasattr(p, "video_engine"))
if p:
    p.video_engine = "ltx_director"
    s.commit()
    pid = p.id
s.close()

from app.config import settings
from app.services.pipeline import PipelineOrchestrator as Pipeline
import inspect
sig_ok = "_project_video_engine" in dir(Pipeline) and "_ensure_scene_stills" in dir(Pipeline)
check("pipeline has engine helpers", sig_ok)
src = inspect.getsource(Pipeline._run_full_auto_impl)
check("full-auto routes on ltx_director", "generate_ltx_director" in src and "_ensure_scene_stills" in src)
src2 = inspect.getsource(Pipeline._run_full_auto_narration_impl)
check("narration full-auto routes too", "generate_ltx_director" in src2)

# _project_video_engine without instantiating the heavy Pipeline: call unbound
class _Stub:
    pass
stub = _Stub()
val = Pipeline._project_video_engine(stub, pid) if p else None
check("_project_video_engine returns ltx_director", val == "ltx_director", str(val))

# ── 2. workflow build ──────────────────────────────────────────────────────
print("2) LTX Director multi-director workflow build")
from app.services.ltx_director import LTXDirectorService

svc = LTXDirectorService(model_manager=None, config=settings)
N = 10
segs = [{"prompt": f"scene {i} prompt", "dialogue": f'The narrator says: "line {i}"',
         "image_path": f"C:/fake/scene_{i:03d}_v1.png", "seconds": 5.0}
        for i in range(N)]
api = svc.build_workflow(segs, fps=24)
directors = [(nid, n) for nid, n in api.items() if n["class_type"] == "LTXDirector"]
check("2 LTXDirector nodes in API prompt", len(directors) == 2, f"got {len(directors)}")

tot_segs, all_imgs = 0, []
for nid, n in directors:
    tl = json.loads(n["inputs"]["timeline_data"])
    segs_tl = tl["segments"]
    tot_segs += len(segs_tl)
    frames = [int(x["length"]) for x in segs_tl]
    lengths_csv = [int(x) for x in n["inputs"]["segment_lengths"].split(",")]
    check(f"node {nid}: lengths csv matches timeline", frames == lengths_csv)
    check(f"node {nid}: duration_frames = sum(segments)",
          int(n["inputs"]["duration_frames"]) == sum(frames))
    imgs = [x.get("imageFile") for x in segs_tl]
    check(f"node {nid}: every segment has an image", all(imgs),
          f"{len(segs_tl)} segs, ~{sum(frames)//24}s")
    all_imgs += imgs
check("segments split across directors (5+5)", tot_segs == N, f"got {tot_segs}")
check("all 10 reference images attached, in order",
      all_imgs == [f"scene_{i:03d}_v1.png" for i in range(N)])

# per-director duration in the 20-30s band for a typical 10x5s project
per_dir = [sum(int(x["length"]) for x in json.loads(n["inputs"]["timeline_data"])["segments"]) / 24
           for _, n in directors]
check("each director ~20-30s", all(20 <= d <= 30 for d in per_dir), str(per_dir))

# odd count: 7 segments -> 4+3
api2 = svc.build_workflow(segs[:7], fps=24)
d2 = [n for n in api2.values() if n["class_type"] == "LTXDirector"]
counts = sorted(len(json.loads(n["inputs"]["timeline_data"])["segments"]) for n in d2)
check("odd split 7 -> 3+4", counts == [3, 4], str(counts))

# ── 3. proven config (usman's 720p / 25min recipe) ─────────────────────────
print("3) proven-config parity (usman created ltx.json)")
segs6 = [{"prompt": f"scene {i}", "dialogue": f'The girl says: "line {i}"',
          "image_path": f"C:/fake/{i}.png", "seconds": 6.0} for i in range(10)]
api3 = svc.build_workflow(segs6, fps=24)
d3 = [n for n in api3.values() if n["class_type"] == "LTXDirector"]
for n in d3:
    inp = n["inputs"]
    nseg = len(json.loads(inp["timeline_data"])["segments"])
    check("guide_strength has one entry per segment",
          len(inp["guide_strength"].split(",")) == nseg,
          f"{inp['guide_strength']} for {nseg} segs")
    check("epsilon preserved (0.001)", float(inp["epsilon"]) == 0.001)
    check("resolution preserved 1280x720",
          inp["custom_width"] == 1280 and inp["custom_height"] == 720)
    check("resize crop / divisible 32 / compression 18 preserved",
          inp["resize_method"] == "crop" and inp["divisible_by"] == 32
          and inp["img_compression"] == 18)
    check("30s per director (720 frames @24fps)",
          int(inp["duration_frames"]) == 720 and int(inp["duration_seconds"]) == 30,
          f"{inp['duration_frames']}f/{inp['duration_seconds']}s")
# >12 scenes must be refused with a clear error
try:
    svc.build_workflow(segs6 + segs6[:3], fps=24)
    check(">12 segments rejected", False, "no error raised")
except ValueError as e:
    check(">12 segments rejected", "at most 12" in str(e))

# single-director mode: <=6 segments -> ONE director node, joiners gone
api4 = svc.build_workflow([{"prompt": f"s{i}", "image_path": f"C:/f/{i}.png",
                            "seconds": 5.0} for i in range(6)], fps=24)
d4 = [n for n in api4.values() if n["class_type"] == "LTXDirector"]
check("<=6 segs -> exactly 1 LTXDirector node", len(d4) == 1, f"got {len(d4)}")
joiners = [n for n in api4.values()
           if n["class_type"] in ("ImageBatchMulti", "AudioConcatenate")]
check("joiners bypassed in single-director mode", not joiners)
if d4:
    i4 = d4[0]["inputs"]
    n4 = len(json.loads(i4["timeline_data"])["segments"])
    check("single director holds all 6 segments (30s)",
          n4 == 6 and int(i4["duration_frames"]) == 720,
          f"{n4} segs {i4['duration_frames']}f")
    check("single-director guides = 6", len(i4["guide_strength"].split(",")) == 6)
vhs = [n for n in api4.values() if n["class_type"] == "VHS_VideoCombine"]
check("VHS combine still wired (images+audio links)",
      bool(vhs) and isinstance(vhs[0]["inputs"].get("images"), list)
      and isinstance(vhs[0]["inputs"].get("audio"), list))
# global style prompt from template survives when style_prompt=""
check("template global_prompt kept when no style override",
      d3[0]["inputs"]["global_prompt"].startswith("masterpiece, cinematic"))

print()
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
