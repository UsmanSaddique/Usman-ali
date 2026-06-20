"""
AI Director — FULL END-TO-END QA TEST
Produces a ~40s video: 5 clips @ 720p base, upscaled to 1080p, assembled with a
custom ("bring your own") audio track muxed in.

Acts like QA: runs each pipeline phase, times it, and asserts on the real output.

Stages:
  1. Create a fresh project (channel: urdu-moral-stories, 40s, 5 scenes)
  2. Script generation  (elite director; checks Roman-Urdu + SEO + GPU speed)
  3. Asset generation   (5 clips @ 1280x720)
  4. Upscale            (-> 1080p)
  5. Custom audio       (synthesize a gentle track = "your own song")
  6. Render/assemble    (mux clips + custom audio)
  7. QA assertions on final_render.mp4 (exists, ~40s, 1080p, has audio)

Run (ComfyUI must be running):
  PYTHONIOENCODING=utf-8 python_embeded\\python.exe test_e2e.py
"""
import sys, os, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.config import settings
from app.database import (
    get_session, Channel, Project, Scene, ProjectStatus, SceneStatus,
)
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.pipeline import PipelineOrchestrator

CHANNEL_SLUG = "urdu-moral-stories"
DURATION = 40
NUM_SCENES = 5
BASE_W, BASE_H = 1280, 720

report = []
def stage(name, ok, detail=""):
    report.append((name, ok, detail))
    print(f"{'[OK]' if ok else '[XX]'} {name}" + (f" — {detail}" if detail else ""))

def ffprobe_video(path):
    """Return (duration_s, width, height, has_audio) via ffmpeg -i parsing."""
    ff = settings.paths.ffmpeg_bin
    out = subprocess.run([ff, "-i", path], capture_output=True, text=True)
    txt = out.stderr
    dur, w, h, has_audio = 0.0, 0, 0, False
    for line in txt.splitlines():
        if "Duration:" in line:
            t = line.split("Duration:")[1].split(",")[0].strip()
            hh, mm, ss = t.split(":")
            dur = int(hh) * 3600 + int(mm) * 60 + float(ss)
        if "Video:" in line and "x" in line:
            import re
            m = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if m:
                w, h = int(m.group(1)), int(m.group(2))
        if "Audio:" in line:
            has_audio = True
    return dur, w, h, has_audio


def main():
    print("=" * 66)
    print("AI DIRECTOR — FULL E2E QA TEST")
    print("=" * 66)

    mm = ModelManager()
    register_all_loaders(mm, settings)
    orch = PipelineOrchestrator(settings, mm)

    # ── Stage 1: project ──────────────────────────────────────────────
    session = get_session()
    ch = session.query(Channel).filter(Channel.slug == CHANNEL_SLUG).first()
    if not ch:
        stage("1. project setup", False, f"channel '{CHANNEL_SLUG}' not in DB — run seed_channels.py")
        return
    proj = Project(
        title="The kind little rabbit who helped a lost duckling",
        channel_id=ch.id,
        duration_target=DURATION,
        num_scenes_target=NUM_SCENES,
        context="A warm 40-second moral story for young kids with a clear lesson about kindness.",
        status=ProjectStatus.DRAFT,
        video_model="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
    )
    session.add(proj)
    session.commit()
    pid = proj.id
    session.close()
    stage("1. project setup", True, f"project {pid[:8]} on {CHANNEL_SLUG}")

    # ── Stage 2: script ───────────────────────────────────────────────
    t0 = time.time()
    try:
        script = orch.generate_script(pid)
        dt = time.time() - t0
        narr = " ".join(s.narration_text for s in script.scenes)
        roman_ok = all(ord(c) < 256 for c in narr) and len(narr.strip()) > 0  # Roman Urdu = ASCII
        seo_ok = bool(script.description) and len(script.tags) >= 5
        stage("2. script generation", len(script.scenes) > 0,
              f"{len(script.scenes)} scenes in {dt:.0f}s "
              f"({len(script.scenes)*1.0:.0f} scenes)")
        stage("2a. Roman-Urdu narration (no Urdu script)", roman_ok,
              f"sample: {narr[:60]!r}")
        stage("2b. SEO present (desc + >=5 tags)", seo_ok,
              f"{len(script.tags)} tags, desc {len(script.description)} chars")
        # GPU speed heuristic: tuned full-GPU fit should be well under 3 min
        stage("2c. script speed (GPU fit)", dt < 180,
              f"{dt:.0f}s ({'fast — likely full GPU' if dt < 120 else 'check offload' if dt < 180 else 'SLOW — still CPU-spilling'})")
    except Exception as e:
        stage("2. script generation", False, f"{type(e).__name__}: {e}")
        return

    # auto-approve all scenes
    session = get_session()
    for s in session.query(Scene).filter(Scene.project_id == pid).all():
        s.status = SceneStatus.PENDING
    session.commit(); session.close()

    # ── Stage 3: generation @ 720p ────────────────────────────────────
    t0 = time.time()
    try:
        orch.start_generation(pid, width=BASE_W, height=BASE_H)
        dt = time.time() - t0
        session = get_session()
        gen_ok = session.query(Scene).filter(
            Scene.project_id == pid, Scene.status == SceneStatus.GENERATED).count()
        total = session.query(Scene).filter(Scene.project_id == pid).count()
        session.close()
        stage("3. clip generation @720p", gen_ok > 0,
              f"{gen_ok}/{total} scenes generated in {dt:.0f}s")
    except Exception as e:
        stage("3. clip generation @720p", False, f"{type(e).__name__}: {e}")

    # ── Stage 4: upscale -> 1080p ─────────────────────────────────────
    t0 = time.time()
    try:
        orch.start_upscale(pid)
        stage("4. upscale -> 1080p", True, f"{time.time()-t0:.0f}s")
    except Exception as e:
        stage("4. upscale -> 1080p", False, f"{type(e).__name__}: {e}")

    # ── Stage 5: custom "your own song" ───────────────────────────────
    proj_dir = settings.paths.projects_dir / pid
    proj_dir.mkdir(parents=True, exist_ok=True)
    my_song = str(proj_dir / "my_custom_song.wav")
    try:
        ff = settings.paths.ffmpeg_bin
        subprocess.run([ff, "-y", "-f", "lavfi",
                        "-i", f"sine=frequency=330:duration={DURATION+5}",
                        "-af", "volume=0.2", my_song],
                       capture_output=True, text=True, timeout=60, check=True)
        stage("5. custom audio (BYO song)", os.path.exists(my_song), my_song)
    except Exception as e:
        my_song = None
        stage("5. custom audio (BYO song)", False, f"{type(e).__name__}: {e}")

    # ── Stage 6: render ───────────────────────────────────────────────
    t0 = time.time()
    try:
        orch.render(pid, music_path=my_song)
        stage("6. render/assemble", True, f"{time.time()-t0:.0f}s")
    except Exception as e:
        stage("6. render/assemble", False, f"{type(e).__name__}: {e}")

    # ── Stage 7: QA the final file ────────────────────────────────────
    final = str(proj_dir / "final_render.mp4")
    if os.path.exists(final):
        dur, w, h, has_audio = ffprobe_video(final)
        size_mb = os.path.getsize(final) / (1024*1024)
        stage("7. final exists", True, f"{size_mb:.1f}MB")
        stage("7a. duration ~40s", 25 <= dur <= 55, f"{dur:.1f}s")
        stage("7b. resolution 1080p", h >= 1000, f"{w}x{h}")
        stage("7c. has audio (custom song muxed)", has_audio, "audio stream present" if has_audio else "NO audio")
    else:
        stage("7. final exists", False, f"missing: {final}")

    # ── summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    passed = sum(1 for _, ok, _ in report if ok)
    failed = [n for n, ok, _ in report if not ok]
    print(f"QA RESULT: {passed}/{len(report)} checks passed")
    if failed:
        print("FAILED:", ", ".join(failed))
    print(f"Final video: {final}")
    print("=" * 66)


if __name__ == "__main__":
    main()
