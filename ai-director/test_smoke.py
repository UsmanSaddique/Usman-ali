"""
AI Director — Smoke Test
Validates wiring without needing GPU model servers (ComfyUI/TTS/ACE-Step).

What it checks:
  1. All service modules import cleanly (catches stale refs like ltx_python)
  2. Config loads and key paths resolve
  3. VideoGenService.ken_burns exists with the signature pipeline.py expects
  4. PipelineOrchestrator constructs (all services wire together)
  5. FastAPI app imports and the documented routes are registered
  6. ffmpeg is reachable, AND a real Ken Burns render works (CPU-only)

Run:  C:\\...\\python_embeded\\python.exe test_smoke.py
Exit code 0 = all critical checks passed.
"""
import sys
import os
import inspect
import tempfile
import subprocess
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def record(name, status, detail=""):
    results.append((name, status, detail))
    icon = {"PASS": "[OK]", "FAIL": "[XX]", "WARN": "[!!]"}[status]
    print(f"{icon} {name}" + (f" — {detail}" if detail else ""))


def check(name, fn, critical=True):
    try:
        detail = fn() or ""
        record(name, PASS, detail)
        return True
    except Exception as e:
        record(name, FAIL if critical else WARN, f"{type(e).__name__}: {e}")
        if critical:
            traceback.print_exc()
        return False


# ── 1. Imports ───────────────────────────────────────────────────────────────
def t_imports():
    import app.config            # noqa
    import app.database          # noqa
    import app.services.model_manager   # noqa
    import app.services.director         # noqa
    import app.services.image_gen        # noqa
    import app.services.video_gen        # noqa
    import app.services.upscaler         # noqa
    import app.services.assembler        # noqa
    import app.services.music_gen        # noqa
    import app.services.tts              # noqa
    import app.services.comfyui_client   # noqa
    import app.services.pipeline         # noqa
    return "all service modules import"


# ── 2. Config ─────────────────────────────────────────────────────────────────
def t_config():
    from app.config import settings
    assert settings.port == 8000
    assert settings.video.model_path is not None
    return f"port={settings.port}, video={settings.video.model_path.name}"


# ── 3. ken_burns signature ──────────────────────────────────────────────────
def t_ken_burns_sig():
    from app.services.video_gen import VideoGenService
    assert hasattr(VideoGenService, "ken_burns"), "ken_burns missing"
    sig = inspect.signature(VideoGenService.ken_burns)
    params = list(sig.parameters)
    assert params[0] == "self", f"first param should be self, got {params[0]}"
    # pipeline.py calls these kwargs:
    for needed in ("image_path", "output_path", "duration", "fps",
                   "zoom_start", "zoom_end", "pan_x", "pan_y", "ffmpeg_bin"):
        assert needed in params, f"ken_burns missing kwarg: {needed}"
    return f"signature OK ({len(params)} params)"


# ── 4. Pipeline construction ──────────────────────────────────────────────────
def t_pipeline():
    from app.config import settings
    from app.services.model_manager import ModelManager
    from app.services.pipeline import PipelineOrchestrator
    mm = ModelManager()
    orch = PipelineOrchestrator(settings, mm)
    assert orch.video_gen is not None
    assert hasattr(orch.video_gen, "ken_burns")
    return "PipelineOrchestrator + all services constructed"


# ── 5. FastAPI routes ─────────────────────────────────────────────────────────
def t_routes():
    from app.main import app
    paths = {r.path for r in app.routes}
    expected = [
        "/api/projects",
        "/api/system/engine-status",
    ]
    missing = [p for p in expected if p not in paths]
    assert not missing, f"missing routes: {missing}"
    return f"{len(paths)} routes registered"


# ── 6. ffmpeg + real Ken Burns render (CPU only) ──────────────────────────────
def t_ffmpeg_available():
    from app.config import settings
    ff = settings.paths.ffmpeg_bin
    out = subprocess.run([ff, "-version"], capture_output=True, text=True, timeout=20)
    assert out.returncode == 0
    return out.stdout.splitlines()[0]


def t_ken_burns_render():
    from app.config import settings
    from app.services.model_manager import ModelManager
    from app.services.video_gen import VideoGenService

    ff = settings.paths.ffmpeg_bin
    tmp = Path(tempfile.mkdtemp(prefix="kbtest_"))
    img = tmp / "src.png"
    vid = tmp / "out.mp4"

    # make a 1280x720 test image with ffmpeg's testsrc
    gen = subprocess.run(
        [ff, "-y", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=1",
         "-frames:v", "1", str(img)],
        capture_output=True, text=True, timeout=30,
    )
    assert gen.returncode == 0 and img.exists(), "failed to make test image"

    svc = VideoGenService(ModelManager(), settings)
    res = svc.ken_burns(
        image_path=str(img), output_path=str(vid),
        duration=2.0, fps=24, zoom_start=1.0, zoom_end=1.15,
        pan_x=0.1, pan_y=0.0, ffmpeg_bin=ff,
    )
    assert Path(res.path).exists(), "output video not created"
    size_kb = Path(res.path).stat().st_size / 1024
    assert size_kb > 1, "output video is empty"
    return f"rendered {size_kb:.0f}KB clip in {res.generation_time:.2f}s"


if __name__ == "__main__":
    print("=" * 60)
    print("AI Director — Smoke Test")
    print("=" * 60)

    check("1. service imports", t_imports)
    check("2. config loads", t_config)
    check("3. ken_burns signature", t_ken_burns_sig)
    check("4. pipeline constructs", t_pipeline)
    check("5. fastapi routes", t_routes)
    ff_ok = check("6a. ffmpeg available", t_ffmpeg_available, critical=False)
    if ff_ok:
        check("6b. ken_burns real render", t_ken_burns_render, critical=False)
    else:
        record("6b. ken_burns real render", WARN, "skipped (no ffmpeg)")

    print("=" * 60)
    crit_fail = [r for r in results if r[1] == FAIL]
    n_pass = sum(1 for r in results if r[1] == PASS)
    n_warn = sum(1 for r in results if r[1] == WARN)
    print(f"{n_pass} passed, {len(crit_fail)} failed, {n_warn} warnings")
    print("=" * 60)
    sys.exit(1 if crit_fail else 0)
