"""
Phase 0 characterization shim (parity tooling for the C# port).

Import this BEFORE running the Python pipeline to record, as golden fixtures,
every workflow JSON submitted to ComfyUI and every ffmpeg/ffprobe command line
executed. The C# WorkflowBuilder / Assembler must reproduce these byte-for-byte
(modulo temp paths + seeds).

Usage (from the repo root, on the ComfyUI embedded python that runs the app):
    python -c "import csharp.parity.characterize as c; c.install()"  # then run pipeline
or set AIDIR_CHARACTERIZE=1 and import it at app startup.

Fixtures land in csharp/parity/fixtures/:
    comfyui_submissions/<n>_<prompt_prefix>.json
    process_calls.jsonl        (argv of ffmpeg/ffprobe/subprocess runs)
"""
import json
import time
import functools
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
SUBMISSIONS = FIXTURES / "comfyui_submissions"
PROC_LOG = FIXTURES / "process_calls.jsonl"
_counter = [0]


def _ensure_dirs():
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)


def _record_submission(workflow: dict):
    _ensure_dirs()
    _counter[0] += 1
    # Name by the first text-bearing node so fixtures are recognizable.
    label = "wf"
    for node in workflow.values():
        t = (node.get("inputs", {}) or {}).get("text") or (node.get("inputs", {}) or {}).get("tags")
        if t:
            label = "".join(ch for ch in t[:24] if ch.isalnum() or ch in " _-").strip().replace(" ", "_")
            break
    path = SUBMISSIONS / f"{_counter[0]:03d}_{label}.json"
    path.write_text(json.dumps(workflow, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _record_process(argv):
    _ensure_dirs()
    with open(PROC_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "argv": [str(a) for a in argv]}) + "\n")


def install():
    """Monkeypatch ComfyUIClient.submit and subprocess.run/Popen to record."""
    _ensure_dirs()

    # 1. ComfyUI submissions.
    try:
        from app.services import comfyui_client as cc
        orig_submit = cc.ComfyUIClient.submit

        @functools.wraps(orig_submit)
        def submit(self, workflow):
            try:
                _record_submission(workflow)
            except Exception:
                pass
            return orig_submit(self, workflow)

        cc.ComfyUIClient.submit = submit
    except Exception as e:
        print(f"[characterize] could not patch ComfyUIClient.submit: {e}")

    # 2. ffmpeg / ffprobe / subprocess argv.
    import subprocess
    orig_run = subprocess.run
    orig_popen = subprocess.Popen

    @functools.wraps(orig_run)
    def run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args")
        if isinstance(cmd, (list, tuple)) and cmd and _is_media(cmd[0]):
            _record_process(cmd)
        return orig_run(*args, **kwargs)

    class Popen(orig_popen):
        def __init__(self, *args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            if isinstance(cmd, (list, tuple)) and cmd and _is_media(cmd[0]):
                _record_process(cmd)
            super().__init__(*args, **kwargs)

    subprocess.run = run
    subprocess.Popen = Popen
    print(f"[characterize] recording to {FIXTURES}")


def _is_media(exe) -> bool:
    e = str(exe).lower()
    return "ffmpeg" in e or "ffprobe" in e


if __name__ == "__main__":
    install()
    print("Characterization shim installed. Now run the pipeline in this process.")
