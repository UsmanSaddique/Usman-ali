"""
AI Director — Automation CLI
One command = one finished video, through the real app pipeline with QA gates.

Usage:
  python autoproduce.py --channel little-muslim-nation --title "Sabr ki Kahani" --duration 120
  python autoproduce.py --channel little-muslim-nation --title "20s Sample" --duration 20 --scenes 5 --wait

Designed for Windows Task Scheduler / overnight batches: exits 0 only when the
final render passed QA, non-zero otherwise, and prints the output path.
Requires the app server (run.py) and ComfyUI to be running — preflight will
tell you exactly what's missing if not.
"""
import argparse
import json
import sys
import time
import urllib.request
import urllib.error

API = "http://127.0.0.1:8000"


def call(method: str, path: str, body: dict = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"{API}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"API {e.code} on {path}: {detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"App server not reachable at {API} ({e.reason}). Start it with run.py.")


def main():
    ap = argparse.ArgumentParser(description="Produce one video hands-off via the AI Director pipeline")
    ap.add_argument("--channel", required=True, help="channel slug, e.g. little-muslim-nation")
    ap.add_argument("--title", required=True)
    ap.add_argument("--duration", type=int, default=60, help="target seconds")
    ap.add_argument("--scenes", type=int, default=None, help="target scene count (optional)")
    ap.add_argument("--context", default="", help="creative brief for the director LLM")
    ap.add_argument("--video-model", default=None)
    ap.add_argument("--wait", action="store_true", help="block until rendered/failed and set exit code")
    ap.add_argument("--poll-sec", type=int, default=30)
    args = ap.parse_args()

    # Preflight first so a bad night fails in seconds, with reasons.
    pf = call("GET", "/api/system/preflight")
    for c in pf["checks"]:
        mark = "OK " if c["ok"] else ("!! " if c["critical"] else "warn")
        print(f"  [{mark}] {c['name']}{': ' + c['detail'] if c['detail'] else ''}")
    if not pf["ok"]:
        raise SystemExit("Preflight failed — fix the checks above and rerun.")

    r = call("POST", "/api/automation/produce", {
        "channel_slug": args.channel,
        "title": args.title,
        "duration": args.duration,
        "num_scenes": args.scenes,
        "context": args.context,
        "video_model": args.video_model,
    })
    pid = r["id"]
    print(f"\nProject {pid} started: {r['title']}")

    if not args.wait:
        print(f"Poll: {API}{r['poll']}")
        return

    t0 = time.time()
    last = ""
    while True:
        time.sleep(args.poll_sec)
        p = call("GET", f"/api/projects/{pid}")
        status = p.get("status", "?")
        line = f"{status} scenes={p.get('completed_scenes', 0)}/{p.get('total_scenes', 0)}"
        if line != last:
            print(f"  [{(time.time()-t0)/60:5.1f}m] {line}")
            last = line
        if status == "rendered":
            print(f"\nDONE: {p.get('output_path')}")
            return
        if status == "failed":
            raise SystemExit(f"FAILED: {p.get('error_log', 'see server logs')}")


if __name__ == "__main__":
    main()
