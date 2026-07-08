"""Submit ACE-Step v1 song generation and wait."""
import sys
import time
import json
import uuid
import shutil
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

COMFYUI = "http://127.0.0.1:8188"
OUTPUT_DIR = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\output")
DEST = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\projects\song_merge_output")
DEST.mkdir(parents=True, exist_ok=True)

lyrics = (
    "[verse]\n"
    "The kids on the bus say Bismillah\n"
    "Bismillah Bismillah\n"
    "The kids on the bus say Bismillah\n"
    "All through the town\n"
    "\n"
    "[chorus]\n"
    "The wheels on the bus go round and round\n"
    "Round and round round and round\n"
    "The wheels on the bus go round and round\n"
    "All through the town\n"
    "\n"
    "[verse]\n"
    "The mommies on the bus say Alhamdulillah\n"
    "Alhamdulillah Alhamdulillah\n"
    "The mommies on the bus say Alhamdulillah\n"
    "All through the town\n"
    "\n"
    "[chorus]\n"
    "The wheels on the bus go round and round\n"
    "Round and round round and round\n"
    "The wheels on the bus go round and round\n"
    "All through the town\n"
    "\n"
    "[verse]\n"
    "The babies on the bus say Masha Allah\n"
    "Masha Allah Masha Allah\n"
    "The babies on the bus say Masha Allah\n"
    "All through the town\n"
    "\n"
    "[verse]\n"
    "The friends on the bus say Assalamu Alaikum\n"
    "Alaikum Alaikum\n"
    "The friends on the bus say Assalamu Alaikum\n"
    "All through the town\n"
    "\n"
    "[chorus]\n"
    "The wheels on the bus go round and round\n"
    "Round and round round and round\n"
    "The wheels on the bus go round and round\n"
    "All through the town"
)

tags = "pop, female vocal, happy, bright, children's music, piano, ukulele, xylophone, bells"

# Wait for queue to be empty
print("Waiting for ComfyUI queue to clear...", flush=True)
for i in range(120):
    try:
        r = urllib.request.urlopen(f"{COMFYUI}/queue", timeout=5)
        q = json.loads(r.read())
        running = len(q.get("queue_running", []))
        pending = len(q.get("queue_pending", []))
        if running == 0 and pending == 0:
            print("Queue clear!", flush=True)
            break
        if i % 12 == 0:
            print(f"  Queue: {running} running, {pending} pending... ({i*5}s)", flush=True)
    except Exception:
        pass
    time.sleep(5)
else:
    print("Queue never cleared — submitting anyway", flush=True)

# Free VRAM
try:
    req = urllib.request.Request(
        f"{COMFYUI}/free",
        data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=10)
    print("VRAM freed", flush=True)
    time.sleep(3)
except Exception:
    pass

# Submit ACE-Step v1 workflow
wf = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
        "ckpt_name": "ace_step_v1_3.5b.safetensors"}},
    "2": {"class_type": "TextEncodeAceStepAudio", "inputs": {
        "clip": ["1", 1], "tags": tags, "lyrics": lyrics,
        "lyrics_strength": 1.0}},
    "3": {"class_type": "TextEncodeAceStepAudio", "inputs": {
        "clip": ["1", 1], "tags": "", "lyrics": "", "lyrics_strength": 1.0}},
    "4": {"class_type": "EmptyAceStepLatentAudio", "inputs": {
        "seconds": 120.0, "batch_size": 1}},
    "5": {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
        "latent_image": ["4", 0], "seed": 42, "steps": 60, "cfg": 5.0,
        "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
    "6": {"class_type": "VAEDecodeAudio", "inputs": {
        "samples": ["5", 0], "vae": ["1", 2]}},
    "7": {"class_type": "SaveAudio", "inputs": {
        "audio": ["6", 0], "filename_prefix": "song_merge_final"}},
}

print(f"\nSubmitting ACE-Step v1 (120s song, real singing)...", flush=True)
cid = uuid.uuid4().hex[:12]
payload = json.dumps({"prompt": wf, "client_id": cid}).encode()
req = urllib.request.Request(
    f"{COMFYUI}/prompt", data=payload,
    headers={"Content-Type": "application/json"},
)
try:
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read())
    pid = result.get("prompt_id")
    if not pid:
        print(f"REJECTED: {result}", flush=True)
        sys.exit(1)
    print(f"Submitted: {pid}", flush=True)
except urllib.error.HTTPError as e:
    body = e.read().decode()[:500]
    print(f"HTTP {e.code}: {body}", flush=True)
    sys.exit(1)

print("Generating with ACE-Step v1 (real AI singing, ~5-10 min)...", flush=True)
t0 = time.time()
while time.time() - t0 < 900:  # 15 min timeout
    try:
        resp = urllib.request.urlopen(f"{COMFYUI}/history/{pid}", timeout=10)
        data = json.loads(resp.read())
        hist = data.get(pid)
        if hist:
            status = hist.get("status", {})
            if status.get("completed"):
                elapsed = time.time() - t0
                print(f"\nCOMPLETED in {elapsed:.0f}s!", flush=True)

                outputs = hist.get("outputs", {})
                for nid, nout in outputs.items():
                    for entry in nout.get("audio", []):
                        fname = entry.get("filename", "")
                        subfolder = entry.get("subfolder", "")
                        src = OUTPUT_DIR / subfolder / fname if subfolder else OUTPUT_DIR / fname
                        if src.exists():
                            dest_wav = str(DEST / "song_merged.wav")
                            shutil.copy2(str(src), dest_wav)
                            size_mb = src.stat().st_size / 1024 / 1024
                            print(f"WAV: {dest_wav} ({size_mb:.1f}MB)", flush=True)

                            # Master it
                            mastered = str(DEST / "song_mastered.wav")
                            dur_probe = subprocess.run(
                                ["ffprobe", "-v", "quiet", "-print_format", "json",
                                 "-show_format", dest_wav],
                                capture_output=True, text=True
                            )
                            try:
                                dur = float(json.loads(dur_probe.stdout)["format"]["duration"])
                            except Exception:
                                dur = 120.0
                            fade_out = max(0, dur - 3)
                            subprocess.run([
                                "ffmpeg", "-y", "-i", dest_wav,
                                "-af", (
                                    f"highpass=f=60,"
                                    f"equalizer=f=3500:t=h:w=3000:g=2,"
                                    f"equalizer=f=12000:t=h:w=4000:g=1.5,"
                                    f"alimiter=limit=0.95:attack=5:release=50,"
                                    f"afade=t=in:st=0:d=0.5,afade=t=out:st={fade_out}:d=3"
                                ),
                                "-ar", "44100", mastered
                            ], capture_output=True, text=True)

                            # MP3
                            mp3 = str(DEST / "song_merged.mp3")
                            subprocess.run([
                                "ffmpeg", "-y", "-i", mastered,
                                "-codec:a", "libmp3lame", "-b:a", "320k", mp3
                            ], capture_output=True, text=True)

                            if Path(mp3).exists():
                                mp3_size = Path(mp3).stat().st_size / 1024 / 1024
                                print(f"MP3: {mp3} ({mp3_size:.1f}MB)", flush=True)

                            print(f"\nDONE! Real AI singing generated.", flush=True)
                            sys.exit(0)

                print("No audio output found", flush=True)
                sys.exit(1)

            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                for m in msgs:
                    if isinstance(m, list) and m[0] == "execution_error":
                        err = m[1]
                        print(f"ERROR: {err.get('node_type','?')}: {err.get('exception_message','?')[:300]}", flush=True)
                sys.exit(1)
    except (urllib.error.URLError, TimeoutError):
        pass

    elapsed = int(time.time() - t0)
    if elapsed % 60 == 0 and elapsed > 0:
        print(f"... {elapsed // 60}min elapsed", flush=True)
    time.sleep(5)

print("TIMEOUT after 15min", flush=True)
