"""
FL-SongGen: Generate complete Islamic nursery rhyme with REAL singing vocals.

Tencent LeVo model — generates actual singing voice + instruments.
Wheels on the Bus structure with Islamic lyrics.
"""
import json
import time
import uuid
import shutil
import urllib.request
from pathlib import Path

COMFYUI = "http://127.0.0.1:8188"
OUTPUT_DIR = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\output")
DEST = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\projects\3efaca79-b3ae-4c20-a6a7-9c294bb368d1")

# FL-SongGen lyrics format:
# - Sections separated by " ; "
# - Lines within sections separated by "."
# - Section tags: [intro-short], [verse], [chorus], [outro-short], [inst-short]
# - NO punctuation (commas, apostrophes, quotes get stripped)

LYRICS = (
    "[intro-short] ; "
    "[verse] "
    "The kids on the bus say Bismillah."
    "Bismillah Bismillah."
    "The kids on the bus say Bismillah."
    "All through the town"
    " ; "
    "[chorus] "
    "The wheels on the bus go round and round."
    "Round and round round and round."
    "The wheels on the bus go round and round."
    "All through the town"
    " ; "
    "[verse] "
    "The mommies on the bus say Alhamdulillah."
    "Alhamdulillah Alhamdulillah."
    "The mommies on the bus say Alhamdulillah."
    "All through the town"
    " ; "
    "[chorus] "
    "The wheels on the bus go round and round."
    "Round and round round and round."
    "The wheels on the bus go round and round."
    "All through the town"
    " ; "
    "[verse] "
    "The babies on the bus say Masha Allah."
    "Masha Allah Masha Allah."
    "The babies on the bus say Masha Allah."
    "All through the town"
    " ; "
    "[verse] "
    "The friends on the bus say Assalamu Alaikum."
    "Alaikum Alaikum."
    "The friends on the bus say Assalamu Alaikum."
    "All through the town"
    " ; "
    "[chorus] "
    "The wheels on the bus go round and round."
    "Round and round round and round."
    "The wheels on the bus go round and round."
    "All through the town"
    " ; "
    "[outro-short]"
)

# CoComelon-style description
DESCRIPTION = "female, bright, pop, happy, piano and ukulele and xylophone and bells, the bpm is 120"


def submit(workflow):
    client_id = uuid.uuid4().hex[:12]
    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{COMFYUI}/prompt", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    pid = result.get("prompt_id")
    if not pid:
        raise RuntimeError(f"Rejected: {result}")
    return pid


def wait(prompt_id, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            url = f"{COMFYUI}/history/{prompt_id}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            hist = data.get(prompt_id)
            if hist:
                status = hist.get("status", {})
                if status.get("completed"):
                    return hist
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"Error: {msgs}")
        except (urllib.error.URLError, TimeoutError):
            pass
        elapsed = int(time.time() - t0)
        if elapsed > 0 and elapsed % 30 == 0:
            print(f"    ... {elapsed}s elapsed")
        time.sleep(5)
    raise TimeoutError(f"Timed out after {timeout}s")


def collect_all(history, prefix):
    """Collect all audio outputs (mixed, vocal, bgm)."""
    outputs = history.get("outputs", {})
    saved = []
    for node_id, node_out in sorted(outputs.items()):
        for key in ("audio",):
            for i, entry in enumerate(node_out.get(key, [])):
                fname = entry.get("filename", "")
                subfolder = entry.get("subfolder", "")
                if not fname:
                    continue
                src = OUTPUT_DIR / subfolder / fname if subfolder else OUTPUT_DIR / fname
                if src.exists():
                    dest = str(DEST / f"{prefix}_{fname}")
                    shutil.copy2(str(src), dest)
                    print(f"    Saved: {dest}")
                    saved.append(dest)
    return saved


def free_vram():
    try:
        req = urllib.request.Request(
            f"{COMFYUI}/free", data=json.dumps({"unload_models": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def generate_song(lyrics, description, duration, seed, prefix):
    """Generate a complete song using FL-SongGen."""
    print(f"\n  Generating: {prefix}")
    print(f"  Description: {description}")
    print(f"  Duration: {duration}s, Seed: {seed}")

    free_vram()

    workflow = {
        # Node 1: Model Loader
        "1": {
            "class_type": "FL_SongGen_ModelLoader",
            "inputs": {
                "model_variant": "base",
                "memory_mode": "auto",
                "keep_loaded": True,
            }
        },
        # Node 2: Generate Song
        "2": {
            "class_type": "FL_SongGen_Generate",
            "inputs": {
                "model": ["1", 0],
                "lyrics": lyrics,
                "description": description,
                "duration": duration,
                "temperature": 0.9,
                "cfg_coef": 1.5,
                "top_k": 50,
                "gen_type": "mixed",
                "seed": seed,
            }
        },
        # Node 3: Save mixed audio
        "3": {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["2", 0],
                "filename_prefix": prefix,
            }
        },
    }

    pid = submit(workflow)
    print(f"  Submitted: {pid}")
    print(f"  Waiting (first run downloads ~5GB model)...")
    hist = wait(pid, timeout=3600)
    files = collect_all(hist, prefix)
    return files


if __name__ == "__main__":
    print("=" * 60)
    print("FL-SONGGEN: Islamic Wheels on the Bus")
    print("Complete song with REAL singing vocals")
    print("=" * 60)

    print("\nWaiting for ComfyUI...")
    for i in range(30):
        try:
            with urllib.request.urlopen(f"{COMFYUI}/system_stats", timeout=3):
                break
        except Exception:
            time.sleep(2)
    else:
        print("ComfyUI not available!")
        exit(1)
    print("ComfyUI ready!\n")

    # Generate the song
    print("Generating Islamic nursery rhyme with singing vocals...")
    print("(First run will download the model - ~5GB - please be patient)\n")

    files = generate_song(
        lyrics=LYRICS,
        description=DESCRIPTION,
        duration=120.0,
        seed=42,
        prefix="songgen_islamic",
    )

    if files:
        print(f"\nDONE! Generated {len(files)} audio file(s):")
        for f in files:
            print(f"  {f}")

        # Copy best to music.wav
        if files:
            best = files[0]
            music_wav = str(DEST / "music.wav")
            shutil.copy2(best, music_wav)
            print(f"\nCopied to music.wav for video render")
    else:
        print("\nFAILED - no audio output")
