"""Generate higher quality song with more steps and different seeds."""
import json
import time
import uuid
import shutil
import urllib.request
from pathlib import Path

COMFYUI = "http://127.0.0.1:8188"
OUTPUT_DIR = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\output")
DEST = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\projects\3efaca79-b3ae-4c20-a6a7-9c294bb368d1")

LYRICS = """[Verse 1]
Bismillah bismillah, we start our day,
Thank you Allah, for letting us play,
The sun is shining, the birds all sing,
Alhamdulillah for everything!

[Chorus]
Be kind, be kind, be kind today,
Share and care in every way,
Say Salam with a big bright smile,
Being good is always worthwhile!

[Verse 2]
Yusuf helps his friends at school,
Being gentle is so cool,
Amina shares her toys with love,
Thanking Allah up above!

[Chorus]
Be kind, be kind, be kind today,
Share and care in every way,
Say Salam with a big bright smile,
Being good is always worthwhile!

[Bridge]
SubhanAllah, the flowers grow,
MashaAllah, the rivers flow,
Allah made this world so bright,
Full of colors, full of light!

[Outro]
Be kind, be kind, be kind today,
Share and care in every way,
Alhamdulillah, alhamdulillah,
Thank you, thank you, dear Allah!
"""

TAGS = "children's music, nursery rhyme, acoustic, female vocal, happy, uplifting, playful, piano, ukulele, xylophone, bells, clapping, warm, gentle, sing-along, kids song, catchy melody"


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


def wait(prompt_id, timeout=1800):
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
                    raise RuntimeError(f"Error: {status.get('messages', [])}")
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(3)
    raise TimeoutError(f"Timed out after {timeout}s")


def collect(history, dest_path):
    outputs = history.get("outputs", {})
    for node_id, node_out in outputs.items():
        for key in ("audio", "gifs", "images", "videos"):
            for entry in node_out.get(key, []):
                fname = entry.get("filename", "")
                subfolder = entry.get("subfolder", "")
                if not fname:
                    continue
                src = OUTPUT_DIR / subfolder / fname if subfolder else OUTPUT_DIR / fname
                if src.exists():
                    shutil.copy2(str(src), dest_path)
                    print(f"  Saved: {dest_path}")
                    return
    raise RuntimeError("No audio output found")


def free_vram():
    try:
        req = urllib.request.Request(
            f"{COMFYUI}/free", data=json.dumps({"unload_models": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def generate(seed, steps, cfg, prefix, suffix=""):
    """Generate a song with ACE-Step v1.0."""
    free_vram()
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
            "ckpt_name": "ace_step_v1_3.5b.safetensors"}},
        "2": {"class_type": "TextEncodeAceStepAudio", "inputs": {
            "clip": ["1", 1],
            "tags": TAGS + (f", {suffix}" if suffix else ""),
            "lyrics": LYRICS,
            "lyrics_strength": 1.0,
        }},
        "3": {"class_type": "EmptyAceStepLatentAudio", "inputs": {
            "seconds": 130.0, "batch_size": 1}},
        "4": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["2", 0],
            "latent_image": ["3", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "5": {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["4", 0], "vae": ["1", 2]}},
        "6": {"class_type": "SaveAudio", "inputs": {
            "audio": ["5", 0], "filename_prefix": prefix}},
    }
    pid = submit(wf)
    print(f"  Submitted {prefix} (seed={seed}, steps={steps}, cfg={cfg})")
    hist = wait(pid)
    dest = str(DEST / f"{prefix}.wav")
    collect(hist, dest)
    return dest


if __name__ == "__main__":
    print("Waiting for ComfyUI...")
    for i in range(30):
        try:
            with urllib.request.urlopen(f"{COMFYUI}/system_stats", timeout=3):
                break
        except Exception:
            time.sleep(2)

    print("Generating 3 song variations...\n")

    # Variation 1: More steps for better quality
    print("[1/3] High quality - 100 steps, cfg=5.0")
    p1 = generate(seed=42, steps=100, cfg=5.0, prefix="song_hq")

    # Variation 2: Different seed, pop-leaning
    print("[2/3] Pop style - seed 777")
    p2 = generate(seed=777, steps=80, cfg=4.0, prefix="song_pop", suffix="pop, catchy chorus, bright")

    # Variation 3: Softer nasheed style
    print("[3/3] Nasheed style - gentle")
    p3 = generate(seed=2024, steps=80, cfg=4.0, prefix="song_nasheed", suffix="nasheed, middle eastern, peaceful, reverb")

    print(f"\nDone! 3 songs generated:")
    print(f"  1. {p1}")
    print(f"  2. {p2}")
    print(f"  3. {p3}")
