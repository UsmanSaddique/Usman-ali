"""Generate CoComelon-style nursery rhyme with Islamic lyrics.

Uses well-known nursery rhyme rhythm patterns (Twinkle Twinkle / Wheels on the Bus)
mapped to Islamic lyrics, with very specific musical tags targeting that bright,
bouncy, children's TV show sound.
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

# Lyrics following "Wheels on the Bus" rhythm pattern
LYRICS_WHEELS = """[Verse 1]
The kids in the class say Bismillah,
Bismillah, Bismillah,
The kids in the class say Bismillah,
All day long!

[Verse 2]
The friends at the park share everything,
Everything, everything,
The friends at the park share everything,
Alhamdulillah!

[Chorus]
Be kind, be kind, be kind to all,
Big and small, big and small,
Be kind, be kind, be kind to all,
Allah loves the kind!

[Verse 3]
Yusuf and Amina help their friends,
Help their friends, help their friends,
Yusuf and Amina help their friends,
Every single day!

[Verse 4]
We say Salam when we meet someone,
Meet someone, meet someone,
We say Salam when we meet someone,
As-salamu alaykum!

[Outro]
The kids in the class say Alhamdulillah,
Alhamdulillah, Alhamdulillah,
The kids in the class say Alhamdulillah,
Thank you Allah!
"""

# Lyrics following "Twinkle Twinkle Little Star" rhythm
LYRICS_TWINKLE = """[Verse 1]
Bismillah Bismillah start your day,
Thank you Allah when we play,
Up above the sky so high,
Allah watches from the sky,
Bismillah Bismillah start your day,
Thank you Allah when we play!

[Verse 2]
Be kind be kind to everyone,
Sharing caring having fun,
Yusuf helps his friends at school,
Being gentle that's the rule,
Be kind be kind to everyone,
Sharing caring having fun!

[Bridge]
SubhanAllah look and see,
Flowers birds and every tree,
MashaAllah the world is bright,
Full of colors full of light!

[Outro]
Alhamdulillah for this day,
Alhamdulillah when we play,
Say Salam with a big smile,
Being good is always worthwhile,
Alhamdulillah for this day,
Thank you thank you dear Allah!
"""

# Lyrics following "Baby Shark" pattern - ultra catchy
LYRICS_SHARK = """[Verse 1]
Say Bismillah, doo doo doo doo doo doo,
Say Bismillah, doo doo doo doo doo doo,
Say Bismillah, doo doo doo doo doo doo,
Say Bismillah!

[Verse 2]
Be kind today, doo doo doo doo doo doo,
Be kind today, doo doo doo doo doo doo,
Be kind today, doo doo doo doo doo doo,
Be kind today!

[Verse 3]
Share with friends, doo doo doo doo doo doo,
Share with friends, doo doo doo doo doo doo,
Share with friends, doo doo doo doo doo doo,
Share with friends!

[Verse 4]
Say Salam, doo doo doo doo doo doo,
Say Salam, doo doo doo doo doo doo,
Say Salam, doo doo doo doo doo doo,
Say Salam!

[Chorus]
Alhamdulillah, doo doo doo doo doo doo,
Alhamdulillah, doo doo doo doo doo doo,
Alhamdulillah, doo doo doo doo doo doo,
Thank you Allah!

[Outro]
That's the end, doo doo doo doo doo doo,
That's the end, doo doo doo doo doo doo,
That's the end, doo doo doo doo doo doo,
Alhamdulillah!
"""


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


def generate(lyrics, tags, seed, steps, cfg, prefix):
    free_vram()
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
            "ckpt_name": "ace_step_v1_3.5b.safetensors"}},
        "2": {"class_type": "TextEncodeAceStepAudio", "inputs": {
            "clip": ["1", 1],
            "tags": tags,
            "lyrics": lyrics,
            "lyrics_strength": 1.0,
        }},
        "3": {"class_type": "EmptyAceStepLatentAudio", "inputs": {
            "seconds": 120.0, "batch_size": 1}},
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
    else:
        print("ComfyUI not available!")
        exit(1)
    print("ComfyUI ready!\n")

    # CoComelon-style tags — bright, bouncy, very specific
    COCOMELON_TAGS = "children's nursery rhyme, kids song, bright, bouncy, cheerful, playful, catchy melody, sing-along, clapping, xylophone, piano, ukulele, bells, tambourine, happy, upbeat, preschool, animated, simple melody, repetitive chorus, 120 bpm"

    print("=== Generating 3 CoComelon-style Islamic songs ===\n")

    # 1. "Wheels on the Bus" rhythm
    print("[1/3] Wheels-on-the-Bus rhythm with Islamic lyrics")
    p1 = generate(
        lyrics=LYRICS_WHEELS,
        tags=COCOMELON_TAGS + ", call and response, repetitive",
        seed=100, steps=120, cfg=5.0, prefix="song_wheels"
    )
    print(f"  Done: {p1}\n")

    # 2. "Twinkle Twinkle" rhythm
    print("[2/3] Twinkle-Twinkle rhythm with Islamic lyrics")
    p2 = generate(
        lyrics=LYRICS_TWINKLE,
        tags=COCOMELON_TAGS + ", lullaby, gentle, sweet, melodic",
        seed=200, steps=120, cfg=5.0, prefix="song_twinkle"
    )
    print(f"  Done: {p2}\n")

    # 3. "Baby Shark" catchy pattern
    print("[3/3] Baby-Shark catchy pattern with Islamic lyrics")
    p3 = generate(
        lyrics=LYRICS_SHARK,
        tags=COCOMELON_TAGS + ", viral, earworm, dance, energetic, fun, pop",
        seed=300, steps=120, cfg=5.0, prefix="song_shark"
    )
    print(f"  Done: {p3}\n")

    print(f"Done! 3 CoComelon-style songs:")
    print(f"  1. {p1} (Wheels on the Bus rhythm)")
    print(f"  2. {p2} (Twinkle Twinkle rhythm)")
    print(f"  3. {p3} (Baby Shark pattern)")
    print(f"\nListen to all 3 and pick the best one!")
