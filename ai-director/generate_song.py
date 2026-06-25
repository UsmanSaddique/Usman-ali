"""Direct music generation with Islamic children's lyrics."""
import json
import time
import uuid
import shutil
import urllib.request
from pathlib import Path

COMFYUI = "http://127.0.0.1:8188"
OUTPUT_DIR = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\output")
DEST = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\projects\3efaca79-b3ae-4c20-a6a7-9c294bb368d1")

# Islamic children's lyrics — catchy nursery-rhyme rhythm
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

[Chorus]
Be kind, be kind, be kind today,
Share and care in every way,
Say Salam with a big bright smile,
Being good is always worthwhile!
"""

TAGS_15XL = "children's music, nursery rhyme, acoustic, female vocal, happy, uplifting, playful, piano, ukulele, xylophone, bells, clapping, warm, gentle, sing-along"
TAGS_V1 = "children's music, nursery rhyme, acoustic, female vocal, happy, uplifting, playful, piano, ukulele, xylophone, bells, clapping, warm, gentle, sing-along, kids song"


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
    print(f"Submitted: {pid}")
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
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"Error: {msgs}")
        except (urllib.error.URLError, TimeoutError):
            pass
        if int(time.time() - t0) % 30 == 0:
            print(f"  ... {int(time.time() - t0)}s elapsed")
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
                    print(f"Saved: {src} -> {dest_path}")
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


def try_acestep_15xl():
    """ACE-Step 1.5 XL Turbo — best quality."""
    print("\n=== Trying ACE-Step 1.5 XL Turbo ===")
    free_vram()
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "acestep_v1.5_xl_turbo_bf16.safetensors",
            "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen_1.7b_ace15.safetensors",
            "type": "ace"}},
        "3": {"class_type": "VAELoader", "inputs": {
            "vae_name": "ace_1.5_vae.safetensors"}},
        "4": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {
            "clip": ["2", 0],
            "tags": TAGS_15XL,
            "lyrics": LYRICS,
            "seed": 12345,
            "bpm": 110,
            "duration": 130.0,
            "timesignature": "4",
            "language": "en",
            "keyscale": "C major",
            "generate_audio_codes": True,
            "cfg_scale": 2.0,
            "temperature": 0.85,
            "top_p": 0.9,
            "top_k": 0,
            "min_p": 0.0,
        }},
        "5": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {
            "seconds": 130.0, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["4", 0],
            "latent_image": ["5", 0], "seed": 12345, "steps": 8, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveAudio", "inputs": {
            "audio": ["7", 0], "filename_prefix": "islamic_kids_song"}},
    }
    pid = submit(wf)
    hist = wait(pid)
    dest = str(DEST / "music_15xl.wav")
    collect(hist, dest)
    return dest


def try_acestep_v1():
    """ACE-Step v1.0 — fallback with lyrics."""
    print("\n=== Trying ACE-Step v1.0 with lyrics ===")
    free_vram()
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
            "ckpt_name": "ace_step_v1_3.5b.safetensors"}},
        "2": {"class_type": "TextEncodeAceStepAudio", "inputs": {
            "clip": ["1", 1],
            "tags": TAGS_V1,
            "lyrics": LYRICS,
            "lyrics_strength": 1.0,
        }},
        "3": {"class_type": "EmptyAceStepLatentAudio", "inputs": {
            "seconds": 130.0, "batch_size": 1}},
        "4": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["2", 0],
            "latent_image": ["3", 0], "seed": 12345, "steps": 60, "cfg": 5.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "5": {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["4", 0], "vae": ["1", 2]}},
        "6": {"class_type": "SaveAudio", "inputs": {
            "audio": ["5", 0], "filename_prefix": "islamic_kids_song_v1"}},
    }
    pid = submit(wf)
    hist = wait(pid)
    dest = str(DEST / "music_v1_lyrics.wav")
    collect(hist, dest)
    return dest


if __name__ == "__main__":
    # Wait for ComfyUI
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

    # Try 1.5 XL first, fall back to v1.0
    try:
        path = try_acestep_15xl()
        print(f"\n✓ ACE-Step 1.5 XL song saved: {path}")
    except Exception as e:
        print(f"\n✗ ACE-Step 1.5 XL failed: {e}")
        print("Falling back to v1.0 with lyrics...")
        try:
            path = try_acestep_v1()
            print(f"\n✓ ACE-Step v1.0 song saved: {path}")
        except Exception as e2:
            print(f"\n✗ v1.0 also failed: {e2}")
            exit(1)
