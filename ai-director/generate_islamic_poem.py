"""
Generate a 2.5-minute Islamic poem (nasheed) with a moral story about kindness,
via ACE-Step 1.5 XL SFT (18GB model) on ComfyUI.
Then create a 30-scene project in AI Director for the video.
"""
import json
import time
import uuid
import shutil
import random
import urllib.request
from pathlib import Path

COMFY = "http://127.0.0.1:8188"
COMFY_OUT = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\output")
DEST = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\assets_generated\music\islamic_poem")
DEST.mkdir(parents=True, exist_ok=True)

DIRECTOR_API = "http://127.0.0.1:8000"

# ── The Poem: "The Kindness Tree" ──
# A moral story about a boy who plants seeds of kindness and watches them grow.
# Theme: small acts of kindness multiply into something beautiful.

POEM_TAGS = (
    "nasheed, children's music, islamic, folk, female lead vocal, "
    "children choir backing, soft piano, gentle strings, flute, "
    "duff frame drum, light hand percussion, warm, peaceful, uplifting, "
    "middle eastern, melodic, heartfelt, storytelling, clear vocals, "
    "balanced mix, reverb, lullaby-like, spiritual"
)

POEM_BPM = 80

POEM_LYRICS = """[verse]
There once was a boy with a heart full of light,
He smiled at the world from morning to night.
He carried no gold and he carried no fame,
But everyone knew little Yusuf by name.

[chorus]
Plant a seed of kindness, watch it grow so tall,
One small act of goodness can change the world for all.
Plant a seed of kindness, let it bloom and shine,
Allah loves the gentle heart, yours and also mine.

[verse]
He shared his last bread with a stranger one day,
He helped an old woman who'd lost her way.
He spoke with soft words when the world was too loud,
He never looked down and he never was proud.

[chorus]
Plant a seed of kindness, watch it grow so tall,
One small act of goodness can change the world for all.
Plant a seed of kindness, let it bloom and shine,
Allah loves the gentle heart, yours and also mine.

[bridge]
The Prophet said peace be upon him so dear,
The best of all people bring others good cheer.
A smile is a charity, given for free,
So open your heart like a welcoming tree.

[verse]
One day little Yusuf knelt down in the rain,
And planted a seed in the earth and the grain.
He watered with prayers and he waited with trust,
From that tiny seed something beautiful just —

[chorus]
Grew into a garden where children would play,
Where birds sang their praises at the end of the day.
Plant a seed of kindness, let it bloom and shine,
Allah loves the gentle heart, yours and also mine.

[outro]
So remember dear child when the world seems too wide,
That kindness is the greatest treasure inside.
Be gentle, be patient, be honest, be true,
And Allah will always take care of you."""


def build_workflow(tags: str, lyrics: str, seconds: float, seed: int, bpm: int) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "acestep_v1.5_xl_sft_bf16.safetensors",
            "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "qwen_0.6b_ace15.safetensors",
            "clip_name2": "qwen_1.7b_ace15.safetensors",
            "type": "ace"}},
        "3": {"class_type": "VAELoader", "inputs": {
            "vae_name": "ace_1.5_vae.safetensors"}},
        "4": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {
            "clip": ["2", 0], "tags": tags, "lyrics": lyrics,
            "seed": seed, "bpm": bpm, "duration": float(seconds),
            "timesignature": "4", "language": "en", "keyscale": "C major",
            "generate_audio_codes": True, "cfg_scale": 2.0,
            "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0}},
        "5": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {
            "seconds": float(seconds), "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["4", 0],
            "latent_image": ["5", 0], "seed": seed, "steps": 50, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveAudio", "inputs": {
            "audio": ["7", 0], "filename_prefix": "islamic_poem_kindness"}},
    }


def http_post(url: str, payload: dict, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http_get(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(timeout: float = 300.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{COMFY}/system_stats", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("ComfyUI not reachable")


def submit(wf: dict) -> str:
    cid = uuid.uuid4().hex[:12]
    res = http_post(f"{COMFY}/prompt", {"prompt": wf, "client_id": cid})
    if "prompt_id" not in res:
        raise RuntimeError(f"Rejected: {res}")
    return res["prompt_id"]


def wait_done(pid: str, timeout: int = 1800) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            hist = http_get(f"{COMFY}/history/{pid}")
            if pid in hist:
                h = hist[pid]
                if h.get("status", {}).get("completed"):
                    return h
                if h.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(f"Error: {h['status'].get('messages')}")
        except urllib.error.HTTPError:
            pass
        time.sleep(3)
    raise TimeoutError(f"Timeout on {pid}")


def collect(hist: dict, dest: Path) -> bool:
    for nid, nout in hist.get("outputs", {}).items():
        for entry in nout.get("audio", []):
            fn = entry.get("filename", "")
            sub = entry.get("subfolder", "")
            if not fn:
                continue
            src = COMFY_OUT / sub / fn if sub else COMFY_OUT / fn
            if src.exists():
                shutil.copy2(str(src), str(dest))
                return True
    return False


def create_project():
    """Create the 30-scene project in AI Director."""
    payload = {
        "title": "The Kindness Tree — Islamic Poem for Kids",
        "channel_slug": "little-muslim-nation",
        "duration": 150,
        "context": (
            "A 2.5-minute Islamic nasheed poem telling the story of little Yusuf "
            "who plants seeds of kindness. The moral: small acts of goodness — sharing, "
            "helping, gentle words — grow into something beautiful. "
            "30 scenes x 5 seconds each. Soft 3D Pixar-style cartoon visuals. "
            "Music already generated via ACE-Step SFT."
        ),
        "num_scenes": 30,
        "video_model": "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
    }
    try:
        res = http_post(f"{DIRECTOR_API}/api/projects", payload)
        print(f"[Project] Created: id={res['id']}, title={res['title']}")
        return res["id"]
    except Exception as e:
        print(f"[Project] Could not create project in Director: {e}")
        return None


def main():
    print("=" * 60)
    print("  The Kindness Tree — Islamic Poem Generator")
    print("  ACE-Step 1.5 XL SFT (18GB, 50 steps)")
    print("  Duration: 150s (~2.5 minutes)")
    print("=" * 60)

    # Step 1: Create project in AI Director
    print("\n[1/3] Creating project in AI Director...")
    project_id = create_project()

    # Step 2: Wait for ComfyUI
    print("\n[2/3] Waiting for ComfyUI...")
    wait_ready(timeout=300)
    print("       ComfyUI ready.")

    # Step 3: Generate the poem
    out_wav = DEST / "the_kindness_tree.flac"
    out_lyrics = DEST / "the_kindness_tree_lyrics.txt"
    out_lyrics.write_text(POEM_LYRICS, encoding="utf-8")

    seed = random.randint(0, 2**31 - 1)
    duration = 150.0  # 2.5 minutes

    wf = build_workflow(POEM_TAGS, POEM_LYRICS, duration, seed, POEM_BPM)

    print(f"\n[3/3] Generating poem...")
    print(f"       BPM: {POEM_BPM}")
    print(f"       Duration: {duration}s")
    print(f"       Seed: {seed}")
    print(f"       Steps: 50 (SFT quality)")

    t0 = time.time()
    try:
        pid = submit(wf)
        print(f"       prompt_id: {pid}")
        print(f"       Waiting for generation (this takes ~10-20 minutes)...")
        hist = wait_done(pid, timeout=2400)
        if collect(hist, out_wav):
            elapsed = time.time() - t0
            size_mb = out_wav.stat().st_size / 1e6
            print(f"\n  DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
            print(f"    Output: {out_wav}")
            print(f"    Size: {size_mb:.1f} MB")
            print(f"    Lyrics: {out_lyrics}")
            if project_id:
                print(f"    Project: {DIRECTOR_API}/api/projects/{project_id}")
                print(f"\n  Next steps:")
                print(f"    1. Open http://localhost:8000 in browser")
                print(f"    2. Click the project 'The Kindness Tree'")
                print(f"    3. Generate Script -> generates 30 scene prompts")
                print(f"    4. Approve & Generate -> creates all video clips")
        else:
            print(f"\n  FAIL: No audio found in outputs")
    except Exception as e:
        print(f"\n  FAIL: {e}")

    print()


if __name__ == "__main__":
    main()
