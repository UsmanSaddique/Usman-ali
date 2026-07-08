"""
Generate a 2.5-minute Urdu Islamic nasheed (proper song, not narration)
via ACE-Step 1.5 XL SFT, then trigger full video pipeline in AI Director.
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
DEST = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\assets_generated\music\urdu_nasheed")
DEST.mkdir(parents=True, exist_ok=True)

DIRECTOR_API = "http://127.0.0.1:8000"

# Proper SONG tags - melodic, catchy, musical (NOT narration)
SONG_TAGS = (
    "nasheed, islamic song, urdu pop, melodic, catchy chorus, "
    "female lead vocal, children choir, sing-along, "
    "soft piano, acoustic guitar, duff frame drum, light tabla, "
    "gentle strings, flute, glockenspiel, "
    "warm, uplifting, happy, cheerful, bright, "
    "middle eastern, south asian, kids song, "
    "clear vocals, polished production, radio quality, "
    "verse chorus structure, hook, earworm melody"
)

SONG_BPM = 100

# Proper Urdu song with story - "Nek Kaam" (Good Deeds)
# A catchy song about a kind boy who helps everyone and learns
# that every small good deed is seen by Allah.
SONG_LYRICS = """[intro]
La la la la, la la la la,
Nek kaam karo, nek kaam karo!

[verse]
Ek chhota sa bacha tha Yusuf naam,
Subah uthta tha karta Bismillah se kaam.
Ammi ko help karta, Abbu ka kehna maanta,
Har chehra dekh ke pyaara sa muskaata.

[chorus]
Nek kaam karo, nek kaam karo,
Dil mein mohabbat ka deep jalao.
Nek kaam karo, nek kaam karo,
Allah tumse bahut khush ho jaaye!

[verse]
School jaate waqt raaste mein dekha,
Ek boodhi Dadi gir gayi akela.
Yusuf ne haath badhaya, pyaar se uthaya,
Dadi ne dua di, beta khush ho jaaye!

[chorus]
Nek kaam karo, nek kaam karo,
Dil mein mohabbat ka deep jalao.
Nek kaam karo, nek kaam karo,
Allah tumse bahut khush ho jaaye!

[bridge]
Chhoti si muskaan bhi sadqa hai,
Pyaare Nabi ne farmaya hai.
Kisi ka dard banto, kisi ko khana do,
Har nek kaam mein jannat ka raasta hai!

[verse]
Ek billi royi thi baarish mein bheegi,
Yusuf ne ghar diya, doodh bhi peela.
Ek phool toda nahi, ek parinda sataya nahi,
Har makhlooq se kiya usne pyaar ka wada.

[chorus]
Nek kaam karo, nek kaam karo,
Dil mein mohabbat ka deep jalao.
Nek kaam karo, nek kaam karo,
Allah tumse bahut khush ho jaaye!

[outro]
La la la la, nek kaam karo,
La la la la, khushi phailao.
Allah dekh raha hai, Allah sun raha hai,
Har chhota nek kaam bada ho jaaye!"""


def build_workflow(tags, lyrics, seconds, seed, bpm):
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
            "timesignature": "4", "language": "ur", "keyscale": "C major",
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
            "audio": ["7", 0], "filename_prefix": "urdu_nasheed_nekkaam"}},
    }


def http_post(url, payload, timeout=30):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http_get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(timeout=300.0):
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


def submit(wf):
    cid = uuid.uuid4().hex[:12]
    res = http_post(f"{COMFY}/prompt", {"prompt": wf, "client_id": cid})
    if "prompt_id" not in res:
        raise RuntimeError(f"Rejected: {res}")
    return res["prompt_id"]


def wait_done(pid, timeout=2400):
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


def collect(hist, dest):
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
    payload = {
        "title": "Nek Kaam - Urdu Islamic Nasheed for Kids",
        "channel_slug": "little-muslim-nation",
        "duration": 150,
        "context": (
            "A 2.5-minute Urdu Islamic nasheed (song) called 'Nek Kaam' (Good Deeds). "
            "Story of little Yusuf who helps his parents, rescues a grandmother, "
            "shelters a wet kitten, and learns every small good deed is seen by Allah. "
            "Catchy sing-along chorus: 'Nek kaam karo, Allah tumse khush ho jaaye!' "
            "30 scenes x 5 seconds each. Soft 3D Pixar-style cartoon visuals. "
            "Music is a proper Urdu song with melody, chorus, verses."
        ),
        "num_scenes": 30,
        "video_model": "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
    }
    try:
        res = http_post(f"{DIRECTOR_API}/api/projects", payload)
        print(f"[Project] Created: id={res['id']}, title={res['title']}")
        return res["id"]
    except Exception as e:
        print(f"[Project] Could not create: {e}")
        return None


def trigger_full_auto(project_id):
    try:
        res = http_post(f"{DIRECTOR_API}/api/projects/{project_id}/generate-script", {})
        print(f"[Pipeline] Script generation started: {res}")
        return True
    except Exception as e:
        print(f"[Pipeline] Failed to start: {e}")
        return False


def main():
    print("=" * 60)
    print("  Nek Kaam - Urdu Islamic Nasheed Generator")
    print("  ACE-Step 1.5 XL SFT | 50 steps | 150s | Urdu")
    print("=" * 60)

    # Step 1: Create project
    print("\n[1/4] Creating project in AI Director...")
    project_id = create_project()

    # Step 2: ComfyUI
    print("\n[2/4] Checking ComfyUI...")
    wait_ready(timeout=300)
    print("       ComfyUI ready.")

    # Step 3: Generate song
    out_flac = DEST / "nek_kaam.flac"
    out_lyrics = DEST / "nek_kaam_lyrics.txt"
    out_lyrics.write_text(SONG_LYRICS, encoding="utf-8")

    seed = random.randint(0, 2**31 - 1)
    duration = 150.0

    wf = build_workflow(SONG_TAGS, SONG_LYRICS, duration, seed, SONG_BPM)

    print(f"\n[3/4] Generating Urdu nasheed...")
    print(f"       BPM: {SONG_BPM} | Duration: {duration}s | Seed: {seed}")
    print(f"       Language: Urdu | Steps: 50 (SFT)")

    t0 = time.time()
    try:
        pid = submit(wf)
        print(f"       prompt_id: {pid}")
        print(f"       Generating... (10-20 min)")
        hist = wait_done(pid, timeout=2400)
        if collect(hist, out_flac):
            elapsed = time.time() - t0
            size_mb = out_flac.stat().st_size / 1e6
            print(f"\n       DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
            print(f"       Output: {out_flac}")
            print(f"       Size: {size_mb:.1f} MB")
        else:
            print(f"\n       FAIL: No audio in outputs")
            return
    except Exception as e:
        print(f"\n       FAIL: {e}")
        return

    # Step 4: Trigger script generation for video
    if project_id:
        print(f"\n[4/4] Triggering script generation for video...")
        trigger_full_auto(project_id)
        print(f"       Project: http://localhost:8000")
        print(f"       Open browser to monitor progress.")

    print("\nAll done!")


if __name__ == "__main__":
    main()
