"""
Ultimate CoComelon-style Islamic nursery rhyme generator.

RESEARCH-BASED APPROACH:
========================
CoComelon "Wheels on the Bus" musical DNA (from Tunebat/SongBPM analysis):
- Key: A major (bright, happy, children-friendly key)
- BPM: 120-125 (bouncy, danceable but not frantic)
- Time signature: 4/4 (standard, predictable for kids)
- Energy: Low-to-medium (gentle but engaging)
- Danceability: Very high (repetitive, predictable rhythm)
- Chord progression: I-IV-V-I (A-D-E-A) — simplest happy progression
- Mode: Major (always major for happy children's content)

CoComelon Production Signature:
- Bright, clean mix — no reverb, no distortion
- Lead: Grand piano playing melody + chord hits
- Rhythm: Ukulele strumming pattern (down-up-down-up, 8th notes)
- Percussion: Light kick-snare, tambourine on 2&4, finger snaps, hand claps
- Color instruments: Glockenspiel/xylophone (melody doubling), toy piano accents
- Bass: Simple root-note bass guitar or synth bass
- Pads: Warm string pad underneath (barely audible, adds fullness)
- Vocal: Young female, clear enunciation, slightly breathy, close-mic'd
- Structure: Ultra-repetitive AABA per verse, same melody every verse

ACE-Step v1.0 Prompting Best Practices (from guides):
- 5-12 specific tags, lead with genre
- Specific instruments > generic ("grand piano" > "piano")
- Include BPM in tags
- Lines of 4-8 syllables flow best
- 2-3 words per second
- AABB/ABAB rhyme scheme sounds most natural
- Repeating chorus helps model lock onto melody
- lyrics_strength=1.0 for vocal songs
- Steps 27 is sweet spot, above 60 gains diminish
- CFG 5-7 for musical interpretation, 10-12 for strict adherence
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

# ══════════════════════════════════════════════════════════════════════
# LYRICS — Exact "Wheels on the Bus" structure
# Pattern: "[Subject] [action verb] [Islamic word], [word], [word],"
#          same line repeated, then resolution line
# Each verse = ONE Islamic concept, naturally embedded
# 4-8 syllables per line, AABB rhyme, 2-3 words/second
# ══════════════════════════════════════════════════════════════════════

LYRICS_PERFECT = """[verse]
The kids on the bus say Bismillah,
Bismillah, Bismillah,
The kids on the bus say Bismillah,
All through the town!

[verse]
The mommies on the bus say Alhamdulillah,
Alhamdulillah, Alhamdulillah,
The mommies on the bus say Alhamdulillah,
All through the town!

[verse]
The babies on the bus say MashaAllah,
MashaAllah, MashaAllah,
The babies on the bus say MashaAllah,
All through the town!

[chorus]
The wheels on the bus go round and round,
Round and round, round and round,
The wheels on the bus go round and round,
All through the town!

[verse]
The friends on the bus say Assalamu Alaikum,
Alaikum, Alaikum,
The friends on the bus say Assalamu Alaikum,
All through the town!

[verse]
The daddies on the bus say SubhanAllah,
SubhanAllah, SubhanAllah,
The daddies on the bus say SubhanAllah,
All through the town!

[verse]
The kids on the bus say InshaAllah,
InshaAllah, InshaAllah,
The kids on the bus say InshaAllah,
All through the town!

[chorus]
The wheels on the bus go round and round,
Round and round, round and round,
The wheels on the bus go round and round,
All through the town!
"""

# ══════════════════════════════════════════════════════════════════════
# TAGS — Precision-engineered CoComelon sonic signature
# Format: genre, specific instruments, vocal type, mood, production, BPM
# ══════════════════════════════════════════════════════════════════════

# Tag set 1: Maximum CoComelon fidelity
TAGS_COCOMELON = (
    "children's nursery rhyme, "
    "grand piano, ukulele strumming, glockenspiel, xylophone, "
    "tambourine, hand claps, finger snaps, light drums, "
    "young female vocal, bright, bouncy, cheerful, "
    "sing-along, catchy melody, repetitive, "
    "clean mix, hi-fi, polished, "
    "120 bpm, A major"
)

# Tag set 2: Simplified (fewer tags, less dilution risk)
TAGS_SIMPLE = (
    "children's nursery rhyme, "
    "grand piano, ukulele, glockenspiel, tambourine, hand claps, "
    "young female vocal, bright, bouncy, cheerful, catchy, "
    "sing-along, 120 bpm"
)

# Tag set 3: Pop-nursery hybrid (CoComelon's modern sound)
TAGS_POP = (
    "children's pop, nursery rhyme, "
    "grand piano, ukulele, bells, xylophone, light percussion, "
    "young female vocal, sweet, bright, happy, uplifting, "
    "catchy chorus, earworm, polished, "
    "125 bpm"
)


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
        elapsed = int(time.time() - t0)
        if elapsed > 0 and elapsed % 60 == 0:
            print(f"    ... {elapsed}s elapsed")
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
                    print(f"    Saved: {dest_path}")
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


def generate(lyrics, tags, seed, steps, cfg, lyrics_strength, prefix, desc):
    """Generate a song with ACE-Step v1.0."""
    print(f"\n  [{desc}]")
    print(f"  Tags: {tags[:80]}...")
    print(f"  Params: seed={seed}, steps={steps}, cfg={cfg}, lyrics_str={lyrics_strength}")
    free_vram()
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
            "ckpt_name": "ace_step_v1_3.5b.safetensors"}},
        "2": {"class_type": "TextEncodeAceStepAudio", "inputs": {
            "clip": ["1", 1],
            "tags": tags,
            "lyrics": lyrics,
            "lyrics_strength": lyrics_strength,
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
    print(f"  Submitted: {pid}")
    hist = wait(pid)
    dest = str(DEST / f"{prefix}.wav")
    collect(hist, dest)
    return dest


if __name__ == "__main__":
    print("=" * 60)
    print("ULTIMATE CoComelon Islamic Nursery Rhyme Generator")
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

    print("Generating 6 variations -- exploring parameter space:\n")

    # -- Round 1: Tag variations (same lyrics, different sonic signatures) --

    print("-" * 50)
    print("ROUND 1: Tag variations (best tags win)")
    print("-" * 50)

    # V1: Full CoComelon tag set, sweet-spot steps
    p1 = generate(
        lyrics=LYRICS_PERFECT, tags=TAGS_COCOMELON,
        seed=42, steps=50, cfg=7.0, lyrics_strength=1.0,
        prefix="ultimate_v1_cocomelon",
        desc="1/6 — Full CoComelon tags, cfg=7 (musical), 50 steps"
    )

    # V2: Simplified tags (less dilution), higher cfg for stricter adherence
    p2 = generate(
        lyrics=LYRICS_PERFECT, tags=TAGS_SIMPLE,
        seed=42, steps=50, cfg=10.0, lyrics_strength=1.0,
        prefix="ultimate_v2_simple",
        desc="2/6 — Simplified tags, cfg=10 (strict), 50 steps"
    )

    # V3: Pop-nursery hybrid
    p3 = generate(
        lyrics=LYRICS_PERFECT, tags=TAGS_POP,
        seed=42, steps=50, cfg=7.0, lyrics_strength=1.0,
        prefix="ultimate_v3_pop",
        desc="3/6 — Pop-nursery hybrid tags, cfg=7, 50 steps"
    )

    # -- Round 2: Parameter variations (best tags from theory + different seeds) --

    print("\n" + "-" * 50)
    print("ROUND 2: Seed & parameter variations")
    print("-" * 50)

    # V4: Different seed, slightly lower lyrics_strength for more musical freedom
    p4 = generate(
        lyrics=LYRICS_PERFECT, tags=TAGS_SIMPLE,
        seed=888, steps=50, cfg=7.0, lyrics_strength=0.8,
        prefix="ultimate_v4_free",
        desc="4/6 — Simple tags, seed=888, lyrics_str=0.8 (more musical)"
    )

    # V5: High steps for max fidelity
    p5 = generate(
        lyrics=LYRICS_PERFECT, tags=TAGS_COCOMELON,
        seed=555, steps=80, cfg=7.0, lyrics_strength=1.0,
        prefix="ultimate_v5_hifi",
        desc="5/6 — Full CoComelon tags, seed=555, 80 steps (hi-fi)"
    )

    # V6: The wildcard — lower cfg for more creative interpretation
    p6 = generate(
        lyrics=LYRICS_PERFECT, tags=TAGS_SIMPLE,
        seed=1234, steps=50, cfg=5.0, lyrics_strength=1.0,
        prefix="ultimate_v6_creative",
        desc="6/6 — Simple tags, seed=1234, cfg=5 (creative), 50 steps"
    )

    print("\n" + "=" * 60)
    print("DONE! 6 variations generated:")
    print("=" * 60)
    print(f"\n  TAG VARIATIONS (same seed=42):")
    print(f"  1. {p1}  — Full CoComelon tags, cfg=7")
    print(f"  2. {p2}  — Simplified tags, cfg=10 (strict)")
    print(f"  3. {p3}  — Pop-nursery hybrid, cfg=7")
    print(f"\n  PARAMETER VARIATIONS:")
    print(f"  4. {p4}  — lyrics_str=0.8 (more musical freedom)")
    print(f"  5. {p5}  — 80 steps (maximum fidelity)")
    print(f"  6. {p6}  — cfg=5 (creative interpretation)")
    print(f"\n  Listen to all 6 and pick the best!")
