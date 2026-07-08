"""
Song Merger — Keep YOUR instrumental unchanged, AI sings vocals on top.

Pipeline:
1. ACE-Step v1.0 generates full song (vocals + instruments) — proven working
2. Demucs separates vocals from the generated song
3. FFmpeg overlays extracted vocals on YOUR unchanged instrumental
4. Studio master (EQ, limiter, fade)

Usage:
    python generate_song_merge.py
    python generate_song_merge.py --instrumental path/to/your/track.mp3
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import time
import uuid
import urllib.request
import urllib.error
from pathlib import Path

COMFYUI = "http://127.0.0.1:8188"
COMFYUI_OUTPUT = Path(
    r"C:\ComfyUI_windows_portable_nvidia_cu126"
    r"\ComfyUI_windows_portable\ComfyUI\output"
)
OUTPUT_DIR = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\projects\song_merge_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INSTRUMENTAL = r"C:\Users\PC\Desktop\VideoMaker\ai-director\projects\3efaca79-b3ae-4c20-a6a7-9c294bb368d1\wheels_instrumental_v1.mp3"

DEFAULT_LYRICS = """\
[intro]
La la la la la la

[verse]
Allah made the sun that shines so bright
Allah made the moon that glows at night
Allah made the stars up in the sky
Twinkling twinkling way up high

[chorus]
SubhanAllah the world is beautiful
SubhanAllah so wonderful
Look around at all that you can see
Allah made it all for you and me

[verse]
Allah made the rivers and the sea
Allah made the flowers and the trees
Allah made the birds that sing their song
We can sing and clap along

[chorus]
SubhanAllah the world is beautiful
SubhanAllah so wonderful
Look around at all that you can see
Allah made it all for you and me

[bridge]
Every colour every shape and sound
So much beauty all around
Thank you Allah for this lovely day
We love you in every way

[chorus]
SubhanAllah the world is beautiful
SubhanAllah so wonderful
Look around at all that you can see
Allah made it all for you and me

[outro]
SubhanAllah SubhanAllah
SubhanAllah
"""


# ─── Helpers ───────────────────────────────────────────────────────

def get_duration(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", str(path)], capture_output=True, text=True)
    try: return float(json.loads(r.stdout)["format"]["duration"])
    except: return 0.0

def comfyui_ping():
    try:
        with urllib.request.urlopen(f"{COMFYUI}/system_stats", timeout=3):
            return True
    except Exception:
        return False

def comfyui_free_vram():
    try:
        req = urllib.request.Request(f"{COMFYUI}/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass

def comfyui_submit(workflow):
    client_id = uuid.uuid4().hex[:12]
    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(f"{COMFYUI}/prompt", data=payload,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:500]}")
    pid = result.get("prompt_id")
    if not pid:
        raise RuntimeError(f"Rejected: {json.dumps(result)[:500]}")
    return pid

def comfyui_wait(prompt_id, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{COMFYUI}/history/{prompt_id}", timeout=10) as resp:
                hist = json.loads(resp.read()).get(prompt_id)
            if hist:
                status = hist.get("status", {})
                if status.get("completed"):
                    return hist
                if status.get("status_str") == "error":
                    for msg in status.get("messages", []):
                        if isinstance(msg, list) and msg[0] == "execution_error":
                            raise RuntimeError(f"{msg[1].get('node_type','?')}: {msg[1].get('exception_message','?')[:300]}")
                    raise RuntimeError(f"Error: {status.get('messages','?')}")
        except (urllib.error.URLError, TimeoutError):
            pass
        elapsed = int(time.time() - t0)
        if elapsed % 30 == 0 and elapsed > 0:
            print(f"    ... {elapsed}s", flush=True)
        time.sleep(5)
    raise TimeoutError(f"Timed out after {timeout}s")

def comfyui_collect_first(history, dest_path):
    all_outputs = []
    for nid, nout in sorted(history.get("outputs", {}).items()):
        for entry in nout.get("audio", []):
            fname = entry.get("filename", "")
            sub = entry.get("subfolder", "")
            src = COMFYUI_OUTPUT / sub / fname if sub else COMFYUI_OUTPUT / fname
            if src.exists():
                dur = get_duration(str(src))
                print(f"    Found: {fname} ({dur:.1f}s)", flush=True)
                if dur > 5.0:
                    all_outputs.append((str(src), fname, dur))
    if not all_outputs:
        raise FileNotFoundError("No audio output found")
    # Save all batch outputs for comparison
    dest_dir = Path(dest_path).parent
    for i, (src, fname, dur) in enumerate(all_outputs):
        variant = dest_dir / f"song_variant_{i+1}.wav"
        shutil.copy2(src, str(variant))
        # Also export MP3
        subprocess.run(["ffmpeg", "-y", "-i", str(variant),
            "-codec:a", "libmp3lame", "-b:a", "320k",
            str(dest_dir / f"song_variant_{i+1}.mp3")],
            capture_output=True, text=True)
        print(f"    Saved variant {i+1}: {dur:.1f}s", flush=True)
    # Use first as default
    shutil.copy2(all_outputs[0][0], dest_path)
    return dest_path


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Generate full song with ACE-Step v1.0 (proven working)
# ═══════════════════════════════════════════════════════════════════════

def generate_full_song(lyrics, tags, duration=90.0, seed=None, ref_audio=None):
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    use_ref = ref_audio is not None
    denoise = 0.85 if use_ref else 1.0
    steps = 65

    print(f"  Model: ACE-Step v1.0 (3.5B)", flush=True)
    print(f"  Tags: {tags}", flush=True)
    print(f"  Duration: {duration:.0f}s, Seed: {seed}", flush=True)
    if use_ref:
        print(f"  Reference audio: {ref_audio} (denoise={denoise})", flush=True)
        print(f"  -> Keeps rhythm/tempo from reference, generates new vocals", flush=True)

    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
            "ckpt_name": "ace_step_v1_3.5b.safetensors"}},
        "2": {"class_type": "TextEncodeAceStepAudio", "inputs": {
            "clip": ["1", 1], "tags": tags, "lyrics": lyrics,
            "lyrics_strength": 1.0}},
        "3": {"class_type": "TextEncodeAceStepAudio", "inputs": {
            "clip": ["1", 1], "tags": "", "lyrics": "", "lyrics_strength": 1.0}},
    }

    if use_ref:
        # Audio-to-audio: encode reference -> use as starting latent
        workflow["8"] = {"class_type": "LoadAudio", "inputs": {
            "audio": ref_audio}}
        workflow["9"] = {"class_type": "VAEEncodeAudio", "inputs": {
            "audio": ["8", 0], "vae": ["1", 2]}}
        latent_source = ["9", 0]
    else:
        workflow["4"] = {"class_type": "EmptyAceStepLatentAudio", "inputs": {
            "seconds": float(duration), "batch_size": 1}}
        latent_source = ["4", 0]

    workflow["5"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
        "latent_image": latent_source, "seed": seed, "steps": steps, "cfg": 4.0,
        "sampler_name": "er_sde", "scheduler": "linear_quadratic", "denoise": denoise}}
    workflow["6"] = {"class_type": "VAEDecodeAudio", "inputs": {
        "samples": ["5", 0], "vae": ["1", 2]}}
    workflow["7"] = {"class_type": "SaveAudio", "inputs": {
        "audio": ["6", 0], "filename_prefix": "song_full"}}

    comfyui_free_vram()
    time.sleep(3)
    pid = comfyui_submit(workflow)
    print(f"  Submitted: {pid}", flush=True)
    print(f"  Generating full song with real AI singing...", flush=True)

    t0 = time.time()
    hist = comfyui_wait(pid, timeout=1800)
    print(f"  Generated in {time.time()-t0:.0f}s!", flush=True)

    full_song = str(OUTPUT_DIR / "song_full.wav")
    return comfyui_collect_first(hist, full_song)


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Extract vocals using Demucs
# ═══════════════════════════════════════════════════════════════════════

def extract_vocals(song_path):
    print(f"  Running Demucs vocal separation...", flush=True)
    sep_dir = str(OUTPUT_DIR / "separated")

    # Convert input to WAV first (soundfile can't read FLAC reliably)
    wav_input = str(OUTPUT_DIR / "song_full_converted.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(song_path),
        "-ar", "44100", "-ac", "2", wav_input
    ], capture_output=True, text=True)

    # Run demucs via inline script using soundfile directly (bypass torchcodec)
    script = f"""
import os, sys
import numpy as np
import soundfile as sf
import torch

from demucs.pretrained import get_model
from demucs.apply import apply_model

model = get_model("htdemucs")
model.to("cuda")

data, sr = sf.read("{wav_input.replace(chr(92), '/')}", dtype="float32")
# data shape: (samples, channels)
wav = torch.from_numpy(data.T)  # -> (channels, samples)
if wav.dim() == 1:
    wav = wav.unsqueeze(0)
if wav.shape[0] == 1:
    wav = wav.repeat(2, 1)

ref = wav.mean(0)
wav_norm = (wav - ref.mean()) / ref.std()
wav_norm = wav_norm.unsqueeze(0).to("cuda")

sources = apply_model(model, wav_norm, device="cuda")
vocals = sources[0, -1]  # last source = vocals
vocals = vocals * ref.std() + ref.mean()

os.makedirs("{sep_dir.replace(chr(92), '/')}", exist_ok=True)
out_path = "{sep_dir.replace(chr(92), '/')}/vocals.wav"
sf.write(out_path, vocals.cpu().numpy().T, sr)
print(f"Saved vocals to {{out_path}}")
"""

    r = subprocess.run(
        ["python", "-c", script],
        capture_output=True, text=True, timeout=600
    )

    if r.returncode != 0:
        print(f"  stdout: {r.stdout[-300:]}", flush=True)
        print(f"  stderr: {r.stderr[-500:]}", flush=True)
        raise RuntimeError("Demucs failed")

    print(f"  {r.stdout.strip()}", flush=True)

    vocals_path = Path(sep_dir) / "vocals.wav"
    if vocals_path.exists():
        dur = get_duration(str(vocals_path))
        print(f"  Extracted vocals: {dur:.1f}s", flush=True)
        dest = str(OUTPUT_DIR / "vocals_extracted.wav")
        shutil.copy2(str(vocals_path), dest)
        return dest

    # Fallback: check rglob
    for vf in Path(sep_dir).rglob("vocals.wav"):
        dur = get_duration(str(vf))
        dest = str(OUTPUT_DIR / "vocals_extracted.wav")
        shutil.copy2(str(vf), dest)
        return dest

    raise FileNotFoundError("Demucs didn't produce vocals.wav")


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Overlay vocals on YOUR instrumental
# ═══════════════════════════════════════════════════════════════════════

def overlay_vocals_on_instrumental(instrumental_path, vocals_path, output_path,
                                    inst_volume=0.55, vocal_volume=1.8):
    inst_dur = get_duration(instrumental_path)
    vox_dur = get_duration(vocals_path)
    total_dur = max(inst_dur, vox_dur)
    fade_out = max(0, total_dur - 3.0)

    print(f"  Instrumental: {inst_dur:.1f}s (vol={inst_volume})", flush=True)
    print(f"  Vocals: {vox_dur:.1f}s (vol={vocal_volume})", flush=True)

    filt = (
        f"[0:a]volume={inst_volume},aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[inst];"
        f"[1:a]volume={vocal_volume},aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[vox];"
        f"[inst][vox]amix=inputs=2:duration=first:normalize=0,"
        f"highpass=f=60,"
        f"equalizer=f=3500:t=h:w=3000:g=2,"
        f"equalizer=f=12000:t=h:w=4000:g=1.5,"
        f"alimiter=limit=0.95:attack=5:release=50,"
        f"afade=t=in:st=0:d=0.5,afade=t=out:st={fade_out}:d=3[out]"
    )

    r = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(instrumental_path),
        "-i", str(vocals_path),
        "-filter_complex", filt,
        "-map", "[out]", "-ar", "44100", str(output_path),
    ], capture_output=True, text=True)

    if r.returncode != 0:
        raise RuntimeError(f"Mix failed: {r.stderr[-300:]}")

    dur = get_duration(str(output_path))
    size = os.path.getsize(str(output_path)) / 1024 / 1024
    print(f"  Output: {dur:.1f}s, {size:.1f}MB", flush=True)


# ═══════════════════════════════════════════════════════════════════════

STYLES = [
    {
        "name": "01_cocomelon_female",
        "tags": "children's pop, catchy nursery rhyme, sweet female vocalist, crystal clear vocals, major key, glockenspiel, acoustic guitar, ukulele, bright piano, light percussion, handclaps, finger snaps, bouncy bass, 115 bpm, cheerful, singalong, simple catchy melody, warm and playful, polished studio mix, radio quality, reverb on vocals",
    },
    {
        "name": "02_lullaby_gentle",
        "tags": "lullaby, gentle female vocals, soft and soothing, music box, celesta, gentle piano, acoustic guitar fingerpicking, warm pad, 80 bpm, major key, dreamy, peaceful, tender, delicate arrangement, minimal percussion, reverb, bedtime song, studio quality",
    },
    {
        "name": "03_upbeat_male",
        "tags": "children's pop, cheerful male vocalist, clear warm voice, acoustic guitar, clapping, tambourine, bass guitar, bright piano, 120 bpm, major key, fun, energetic, bouncy, catchy hook, singalong, clean mix, studio quality production",
    },
    {
        "name": "04_choir_kids",
        "tags": "children's choir, group singing, kids vocals, joyful, bright, clapping, stomping, acoustic guitar, piano, xylophone, recorder, 110 bpm, major key, school concert, innocent, pure voices, warm reverb, polished mix",
    },
    {
        "name": "05_disney_epic",
        "tags": "cinematic children's pop, powerful female vocalist, soaring melody, orchestral strings, grand piano, building chorus, inspirational, choir harmonies, 100 bpm, major key, warm reverb, lush production, anthem, dynamic, soft verse loud chorus, studio quality",
    },
    {
        "name": "06_reggae_island",
        "tags": "reggae, children's song, sweet female vocals, island vibes, steel drums, acoustic guitar, bass guitar, light drums, offbeat rhythm, 95 bpm, major key, sunny, tropical, relaxed, happy, warm, clean vocals, studio mix",
    },
    {
        "name": "07_jazz_swing",
        "tags": "jazz, swing, children's song, smooth female vocalist, clear vocals, upright bass, brushed drums, piano trio, trumpet, 130 bpm, major key, playful, sophisticated, vintage, warm analog sound, clean production",
    },
    {
        "name": "08_folk_acoustic",
        "tags": "folk, acoustic, children's song, warm female vocalist, clear intimate vocals, acoustic guitar strumming, banjo, fiddle, light percussion, 105 bpm, major key, heartfelt, storytelling, campfire, natural reverb, organic sound",
    },
    {
        "name": "09_electronic_cute",
        "tags": "electronic pop, children's song, cute female vocals, synth melody, chiptune, music box, soft synth bass, electronic drums, 125 bpm, major key, kawaii, sparkly, fun, modern, bright, clean digital production",
    },
    {
        "name": "10_bollywood_kids",
        "tags": "bollywood pop, children's song, expressive female vocalist, clear vocals, sitar, tabla, dholak, flute, strings, harmonium, 115 bpm, major key, festive, colorful, celebratory, dancing, rich arrangement, studio quality",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Generate 10 style variants")
    parser.add_argument("--instrumental", type=str, default=DEFAULT_INSTRUMENTAL)
    parser.add_argument("--duration", type=float, default=120.0)
    args = parser.parse_args()

    if not os.path.isfile(args.instrumental):
        print(f"ERROR: File not found: {args.instrumental}")
        return

    lyrics = DEFAULT_LYRICS
    inst_dur = get_duration(args.instrumental)

    print("=" * 70, flush=True)
    print("  SONG FACTORY -- 10 Styles, Same Lyrics, Best One Wins", flush=True)
    print("=" * 70, flush=True)
    print(f"  Instrumental: {args.instrumental} ({inst_dur:.0f}s)", flush=True)
    print(f"  Generating 10 different versions...", flush=True)

    print("\n[INIT] Waiting for ComfyUI...", flush=True)
    for i in range(60):
        if comfyui_ping():
            print("  Ready!", flush=True)
            break
        if i % 10 == 0 and i > 0: print(f"  ... {i*2}s", flush=True)
        time.sleep(2)
    else:
        print("  ComfyUI not available"); return

    results_dir = OUTPUT_DIR / "10_versions"
    results_dir.mkdir(parents=True, exist_ok=True)

    for idx, style in enumerate(STYLES):
        num = idx + 1
        name = style["name"]
        tags = style["tags"]

        print(f"\n{'='*70}", flush=True)
        print(f"  [{num}/10] {name}", flush=True)
        print(f"  Tags: {tags[:80]}...", flush=True)
        print(f"{'='*70}", flush=True)

        # Generate
        try:
            seed = random.randint(0, 2**31 - 1)
            full_path = str(results_dir / f"{name}_full.wav")
            generate_full_song(lyrics, tags, args.duration, seed)
            # Collect output
            with urllib.request.urlopen(f"{COMFYUI}/history", timeout=10) as resp:
                pass
            # The generate_full_song already saves to song_full.wav via collect
            src = str(OUTPUT_DIR / "song_full.wav")
            shutil.copy2(src, full_path)
        except Exception as e:
            print(f"  FAILED generation: {e}", flush=True)
            continue

        # Free VRAM
        comfyui_free_vram()
        time.sleep(1)

        # Extract vocals
        try:
            vocals_path = str(results_dir / f"{name}_vocals.wav")
            extracted = extract_vocals(full_path)
            shutil.copy2(extracted, vocals_path)
        except Exception as e:
            print(f"  FAILED demucs: {e}", flush=True)
            continue

        # Mix with instrumental
        try:
            mixed_wav = str(results_dir / f"{name}_mixed.wav")
            overlay_vocals_on_instrumental(args.instrumental, vocals_path, mixed_wav)
        except Exception as e:
            print(f"  FAILED mix: {e}", flush=True)
            continue

        # Export MP3s
        for src_file, suffix in [(full_path, "_full"), (mixed_wav, "_mixed")]:
            mp3_path = str(results_dir / f"{name}{suffix}.mp3")
            subprocess.run(["ffmpeg", "-y", "-i", src_file,
                "-codec:a", "libmp3lame", "-b:a", "320k", mp3_path],
                capture_output=True, text=True)

        dur = get_duration(mixed_wav)
        print(f"  OK! {name}_full.mp3 + {name}_mixed.mp3 ({dur:.0f}s)", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  ALL DONE! 10 versions saved to:", flush=True)
    print(f"  {results_dir}", flush=True)
    print(f"", flush=True)
    print(f"  *_full.mp3  = raw AI song (vocals + instruments)", flush=True)
    print(f"  *_mixed.mp3 = vocals overlaid on YOUR instrumental", flush=True)
    print(f"{'='*70}", flush=True)

if __name__ == "__main__":
    main()
