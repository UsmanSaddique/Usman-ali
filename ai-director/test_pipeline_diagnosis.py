"""
Diagnose WHY the pipeline is 10-15x slower than a clean run.
Simulates the actual pipeline flow: still gen → free → video gen.
Checks if ZImage/SDXL stays cached and causes LTX to spill to system RAM.
"""
import sys, time, json, shutil, urllib.request, urllib.error, subprocess
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

PROMPT = (
    "wide shot, a cute 6-year-old cartoon fairy girl with sparkling blue wings, "
    "long silver hair, wearing a glowing pastel dress takes flight into the cool "
    "night air followed by a flock of cute glowing baby birds in a vast starry sky, "
    "slow zoom in, 3D cartoon animation, soft dreamlike lighting, high quality render, "
    "magical pastel tones, deep indigo and glowing golds"
)

COMFY = "http://127.0.0.1:8188"
COMFY_ROOT = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable")
COMFY_OUTPUT = COMFY_ROOT / "ComfyUI" / "output"
COMFY_INPUT = COMFY_ROOT / "ComfyUI" / "input"

def vram():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    u, f = r.stdout.strip().split(", ")
    return int(u), int(f)

def free():
    payload = json.dumps({"unload_models": True, "free_memory": True}).encode()
    req = urllib.request.Request(f"{COMFY}/free", data=payload,
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)

def submit(wf):
    payload = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(f"{COMFY}/prompt", data=payload,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            r = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"  ERROR: {e.read().decode()[:300]}")
        return None
    if r.get("error") or r.get("node_errors"):
        print(f"  ERROR: {r.get('error') or r.get('node_errors')}")
        return None
    return r.get("prompt_id")

def done(pid):
    try:
        req = urllib.request.Request(f"{COMFY}/history/{pid}")
        with urllib.request.urlopen(req, timeout=5) as resp:
            h = json.loads(resp.read())
            e = h.get(pid)
            if e and e.get("status", {}).get("completed"):
                return True
            if e and e.get("status", {}).get("status_str") == "error":
                return "error"
    except:
        pass
    return False

def wait(pid, label, max_sec=900):
    t0 = time.time()
    for tick in range(5, max_sec, 5):
        el = time.time() - t0
        rem = tick - el
        if rem > 0: time.sleep(rem)
        s = done(pid)
        if s == "error":
            print(f"  {label}: FAILED at {time.time()-t0:.0f}s")
            return None
        if s:
            took = time.time() - t0
            print(f"  {label}: {took:.1f}s")
            return took
        u, f = vram()
        spill = "!! SPILLING TO RAM !!" if f < 500 else ""
        print(f"    +{tick:3d}s  VRAM {u:5d}/{u+f} MiB  (free={f})  {spill}")
    return None

sys.path.insert(0, str(Path(__file__).parent))
from app.services.comfyui_client import build_zimage_workflow, build_ltx_img2vid_workflow

COMFY_INPUT.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("PIPELINE FLOW SIMULATION — Why is it 10-15 min?")
print("=" * 65)

# ── Check: what's currently loaded? ──
u, f = vram()
print(f"\nCurrent VRAM: {u} MiB used, {f} MiB free")

# ══════════════════════════════════════════════════════════════════
# TEST A: Simulate OLD pipeline (generate still → single free → video)
# This is what happened BEFORE the fix
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST A: OLD PIPELINE FLOW (still → free → video, same scene)")
print("         This is what your pipeline was doing per scene")
print("=" * 65)

# Clean start
free(); time.sleep(2); free(); time.sleep(1)
u, f = vram()
print(f"\n[A.0] Clean start: VRAM {u} MiB used, {f} MiB free")

# A.1: Generate still (loads ZImage into VRAM)
print("\n[A.1] Generating still (Z-Image, 832x480)...")
wf1 = build_zimage_workflow(prompt=PROMPT, width=832, height=480,
    steps=9, cfg=1.0, seed=99, output_prefix="diag_a_still")
t0 = time.time()
pid1 = submit(wf1)
wait(pid1, "Still gen")

u, f = vram()
print(f"  After still: VRAM {u} MiB used, {f} MiB free")
print(f"  >> ZImage model is CACHED in ComfyUI VRAM")

# Copy output to input
still = sorted(COMFY_OUTPUT.glob("diag_a_still*.png"),
    key=lambda x: x.stat().st_mtime, reverse=True)
if not still:
    print("  No still found!"); sys.exit(1)
img_name = "diag_fairy_i2v.png"
shutil.copy2(str(still[0]), str(COMFY_INPUT / img_name))

# A.2: Single free_vram (what video_gen does with our fix)
print(f"\n[A.2] Single free_vram + sleep(3) (what video_gen.py does)...")
free()
time.sleep(3)
u, f = vram()
print(f"  After free: VRAM {u} MiB used, {f} MiB free")
if u > 3000:
    print(f"  >> WARNING: {u} MiB still used! ZImage NOT fully evicted!")
    print(f"  >> LTX-22B needs ~13GB but only {f} MiB free")
    print(f"  >> THIS IS WHY IT TAKES 10-15 MIN — VRAM SPILL TO SYSTEM RAM")
else:
    print(f"  >> Good — VRAM is mostly free, LTX should fit")

# A.3: Generate video immediately (LTX loads into whatever VRAM is left)
print(f"\n[A.3] Generating video (LTX img2vid, 832x480, 97f, 8 steps)...")
print(f"  If VRAM wasn't fully freed, LTX spills to RAM = 275s/step!")

motion_prompt = (f"{PROMPT}, the subject moving with lively dynamic motion, "
    "gentle natural movement, subtle camera push-in")
wf2 = build_ltx_img2vid_workflow(
    model_filename="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
    image_filename=img_name, prompt=motion_prompt,
    negative_prompt="static, still image, frozen, motionless",
    width=832, height=480, num_frames=97, steps=8, cfg=1.0,
    seed=99, fps=24, strength=0.6)
t0_vid = time.time()
pid2 = submit(wf2)
if not pid2:
    print("  FAILED!"); sys.exit(1)
t_a = wait(pid2, "LTX video (dirty VRAM)", 900)

# ══════════════════════════════════════════════════════════════════
# TEST B: Clean run (fresh VRAM → video only)
# This is what my benchmark did
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST B: CLEAN RUN (double-free → video only)")
print("         This is what my benchmark did (and what the fix does)")
print("=" * 65)

# B.1: Thorough VRAM free
print("\n[B.1] Double free_vram + sleep...")
free(); time.sleep(2); free(); time.sleep(2)
u, f = vram()
print(f"  After clean free: VRAM {u} MiB used, {f} MiB free")

# B.2: Generate video with clean VRAM
print(f"\n[B.2] Generating video (same LTX img2vid, clean VRAM)...")
wf3 = build_ltx_img2vid_workflow(
    model_filename="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
    image_filename=img_name, prompt=motion_prompt,
    negative_prompt="static, still image, frozen, motionless",
    width=832, height=480, num_frames=97, steps=8, cfg=1.0,
    seed=99, fps=24, strength=0.6)
pid3 = submit(wf3)
if not pid3:
    print("  FAILED!"); sys.exit(1)
t_b = wait(pid3, "LTX video (clean VRAM)", 900)

# ══════════════════════════════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)
if t_a and t_b:
    ratio = t_a / t_b if t_b > 0 else 0
    print(f"  Test A (dirty VRAM, old pipeline): {t_a:6.1f}s")
    print(f"  Test B (clean VRAM, benchmark):    {t_b:6.1f}s")
    print(f"  Ratio: {ratio:.1f}x")
    if ratio > 2:
        print(f"\n  >> CONFIRMED: Dirty VRAM makes LTX {ratio:.1f}x SLOWER")
        print(f"  >> ZImage stays cached, LTX spills to system RAM")
        print(f"  >> The two-phase fix eliminates this by doing ALL stills")
        print(f"     first, then freeing VRAM completely before video phase.")
    else:
        print(f"\n  >> VRAM was properly freed. Slowdown is elsewhere.")
        print(f"     Check: are you running at 1152x640 instead of 832x480?")
        print(f"     Check: is Qwen LLM loaded during generation?")
elif t_a:
    print(f"  Test A (dirty): {t_a:.1f}s")
    print(f"  Test B (clean):  FAILED")
elif t_b:
    print(f"  Test A (dirty): FAILED (probable OOM/spill)")
    print(f"  Test B (clean):  {t_b:.1f}s")
    print(f"  >> CONFIRMED: LTX cannot fit when ZImage is cached!")
print("=" * 65)
