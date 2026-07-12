"""
Test: img2vid pipeline benchmark with the fairy girl prompt.
Times every step to identify exactly where time is spent.
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

COMFY_BASE = "http://127.0.0.1:8188"
COMFY_ROOT = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable")
COMFY_OUTPUT = COMFY_ROOT / "ComfyUI" / "output"
COMFY_INPUT = COMFY_ROOT / "ComfyUI" / "input"

# ── Helpers ──

def vram():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    used, free = r.stdout.strip().split(", ")
    return int(used), int(free)

def comfy_ping():
    try:
        req = urllib.request.Request(f"{COMFY_BASE}/system_stats")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except:
        return False

def comfy_free():
    payload = json.dumps({"unload_models": True, "free_memory": True}).encode()
    req = urllib.request.Request(
        f"{COMFY_BASE}/free", data=payload,
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=10)

def comfy_submit(workflow):
    payload = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFY_BASE}/prompt", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  COMFYUI ERROR: {body[:500]}")
        return None
    err = result.get("error") or result.get("node_errors")
    if err:
        print(f"  COMFYUI ERROR: {err}")
        return None
    return result.get("prompt_id")

def comfy_done(pid):
    try:
        req = urllib.request.Request(f"{COMFY_BASE}/history/{pid}")
        with urllib.request.urlopen(req, timeout=5) as resp:
            h = json.loads(resp.read())
            entry = h.get(pid)
            if entry:
                status = entry.get("status", {})
                if status.get("completed"):
                    return True
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    print(f"  COMFYUI JOB ERROR: {msgs}")
                    return "error"
            return False
    except:
        return False

def wait_job(pid, label, max_sec=900):
    t0 = time.time()
    for tick in range(5, max_sec, 5):
        elapsed = time.time() - t0
        remaining = tick - elapsed
        if remaining > 0:
            time.sleep(remaining)
        status = comfy_done(pid)
        if status == "error":
            print(f"  {label}: FAILED after {time.time()-t0:.0f}s")
            return None
        if status:
            took = time.time() - t0
            print(f"  {label}: DONE in {took:.1f}s")
            return took
        used, free = vram()
        print(f"    +{tick:3d}s: VRAM {used} MiB used, {free} MiB free")
    print(f"  {label}: TIMED OUT after {max_sec}s")
    return None

def find_latest(prefix, ext="png"):
    files = sorted(COMFY_OUTPUT.glob(f"{prefix}*.{ext}"), key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0] if files else None

# ── Main ──

print("=" * 60)
print("IMG2VID BENCHMARK - Fairy Girl Prompt")
print("=" * 60)

if not comfy_ping():
    print("\nERROR: ComfyUI not running at", COMFY_BASE)
    sys.exit(1)
print("ComfyUI: online\n")

total_t0 = time.time()

# ── Step 1: Free VRAM ──
print("[STEP 1] Freeing VRAM...")
t0 = time.time()
comfy_free()
time.sleep(2)
comfy_free()
time.sleep(1)
used, free = vram()
step1_time = time.time() - t0
print(f"  Done in {step1_time:.1f}s - VRAM: {used} MiB used, {free} MiB free\n")

# ── Step 2: Generate still image (Z-Image-Turbo) ──
print("[STEP 2] Generating still image (Z-Image-Turbo, 832x480, 9 steps)...")
sys.path.insert(0, str(Path(__file__).parent))
from app.services.comfyui_client import build_zimage_workflow

# Using the wizard's resolution: 832x480
img_wf = build_zimage_workflow(
    prompt=PROMPT,
    width=832, height=480,
    steps=9, cfg=1.0, seed=12345,
    output_prefix="bench_fairy_still",
)

t0 = time.time()
pid_img = comfy_submit(img_wf)
if not pid_img:
    print("  FAILED to submit image workflow")
    sys.exit(1)
print(f"  Submitted: {pid_img}")
step2_time = wait_job(pid_img, "Z-Image still", 120)

still_file = find_latest("bench_fairy_still", "png")
if not still_file:
    print("  ERROR: No still image output found!")
    sys.exit(1)
print(f"  Output: {still_file.name}")

# Copy to ComfyUI input for img2vid
COMFY_INPUT.mkdir(parents=True, exist_ok=True)
img_name = "bench_fairy_i2v.png"
shutil.copy2(str(still_file), str(COMFY_INPUT / img_name))
print(f"  Copied to input: {img_name}\n")

# ── Step 3: Free VRAM for video model ──
print("[STEP 3] Freeing VRAM before video gen...")
t0 = time.time()
comfy_free()
time.sleep(3)
used, free = vram()
step3_time = time.time() - t0
print(f"  Done in {step3_time:.1f}s - VRAM: {used} MiB used, {free} MiB free\n")

# ── Step 4: Generate video (LTX img2vid) ──
# 5 seconds at 24fps = 120 frames. Cap at 97 (LTX default for VRAM safety).
NUM_FRAMES = 97  # ~4s at 24fps - the LTX default
FPS = 24

print(f"[STEP 4] Generating video (LTX img2vid, 832x480, {NUM_FRAMES} frames, 8 steps)...")
from app.services.comfyui_client import build_ltx_img2vid_workflow

motion_prompt = (
    f"{PROMPT}, the subject moving and acting with lively dynamic motion, "
    f"gentle natural movement, subtle camera push-in"
)
motion_neg = "static, still image, frozen, motionless, no movement"

vid_wf = build_ltx_img2vid_workflow(
    model_filename="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
    image_filename=img_name,
    prompt=motion_prompt,
    negative_prompt=motion_neg,
    width=832, height=480,
    num_frames=NUM_FRAMES,
    steps=8, cfg=1.0, seed=12345, fps=FPS,
    strength=0.6,
)

t0 = time.time()
pid_vid = comfy_submit(vid_wf)
if not pid_vid:
    print("  FAILED to submit video workflow")
    sys.exit(1)
print(f"  Submitted: {pid_vid}")
step4_time = wait_job(pid_vid, "LTX img2vid", 900)

vid_file = find_latest("ai_director_i2v", "mp4")
if vid_file:
    print(f"  Output: {vid_file.name}")
else:
    print("  WARNING: No video output found")

# ── Summary ──
total_time = time.time() - total_t0
print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)
print(f"  Step 1 - VRAM free:         {step1_time:6.1f}s")
print(f"  Step 2 - Z-Image still:     {step2_time:6.1f}s" if step2_time else "  Step 2 - Z-Image still:     FAILED")
print(f"  Step 3 - VRAM free (video): {step3_time:6.1f}s")
print(f"  Step 4 - LTX img2vid:       {step4_time:6.1f}s" if step4_time else "  Step 4 - LTX img2vid:       FAILED")
print(f"  ---------------------------------------")
print(f"  TOTAL:                      {total_time:6.1f}s")
if step2_time and step4_time:
    overhead = step1_time + step3_time
    gen_time = step2_time + step4_time
    print(f"\n  Overhead (VRAM mgmt):  {overhead:.1f}s ({overhead/total_time*100:.0f}%)")
    print(f"  Actual generation:     {gen_time:.1f}s ({gen_time/total_time*100:.0f}%)")
    print(f"  GPU inference only:    ~{step4_time:.0f}s for {NUM_FRAMES} frames = {NUM_FRAMES/FPS:.1f}s of video")
print("=" * 60)
