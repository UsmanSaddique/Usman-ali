"""Test img2vid workflow at realistic frame count to find the slowdown."""
import time, subprocess, json, urllib.request, sys, shutil
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

def nvidia_vram():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    used, free = r.stdout.strip().split(", ")
    return int(used), int(free)

def comfy_free():
    payload = json.dumps({"unload_models": True, "free_memory": True}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8188/free", data=payload,
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=10)

def comfy_submit(workflow):
    payload = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8188/prompt", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    err = result.get("error") or result.get("node_errors")
    if err:
        print(f"  COMFYUI ERROR: {err}")
    return result.get("prompt_id")

def comfy_done(pid):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:8188/history/{pid}")
        with urllib.request.urlopen(req, timeout=3) as resp:
            h = json.loads(resp.read())
            return pid in h
    except:
        return False

def wait_and_monitor(pid, t0, max_sec=900):
    for check_at in range(10, max_sec, 10):
        elapsed = time.time() - t0
        remaining = check_at - elapsed
        if remaining > 0:
            time.sleep(remaining)
        used, free = nvidia_vram()
        print(f"    +{check_at:3d}s: VRAM used={used} MiB  free={free} MiB")
        if comfy_done(pid):
            return time.time() - t0
    return time.time() - t0

from app.services.comfyui_client import build_ltx_workflow, build_ltx_img2vid_workflow

print("=== IMG2VID vs TXT2VID COMPARISON ===\n")

# ── Test 1: txt2vid at 97 frames (realistic) ──
print("[TEST 1] txt2vid 832x480, 97 frames, 8 steps")
comfy_free(); time.sleep(2)
used, free = nvidia_vram()
print(f"  Before: VRAM used={used} MiB  free={free} MiB")

wf1 = build_ltx_workflow(
    model_filename="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
    prompt="a cat walking across a sunny garden, cinematic",
    negative_prompt="", width=832, height=480, num_frames=97,
    steps=8, cfg=1.0, seed=42, fps=24,
)
t0 = time.time()
pid1 = comfy_submit(wf1)
print(f"  Submitted: {pid1}")
t1 = wait_and_monitor(pid1, t0, 600)
print(f"  >> txt2vid 97 frames: {t1:.0f}s\n")

# ── Test 2: img2vid at 97 frames ──
print("[TEST 2] img2vid 832x480, 97 frames, 8 steps")

# First generate a test image
print("  Generating test image first...")
comfy_free(); time.sleep(2)
from app.services.comfyui_client import build_zimage_workflow
img_wf = build_zimage_workflow(
    prompt="a cute cat sitting in a sunny garden, photorealistic",
    width=832, height=480, steps=9, cfg=1.0, seed=42,
)
pid_img = comfy_submit(img_wf)
time.sleep(20)  # wait for image
for _ in range(30):
    if comfy_done(pid_img):
        break
    time.sleep(2)
print("  Image done.")

# Find the generated image and copy to ComfyUI input
comfy_output = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\output")
comfy_input = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\input")
comfy_input.mkdir(parents=True, exist_ok=True)
# Find latest zimage output
zimage_files = sorted(comfy_output.glob("zimage_*.png"), key=lambda f: f.stat().st_mtime, reverse=True)
if not zimage_files:
    print("  ERROR: No zimage output found!")
    sys.exit(1)
test_img = zimage_files[0]
img_name = "test_i2v_input.png"
shutil.copy2(test_img, comfy_input / img_name)
print(f"  Using image: {test_img.name} -> {img_name}")

# Now free VRAM and run img2vid
print("  Freeing VRAM for video gen...")
comfy_free(); time.sleep(2); comfy_free(); time.sleep(2)
used, free = nvidia_vram()
print(f"  Before: VRAM used={used} MiB  free={free} MiB")

wf2 = build_ltx_img2vid_workflow(
    model_filename="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
    image_filename=img_name,
    prompt="a cat walking across a sunny garden, lively dynamic motion, gentle natural movement",
    negative_prompt="static, still image, frozen, motionless",
    width=832, height=480, num_frames=97,
    steps=8, cfg=1.0, seed=42, fps=24, strength=0.6,
)
t0 = time.time()
pid2 = comfy_submit(wf2)
if not pid2:
    print("  SUBMIT FAILED!")
    sys.exit(1)
print(f"  Submitted: {pid2}")
t2 = wait_and_monitor(pid2, t0, 900)
if comfy_done(pid2):
    print(f"  >> img2vid 97 frames: {t2:.0f}s\n")
else:
    print(f"  >> img2vid STILL RUNNING after {t2:.0f}s!\n")

# ── Summary ──
print("=== SUMMARY ===")
print(f"txt2vid 97 frames: {t1:.0f}s")
if comfy_done(pid2):
    print(f"img2vid 97 frames: {t2:.0f}s")
    if t2 > t1 * 3:
        print(f"\nimg2vid is {t2/t1:.1f}x SLOWER than txt2vid!")
        print("The LTXVImgToVideo node itself is the bottleneck, not VRAM.")
    else:
        print("\nBoth are similar speed. The slowdown must be in the pipeline code, not ComfyUI.")
else:
    print(f"img2vid: TIMED OUT (>{t2:.0f}s)")
    print("\nThe LTXVImgToVideo workflow is fundamentally broken/slow.")
