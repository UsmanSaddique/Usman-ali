"""
Benchmark: OLD pipeline (GGUF img2vid) vs NEW (43GB distilled + IC-LoRA Ingredients)
====================================================================================
Produces two ~1-min videos (12 clips x ~5s @ 832x480 -> ESRGAN 1080p) from the SAME
12 scene prompts, with per-phase timing stats.

  A (old): SDXL still -> LTX GGUF img2vid (8 steps) -> ESRGAN 1080p
  B (new): reference sheet -> 43GB ckpt + IC-LoRA Ingredients (8-step distilled
           sigmas, video-only sampling) -> ESRGAN 1080p

Run on ComfyUI embedded python:
  C:\\ComfyUI_windows_portable_nvidia_cu126\\ComfyUI_windows_portable\\python_embeded\\python.exe benchmark_iclora.py
"""
import json
import shutil
import subprocess
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, ".")

from app.services.comfyui_client import (
    ComfyUIClient,
    COMFYUI_OUTPUT,
    build_sdxl_workflow,
    build_zimage_workflow,
    build_ltx_img2vid_workflow,
    build_esrgan_video_upscale_workflow,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bench")

COMFY_INPUT = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\input")
OUT = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\assets_generated\benchmark_iclora")
OUT.mkdir(parents=True, exist_ok=True)

FFMPEG = "ffmpeg"
try:
    subprocess.run([FFMPEG, "-version"], capture_output=True, timeout=5)
except Exception:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

W, H, FRAMES, FPS = 832, 480, 121, 24
SEED = 777001
GGUF_MODEL = "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf"
CKPT_43G = "ltx-2.3-22b-distilled.safetensors"
INGREDIENTS_LORA = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

STYLE = ("soft 3D Pixar-style cartoon render, big expressive eyes, rounded chubby "
         "features, smooth subsurface skin shading, glossy highlights, polished "
         "children's animation movie look, bright cheerful pastels, sunny and warm")
YUSUF = ("Yusuf, a cute chubby-cheeked 3D cartoon boy age 4, white knit prayer cap, "
         "short brown curly hair, big warm brown eyes, sky-blue kurta with white "
         "embroidery, friendly smile")
NEG = ("photorealistic, realistic skin, real human photo, uncanny, text, watermark, "
       "scary, dark, harsh lighting, deformed, extra limbs, extra fingers, blurry, "
       "low quality, asymmetric eyes, melted face, mutated hands")

SETTINGS = {
    "garden": "a sunny garden with blooming pink flowers, soft green grass and a small fountain",
    "room": "a cozy sunlit room with arched windows, cream walls and colorful cushions",
    "courtyard": "a peaceful mosque courtyard with ornate white arches and warm golden light",
}

# 12 scenes: shot, continuous-action, location, camera-move.
# Actions describe ONE clear continuous motion (LTX follows verbs, not adjectives).
SCENES = [
    ("medium shot", "slowly sits up and stretches both arms high above his head, then rubs his eyes", "room", "slow push in"),
    ("medium wide shot", "walks forward along the grass path, arms swinging, taking clear steps toward camera", "garden", "slow dolly back following him"),
    ("medium close-up", "raises both open hands up to chest level for dua, looking upward, lips moving", "courtyard", "gentle tilt up"),
    ("medium shot", "tips a tiny watering can, water streams onto a pink flower that sways", "garden", "static, slight zoom in"),
    ("medium wide shot", "turns a page of a colorful book, then looks up and smiles", "room", "slow pan right"),
    ("medium shot", "lifts one arm and waves hello broadly, head tilting", "courtyard", "static"),
    ("medium close-up", "throws head back laughing, shoulders bouncing", "garden", "slow push in"),
    ("medium wide shot", "reaches out and places a book onto a low shelf, arm extending fully", "room", "slow pan left"),
    ("medium shot", "claps hands together repeatedly in a steady rhythm", "courtyard", "static"),
    ("medium wide shot", "runs a few steps chasing a fluttering butterfly, arms reaching out", "garden", "camera pans following the butterfly"),
    ("medium close-up", "lifts a small snack toward his mouth, pausing to say Bismillah", "room", "static, slight zoom in"),
    ("medium shot", "waves goodbye with a big sweeping arm as warm sunset light glows", "courtyard", "slow dolly back"),
]

REF_SHEET_DESC = (
    "Reference sheet: a character model sheet of a cute chubby-cheeked 3D cartoon boy "
    "shown from multiple angles (front view, side view, three-quarter view), wearing a "
    "white knit prayer cap, short brown curly hair, big warm brown eyes, sky-blue kurta "
    "with white embroidery; plus panels of a sunny flower garden with a fountain, a cozy "
    "sunlit room with arched windows, and a white mosque courtyard with ornate arches, "
    "all in soft 3D Pixar-style cartoon render with bright cheerful pastel colors."
)

client = ComfyUIClient()


def scene_prompt(i, motion=False):
    shot, action, loc, cam = SCENES[i]
    base = f"{shot}, {YUSUF}, {action}, in {SETTINGS[loc]}, {STYLE}"
    if motion:
        base += (f", {cam}, smooth fluid natural motion, lively fluid character animation, "
                 f"clear body movement, cinematic camera work, 24fps")
    return base


def collect(history, dest, keys=("gifs", "videos", "images")):
    for _nid, nout in history.get("outputs", {}).items():
        for key in keys:
            for entry in nout.get(key, []):
                fn, sub = entry.get("filename", ""), entry.get("subfolder", "")
                if not fn:
                    continue
                src = COMFYUI_OUTPUT / sub / fn if sub else COMFYUI_OUTPUT / fn
                if src.exists():
                    shutil.copy2(str(src), str(dest))
                    return True
    return False


def build_ingredients_workflow(prompt, negative, ref_image, seed, num_frames=FRAMES,
                               width=W, height=H, fps=FPS):
    """API-format Ingredients workflow: 43GB distilled ckpt + IC-LoRA, video-only,
    8-step distilled manual sigmas — mirrors the official
    LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json minus audio branch."""
    wf = {}
    n = [0]
    def nid():
        n[0] += 1
        return str(n[0])

    ck = nid(); wf[ck] = {"class_type": "CheckpointLoaderSimple",
                          "inputs": {"ckpt_name": CKPT_43G}}
    ic = nid(); wf[ic] = {"class_type": "LTXICLoRALoaderModelOnly",
                          "inputs": {"model": [ck, 0], "lora_name": INGREDIENTS_LORA,
                                     "strength_model": 1.0}}
    clip = nid(); wf[clip] = {"class_type": "DualCLIPLoader", "inputs": {
        "clip_name1": "gemma_3_12B_it_fp4_mixed.safetensors",
        "clip_name2": "ltx-2.3_text_projection_bf16.safetensors", "type": "ltxv"}}
    pos = nid(); wf[pos] = {"class_type": "CLIPTextEncode",
                            "inputs": {"text": prompt, "clip": [clip, 0]}}
    neg = nid(); wf[neg] = {"class_type": "CLIPTextEncode",
                            "inputs": {"text": negative, "clip": [clip, 0]}}
    cond = nid(); wf[cond] = {"class_type": "LTXVConditioning", "inputs": {
        "positive": [pos, 0], "negative": [neg, 0], "frame_rate": float(fps)}}

    img = nid(); wf[img] = {"class_type": "LoadImage", "inputs": {"image": ref_image}}
    rep = nid(); wf[rep] = {"class_type": "RepeatImageBatch",
                            "inputs": {"image": [img, 0], "amount": num_frames}}
    pre = nid(); wf[pre] = {"class_type": "LTXVPreprocess",
                            "inputs": {"image": [img, 0], "img_compression": 18}}
    lat = nid(); wf[lat] = {"class_type": "EmptyLTXVLatentVideo", "inputs": {
        "width": width, "height": height, "length": num_frames, "batch_size": 1}}
    co = nid(); wf[co] = {"class_type": "LTXVImgToVideoConditionOnly", "inputs": {
        "vae": [ck, 2], "image": [pre, 0], "latent": [lat, 0],
        "strength": 1.0, "bypass": True}}
    guide = nid(); wf[guide] = {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
        "positive": [cond, 0], "negative": [cond, 1], "vae": [ck, 2],
        "latent": [co, 0], "image": [rep, 0], "frame_idx": 0, "strength": 1.0,
        "latent_downscale_factor": [ic, 1], "crop": "disabled",
        "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64}}

    noise = nid(); wf[noise] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    guider = nid(); wf[guider] = {"class_type": "CFGGuider", "inputs": {
        "model": [ic, 0], "positive": [guide, 0], "negative": [guide, 1], "cfg": 1.0}}
    samp_sel = nid(); wf[samp_sel] = {"class_type": "KSamplerSelect",
                                      "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}}
    sig = nid(); wf[sig] = {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}}
    samp = nid(); wf[samp] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": [noise, 0], "guider": [guider, 0], "sampler": [samp_sel, 0],
        "sigmas": [sig, 0], "latent_image": [guide, 2]}}
    crop = nid(); wf[crop] = {"class_type": "LTXVCropGuides", "inputs": {
        "positive": [guide, 0], "negative": [guide, 1], "latent": [samp, 0]}}
    dec = nid(); wf[dec] = {"class_type": "LTXVTiledVAEDecode", "inputs": {
        "vae": [ck, 2], "latents": [crop, 2], "horizontal_tiles": 2,
        "vertical_tiles": 2, "overlap": 6, "last_frame_fix": False}}
    save = nid(); wf[save] = {"class_type": "VHS_VideoCombine", "inputs": {
        "images": [dec, 0], "frame_rate": fps, "loop_count": 0,
        "filename_prefix": "bench_iclora", "format": "video/h264-mp4",
        "save_output": True, "pingpong": False}}
    return wf


def build_ltx43_img2vid_workflow(image_filename, prompt, negative, seed,
                                 num_frames=FRAMES, width=W, height=H, fps=FPS,
                                 strength=0.6):
    """img2vid from a still using the 43GB FULL distilled checkpoint (no IC-LoRA).
    Same task/settings as pipeline A's GGUF img2vid — only the model differs, so this
    isolates GGUF-quant vs full-precision. CheckpointLoaderSimple gives MODEL(0)+VAE(2);
    Gemma text encoder via DualCLIPLoader; distilled 8-step sigmas."""
    wf = {}
    n = [0]
    def nid():
        n[0] += 1
        return str(n[0])

    ck = nid(); wf[ck] = {"class_type": "CheckpointLoaderSimple",
                          "inputs": {"ckpt_name": CKPT_43G}}
    clip = nid(); wf[clip] = {"class_type": "DualCLIPLoader", "inputs": {
        "clip_name1": "gemma_3_12B_it_fp4_mixed.safetensors",
        "clip_name2": "ltx-2.3_text_projection_bf16.safetensors", "type": "ltxv"}}
    pos = nid(); wf[pos] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": [clip, 0]}}
    neg = nid(); wf[neg] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": [clip, 0]}}
    img = nid(); wf[img] = {"class_type": "LoadImage", "inputs": {"image": image_filename}}
    i2v = nid(); wf[i2v] = {"class_type": "LTXVImgToVideo", "inputs": {
        "positive": [pos, 0], "negative": [neg, 0], "vae": [ck, 2],
        "image": [img, 0], "width": width, "height": height,
        "length": num_frames, "batch_size": 1, "strength": strength}}
    noise = nid(); wf[noise] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    guider = nid(); wf[guider] = {"class_type": "CFGGuider", "inputs": {
        "model": [ck, 0], "positive": [i2v, 0], "negative": [i2v, 1], "cfg": 1.0}}
    ssel = nid(); wf[ssel] = {"class_type": "KSamplerSelect",
                              "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}}
    sig = nid(); wf[sig] = {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}}
    samp = nid(); wf[samp] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": [noise, 0], "guider": [guider, 0], "sampler": [ssel, 0],
        "sigmas": [sig, 0], "latent_image": [i2v, 2]}}
    dec = nid(); wf[dec] = {"class_type": "VAEDecode", "inputs": {"samples": [samp, 0], "vae": [ck, 2]}}
    save = nid(); wf[save] = {"class_type": "VHS_VideoCombine", "inputs": {
        "images": [dec, 0], "frame_rate": fps, "loop_count": 0,
        "filename_prefix": "bench_c43", "format": "video/h264-mp4",
        "save_output": True, "pingpong": False}}
    return wf


def pipeline_c(cp):
    """Same Z-Image stills as A, img2vid with the 43GB full checkpoint."""
    stats = cp.setdefault("C", {"clips": {}, "ups": {}})
    (OUT / "C").mkdir(exist_ok=True)
    client.free_vram(); time.sleep(2)
    for i in range(len(SCENES)):
        dest = OUT / "C" / f"clip_{i:02d}.mp4"
        if str(i) in stats["clips"]:
            continue
        still = OUT / "A" / f"still_{i:02d}.png"   # reuse A's Z-Image stills
        if not still.exists():
            stats["clips"][str(i)] = None
            continue
        shutil.copy2(still, COMFY_INPUT / f"bench_c_{i:02d}.png")
        wf = build_ltx43_img2vid_workflow(
            image_filename=f"bench_c_{i:02d}.png",
            prompt=scene_prompt(i, motion=True),
            negative=NEG + ", static, frozen, motionless, still image, no movement, stiff, jerky, warping, morphing, flickering",
            seed=SEED + i)
        try:
            ok, dt = timed_submit(wf, dest, 2400)
        except Exception as e:
            log.error(f"[C clip {i:02d}] ERROR: {e}")
            ok, dt = False, -1
        stats["clips"][str(i)] = round(dt, 1) if ok else None
        save_cp(cp)
        log.info(f"[C clip {i:02d}] {'OK' if ok else 'FAIL'} {dt:.0f}s")
    upscale_phase(cp, "C")


def make_graph(cp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def clip_times(k):
        return [v for v in cp.get(k, {}).get("clips", {}).values() if v]

    series = [("A: GGUF img2vid", clip_times("A")),
              ("B: 43GB + IC-LoRA", clip_times("B")),
              ("C: 43GB img2vid", clip_times("C"))]
    series = [(n, v) for n, v in series if v]
    if not series:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#4C9AFF", "#FF8B6B", "#57D9A3"]
    # per-clip line
    for (name, vals), c in zip(series, colors):
        ax1.plot(range(len(vals)), vals, marker="o", label=name, color=c)
    ax1.set_title("Per-clip generation time (5s clip @832x480)")
    ax1.set_xlabel("clip #"); ax1.set_ylabel("seconds"); ax1.legend(); ax1.grid(alpha=0.3)
    # avg bar
    names = [n for n, _ in series]
    avgs = [sum(v) / len(v) for _, v in series]
    bars = ax2.bar(names, avgs, color=colors[:len(series)])
    ax2.set_title("Average time per clip"); ax2.set_ylabel("seconds")
    for b, a in zip(bars, avgs):
        ax2.text(b.get_x() + b.get_width() / 2, a, f"{a:.0f}s", ha="center", va="bottom")
    plt.tight_layout()
    out = OUT / "compare_graph.png"
    plt.savefig(out, dpi=120)
    log.info(f"[Graph] -> {out}")
    return out


def load_cp():
    cp = OUT / "checkpoint.json"
    return json.loads(cp.read_text()) if cp.exists() else {}


def save_cp(d):
    (OUT / "checkpoint.json").write_text(json.dumps(d, indent=2))


def timed_submit(wf, dest, timeout, keys=("gifs", "videos", "images")):
    t0 = time.time()
    pid = client.submit(wf)
    hist = client.wait_for_completion(pid, timeout=timeout, poll=2.0)
    ok = collect(hist, dest, keys)
    return ok, time.time() - t0


def make_ref_sheet(cp):
    if cp.get("ref_sheet"):
        return cp
    log.info("[RefSheet] Generating Yusuf reference sheet (SDXL)...")
    client.free_vram(); time.sleep(2)
    prompt = (
        "character reference sheet, model sheet, turnaround of the SAME cute chubby-cheeked "
        "3D cartoon boy shown three times: front view, side view, three-quarter view, "
        f"{YUSUF}, standing in T-pose and neutral pose on a plain cream background, "
        "small corner panels showing: a sunny flower garden with fountain, a cozy sunlit "
        f"room with arched windows, a white mosque courtyard with ornate arches, {STYLE}, "
        "clean layout, organized grid"
    )
    wf = build_sdxl_workflow(
        prompt=prompt, negative_prompt=NEG + ", different characters, inconsistent face",
        width=1344, height=768, steps=35, cfg=7.0, seed=SEED,
        ckpt_name="sd_xl_base_1.0.safetensors",
        loras=[("samaritan-3d-cartoon-lora.safetensors", 0.9)],
        hires=True, hires_scale=1.5, hires_denoise=0.45, hires_steps=18,
    )
    dest = OUT / "ref_sheet_raw.png"
    ok, dt = timed_submit(wf, dest, 600, keys=("images",))
    if not ok:
        raise RuntimeError("ref sheet generation failed")
    from PIL import Image
    im = Image.open(dest).convert("RGB").resize((W, H), Image.LANCZOS)
    sheet = OUT / "ref_sheet.png"
    im.save(sheet)
    shutil.copy2(sheet, COMFY_INPUT / "bench_ref_sheet.png")
    cp["ref_sheet"] = True
    cp["t_ref_sheet"] = dt
    save_cp(cp)
    log.info(f"[RefSheet] done in {dt:.0f}s")
    return cp


def pipeline_a(cp):
    stats = cp.setdefault("A", {"stills": {}, "clips": {}, "ups": {}})
    (OUT / "A").mkdir(exist_ok=True)

    # stills
    client.free_vram(); time.sleep(2)
    for i in range(len(SCENES)):
        dest = OUT / "A" / f"still_{i:02d}.png"
        if str(i) in stats["stills"]:
            continue
        wf = build_zimage_workflow(
            prompt=scene_prompt(i), width=W, height=H,
            steps=8, cfg=1.0, seed=SEED + i)
        ok, dt = timed_submit(wf, dest, 600, keys=("images",))
        stats["stills"][str(i)] = round(dt, 1) if ok else None
        save_cp(cp)
        log.info(f"[A still {i:02d}] {'OK' if ok else 'FAIL'} {dt:.0f}s")

    # clips
    client.free_vram(); time.sleep(2)
    for i in range(len(SCENES)):
        dest = OUT / "A" / f"clip_{i:02d}.mp4"
        if str(i) in stats["clips"]:
            continue
        still = OUT / "A" / f"still_{i:02d}.png"
        if not still.exists():
            stats["clips"][str(i)] = None
            continue
        shutil.copy2(still, COMFY_INPUT / f"bench_a_{i:02d}.png")
        wf = build_ltx_img2vid_workflow(
            model_filename=GGUF_MODEL, image_filename=f"bench_a_{i:02d}.png",
            prompt=scene_prompt(i, motion=True),
            negative_prompt=NEG + ", static, frozen, motionless, still image, no movement, stiff, jerky, warping, morphing, flickering",
            width=W, height=H, num_frames=FRAMES, steps=8, cfg=1.0,
            seed=SEED + i, fps=FPS, strength=0.6)
        ok, dt = timed_submit(wf, dest, 1200)
        stats["clips"][str(i)] = round(dt, 1) if ok else None
        save_cp(cp)
        log.info(f"[A clip {i:02d}] {'OK' if ok else 'FAIL'} {dt:.0f}s")

    upscale_phase(cp, "A")


def pipeline_b(cp):
    stats = cp.setdefault("B", {"clips": {}, "ups": {}})
    (OUT / "B").mkdir(exist_ok=True)

    client.free_vram(); time.sleep(2)
    for i in range(len(SCENES)):
        dest = OUT / "B" / f"clip_{i:02d}.mp4"
        if str(i) in stats["clips"]:
            continue
        prompt = (REF_SHEET_DESC + " Generated video: " + scene_prompt(i, motion=True))
        wf = build_ingredients_workflow(
            prompt=prompt, negative=NEG + ", static, frozen, motionless, still image, no movement, stiff, jerky, warping, morphing, flickering",
            ref_image="bench_ref_sheet.png", seed=SEED + i)
        try:
            ok, dt = timed_submit(wf, dest, 2400)
        except Exception as e:
            log.error(f"[B clip {i:02d}] ERROR: {e}")
            ok, dt = False, -1
        stats["clips"][str(i)] = round(dt, 1) if ok else None
        save_cp(cp)
        log.info(f"[B clip {i:02d}] {'OK' if ok else 'FAIL'} {dt:.0f}s")

    upscale_phase(cp, "B")


def upscale_phase(cp, which):
    stats = cp[which]
    client.free_vram(); time.sleep(2)
    for i in range(len(SCENES)):
        src = OUT / which / f"clip_{i:02d}.mp4"
        dest = OUT / which / f"clip_{i:02d}_hd.mp4"
        if str(i) in stats["ups"] or not src.exists():
            continue
        name = f"bench_{which}_up_{i:02d}.mp4"
        shutil.copy2(src, COMFY_INPUT / name)
        wf = build_esrgan_video_upscale_workflow(
            video_filename=name, target_width=1920, target_height=1080,
            fps=float(FPS), model_name="4x-UltraSharp.pth")
        try:
            ok, dt = timed_submit(wf, dest, 900)
        except Exception as e:
            log.error(f"[{which} up {i:02d}] ERROR: {e}")
            ok, dt = False, -1
        stats["ups"][str(i)] = round(dt, 1) if ok else None
        save_cp(cp)
        log.info(f"[{which} up {i:02d}] {'OK' if ok else 'FAIL'} {dt:.0f}s")


def assemble(which):
    clips = sorted((OUT / which).glob("clip_*_hd.mp4"))
    if not clips:
        clips = sorted(p for p in (OUT / which).glob("clip_*.mp4") if "_hd" not in p.name)
    if not clips:
        return None
    lst = OUT / f"concat_{which}.txt"
    lst.write_text("\n".join(f"file '{c.as_posix()}'" for c in clips))
    final = OUT / f"benchmark_{which}_final.mp4"
    subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(final)], capture_output=True, timeout=900)
    return final if final.exists() else None


def report(cp):
    def agg(d):
        vals = [v for v in d.values() if v]
        return (sum(vals), len(vals), (sum(vals) / len(vals)) if vals else 0)

    lines = ["# Benchmark: GGUF (A) vs 43GB + IC-LoRA Ingredients (B)", ""]
    a, b = cp.get("A", {}), cp.get("B", {})
    rows = []
    if a:
        s_t, s_n, s_avg = agg(a.get("stills", {}))
        c_t, c_n, c_avg = agg(a.get("clips", {}))
        u_t, u_n, u_avg = agg(a.get("ups", {}))
        rows.append(("A stills (SDXL)", s_n, s_avg, s_t))
        rows.append(("A clips (GGUF i2v)", c_n, c_avg, c_t))
        rows.append(("A upscale", u_n, u_avg, u_t))
        rows.append(("A TOTAL", "-", "-", s_t + c_t + u_t))
    if b:
        c_t, c_n, c_avg = agg(b.get("clips", {}))
        u_t, u_n, u_avg = agg(b.get("ups", {}))
        rows.append(("B ref sheet", 1, cp.get("t_ref_sheet", 0), cp.get("t_ref_sheet", 0)))
        rows.append(("B clips (Ingredients)", c_n, c_avg, c_t))
        rows.append(("B upscale", u_n, u_avg, u_t))
        rows.append(("B TOTAL", "-", "-", cp.get("t_ref_sheet", 0) + c_t + u_t))
    lines.append("| Phase | ok | avg (s) | total (s) |")
    lines.append("|-------|----|---------|-----------|")
    for name, n, avg, tot in rows:
        avg_s = f"{avg:.0f}" if isinstance(avg, float) else avg
        tot_s = f"{tot:.0f}" if isinstance(tot, (int, float)) else tot
        lines.append(f"| {name} | {n} | {avg_s} | {tot_s} |")
    lines.append("")
    lines.append("Per-clip seconds A: " + json.dumps(a.get("clips", {})))
    lines.append("")
    lines.append("Per-clip seconds B: " + json.dumps(b.get("clips", {})))
    md = "\n".join(lines)
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")
    print("\n" + md)


def main():
    if not client.wait_ready(600):
        raise RuntimeError("ComfyUI not reachable")
    cp = load_cp()
    t0 = time.time()
    cp = make_ref_sheet(cp)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only in (None, "A"):
        pipeline_a(cp)
    if only in (None, "B"):
        pipeline_b(cp)
    if only in (None, "C"):
        pipeline_c(cp)
    fa = assemble("A")
    fb = assemble("B")
    fc = assemble("C")
    cp["final_A"], cp["final_B"], cp["final_C"] = str(fa), str(fb), str(fc)
    save_cp(cp)
    report(cp)
    try:
        make_graph(cp)
    except Exception as e:
        log.error(f"[Graph] failed: {e}")
    log.info(f"ALL DONE in {(time.time()-t0)/60:.1f} min")
    log.info(f"A: {fa}")
    log.info(f"B: {fb}")
    log.info(f"C: {fc}")


if __name__ == "__main__":
    main()
