"""
20-Second Sample — 5 clips x 4s
================================
High-quality demo of the full stack:
  ACE-Step 1.5 XL SFT music (20s) -> Z-Image-Turbo stills (5) ->
  LTX img2vid clips (5 x 4s) -> ESRGAN 1080p upscale -> FFmpeg assembly
Registers the result as a RENDERED project in the app DB.

Run with ComfyUI embedded python:
  C:\\ComfyUI_windows_portable_nvidia_cu126\\ComfyUI_windows_portable\\python_embeded\\python.exe sample_20s_5x4.py
"""
import json
import logging
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from app.services.comfyui_client import (
    ComfyUIClient,
    COMFYUI_OUTPUT,
    build_acestep15xl_sft_workflow,
    build_zimage_workflow,
    build_ltx_img2vid_workflow,
    build_esrgan_video_upscale_workflow,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sample20s")

COMFY_INPUT = Path(
    r"C:\ComfyUI_windows_portable_nvidia_cu126"
    r"\ComfyUI_windows_portable\ComfyUI\input"
)
PROJECT_DIR = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\assets_generated\sample_20s_5x4")

SLUG = "sample_20s_5x4"
TITLE = "Bismillah Garden — 20s Sample (5x4s)"
SEED = 424242

CLIP_SECONDS = 4.0
NUM_CLIPS = 5
FPS = 24
NUM_FRAMES = 97           # 8n+1; 97f @ 24fps = 4.04s
VIDEO_W, VIDEO_H = 832, 480
STILL_W, STILL_H = 1248, 720   # 16:9-ish at Z-Image's native ~1MP
UPSCALE_W, UPSCALE_H = 1920, 1080

ART_STYLE = ("soft 3D Pixar-style cartoon render, big expressive eyes, rounded chubby "
             "features, smooth subsurface skin shading, glossy highlights, polished "
             "children's animation movie look")
PALETTE = "bright cheerful pastels, sky blue, mint, pink, cream, soft gold, sunny and warm"
QUALITY = ("highly detailed, cinematic, soft volumetric lighting, depth of field, "
           "warm rim light, gentle bokeh background, beautifully rendered")
YUSUF = ("cute chubby-cheeked 3D cartoon boy Yusuf, 4 years old, white knit prayer cap "
         "(taqiyah), short brown curly hair, big warm brown eyes, sky-blue kurta with "
         "white embroidery, friendly smile")
MARYAM = ("cute 3D cartoon girl Maryam, 4 years old, soft mint-teal hijab, big bright "
          "green eyes, rosy cheeks, gentle teal dress, sweet smile")
NEG = ("photorealistic, realistic skin, real human photo, uncanny, text, watermark, "
       "scary, dark, harsh lighting, deformed, extra limbs, extra fingers, blurry, "
       "low quality, extreme close-up, face only, cropped frame")

SCENES = [
    {
        "prompt": (f"wide establishing shot, golden sunrise over a charming pastel old-town "
                   f"street with a soft mosque dome in the background, {YUSUF} walking happily "
                   f"along the cobblestone path, warm light rays through morning mist, "
                   f"{ART_STYLE}, {PALETTE}, {QUALITY}"),
        "motion": "the boy walking forward with a gentle bounce, morning mist drifting, subtle camera push-in",
    },
    {
        "prompt": (f"medium wide shot, {YUSUF} in a lush garden with blooming pink and white "
                   f"flowers, reaching up gently toward a small friendly bluebird on a branch, "
                   f"soft petals floating in the air, {ART_STYLE}, {PALETTE}, {QUALITY}"),
        "motion": "the boy reaching up slowly, the bird fluttering its wings, petals drifting in a gentle breeze",
    },
    {
        "prompt": (f"wide shot, {YUSUF} and {MARYAM} sitting together by an ornate courtyard "
                   f"fountain sharing a bowl of dates, sparkling water drops, sunny mosque "
                   f"courtyard with arches, {ART_STYLE}, {PALETTE}, {QUALITY}"),
        "motion": "the children smiling and passing the bowl to each other, fountain water gently splashing",
    },
    {
        "prompt": (f"medium shot, {YUSUF} kneeling on a soft red prayer rug in a cozy sunlit "
                   f"room with an arched window, hands raised in dua, dust motes floating in "
                   f"a warm light beam, peaceful expression, {ART_STYLE}, {PALETTE}, {QUALITY}"),
        "motion": "the boy raising his small hands slowly in prayer, light rays shimmering, dust motes drifting",
    },
    {
        "prompt": (f"wide shot, {YUSUF} and {MARYAM} standing on a grassy hill waving happily "
                   f"at a beautiful golden sunset sky with soft pink clouds, mosque silhouette "
                   f"on the horizon, fireflies beginning to glow, {ART_STYLE}, {PALETTE}, {QUALITY}"),
        "motion": "the children waving cheerfully, clouds drifting slowly, fireflies twinkling, subtle camera pull-back",
    },
]

MUSIC_STYLE = ("children's song, kids music, cheerful female lead vocal, children choir "
               "harmony, ukulele, glockenspiel, soft piano, gentle hand claps, warm "
               "strings, happy, uplifting, sing-along, bouncy, professional studio mix")
MUSIC_LYRICS = """[chorus]
Bismillah, Bismillah, say it with a smile,
Bismillah, Bismillah, every little while.
In the name of Allah, kind and true,
Bismillah, Bismillah, He takes care of you!
"""

client = ComfyUIClient()


def load_checkpoint() -> dict:
    cp = PROJECT_DIR / "checkpoint.json"
    return json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}


def save_checkpoint(data: dict):
    (PROJECT_DIR / "checkpoint.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8")


def free_vram():
    client.free_vram()
    time.sleep(3)


def collect_output(history: dict, dest: Path, keys: tuple) -> bool:
    for _nid, nout in history.get("outputs", {}).items():
        for key in keys:
            for entry in nout.get(key, []):
                fn = entry.get("filename", "")
                sub = entry.get("subfolder", "")
                if not fn:
                    continue
                src = COMFYUI_OUTPUT / sub / fn if sub else COMFYUI_OUTPUT / fn
                if src.exists():
                    shutil.copy2(str(src), str(dest))
                    return True
    return False


def phase_music(cp: dict) -> str:
    dest = PROJECT_DIR / "music.flac"
    if cp.get("music") and dest.exists():
        log.info("[Music] resuming — already done")
        return str(dest)

    log.info("[Music] ACE-Step 1.5 XL SFT, 20s, 150 steps (high quality)...")
    free_vram()
    wf = build_acestep15xl_sft_workflow(
        style_tags=MUSIC_STYLE,
        lyrics=MUSIC_LYRICS,
        seconds=22.0,           # slight overrun so assembly never runs dry
        seed=SEED,
        steps=150,
        cfg=1.0,
        bpm=100,
        language="en",
        keyscale="C major",
    )
    pid = client.submit(wf)
    log.info(f"[Music] prompt_id={pid}")
    history = client.wait_for_completion(pid, timeout=1800, poll=3.0)
    if not collect_output(history, dest, ("audio",)):
        raise RuntimeError("Music generation produced no audio")
    log.info(f"[Music] OK -> {dest}")
    cp["music"] = True
    save_checkpoint(cp)
    return str(dest)


def phase_stills(cp: dict) -> list[str]:
    images_dir = PROJECT_DIR / "images"
    images_dir.mkdir(exist_ok=True)
    if cp.get("stills"):
        paths = [str(images_dir / f"scene_{i}.png") for i in range(NUM_CLIPS)]
        if all(Path(p).exists() for p in paths):
            log.info("[Stills] resuming — already done")
            return paths

    log.info(f"[Stills] Z-Image-Turbo x{NUM_CLIPS} at {STILL_W}x{STILL_H}...")
    free_vram()
    paths = []
    for i, scene in enumerate(SCENES):
        dest = images_dir / f"scene_{i}.png"
        if dest.exists():
            paths.append(str(dest))
            log.info(f"  [{i}] exists, skipping")
            continue
        wf = build_zimage_workflow(
            prompt=scene["prompt"],
            width=STILL_W,
            height=STILL_H,
            steps=9,
            cfg=1.0,
            seed=SEED + i,
            output_prefix=f"s20_{SLUG}_{i}",
        )
        pid = client.submit(wf)
        history = client.wait_for_completion(pid, timeout=600, poll=1.5)
        if not collect_output(history, dest, ("images",)):
            raise RuntimeError(f"Still {i} produced no image")
        log.info(f"  [{i}] OK -> {dest.name}")
        paths.append(str(dest))

    cp["stills"] = True
    save_checkpoint(cp)
    return paths


def phase_clips(cp: dict, image_paths: list[str]) -> list[str]:
    clips_dir = PROJECT_DIR / "clips"
    clips_dir.mkdir(exist_ok=True)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    if cp.get("clips"):
        paths = [str(clips_dir / f"scene_{i}.mp4") for i in range(NUM_CLIPS)]
        if all(Path(p).exists() for p in paths):
            log.info("[Clips] resuming — already done")
            return paths

    model = "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf"
    log.info(f"[Clips] LTX img2vid x{NUM_CLIPS}, {NUM_FRAMES}f @ {FPS}fps ({VIDEO_W}x{VIDEO_H})...")
    free_vram()
    paths = []
    for i, (scene, img) in enumerate(zip(SCENES, image_paths)):
        dest = clips_dir / f"scene_{i}.mp4"
        if dest.exists():
            paths.append(str(dest))
            log.info(f"  [{i}] exists, skipping")
            continue
        img_name = f"s20_{SLUG}_{i}.png"
        shutil.copy2(img, str(COMFY_INPUT / img_name))
        wf = build_ltx_img2vid_workflow(
            model_filename=model,
            image_filename=img_name,
            prompt=f"{scene['prompt']}, {scene['motion']}",
            negative_prompt=f"{NEG}, static, still image, frozen, motionless, no movement",
            width=VIDEO_W,
            height=VIDEO_H,
            num_frames=NUM_FRAMES,
            steps=10,
            cfg=1.0,
            seed=SEED + i,
            fps=FPS,
            strength=0.6,
            output_prefix=f"s20clip_{i}",
        )
        pid = client.submit(wf)
        t0 = time.time()
        history = client.wait_for_completion(pid, timeout=900, poll=2.0)
        if not collect_output(history, dest, ("gifs", "videos", "images")):
            raise RuntimeError(f"Clip {i} produced no video")
        log.info(f"  [{i}] OK ({time.time()-t0:.0f}s) -> {dest.name}")
        paths.append(str(dest))

    cp["clips"] = True
    save_checkpoint(cp)
    return paths


def phase_upscale(cp: dict, clip_paths: list[str]) -> list[str]:
    hd_dir = PROJECT_DIR / "clips_hd"
    hd_dir.mkdir(exist_ok=True)
    if cp.get("upscale"):
        paths = [str(hd_dir / f"scene_{i}_hd.mp4") for i in range(NUM_CLIPS)]
        if all(Path(p).exists() for p in paths):
            log.info("[Upscale] resuming — already done")
            return paths

    log.info(f"[Upscale] ESRGAN anime x{NUM_CLIPS} -> {UPSCALE_W}x{UPSCALE_H}...")
    free_vram()
    paths = []
    for i, clip in enumerate(clip_paths):
        dest = hd_dir / f"scene_{i}_hd.mp4"
        if dest.exists():
            paths.append(str(dest))
            continue
        vid_name = f"s20up_{SLUG}_{i}.mp4"
        shutil.copy2(clip, str(COMFY_INPUT / vid_name))
        wf = build_esrgan_video_upscale_workflow(
            video_filename=vid_name,
            target_width=UPSCALE_W,
            target_height=UPSCALE_H,
            fps=float(FPS),
            model_name="RealESRGAN_x4plus_anime_6B.pth",  # clean cartoon edges
            output_prefix=f"s20hd_{i}",
        )
        try:
            pid = client.submit(wf)
            history = client.wait_for_completion(pid, timeout=900, poll=2.0)
            if collect_output(history, dest, ("gifs", "videos", "images")):
                log.info(f"  [{i}] OK -> {dest.name}")
                paths.append(str(dest))
            else:
                log.warning(f"  [{i}] no upscaled output, using raw clip")
                paths.append(clip)
        except Exception as e:
            log.warning(f"  [{i}] upscale failed ({e}), using raw clip")
            paths.append(clip)

    cp["upscale"] = True
    save_checkpoint(cp)
    return paths


def phase_assemble(music_path: str, clip_paths: list[str]) -> str:
    from app.services.assembler import AssemblerService, ClipEntry
    from app.config import settings

    log.info("[Assemble] 5 x 4s clips + music -> final 20s video")
    assembler = AssemblerService(settings)
    clips = [ClipEntry(path=p, duration=CLIP_SECONDS,
                       transition_in="crossfade", transition_out="crossfade")
             for p in clip_paths if p and Path(p).exists()]
    if not clips:
        raise RuntimeError("No clips for assembly")
    clips[0].transition_in = "fade_from_black"

    output = str(PROJECT_DIR / "final_render.mp4")
    result = assembler.assemble(
        clips=clips,
        output_path=output,
        narration_path=None,
        music_path=music_path,
        music_volume=1.0,
        transition_duration=0.25,
        resolution="1080p",
    )
    log.info(f"[Assemble] DONE: {result.total_duration:.1f}s, {result.file_size_mb:.1f}MB")
    return output


def register_project(output: str):
    """Insert the sample as a RENDERED project so it appears in the app UI."""
    from app.database import init_db, get_session, Channel, Project, ProjectStatus
    init_db()
    session = get_session()
    try:
        channel = session.query(Channel).filter_by(slug="little-muslim-nation").first()
        if not channel:
            log.warning("[DB] channel little-muslim-nation not found, skipping registration")
            return
        existing = session.query(Project).filter_by(title=TITLE).first()
        if existing:
            existing.status = ProjectStatus.RENDERED
            existing.output_path = output
            existing.total_scenes = NUM_CLIPS
            existing.completed_scenes = NUM_CLIPS
        else:
            session.add(Project(
                title=TITLE,
                channel_id=channel.id,
                duration_target=20,
                context="20s sample: 5 clips x 4s. ACE-Step 1.5 XL music, Z-Image stills, LTX img2vid.",
                status=ProjectStatus.RENDERED,
                total_scenes=NUM_CLIPS,
                completed_scenes=NUM_CLIPS,
                output_path=output,
            ))
        session.commit()
        log.info("[DB] project registered in dashboard")
    finally:
        session.close()


def main():
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log.info("=" * 60)
    log.info(f"  {TITLE}")
    log.info(f"  Output: {PROJECT_DIR}")
    log.info("=" * 60)

    if not client.wait_ready(timeout=900):
        raise RuntimeError("ComfyUI not reachable at 127.0.0.1:8188")

    cp = load_checkpoint()
    music = phase_music(cp)
    stills = phase_stills(cp)
    clips = phase_clips(cp, stills)
    hd = phase_upscale(cp, clips)
    output = phase_assemble(music, hd)
    cp["output"] = output
    save_checkpoint(cp)
    register_project(output)

    log.info(f"\nCOMPLETE in {(time.time()-t0)/60:.1f} min -> {output}")


if __name__ == "__main__":
    main()
