"""
AI Director — LTX Video Generation Subprocess Wrapper

This script is called by VideoGenService when LTX modules can't be imported
into the main Python process. It runs with LTX Desktop's own Python environment.

Usage:
    python ltx_generate.py --args-json '{"mode":"txt2vid","prompt":"...", ...}'

Expected to be run with:
    C:\\Users\\PC\\AppData\\Local\\LTXDesktop\\python\\python.exe scripts/ltx_generate.py ...
"""
import sys
import json
import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="LTX Video Generation")
    parser.add_argument("--args-json", required=True, help="JSON string with generation parameters")
    args = parser.parse_args()

    params = json.loads(args.args_json)

    mode = params["mode"]               # txt2vid or img2vid
    prompt = params["prompt"]
    negative_prompt = params.get("negative_prompt", "")
    width = params.get("width", 768)
    height = params.get("height", 512)
    num_frames = params.get("num_frames", 121)
    fps = params.get("fps", 24)
    steps = params.get("steps", 8)
    cfg_scale = params.get("cfg_scale", 1.0)
    seed = params.get("seed", -1)
    output_path = params["output_path"]
    model_path = params["model_path"]
    use_fp8 = params.get("use_fp8", True)
    image_path = params.get("image_path")  # for img2vid

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    import torch
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    print(f"[LTX] Mode: {mode}")
    print(f"[LTX] Prompt: {prompt[:100]}...")
    print(f"[LTX] Resolution: {width}x{height}, frames: {num_frames}, fps: {fps}")
    print(f"[LTX] Steps: {steps}, CFG: {cfg_scale}, Seed: {seed}")
    print(f"[LTX] Model: {model_path}")
    print(f"[LTX] FP8: {use_fp8}")

    t0 = time.time()

    # ── Load LTX Pipeline ──────────────────────────────────────────────
    try:
        from ltx_video.pipelines.distilled_pipeline import DistilledPipeline
        from ltx_video.utils.model_builder import SingleGPUModelBuilder
    except ImportError:
        # Try adding LTX Desktop path
        ltx_desktop = Path(r"C:\Users\PC\AppData\Local\LTXDesktop")
        for candidate in [ltx_desktop / "ltx_video", ltx_desktop]:
            if candidate.exists():
                sys.path.insert(0, str(candidate))
        from ltx_video.pipelines.distilled_pipeline import DistilledPipeline
        from ltx_video.utils.model_builder import SingleGPUModelBuilder

    print("[LTX] Building model...")
    builder = SingleGPUModelBuilder(
        model_path=model_path,
        use_fp8=use_fp8,
    )
    pipeline = DistilledPipeline(builder)

    # ── Generate ───────────────────────────────────────────────────────
    print("[LTX] Generating video...")

    gen_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "guidance_scale": cfg_scale,
        "seed": seed,
        "output_path": output_path,
    }

    if mode == "img2vid" and image_path:
        gen_kwargs["image_path"] = image_path
        print(f"[LTX] Input image: {image_path}")

    with torch.inference_mode():
        pipeline.generate(**gen_kwargs)

    elapsed = time.time() - t0
    print(f"[LTX] Done in {elapsed:.1f}s → {output_path}")

    # ── Verify output exists ───────────────────────────────────────────
    if not Path(output_path).exists():
        # Some LTX versions save with different naming
        # Check for auto-generated output in default location
        import glob
        ltx_output = Path(r"D:/assets_generated/ltx_output")
        recent = sorted(ltx_output.glob("*.mp4"), key=os.path.getmtime, reverse=True)
        if recent:
            import shutil
            shutil.copy2(str(recent[0]), output_path)
            print(f"[LTX] Copied from default output: {recent[0]}")
        else:
            print("[LTX] WARNING: Output file not found!", file=sys.stderr)
            sys.exit(1)

    # Output result as JSON on stdout for parent process
    result = {
        "status": "success",
        "output_path": output_path,
        "seed": seed,
        "generation_time": elapsed,
    }
    print(f"\n__RESULT_JSON__={json.dumps(result)}")


if __name__ == "__main__":
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
