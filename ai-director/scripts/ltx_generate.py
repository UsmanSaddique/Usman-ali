"""
AI Director — LTX Video Generation Subprocess Wrapper

This script is called by VideoGenService to generate video using LTX pipelines.
It runs with LTX Desktop's own Python environment.

Usage:
    python ltx_generate.py --args-json '{"mode":"txt2vid","prompt":"...", ...}'
"""
import sys
import json
import argparse
import time
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-json", required=True)
    args = parser.parse_args()

    params = json.loads(args.args_json)

    mode = params.get("mode", "txt2vid")
    prompt = params["prompt"]
    width = params.get("width", 768)
    height = params.get("height", 512)
    num_frames = params.get("num_frames", 121)
    fps = params.get("fps", 24)
    steps = params.get("steps", 8)
    seed = params.get("seed", -1)
    output_path = params["output_path"]
    model_path = params["model_path"]
    gemma_root = params["gemma_root"]
    spatial_upsampler_path = params["spatial_upsampler_path"]
    loras = params.get("loras", [])  # list of dicts {"path": str, "weight": float}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    print(f"[LTX] Mode: {mode}")
    print(f"[LTX] Prompt: {prompt[:100]}...")
    print(f"[LTX] Resolution: {width}x{height}, frames: {num_frames}, fps: {fps}")
    print(f"[LTX] Steps: {steps}, Seed: {seed}")
    print(f"[LTX] Model: {model_path}")

    t0 = time.time()

    try:
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_core.loader import LoraPathStrengthAndSDOps
        from ltx_core.loader.primitives import SDOps
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    except ImportError as e:
        print(f"[LTX] ERROR: Failed to import ltx_pipelines: {e}", file=sys.stderr)
        sys.exit(1)

    print("[LTX] Building pipeline...")
    
    lora_objs = []
    for lora in loras:
        lora_objs.append(LoraPathStrengthAndSDOps(
            path=lora["path"],
            strength=lora["weight"],
            sd_ops=SDOps(name="default")
        ))

    pipeline = DistilledPipeline(
        distilled_checkpoint_path=model_path,
        gemma_root=gemma_root,
        spatial_upsampler_path=spatial_upsampler_path,
        loras=lora_objs,
    )

    print("[LTX] Generating video...")

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    video, audio = pipeline(
        prompt=prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=fps,
        images=[], # img2vid uses conditioning images
        tiling_config=tiling_config,
    )

    print(f"[LTX] Encoding video to {output_path}...")
    encode_video(
        video=video,
        fps=fps,
        audio=None,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )

    elapsed = time.time() - t0
    print(f"[LTX] Done in {elapsed:.1f}s -> {output_path}")

    result = {
        "status": "success",
        "output_path": output_path,
        "seed": seed,
        "generation_time": elapsed,
    }
    print(f"__RESULT_JSON__={json.dumps(result)}")

if __name__ == "__main__":
    main()
