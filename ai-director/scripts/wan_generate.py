"""
AI Director — Wan 2.2 Video Generation Subprocess Wrapper
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
    negative_prompt = params.get("negative_prompt", "")
    width = params.get("width", 832)
    height = params.get("height", 480)
    num_frames = params.get("num_frames", 81)
    fps = params.get("fps", 16)
    steps = params.get("steps", 30)
    cfg_scale = params.get("cfg_scale", 5.0)
    seed = params.get("seed", -1)
    output_path = params["output_path"]
    model_path = params["model_path"]
    loras = params.get("loras", [])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    print(f"[Wan] Mode: {mode}")
    print(f"[Wan] Prompt: {prompt[:100]}...")
    print(f"[Wan] Resolution: {width}x{height}, frames: {num_frames}, fps: {fps}")
    print(f"[Wan] Steps: {steps}, CFG: {cfg_scale}, Seed: {seed}")
    print(f"[Wan] Model: {model_path}")

    t0 = time.time()

    try:
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video
    except ImportError as e:
        print(f"[Wan] ERROR: Failed to import diffusers.WanPipeline: {e}", file=sys.stderr)
        sys.exit(1)

    print("[Wan] Building pipeline...")
    dtype = torch.float16
    if "fp8" in model_path.lower():
        dtype = torch.float8_e4m3fn
        
    pipe = WanPipeline.from_pretrained(
        model_path, 
        torch_dtype=dtype,
        use_safetensors=True
    ).to("cuda")

    for lora in loras:
        print(f"[Wan] Loading LoRA: {lora['path']} (weight {lora['weight']})")
        pipe.load_lora_weights(lora["path"], adapter_name=Path(lora["path"]).stem)
        pipe.set_adapters(Path(lora["path"]).stem, adapter_weights=lora["weight"])

    print("[Wan] Generating video...")
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=cfg_scale,
            num_inference_steps=steps,
            generator=generator,
        ).frames[0]

    print(f"[Wan] Encoding video to {output_path}...")
    export_to_video(output, output_path, fps=fps)

    elapsed = time.time() - t0
    print(f"[Wan] Done in {elapsed:.1f}s -> {output_path}")

    result = {
        "status": "success",
        "output_path": output_path,
        "seed": seed,
        "generation_time": elapsed,
    }
    print(f"__RESULT_JSON__={json.dumps(result)}")

if __name__ == "__main__":
    main()
