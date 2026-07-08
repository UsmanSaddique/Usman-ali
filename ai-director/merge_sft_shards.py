"""Merge sharded ACE-Step SFT safetensors into a single file for ComfyUI."""
import json, time
from pathlib import Path
from safetensors.torch import load_file, save_file

MODEL_DIR = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\acestep-v15-xl-sft")
OUTPUT = MODEL_DIR.parent / "acestep_v1.5_xl_sft_bf16.safetensors"

index_path = MODEL_DIR / "model.safetensors.index.json"
with open(index_path) as f:
    index = json.load(f)

# Get unique shard files
shard_files = sorted(set(index["weight_map"].values()))
print(f"Merging {len(shard_files)} shards into single file...")
print(f"Output: {OUTPUT}")

t0 = time.time()
all_tensors = {}
for i, shard in enumerate(shard_files):
    shard_path = MODEL_DIR / shard
    print(f"  Loading shard {i+1}/{len(shard_files)}: {shard} ({shard_path.stat().st_size/1024/1024/1024:.2f} GB)...", flush=True)
    tensors = load_file(str(shard_path))
    all_tensors.update(tensors)
    print(f"    Got {len(tensors)} tensors (total so far: {len(all_tensors)})")

print(f"\nSaving merged file ({len(all_tensors)} tensors)...", flush=True)
save_file(all_tensors, str(OUTPUT))

elapsed = time.time() - t0
size_gb = OUTPUT.stat().st_size / 1024 / 1024 / 1024
print(f"\nDone in {elapsed:.0f}s — {size_gb:.2f} GB -> {OUTPUT.name}")
