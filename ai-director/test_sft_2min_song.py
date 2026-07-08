"""
Test ACE-Step 1.5 XL SFT — generate a 2-minute song via ComfyUI.
Uses DualCLIPLoader with both qwen_0.6b and qwen_1.7b CLIP files
as required by the ACE 1.5 architecture.
"""
import sys, time, random, json, urllib.request
sys.path.insert(0, ".")

COMFYUI = "http://127.0.0.1:8188"

def wait_comfyui(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"{COMFYUI}/system_stats", timeout=3)
            return True
        except Exception:
            time.sleep(2)
    return False

def free_vram():
    try:
        req = urllib.request.Request(f"{COMFYUI}/free", method="POST",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        time.sleep(3)
    except Exception as e:
        print(f"  free_vram warning: {e}")

def submit(workflow):
    payload = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(f"{COMFYUI}/prompt", data=payload,
        headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        return resp["prompt_id"], None
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return None, body

def wait_completion(prompt_id, timeout=3600, poll=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            resp = json.loads(urllib.request.urlopen(
                f"{COMFYUI}/history/{prompt_id}", timeout=10).read())
            if prompt_id in resp:
                return resp[prompt_id]
        except Exception:
            pass
        time.sleep(poll)
    raise TimeoutError(f"Timed out after {timeout}s")

def check_result(history):
    status = history.get("status", {})
    if status.get("status_str") == "error":
        msgs = status.get("messages", [])
        for msg in msgs:
            if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "execution_error":
                err = msg[1]
                return f"{err.get('node_type', '?')}: {err.get('exception_message', '?')[:300]}"
        return "Unknown error"
    return None

def collect_output(history, output_path):
    outputs = history.get("outputs", {})
    for node_id, node_out in outputs.items():
        if "audio" in node_out:
            for audio in node_out["audio"]:
                fname = audio["filename"]
                subfolder = audio.get("subfolder", "")
                url = f"{COMFYUI}/view?filename={fname}&subfolder={subfolder}&type=output"
                data = urllib.request.urlopen(url, timeout=60).read()
                with open(output_path, "wb") as f:
                    f.write(data)
                return len(data)
    return 0


def main():
    print("=" * 60)
    print("ACE-Step 1.5 XL SFT — 2 Minute Song Generation")
    print("Using DualCLIPLoader (qwen_0.6b + qwen_1.7b)")
    print("=" * 60)

    if not wait_comfyui(60):
        print("ComfyUI not reachable!")
        return

    seed = random.randint(0, 2**31 - 1)
    tags = "children's music, lullaby, gentle, warm, soft piano, music box, acoustic guitar, instrumental, peaceful, dreamy"
    output_path = r"D:\assets_genrated\ltx_output\sft_2min_song.wav"

    print(f"\nSeed: {seed}")
    print(f"Tags: {tags}")
    print(f"Duration: 120s | Steps: 50 | fp8_e4m3fn")
    print()

    free_vram()

    # ACE 1.5 requires DualCLIPLoader with BOTH clip models:
    # - qwen_0.6b: base text encoder + lyrics encoder
    # - qwen_1.7b: LLM for audio code generation
    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "acestep_v1.5_xl_sft_bf16.safetensors",
                "weight_dtype": "fp8_e4m3fn"
            }
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "qwen_0.6b_ace15.safetensors",
                "clip_name2": "qwen_1.7b_ace15.safetensors",
                "type": "ace"
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "ace_1.5_vae.safetensors"
            }
        },
        "4": {
            "class_type": "TextEncodeAceStepAudio1.5",
            "inputs": {
                "clip": ["2", 0],
                "tags": tags,
                "lyrics": "",
                "seed": seed,
                "bpm": 85,
                "duration": 120.0,
                "timesignature": "4",
                "language": "en",
                "keyscale": "C major",
                "generate_audio_codes": True,
                "cfg_scale": 2.0,
                "temperature": 0.85,
                "top_p": 0.9,
                "top_k": 0,
                "min_p": 0.0
            }
        },
        "5": {
            "class_type": "EmptyAceStep1.5LatentAudio",
            "inputs": {
                "seconds": 120.0,
                "batch_size": 1
            }
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": 50,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            }
        },
        "7": {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["3", 0]
            }
        },
        "8": {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["7", 0],
                "filename_prefix": "sft_2min_song"
            }
        }
    }

    print("Submitting workflow...")
    prompt_id, err = submit(workflow)
    if err:
        print(f"REJECTED: {err[:500]}")
        return

    print(f"Prompt ID: {prompt_id}")
    print(f"Generating... (LLM audio codes + 50 diffusion steps for 120s audio)")
    print(f"This will take a while — SFT model is 18.6GB in fp8 + LLM inference")
    print()

    t0 = time.time()
    try:
        history = wait_completion(prompt_id, timeout=3600, poll=5.0)
    except TimeoutError:
        print("TIMEOUT after 60min")
        return

    elapsed = time.time() - t0
    error = check_result(history)
    if error:
        print(f"FAILED in {elapsed:.0f}s ({elapsed/60:.1f}m): {error}")
        return

    size = collect_output(history, output_path)
    if size > 0:
        print(f"SUCCESS in {elapsed:.0f}s ({elapsed/60:.1f}m)")
        print(f"Output: {output_path} ({size/1024/1024:.1f} MB)")
    else:
        print(f"Completed but no audio output found")


if __name__ == "__main__":
    main()
