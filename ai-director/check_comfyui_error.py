"""Debug: submit the full SFT workflow and print ComfyUI's error response."""
import json, urllib.request, random

seed = random.randint(0, 2**31 - 1)
wf = {
    "1": {"class_type": "UNETLoader", "inputs": {
        "unet_name": "acestep-v15-xl-sft",
        "weight_dtype": "default"
    }},
    "2": {"class_type": "CLIPLoader", "inputs": {
        "clip_name": "qwen_1.7b_ace15.safetensors",
        "type": "ace"
    }},
    "3": {"class_type": "VAELoader", "inputs": {
        "vae_name": "ace_1.5_vae.safetensors"
    }},
    "4": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {
        "clip": ["2", 0],
        "tags": "children's music, lullaby, gentle, piano",
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
        "min_p": 0.0,
    }},
    "5": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {
        "seconds": 120.0,
        "batch_size": 1
    }},
    "6": {"class_type": "KSampler", "inputs": {
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
    }},
    "7": {"class_type": "VAEDecodeAudio", "inputs": {
        "samples": ["6", 0],
        "vae": ["3", 0]
    }},
    "8": {"class_type": "SaveAudio", "inputs": {
        "audio": ["7", 0],
        "filename_prefix": "sft_test"
    }},
}

payload = json.dumps({"prompt": wf}).encode()
req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=payload,
    headers={"Content-Type": "application/json"})
try:
    resp = urllib.request.urlopen(req, timeout=30)
    print("OK:", resp.read().decode()[:500])
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP {e.code}:")
    try:
        err = json.loads(body)
        print(json.dumps(err, indent=2)[:3000])
    except Exception:
        print(body[:3000])
