import urllib.request
import json

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/projects/6801ac66-4d60-4c14-8d46-0fb37bf3e278/generate-scenes",
    data=json.dumps({
        "scene_ids": ["1ae5e474-def0-4588-8281-4b2227728cd1"],
        "video_model": "wan2.1_8B.safetensors",
        "lora_ids": [],
        "lora_weights": []
    }).encode(),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req) as resp:
        print("Status:", resp.status)
        print("Response:", resp.read().decode())
except Exception as e:
    print("Error:", e)
