import urllib.request
import json
import sys
import os
sys.path.append(os.getcwd())

from app.services.comfyui_client import build_ltx_workflow

workflow = build_ltx_workflow(
    model_filename="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
    prompt="A test prompt",
    width=768,
    height=512,
    num_frames=97,
    fps=24,
    steps=8,
    cfg=1.0,
    seed=12345
)

payload = json.dumps({"prompt": workflow, "client_id": "test_client"}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8188/prompt",
    data=payload,
    headers={"Content-Type": "application/json"},
)

try:
    with urllib.request.urlopen(req) as resp:
        print("Response:", resp.read().decode())
except Exception as e:
    print("Error:", e)
    if hasattr(e, 'read'):
        print(e.read().decode())
