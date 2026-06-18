import urllib.request
import json
import time
import subprocess

def wait_for_comfy():
    print("Waiting for ComfyUI to start on port 8188...")
    while True:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2) as resp:
                if resp.status == 200:
                    print("ComfyUI is UP!")
                    return
        except Exception:
            pass
        time.sleep(2)

def print_gpu_stats():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        print(f"GPU Stats: {output} (Used MB, Total MB, Util %)")
    except Exception as e:
        print(f"Failed to read GPU stats: {e}")

def trigger_generation():
    print("Triggering AI Director generation API...")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/projects/6801ac66-4d60-4c14-8d46-0fb37bf3e278/generate-scenes",
        data=json.dumps({
            "scene_ids": ["1ae5e474-def0-4588-8281-4b2227728cd1"],
            "video_model": "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
            "lora_ids": [],
            "lora_weights": []
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print("API Response:", resp.read().decode())
    except Exception as e:
        print("API Error:", e)

def monitor_queue():
    print("Monitoring ComfyUI queue and history...")
    while True:
        print_gpu_stats()
        try:
            with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=2) as resp:
                q = json.loads(resp.read())
                running = len(q.get("queue_running", []))
                pending = len(q.get("queue_pending", []))
                print(f"ComfyUI Queue - Running: {running}, Pending: {pending}")
                if running == 0 and pending == 0:
                    print("Queue is empty. Checking if completed...")
                    break
        except Exception as e:
            print("Queue check error:", e)
        time.sleep(5)

if __name__ == "__main__":
    wait_for_comfy()
    print_gpu_stats()
    trigger_generation()
    time.sleep(2)
    monitor_queue()
    print("Finished monitoring. Check output folder for results.")
