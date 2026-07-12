import requests
import time
import sys

BASE_URL = "http://127.0.0.1:8000/api"

def log(msg):
    print(f"[*] {msg}", flush=True)

def wait_for_project_status(project_id, target_statuses, timeout=21600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{BASE_URL}/projects/{project_id}")
            resp.raise_for_status()
            status = resp.json()["status"]
            log(f"Project status: {status}")
            if status in target_statuses:
                return resp.json()
            if status == "failed":
                raise Exception(f"Project failed: {resp.json().get('error_log')}")
        except requests.exceptions.RequestException as e:
            log(f"Connection error: {e}. Retrying...")
        time.sleep(15)
    raise TimeoutError("Timeout waiting for project status")

def main():
    # 1. Create project
    log("Creating Masterpiece Project...")
    req = {
        "title": "Echoes of Eternity: A Journey to the Edge of the Universe",
        "channel_slug": "little-fairy-dreams", # Assuming this channel exists from run_overnight.py
        "context": "A breathtaking cinematic documentary exploring the mysteries of deep space, black holes, nebulas, and the birth of galaxies. Narrated in a highly engaging, retention-optimized style typical of top-tier YouTube documentaries. The visuals should be hyper-realistic, awe-inspiring, and Oscar-worthy in their composition and lighting.",
        "duration": 210, # 3.5 minutes
        "music_style": "Epic cinematic Hans Zimmer style orchestral track, slow build-up, deep bass, powerful choir, emotional climax, cosmic, awe-inspiring"
    }
    
    # Wait for server to be up
    server_up = False
    for _ in range(20):
        try:
            requests.get(f"{BASE_URL}/projects")
            server_up = True
            break
        except requests.exceptions.ConnectionError:
            time.sleep(5)
            log("Waiting for server to start...")
    
    if not server_up:
        log("Server did not start in time. Exiting.")
        sys.exit(1)

    resp = requests.post(f"{BASE_URL}/projects", json=req)
    resp.raise_for_status()
    project_id = resp.json()["id"]
    log(f"Project created: {project_id}")

    # Wait for lyrics/script generation
    project = wait_for_project_status(project_id, ["draft", "scripted"])

    # 2. Generate Music
    log("Generating music variants...")
    resp = requests.post(f"{BASE_URL}/projects/{project_id}/generate-music-variants")
    resp.raise_for_status()
    
    project = wait_for_project_status(project_id, ["draft", "scripted"])
    
    # Select first music track
    tracks = project.get("music_tracks", [])
    
    # Actually wait_for_project_status only checks the project object, music tracks might need to be refreshed
    # Let's poll until we get a music track generated
    log("Waiting for music track to finish generation...")
    track_id = None
    for _ in range(60):
        proj_resp = requests.get(f"{BASE_URL}/projects/{project_id}").json()
        tracks = proj_resp.get("music_variants", [])
        if tracks:
            # Check if any track is ready (has url/path)
            for t in tracks:
                if t.get("url"):
                    track_id = t["id"]
                    break
        if track_id:
            break
        time.sleep(10)
        
    if not track_id:
        raise Exception("No music tracks generated")
    
    log(f"Selecting music track: {track_id}")
    resp = requests.post(f"{BASE_URL}/projects/{project_id}/select-music/{track_id}")
    resp.raise_for_status()

    # 3. Generate Scenes
    log("Generating scenes from lyrics/script...")
    # Target duration is 210, num clips = 210 / 5 = 42
    resp = requests.post(f"{BASE_URL}/projects/{project_id}/scenes-from-lyrics", json={"num_clips": 42})
    resp.raise_for_status()

    project = requests.get(f"{BASE_URL}/projects/{project_id}").json()
    scene_ids = [s["id"] for s in project["scenes"]]
    log(f"Created {len(scene_ids)} scenes")

    # 4. Generate Video (without inline upscale to speed up base generation and avoid timeouts)
    log("Starting video generation...")
    req = {
        "scene_ids": scene_ids,
        "video_model": "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
        "lora_ids": [],
        "lora_weights": [],
        "width": 832,
        "height": 480,
        "batch": True,
        "upscale_inline": False
    }
    resp = requests.post(f"{BASE_URL}/projects/{project_id}/generate-scenes", json=req)
    resp.raise_for_status()

    project = wait_for_project_status(project_id, ["generated", "approved", "draft", "scripted"])
    
    # 5. Upscale
    log("Starting upscaling...")
    resp = requests.post(f"{BASE_URL}/projects/{project_id}/start-upscale")
    resp.raise_for_status()
    
    project = wait_for_project_status(project_id, ["generated", "approved", "draft", "scripted", "upscaled", "completed"])

    # 6. Merge
    log("Starting merge...")
    resp = requests.post(f"{BASE_URL}/projects/{project_id}/merge")
    resp.raise_for_status()

    project = wait_for_project_status(project_id, ["completed"])
    log("Finished successfully! Video is ready. You will have a cinematic masterpiece waiting for you when you wake up.")

if __name__ == "__main__":
    main()
