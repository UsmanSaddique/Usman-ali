import requests
import time
import sys

BASE_URL = "http://127.0.0.1:8000/api"

def log(msg):
    print(f"[*] {msg}", flush=True)

def wait_for_project_status(project_id, target_statuses, timeout=14400):
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{BASE_URL}/projects/{project_id}")
        resp.raise_for_status()
        status = resp.json()["status"]
        log(f"Project status: {status}")
        if status in target_statuses:
            return resp.json()
        if status == "failed":
            raise Exception(f"Project failed: {resp.json().get('error_log')}")
        time.sleep(10)
    raise TimeoutError("Timeout waiting for project status")

def main():
    # 1. Create project
    log("Creating project...")
    req = {
        "title": "Majestic Lahore",
        "channel_slug": "little-fairy-dreams",
        "context": "A cinematic and majestic journey through the historical streets of Lahore, the Badshahi Mosque, and the vibrant culture of Pakistan.",
        "duration": 210,
        "music_style": "Urdu Hindi soulful emotional cinematic sufi track, soft acoustic, tabla, rabab, male vocal"
    }
    resp = requests.post(f"{BASE_URL}/projects", json=req)
    resp.raise_for_status()
    project_id = resp.json()["id"]
    log(f"Project created: {project_id}")

    # Wait for lyrics generation
    project = wait_for_project_status(project_id, ["draft"])

    # 2. Generate Music
    log("Generating music variants...")
    resp = requests.post(f"{BASE_URL}/projects/{project_id}/generate-music-variants")
    resp.raise_for_status()
    
    project = wait_for_project_status(project_id, ["draft"])
    
    # Select first music track
    tracks = project.get("music_tracks", [])
    if not tracks:
        raise Exception("No music tracks generated")
    
    track_id = tracks[0]["id"]
    log(f"Selecting music track: {track_id}")
    resp = requests.post(f"{BASE_URL}/projects/{project_id}/select-music/{track_id}")
    resp.raise_for_status()

    # 3. Generate Scenes
    log("Generating scenes from lyrics (5s clips)...")
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

    project = wait_for_project_status(project_id, ["generated", "approved", "draft"])
    
    # 5. Upscale
    log("Starting upscaling...")
    resp = requests.post(f"{BASE_URL}/projects/{project_id}/start-upscale")
    resp.raise_for_status()
    
    project = wait_for_project_status(project_id, ["generated", "approved", "draft", "upscaled", "completed"])

    # 6. Merge
    log("Starting merge...")
    resp = requests.post(f"{BASE_URL}/projects/{project_id}/merge")
    resp.raise_for_status()

    project = wait_for_project_status(project_id, ["completed"])
    log("Finished successfully! Video is ready.")

if __name__ == "__main__":
    main()
