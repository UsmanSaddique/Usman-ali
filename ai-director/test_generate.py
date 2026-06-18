import sqlite3
import json
import urllib.request
import os

db_path = os.path.abspath("ai_director.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT project_id, id FROM scenes LIMIT 1")
row = cursor.fetchone()
if not row:
    print("No scenes found in the entire DB.")
    exit()
project_id, scene_id = row

cursor.execute("SELECT id FROM scenes WHERE project_id = ? LIMIT 1", (project_id,))
row = cursor.fetchone()
if not row:
    print("No scenes found.")
    exit()
scene_id = row[0]

print(f"Testing generation for project {project_id}, scene {scene_id}")

payload = {
    "scene_ids": [scene_id],
    "video_model": "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
    "lora_ids": [],
    "lora_weights": []
}

req = urllib.request.Request(
    f"http://localhost:8000/api/projects/{project_id}/generate-scenes",
    data=json.dumps(payload).encode('utf-8'),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req) as response:
        print("Status Code:", response.getcode())
        print("Response:", response.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code, e.read().decode('utf-8'))
except Exception as e:
    print("Error:", e)
