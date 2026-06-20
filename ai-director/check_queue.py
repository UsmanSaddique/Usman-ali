import urllib.request
import json
try:
    req = urllib.request.Request('http://127.0.0.1:8188/queue')
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
        print(f"Pending: {len(data.get('queue_pending', []))}")
        print(f"Running: {len(data.get('queue_running', []))}")
except Exception as e:
    print(f"Error: {e}")
