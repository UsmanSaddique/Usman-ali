"""Check UNETLoader weight_dtype options."""
import json, urllib.request

info = json.loads(urllib.request.urlopen("http://127.0.0.1:8188/object_info/UNETLoader", timeout=10).read())
dtypes = info["UNETLoader"]["input"]["required"]["weight_dtype"][0]
print("UNETLoader weight_dtype options:")
for d in dtypes:
    print(f"  {d}")
