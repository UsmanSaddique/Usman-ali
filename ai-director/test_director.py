"""
AI Director — Director Brain Test
Generates a real script via the (upgraded, elite) DirectorService for a chosen
channel and inspects the result: title language, scene mix, character consistency,
English-visual / native-narration split.

Run:  python_embeded\\python.exe test_director.py [channel_slug] [title]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # let Urdu print on Windows console
except Exception:
    pass

from app.config import settings
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.director import DirectorService

slug = sys.argv[1] if len(sys.argv) > 1 else "urdu-moral-stories"
title = sys.argv[2] if len(sys.argv) > 2 else "The greedy parrot who learned to share"

print(f"Channel: {slug}\nTitle:   {title}\nLoading director LLM (Qwen) — this can take a minute...\n")

mm = ModelManager()
register_all_loaders(mm, settings)
director = DirectorService(mm, settings)
script = director.generate_script(
    title=title,
    duration=60,            # short test video
    context="A warm moral story for young kids. Keep it simple with a clear lesson.",
    channel_slug=slug,
    num_scenes=8,
)
director.manager.unload()

# dump full script to a file (encoding-proof inspection)
dump = {
    "title": script.title, "music_style": script.music_style,
    "music_mood": script.music_mood, "thumbnail_prompt": script.thumbnail_prompt,
    "scenes": [vars(s) for s in script.scenes],
}
with open("last_script.json", "w", encoding="utf-8") as f:
    json.dump(dump, f, ensure_ascii=False, indent=2)
print("(full script written to last_script.json)\n")

print("=" * 64)
print(f"TITLE:        {script.title}")
print(f"SCENES:       {len(script.scenes)} (target 8)")
print(f"MUSIC:        {script.music_style} | mood: {script.music_mood}")
print(f"THUMBNAIL:    {script.thumbnail_prompt[:90]}")
print("=" * 64)

types = {}
for sc in script.scenes:
    types[sc.scene_type] = types.get(sc.scene_type, 0) + 1
print(f"Scene-type mix: {types}")
still = types.get("still_pan", 0)
print(f"still_pan ratio: {still/max(len(script.scenes),1):.0%} (profile target ~65%)")
print("-" * 64)

for sc in script.scenes[:3]:
    print(f"\n[Scene {sc.scene_number}] type={sc.scene_type} motion={sc.camera_motion} {sc.duration}s")
    print(f"  PROMPT (EN): {sc.prompt[:160]}")
    print(f"  NARRATION  : {sc.narration_text[:120]}")

# crude language sanity: narration should contain non-ASCII (Urdu) for urdu channels
joined_narr = " ".join(s.narration_text for s in script.scenes)
has_urdu = any(ord(ch) > 1500 for ch in joined_narr)
joined_prompt = " ".join(s.prompt for s in script.scenes)
prompt_ascii = all(ord(ch) < 256 for ch in joined_prompt)
print("\n" + "=" * 64)
print(f"narration has Urdu script: {has_urdu}")
print(f"visual prompts are English/ASCII: {prompt_ascii}")
print("=" * 64)
