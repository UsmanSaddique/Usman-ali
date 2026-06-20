"""
AI Director — Single Clip End-to-End Test
Generates ONE short LTX clip through VideoGenService -> ComfyUI to prove the
full real generation path works. Short (49 frames) to stay fast.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import settings
from app.services.model_manager import ModelManager
from app.services.video_gen import VideoGenService

OUT = os.path.join(os.path.dirname(__file__), "assets_generated", "smoke_clip.mp4")

svc = VideoGenService(ModelManager(), settings)
model = settings.video.model_path.name
print(f"Generating test clip with {model} ...")
print("(LTX distilled 8-step, 768x512, 49 frames ~2s — first run loads the model, be patient)")

t0 = time.time()
res = svc.txt2vid(
    prompt="a dreamy pastel fairy garden with butterflies, soft watercolor storybook style",
    negative_prompt="blurry, low quality, text, watermark",
    num_frames=49,
    output_path=OUT,
    model_filename=model,
)
dt = time.time() - t0

ok = os.path.exists(res.path) and os.path.getsize(res.path) > 1024
size_mb = os.path.getsize(res.path) / (1024 * 1024) if os.path.exists(res.path) else 0
print("=" * 60)
print(f"{'[OK]' if ok else '[XX]'} output: {res.path}")
print(f"     {size_mb:.2f} MB | {res.width}x{res.height} | {res.num_frames}f @ {res.fps}fps")
print(f"     gen time: {dt:.1f}s")
print("=" * 60)
sys.exit(0 if ok else 1)
