import sys, os, time
sys.path.insert(0, os.getcwd())
from app.config import settings
from app.services.model_manager import ModelManager
from app.services.video_gen import VideoGenService
W,H = int(sys.argv[1]), int(sys.argv[2])
frames = int(sys.argv[3]) if len(sys.argv)>3 else 97
svc = VideoGenService(ModelManager(), settings)
t0=time.time()
r = svc.txt2vid(prompt="a fluffy cream-white rabbit with long ears in a sunny meadow, soft 3D storybook render, pastel",
                negative_prompt="blurry, text", num_frames=frames, seed=12345,
                output_path=f"assets_generated/restest_{W}x{H}.mp4",
                model_filename=settings.video.model_path.name, width=W, height=H)
print(f"RESULT {W}x{H} frames={frames}: {time.time()-t0:.0f}s -> {r.path}")
