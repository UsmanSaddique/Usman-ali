import os
import sys
sys.path.append(os.getcwd())

from app.config import AppConfig
from app.services.model_manager import ModelManager
from app.services.video_gen import VideoGenService

config = AppConfig()
manager = ModelManager(config)
video_gen = VideoGenService(manager, config)

print("Starting video generation...")
try:
    res = video_gen.txt2vid(
        prompt="A dog running in a park",
        model_filename="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf"
    )
    print("Success:", res)
except Exception as e:
    import traceback
    traceback.print_exc()
