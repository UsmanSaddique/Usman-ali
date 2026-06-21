"""Generate a character training set (varied poses, consistent identity) for a LoRA."""
import sys, os
sys.path.insert(0, os.getcwd())
from app.config import settings
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.image_gen import ImageGenService

TRIGGER = "yusufchar"
IDENTITY = ("a cute chubby-cheeked 3D Pixar-style cartoon little boy, white knit prayer cap (taqiyah), "
            "short brown curly hair, big warm brown eyes, sky-blue embroidered kurta, glossy 3D skin, "
            "polished animation movie look")
POSES = [
    "standing and smiling, front view", "waving hello, happy", "sitting cross-legged reading a book",
    "looking up curiously", "three-quarter side view, gentle smile", "laughing joyfully",
    "holding a green apple", "walking in a sunny garden", "hands together praying, peaceful",
    "close-up portrait, big eyes", "pointing at something excitedly", "sitting at a table",
    "standing under a tree", "surprised expression", "waving goodbye", "thinking with finger on chin",
]
out = "assets_generated/lora_train/yusuf"
os.makedirs(out, exist_ok=True)
mm = ModelManager(); register_all_loaders(mm, settings)
ig = ImageGenService(mm, settings)
neg = "photorealistic, realistic skin, text, watermark, deformed, extra fingers, blurry, low quality, two boys, multiple people"
for i, pose in enumerate(POSES):
    prompt = f"{TRIGGER}, {IDENTITY}, {pose}, plain soft pastel background, centered, full character visible"
    p = f"{out}/img_{i:02d}.png"
    ig.generate(prompt=prompt, negative_prompt=neg, width=1024, height=1024, steps=32, cfg_scale=7.0,
                seed=1000+i, output_path=p)
    # caption file for training (trigger + identity)
    with open(f"{out}/img_{i:02d}.txt", "w", encoding="utf-8") as f:
        f.write(f"{TRIGGER}, cute 3D pixar cartoon muslim boy, {pose}")
    print(f"  {i+1}/{len(POSES)} {pose[:30]}")
print(f"DONE: {len(POSES)} training images -> {out}")
