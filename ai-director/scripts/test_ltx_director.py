"""
LTX Director End-to-End Test Script
Generates a 60-second continuous video (10 segments × 6s) at 720p,
then uses FFmpeg to upscale the final video to 1080p.
Acts as a 'Master Prompter' to generate pristine initial still images for the LTX pipeline.
"""
import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path

# Add the project root to sys.path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.services.model_manager import ModelManager
from app.services.image_gen import ImageGenService
from app.services.ltx_director import LTXDirectorService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Master Prompter: A 10-scene Sci-Fi Cyberpunk infiltration story
STORY_SCENES = [
    {
        "prompt": "Cinematic wide shot, a cybernetic samurai standing on a rainy rooftop overlooking a sprawling neon-lit futuristic megacity. Highly detailed, cyberpunk aesthetic, atmospheric fog, 8k resolution, dramatic lighting.",
        "dialogue": "The city of Neo-Kyoto. Beautiful, corrupt, and tonight... my target."
    },
    {
        "prompt": "Extreme close-up, rain drops hitting a glowing blue katana blade as it is slowly drawn from its high-tech scabbard. Glowing neon reflections on the metal, photorealistic, cinematic depth of field.",
        "dialogue": "They think their fortress is impenetrable."
    },
    {
        "prompt": "Dynamic low angle shot, the cyber-samurai leaping down into a dark, misty alleyway, landing gracefully in a puddle. Splashing water, glowing cybernetics, intense action pose, volumetric lighting.",
        "dialogue": "They are about to be proven wrong."
    },
    {
        "prompt": "Cinematic tracking shot, two sleek robotic guard dogs with glowing red eyes patrolling a highly secured metallic corridor. Creepy, sci-fi military aesthetic, harsh overhead fluorescent lighting.",
        "dialogue": "Sector 4. The hounds are already on patrol."
    },
    {
        "prompt": "High-speed action shot, the samurai perfectly deflecting a bright red laser blast with the glowing blue katana. Sparks flying, motion blur, epic combat, highly detailed futuristic armor.",
        "dialogue": "Target locked. Engaging."
    },
    {
        "prompt": "Wide action shot, the samurai executing a spinning kick, sending a robotic guard dog crashing into a flickering neon holographic sign. Shattering glass, sparks, intense cyberpunk street fighting.",
        "dialogue": "Perimeter secured. Moving to the inner sanctum."
    },
    {
        "prompt": "Cinematic shot, the cyber-samurai standing before massive, imposing steel blast doors with glowing warning lights. Heavy industrial sci-fi architecture, smoke on the ground, dramatic shadows.",
        "dialogue": "The mainframe lies just beyond these doors."
    },
    {
        "prompt": "Wide interior shot, a pristine white server room with endless rows of glowing quantum servers, clean futuristic architecture. The samurai stands in the center, contrasting with the bright environment.",
        "dialogue": "The core. All the city's secrets in one place."
    },
    {
        "prompt": "Close-up, a gloved robotic hand plugging a glowing purple data drive into a highly advanced crystal mainframe console. Intricate circuitry, macro photography, cinematic lighting.",
        "dialogue": "Downloading the payload. Almost done."
    },
    {
        "prompt": "Dramatic wide shot, the entire white server room suddenly turning deep crimson red as alarms blare. The samurai draws their katana again, ready for a final stand. Intense red lighting, cinematic tension.",
        "dialogue": "They know I'm here. Time to fight my way out."
    }
]

def run_test():
    logger.info("Initializing AI Director environment...")
    config = settings
    manager = ModelManager()
    image_svc = ImageGenService(manager, config)
    ltx_svc = LTXDirectorService(manager, config)
    
    # Create output directory
    out_dir = config.paths.projects_dir / "ltx_test_project"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("--- PHASE 1: GENERATING REFERENCE STILLS ---")
    segments = []
    for i, scene in enumerate(STORY_SCENES):
        logger.info(f"Generating Still {i+1}/10...")
        img_path = str(out_dir / f"scene_{i+1:02d}.jpg")
        
        # Only generate if it doesn't already exist (allows resuming test)
        if not os.path.exists(img_path):
            try:
                # Fallback to general ComfyUI generation if Z-Image fails or is disabled
                image_svc.generate(
                    prompt=scene["prompt"],
                    negative_prompt="blurry, low quality, deformed, text, watermark",
                    width=1280,
                    height=720,
                    seed=int(time.time()) + i,
                    output_path=img_path
                )
            except Exception as e:
                logger.error(f"Failed to generate image {i+1}: {e}")
                sys.exit(1)
        
        segments.append({
            "prompt": scene["prompt"],
            "dialogue": f'The narrator says: "{scene["dialogue"]}"',
            "image_path": img_path,
            "seconds": 6.0
        })
        
    logger.info("--- PHASE 2: RUNNING LTX DIRECTOR (720p) ---")
    ltx_out_720p = str(out_dir / "ltx_director_720p.mp4")
    
    if not os.path.exists(ltx_out_720p):
        try:
            logger.info("Submitting multi-segment payload to ComfyUI LTX Director...")
            ltx_out_720p = ltx_svc.generate(segments, ltx_out_720p)
            logger.info(f"LTX Director 720p video generated at: {ltx_out_720p}")
        except Exception as e:
            logger.error(f"LTX Director generation failed: {e}")
            sys.exit(1)
    else:
        logger.info(f"Found existing 720p render at {ltx_out_720p}, skipping generation.")

    logger.info("--- PHASE 3: UPSCALING TO 1080p VIA FFMPEG ---")
    ltx_out_1080p = str(out_dir / "ltx_director_1080p.mp4")
    
    ffmpeg_cmd = [
        str(config.paths.ffmpeg_bin), "-y",
        "-i", ltx_out_720p,
        "-vf", "scale=1920:1080:flags=lanczos",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy",
        ltx_out_1080p
    ]
    
    logger.info("Running FFmpeg Lanczos Upscale...")
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        logger.info(f"SUCCESS! Final 1080p video saved to: {ltx_out_1080p}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg upscaling failed: {e.stderr.decode('utf-8')}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
