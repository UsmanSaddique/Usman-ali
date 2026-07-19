"""
build_sfx_library.py

Generates a local library of transition sound effects (whooshes, risers, impacts)
using the ElevenLabs Sound Effects API. These are used by the assembler to
mask hard cuts and motion-graphics transitions, since synced AI foley (MMAudio)
struggles with abstract cinematic transitions.

Usage:
  set ELEVENLABS_API_KEY=your_key
  python scripts/build_sfx_library.py
"""
import os
import sys
import json
import time
import requests
from pathlib import Path

# Required transition types
TRANSITIONS = {
    "whoosh": [
        "A clean, fast cinematic whoosh for a video transition.",
        "Deep low-frequency cinematic swoosh.",
        "Fast air whoosh, light and airy transition.",
        "Sci-fi energy whoosh, futuristic transition.",
        "Fast swipe whoosh."
    ],
    "impact": [
        "A deep cinematic bass hit impact.",
        "Soft muffled sub-bass thud.",
        "Cinematic boom impact.",
        "Electronic glitch hit transition."
    ],
    "riser": [
        "A tension-building cinematic riser.",
        "Short electronic synth riser.",
        "Ethereal ambient swell riser."
    ],
    "pop": [
        "A clean, friendly UI pop sound.",
        "A cute bubble pop sound.",
        "A soft wooden tick UI sound."
    ]
}

def build_sfx_library(output_dir: Path, api_key: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://api.elevenlabs.io/v1/sound-generation"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    total_generated = 0
    for category, prompts in TRANSITIONS.items():
        cat_dir = output_dir / category
        cat_dir.mkdir(exist_ok=True)
        
        print(f"\\n--- Generating category: {category} ---")
        for i, prompt in enumerate(prompts):
            out_file = cat_dir / f"{category}_{i+1:02d}.mp3"
            if out_file.exists():
                print(f"Skipping {out_file.name} (already exists)")
                continue
                
            payload = {
                "text": prompt,
                "duration_seconds": 2.0 if category in ["whoosh", "pop", "impact"] else 4.0,
                "prompt_influence": 0.3
            }
            
            print(f"Generating: {prompt}")
            resp = requests.post(url, headers=headers, json=payload)
            
            if resp.status_code != 200:
                print(f"Error {resp.status_code}: {resp.text}")
                continue
                
            with open(out_file, "wb") as f:
                f.write(resp.content)
            
            print(f"Saved -> {out_file}")
            total_generated += 1
            time.sleep(1)  # Rate limit safety
            
    print(f"\\nDone! Generated {total_generated} new transition SFX in {output_dir}")

if __name__ == "__main__":
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY environment variable not set.")
        sys.exit(1)
        
    root_dir = Path(__file__).parent.parent
    sfx_dir = root_dir / "assets" / "sfx"
    build_sfx_library(sfx_dir, api_key)
