"""Quick test: generate a song with ACE-Step 1.5 XL SFT via ComfyUI."""
import sys, time, random
sys.path.insert(0, ".")

from app.services.comfyui_client import ComfyUIClient, build_acestep15xl_sft_workflow

def main():
    client = ComfyUIClient()
    if not client.wait_ready(30):
        print("ERROR: ComfyUI not reachable at http://127.0.0.1:8188")
        return

    client.free_vram()
    print("VRAM freed, building SFT workflow...")

    wf = build_acestep15xl_sft_workflow(
        style_tags="children's music, instrumental, gentle, soft piano, music box, lullaby",
        lyrics="",
        seconds=30.0,
        seed=random.randint(0, 2**31 - 1),
        steps=50,
        bpm=90,
        language="en",
        keyscale="C major",
    )

    print(f"Submitting ACE-Step 1.5 XL SFT workflow (50 steps, 30s track)...")
    t0 = time.time()
    prompt_id = client.submit(wf)
    print(f"Prompt ID: {prompt_id}")

    history = client.wait_for_completion(prompt_id, timeout=1800, poll=3.0)
    elapsed = time.time() - t0
    print(f"Generation completed in {elapsed:.1f}s")

    out = r"D:\assets_genrated\ltx_output\test_sft_music.wav"
    client.collect_output(history, out)
    print(f"Saved to: {out}")

if __name__ == "__main__":
    main()
