"""Regenerate the Yusuf character reference sheet with Z-Image-Turbo."""
import sys, time, shutil
from pathlib import Path
sys.path.insert(0, ".")
from app.services.comfyui_client import ComfyUIClient, COMFYUI_OUTPUT, build_zimage_workflow

OUT = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\assets_generated\benchmark_iclora")
COMFY_INPUT = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\input")
c = ComfyUIClient()

PROMPT = (
    "A clean character reference sheet on a plain soft cream background, 3D Pixar-style "
    "animated movie render. The SAME cute chubby-cheeked 4-year-old Muslim boy named Yusuf "
    "shown three times in a neat row: front view, side profile view, and three-quarter view. "
    "He wears a white knit prayer cap (taqiyah), has short brown curly hair, big warm brown "
    "eyes, rosy cheeks, a friendly smile, and a sky-blue kurta with delicate white embroidery. "
    "Identical face, outfit and proportions in all three poses. Soft studio lighting, glossy "
    "highlights, smooth subsurface skin shading, bright cheerful pastel colors, high detail, "
    "polished children's animation look, organized clean layout."
)

def main():
    c.wait_ready(300)
    c.free_vram(); time.sleep(2)
    wf = build_zimage_workflow(PROMPT, width=1344, height=768, steps=8, cfg=1.0, seed=777001)
    t0 = time.time()
    pid = c.submit(wf)
    hist = c.wait_for_completion(pid, timeout=300, poll=1.5)
    for _n, o in hist.get("outputs", {}).items():
        for e in o.get("images", []):
            src = COMFYUI_OUTPUT / (e.get("subfolder") or "") / e["filename"]
            if src.exists():
                dest = OUT / "ref_sheet_zimage.png"
                shutil.copy2(src, dest)
                from PIL import Image
                Image.open(dest).convert("RGB").resize((832, 480), Image.LANCZOS).save(OUT / "ref_sheet.png")
                shutil.copy2(OUT / "ref_sheet.png", COMFY_INPUT / "bench_ref_sheet.png")
                print(f"OK {time.time()-t0:.0f}s -> {dest}")
                return
    print("FAIL: no image")

if __name__ == "__main__":
    main()
