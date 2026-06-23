"""
HQ remake of Yusuf's Dance Party. Fixes the 'brush-painting faces/hands' by:
  - MEDIUM/CLOSE framing (waist-up) so the face is large enough for SDXL+LTX to
    actually resolve eyes/nose, and hands stay near the face (large -> clean).
  - Strong anatomy negatives + face/hand quality cues.
  - Real 4x-UltraSharp ESRGAN upscale to true 4K (ComfyUI), then 4K render.
(LTX 'Crisp_Enhance'/HDR LoRAs were tested and attach 0 patches to the GGUF model
 -> no-ops, so they are intentionally NOT used.)
"""
import sys, os, time, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from app.config import settings
from app.database import (get_session, Channel, Project, Scene, SceneType,
                          SceneStatus, ProjectStatus)
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.pipeline import PipelineOrchestrator

mm = ModelManager(); register_all_loaders(mm, settings)
orch = PipelineOrchestrator(settings, mm)
_t = time.time()
orch.on_progress(lambda p: print(f"{time.time()-_t:6.0f}s [{p.phase.value}] {p.message}", flush=True))

ID = ("a cute 3D pixar style cartoon muslim boy, white knit prayer cap, brown curly hair, "
      "big expressive brown eyes, light kurta")
# Framing + quality cues that make faces/hands RESOLVE instead of smearing.
FRAME = ("medium close-up shot from the waist up, face large and clearly visible facing the camera, "
         "highly detailed symmetric face, sharp clean eyes, detailed hands, smooth clean 3D render, "
         "soft studio lighting, vibrant colors, high detail")
SCENES = [
    f"{ID}, smiling and clapping hands up near the face, happily dancing, {FRAME}, in a colorful party room with balloons",
    f"{ID}, raising both arms up and cheering with joy while dancing, {FRAME}, on a bright stage with sparkling lights",
    f"{ID}, waving one hand and bouncing happily to the music, {FRAME}, in a sunny garden with confetti",
]
NEG = ("photorealistic, realistic skin, text, watermark, two boys, extra people, "
       "deformed face, distorted face, asymmetric eyes, crossed eyes, melted eyes, "
       "extra fingers, fused fingers, missing fingers, mutated hands, malformed hands, "
       "blurry, soft focus, grainy, lowres, low detail, painting, sketch")

s = get_session()
ch = s.query(Channel).filter(Channel.slug == "little-muslim-nation").first()
proj = Project(title="Yusuf's Dance Party (HQ)", channel_id=ch.id, duration_target=12,
               num_scenes_target=3, context="12s HQ dance song with Yusuf, tight framing",
               status=ProjectStatus.APPROVED,
               video_model="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf", total_scenes=3)
s.add(proj); s.flush(); pid = proj.id
for i, pr in enumerate(SCENES, 1):
    s.add(Scene(project_id=pid, scene_number=i, scene_type=SceneType.IMG2VID,
                prompt=pr, negative_prompt=NEG, duration=4.0, status=SceneStatus.PENDING))
s.commit(); s.close()
print(f"Project {pid[:8]} | 3 Yusuf HQ dancing scenes (tight framing) | LoRA cast\n")

proj_dir = settings.paths.projects_dir / pid
song = str(proj_dir / "song.mp3")
try:
    from app.services.music_gen import MusicGenService
    MusicGenService(mm, settings).generate(
        style_prompt="upbeat happy kids dance song, catchy rhythm, light percussion and claps, joyful, energetic, wholesome",
        duration=14, lyrics="[verse]\nClap your hands and move your feet\nDance along to the happy beat\n",
        instrumental=False, output_path=song)
    print("[song] ok")
except Exception as e:
    print("song:", e); song = None

# 1. generate (SDXL still pixar+yusuf -> LTX img2vid) @1024x576
orch.start_generation(pid, width=1024, height=576)

# 2. REAL 4x-UltraSharp ESRGAN upscale -> true 4K (per-scene, via ComfyUI)
TW, TH = 3840, 2160
print(f"[upscale] -> {TW}x{TH} (4x-UltraSharp ESRGAN) ...")
s2 = get_session()
scenes = s2.query(Scene).filter(Scene.project_id == pid,
                                Scene.status.in_([SceneStatus.GENERATED, SceneStatus.APPROVED])
                                ).order_by(Scene.scene_number).all()
for sc in scenes:
    gen = sc.active_generation
    if not gen or not gen.output_path:
        continue
    try:
        r = orch.upscaler.upscale_video(input_path=gen.output_path, target_width=TW,
                                        target_height=TH, ffmpeg_bin=settings.paths.ffmpeg_bin)
        gen.upscaled_path = r.output_path
        print(f"    scene {sc.scene_number} ok")
    except Exception as e:
        print(f"    scene {sc.scene_number} upscale err: {type(e).__name__}: {e}")
        gen.upscaled_path = gen.output_path
    s2.commit()
s2.close()

# 3. render @4k with song
try: orch.render(pid, narration_path=None, music_path=song, resolution="4k")
except Exception as e: print("render:", e)

final = str(proj_dir / "final_render.mp4")
print("=" * 60)
if os.path.exists(final):
    ff = settings.paths.ffmpeg_bin
    out = subprocess.run([ff, "-i", final], capture_output=True, text=True).stderr
    dur = 0.0; w = h = 0; audio = "Audio:" in out
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", out)
    if m: dur = int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3))
    mv = re.search(r"Video:.*?(\d{3,5})x(\d{3,5})", out)
    if mv: w, h = int(mv.group(1)), int(mv.group(2))
    print(f"[OK] FINAL: {final}")
    print(f"     {os.path.getsize(final)/1048576:.1f}MB | {w}x{h} | {dur:.1f}s | audio={'YES' if audio else 'no'}")
else:
    print(f"[XX] no final at {final}")
print("=" * 60); print(f"PROJECT_ID={pid}")
