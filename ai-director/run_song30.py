"""
30s SONG video with REAL motion:
  - reuse an existing scripted project, force scenes -> img2vid
  - each scene: consistent SDXL still -> LTX img2vid (character actually MOVES)
  - audio = a generated SONG (ACE-Step), NO TTS voiceover
  - upscale to 1080p, assemble
QA at the end: motion clips present, song muxed, 1080p, ~30s.
"""
import sys, os, time, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from app.config import settings
from app.database import get_session, Project, Scene, SceneType, SceneStatus
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.pipeline import PipelineOrchestrator

SRC = sys.argv[1] if len(sys.argv) > 1 else "ec26a326"
mm = ModelManager(); register_all_loaders(mm, settings)
orch = PipelineOrchestrator(settings, mm)
_t = time.time()
orch.on_progress(lambda p: print(f"{time.time()-_t:6.0f}s [{p.phase.value}] {p.message}", flush=True))

s = get_session()
p = s.query(Project).filter(Project.id.like(SRC + "%")).first()
pid = p.id
scenes = s.query(Scene).filter(Scene.project_id == pid).order_by(Scene.scene_number).all()[:5]
keep = {sc.id for sc in scenes}
for sc in s.query(Scene).filter(Scene.project_id == pid).all():
    if sc.id in keep:
        sc.scene_type = SceneType.IMG2VID   # SDXL still -> LTX animate
        sc.status = SceneStatus.PENDING
        sc.retry_count = 0
        sc.duration = 6.0                    # 5 x 6s = 30s
    else:
        sc.status = SceneStatus.APPROVED  # ignore extras
        sc.scene_type = sc.scene_type
s.query(Scene).filter(Scene.project_id == pid, ~Scene.id.in_(keep)).delete(synchronize_session=False)
s.commit()
print(f"Project {pid[:8]}: 5 scenes -> img2vid @6s (motion), song audio\n")
s.close()

# 1) Generate the SONG first (ACE-Step, ~32s, instrumental song so it doesn't fight a story)
proj_dir = settings.paths.projects_dir / pid
song = str(proj_dir / "song.mp3")
t0 = time.time()
try:
    from app.services.music_gen import MusicGenService
    mg = MusicGenService(mm, settings)
    mg.generate(style_prompt="upbeat cheerful kids song, acoustic guitar, playful, bright, catchy, soft vocals",
                duration=32, lyrics="[verse]\nShare a little, share a lot\nKindness is the best we've got\n",
                instrumental=False, output_path=song)
    print(f"[song] {song} ({time.time()-t0:.0f}s)")
except Exception as e:
    print("song failed:", e); song = None

# 2) Generate motion clips (SDXL still -> LTX img2vid), at 768x512 (VRAM-safe), then upscale
t0 = time.time()
orch.start_generation(pid, width=768, height=512)
print(f"[generation/motion] {time.time()-t0:.0f}s")
try: orch.start_upscale(pid)
except Exception as e: print("upscale:", e)

# 3) Render with SONG only (no narration)
try: orch.render(pid, narration_path=None, music_path=song)
except Exception as e: print("render:", e)

# QA
final = str(proj_dir / "final_render.mp4")
print("=" * 60)
if os.path.exists(final):
    ff = settings.paths.ffmpeg_bin
    out = subprocess.run([ff, "-i", final], capture_output=True, text=True).stderr
    dur=0; w=h=0; audio="Audio:" in out
    m=re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", out);
    if m: dur=int(m.group(1))*3600+int(m.group(2))*60+float(m.group(3))
    mv=re.search(r"Video:.*?(\d{3,5})x(\d{3,5})", out)
    if mv: w,h=int(mv.group(1)),int(mv.group(2))
    clips = list((proj_dir/'clips').glob('*_hd.mp4')) or list((proj_dir/'clips').glob('*.mp4'))
    print(f"[OK] FINAL: {final}")
    print(f"     {os.path.getsize(final)/1048576:.1f}MB | {w}x{h} | {dur:.1f}s | audio(song)={'YES' if audio else 'no'} | 1080p={'YES' if h>=1000 else 'no'}")
    print(f"     motion clips: {len(clips)} | i2v stills: {len(list((proj_dir/'images').glob('*.png')))}")
else:
    print(f"[XX] no final at {final}")
print("=" * 60)
print(f"PROJECT_ID={pid}")
