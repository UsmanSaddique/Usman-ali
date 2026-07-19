"""Generate 4 song variants for The Little Red Bus Song (a599fbc7):
2 English + 2 Urdu/Hindi, using the saved ACE-Step producer-brief recipe.
Tracks are saved as INACTIVE MusicTracks (audition list) — none replaces the
current active song until selected. Run on the ComfyUI embedded python."""
import sys, time, logging
sys.path.insert(0, r"C:\Users\PC\Desktop\VideoMaker\ai-director")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

from app.config import settings
from app.services.model_manager import ModelManager
from app.services.music_gen import MusicGenService
from app.database import get_session, Project, MusicTrack

PID = "a599fbc7-cfef-40f1-84c2-8f8ba5d242f8"

# Template tail kept verbatim per the user-approved popular-rhyme recipe
TAIL = ("soft kids choir echo on the hook only, acoustic guitar strums, ukulele, "
        "glockenspiel, marimba, gentle hand claps, light kick drum on the beat, "
        "bright major key, simple repetitive singable melody toddlers can hum, "
        "wholesome kids pop, professional studio mix, clean bright master")
TAIL_DESI = ("soft kids choir echo on the hook only, dholak, tabla, harmonium, "
             "sitar plucks, glockenspiel, gentle hand claps, bright major key, "
             "simple repetitive singable melody toddlers can hum, wholesome "
             "children's music, professional studio mix, clean bright master")
VOX = ("clear solo female lead vocal with crisp enunciation, vocals loud and "
       "forward in the mix, playful call-and-response sounds sung sweetly "
       "(beep beep, round and round, swish swish, hop hop)")
VOX_DESI = ("clear solo female lead vocal with crisp enunciation, vocals loud and "
            "forward in the mix, playful call-and-response sounds sung sweetly "
            "(beep beep, chali chali, ghoomte ghoomte)")

session = get_session()
project = session.query(Project).get(PID)
lyrics_en = project.lyrics
lyrics_ur = project.lyrics_urdu
assert lyrics_en and lyrics_ur, "missing lyrics"
duration = int(project.duration_target) + 2
session.close()

VARIANTS = [
    ("en_wheels", lyrics_en,
     "classic children's nursery rhyme in the style of The Wheels on the Bus, "
     f"cheerful bouncy sing-along, {VOX}, {TAIL}, 100 bpm"),
    ("en_londonbridge", lyrics_en,
     "classic children's nursery rhyme in the style of London Bridge Is Falling Down, "
     f"cheerful skipping sing-along, {VOX}, {TAIL}, 112 bpm"),
    ("urdu_bouncy", lyrics_ur,
     "classic urdu children's nursery rhyme, cheerful bouncy South Asian kids "
     f"sing-along, {VOX_DESI}, {TAIL_DESI}, 100 bpm"),
    ("hindi_playful", lyrics_ur,
     "classic hindi children's rhyme, playful desi kids song with a skipping "
     f"clap-along rhythm, {VOX_DESI}, {TAIL_DESI}, 95 bpm"),
]

config = settings
manager = ModelManager()
svc = MusicGenService(manager, config)
project_dir = config.paths.projects_dir / PID
tag = int(time.time())

done = []
for name, lyr, style in VARIANTS:
    out = str(project_dir / f"music_v{tag}_{name}.wav")
    print(f"\n=== {name} -> {out}", flush=True)
    t0 = time.time()
    result = svc.generate(
        style_prompt=style, duration=duration, lyrics=lyr,
        output_path=out, instrumental=False,
        channel_profile=None, engine="auto",
    )
    dt = time.time() - t0
    s = get_session()
    s.add(MusicTrack(project_id=PID, style_prompt=style,
                     output_path=result.path, duration=result.duration,
                     is_active=False))
    s.commit(); s.close()
    print(f"=== {name} done in {dt/60:.1f} min: {result.path} ({result.duration:.0f}s)", flush=True)
    done.append((name, result.path, dt))

try:
    manager.unload()
except Exception:
    pass
print("\nALL DONE")
for name, p, dt in done:
    print(f"  {name}: {p} ({dt/60:.1f} min)")
