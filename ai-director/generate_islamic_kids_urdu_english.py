"""
Generate 10 Islamic kids songs via ACE-Step v1.5 XL SFT:
  - 5 Urdu poems (rhythm modeled on popular Urdu nursery rhymes / Iqbal's verses)
  - 5 English poems (rhythm modeled on popular nursery rhymes)

All vocals + music tuned for studio-balanced mix: vocal-forward, clean,
warm instrumentation, gentle percussion. Sequential submission (18GB model).
"""
import json
import time
import uuid
import shutil
import random
import urllib.request
import urllib.error
from pathlib import Path

COMFY = "http://127.0.0.1:8188"
COMFY_OUT = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\output")
DEST = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\assets_generated\music\islamic_kids_bilingual")
DEST.mkdir(parents=True, exist_ok=True)

# Studio-balanced mix tags — engineered for clear vocals over warm instrumentation
STUDIO_MIX = (
    "professional studio mix, crystal clear vocals, vocal forward, "
    "balanced mix, warm mastering, gentle compression, soft reverb, "
    "no harsh frequencies, smooth high end, full low end, broadcast quality"
)

# English kids song base — cocomelon / super simple songs aesthetic
EN_BASE = (
    "children's nursery rhyme, kids song, female lead vocal, "
    "children choir backing, ukulele, glockenspiel, soft piano, "
    "light hand percussion, gentle strings, cheerful, sing-along, "
    "happy, uplifting, simple melody, clear pronunciation"
)

# Urdu nasheed / kids song base — duff frame drum, vocal-forward nasheed
UR_BASE = (
    "urdu nasheed, islamic kids song, female lead vocal, "
    "children choir backing, soft duff frame drum, ney flute, "
    "qanun, soft piano, gentle hand percussion, warm strings, "
    "south asian melodic style, devotional yet playful, "
    "clear urdu pronunciation, sing-along, melodious"
)


SONGS = [
    # ============ ENGLISH (5) ============
    {
        "slug": "en_01_bismillah_twinkle",
        "lang": "en",
        "base": EN_BASE,
        "extra": "in the style of Twinkle Twinkle Little Star, lullaby tempo, sweet",
        "bpm": 80,
        "keyscale": "C major",
        "lyrics": """[verse]
Bismillah before I eat,
Bismillah before I sleep.
In the name of Allah dear,
Every moment He is near.

[chorus]
Bismillah, Bismillah, say it every day,
Bismillah, Bismillah, light along the way.
Bismillah, Bismillah, sweet and soft and true,
In the name of Allah, He takes care of you.

[verse]
Bismillah when I wake,
Bismillah for every sake.
Open hands and open heart,
Bismillah is how I start.

[chorus]
Bismillah, Bismillah, say it every day,
Bismillah, Bismillah, light along the way.""",
    },
    {
        "slug": "en_02_alhamdulillah_mary",
        "lang": "en",
        "base": EN_BASE,
        "extra": "in the style of Mary Had a Little Lamb, walking tempo, gentle bounce",
        "bpm": 100,
        "keyscale": "G major",
        "lyrics": """[verse]
Thank you Allah for the sun,
For the sun, for the sun.
Thank you Allah for the sun,
Shining warm and bright.

[chorus]
Alhamdulillah, Alhamdulillah,
For everything I see.
Alhamdulillah, Alhamdulillah,
Allah cares for me.

[verse]
Thank you Allah for my mom,
For my mom, for my dad.
Thank you Allah for my home,
Every day I'm glad.

[chorus]
Alhamdulillah, Alhamdulillah,
For everything I see.
Alhamdulillah, Alhamdulillah,
Allah cares for me.""",
    },
    {
        "slug": "en_03_prophet_muhammad_wheels",
        "lang": "en",
        "base": EN_BASE,
        "extra": "in the style of The Wheels on the Bus, marching tempo, playful bounce",
        "bpm": 112,
        "keyscale": "D major",
        "lyrics": """[verse]
The Prophet Muhammad was kind and true,
Kind and true, kind and true.
The Prophet Muhammad was kind and true,
Peace be upon him.

[chorus]
He smiled at children passing by,
Passing by, passing by.
He helped the poor and lifted high,
Peace be upon him.

[verse]
He told the truth in everything,
Everything, everything.
He thanked Allah for every spring,
Peace be upon him.

[chorus]
Follow his way and you will see,
You will see, you will see.
A gentle heart is what we'll be,
Peace be upon him.""",
    },
    {
        "slug": "en_04_wudu_old_macdonald",
        "lang": "en",
        "base": EN_BASE,
        "extra": "in the style of Old MacDonald, cheerful, lively folk",
        "bpm": 108,
        "keyscale": "F major",
        "lyrics": """[verse]
Little Ahmed makes his wudu, splish-splash splish-splash.
With a wash here and a wash there,
Here a wash, there a wash, everywhere a clean-wash.
Little Ahmed makes his wudu, splish-splash splish-splash.

[chorus]
Wudu, wudu, clean and bright,
Ready for the prayer tonight.
Wudu, wudu, sparkle clean,
Best little wudu ever seen.

[verse]
Wash my hands and rinse my mouth, splish-splash splish-splash.
Sniff the water, wash my face,
Arms and head and ears with grace, feet so clean in any case.
Little Ahmed makes his wudu, splish-splash splish-splash.

[chorus]
Wudu, wudu, clean and bright,
Ready for the prayer tonight.""",
    },
    {
        "slug": "en_05_names_allah_lullaby",
        "lang": "en",
        "base": EN_BASE,
        "extra": "in the style of a gentle lullaby like Rock-a-bye Baby, sleepy, dreamy",
        "bpm": 72,
        "keyscale": "A minor",
        "lyrics": """[verse]
Ar-Rahman, the most kind,
Ar-Raheem, the loving mind.
As-Salam, the giver of peace,
Whisper softly, troubles cease.

[chorus]
Beautiful names of Allah dear,
Beautiful names, He is near.
Beautiful names, sweet to say,
Closing eyes at end of day.

[verse]
Al-Wadud, the loving one,
An-Noor, the gentle sun.
Al-Hafiz keeps you safe at night,
Sleep my child until the light.

[chorus]
Beautiful names of Allah dear,
Beautiful names, He is near.""",
    },

    # ============ URDU (5) ============
    {
        "slug": "ur_01_lab_pe_dua",
        "lang": "ur",
        "base": UR_BASE,
        "extra": "in the style of Lab Pe Aati Hai Dua by Allama Iqbal, classic urdu nasheed, devotional",
        "bpm": 78,
        "keyscale": "D minor",
        "lyrics": """[verse]
Lab pe aati hai dua ban ke tamanna meri,
Zindagi shama ki soorat ho khudaya meri.

[chorus]
Ya Allah meri dua suniye,
Mere dil mein noor bhar dijiye.
Ya Allah meri dua suniye,
Achhe bachon mein shumar kijiye.

[verse]
Door duniya ka mere dam se andhera ho jaye,
Har jagah mere chamakne se ujala ho jaye.

[chorus]
Ya Allah meri dua suniye,
Mere dil mein noor bhar dijiye.
Ya Allah meri dua suniye,
Achhe bachon mein shumar kijiye.""",
    },
    {
        "slug": "ur_02_bismillah_chanda_mama",
        "lang": "ur",
        "base": UR_BASE,
        "extra": "in the style of Chanda Mama Door Ke, gentle south asian lullaby, sweet",
        "bpm": 84,
        "keyscale": "G major",
        "lyrics": """[verse]
Bismillah keh ke shuru karoon,
Allah ka naam main har dam padhoon.
Khana ho ya ho koi kaam,
Bismillah mera pyara naam.

[chorus]
Bismillah, Bismillah, har kaam se pehle,
Bismillah, Bismillah, har lamhe har pal mein.
Bismillah kehne se barkat aati hai,
Allah ki rehmat saath nibhati hai.

[verse]
Subah uthoon to Bismillah,
Raat soun to Bismillah.
Har qadam pe yeh kalma pyara,
Allah ka noor sab se nyara.

[chorus]
Bismillah, Bismillah, har kaam se pehle,
Bismillah, Bismillah, har lamhe har pal mein.""",
    },
    {
        "slug": "ur_03_alhamdulillah_aloo_bacha",
        "lang": "ur",
        "base": UR_BASE,
        "extra": "in the style of Aloo Bacha kids song, playful bouncy urdu rhyme, cheerful",
        "bpm": 104,
        "keyscale": "C major",
        "lyrics": """[verse]
Suraj nikla, chiriya boli,
Phool khile aur hawa doli.
Allah ne yeh sab banaya,
Shukriya kehna humein sikhaya.

[chorus]
Alhamdulillah, Alhamdulillah,
Har neimat ke liye shukriya.
Alhamdulillah, Alhamdulillah,
Allah ka karam mujh par hua.

[verse]
Ammi abbu pyar karte,
Bhai behen sang khel sajte.
Khana paani, ghar bhi nyara,
Allah ne sab kuch sanwara.

[chorus]
Alhamdulillah, Alhamdulillah,
Har neimat ke liye shukriya.
Alhamdulillah, Alhamdulillah,
Allah ka karam mujh par hua.""",
    },
    {
        "slug": "ur_04_nabi_ka_pyar_bandar_mama",
        "lang": "ur",
        "base": UR_BASE,
        "extra": "in the style of Bandar Mama children's rhyme, march tempo, friendly storytelling",
        "bpm": 100,
        "keyscale": "F major",
        "lyrics": """[verse]
Nabi humare pyare pyare,
Sab bachon ko bohot pukare.
Muskurate, sab se milte,
Khushiyon ke phool sab pe khilte.

[chorus]
Sallalahu alaihi wasallam,
Sallalahu alaihi wasallam.
Pyare Nabi ka rasta apnayen,
Achhe bachon mein hum kehlayen.

[verse]
Sach bolte aur sach sikhate,
Bhookhe ko khana bhi khilate.
Garib yatim ka khayal rakha,
Har dil mein woh ghar bana.

[chorus]
Sallalahu alaihi wasallam,
Sallalahu alaihi wasallam.
Pyare Nabi ka rasta apnayen,
Achhe bachon mein hum kehlayen.""",
    },
    {
        "slug": "ur_05_panch_namazein_iqbal_style",
        "lang": "ur",
        "base": UR_BASE,
        "extra": "in the style of classical urdu taraana, educational, melodic march",
        "bpm": 92,
        "keyscale": "E minor",
        "lyrics": """[verse]
Fajr ki azaan jab hoti hai,
Subah ki rehmat khulti hai.
Zuhr dopahar mein parhte hain,
Asr ko bhi nahi bhulte hain.

[chorus]
Panch namazein, panch khazane,
Allah ke darbar ke pyare nazrane.
Fajr, Zuhr, Asr, Maghrib, Isha,
Sab ko parho, mile gham se nijaat.

[verse]
Maghrib jab suraj dhalta hai,
Isha ko taara nikalta hai.
Allah se baatein karte hain,
Dil ke har dard ko harte hain.

[chorus]
Panch namazein, panch khazane,
Allah ke darbar ke pyare nazrane.
Fajr, Zuhr, Asr, Maghrib, Isha,
Sab ko parho, mile gham se nijaat.""",
    },
]


def build_workflow(tags: str, lyrics: str, seconds: float, seed: int,
                   bpm: int, language: str, keyscale: str) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "acestep_v1.5_xl_sft_bf16.safetensors",
            "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "qwen_0.6b_ace15.safetensors",
            "clip_name2": "qwen_1.7b_ace15.safetensors",
            "type": "ace"}},
        "3": {"class_type": "VAELoader", "inputs": {
            "vae_name": "ace_1.5_vae.safetensors"}},
        "4": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {
            "clip": ["2", 0], "tags": tags, "lyrics": lyrics,
            "seed": seed, "bpm": bpm, "duration": float(seconds),
            "timesignature": "4", "language": language, "keyscale": keyscale,
            "generate_audio_codes": True, "cfg_scale": 2.0,
            "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0}},
        "5": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {
            "seconds": float(seconds), "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["4", 0],
            "latent_image": ["5", 0], "seed": seed, "steps": 50, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveAudio", "inputs": {
            "audio": ["7", 0], "filename_prefix": "islamic_kids_bilingual"}},
    }


def http_post(url, payload, timeout=30):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http_get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(timeout=600.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{COMFY}/system_stats", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("ComfyUI not reachable")


def free_vram():
    try:
        req = urllib.request.Request(
            f"{COMFY}/free", method="POST",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        time.sleep(2)
    except Exception:
        pass


def submit(wf):
    cid = uuid.uuid4().hex[:12]
    res = http_post(f"{COMFY}/prompt", {"prompt": wf, "client_id": cid})
    if "prompt_id" not in res:
        raise RuntimeError(f"Rejected: {res}")
    return res["prompt_id"]


def wait_done(pid, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            hist = http_get(f"{COMFY}/history/{pid}")
            if pid in hist:
                h = hist[pid]
                st = h.get("status", {})
                if st.get("completed"):
                    return h
                if st.get("status_str") == "error":
                    raise RuntimeError(f"Error: {st.get('messages')}")
        except urllib.error.HTTPError:
            pass
        time.sleep(3)
    raise TimeoutError(f"Timeout on {pid}")


def collect(hist, dest):
    for _nid, nout in hist.get("outputs", {}).items():
        for entry in nout.get("audio", []):
            fn = entry.get("filename", "")
            sub = entry.get("subfolder", "")
            if not fn:
                continue
            src = COMFY_OUT / sub / fn if sub else COMFY_OUT / fn
            if src.exists():
                shutil.copy2(str(src), str(dest))
                return True
    return False


def main():
    print("[Setup] Waiting for ComfyUI ...")
    wait_ready(timeout=600)
    print("[Setup] ComfyUI ready.\n")

    duration_sec = 60.0   # 1-minute songs
    results = []
    overall_t0 = time.time()

    for idx, song in enumerate(SONGS, 1):
        out_audio = DEST / f"{song['slug']}.flac"
        out_lyrics = DEST / f"{song['slug']}_lyrics.txt"
        out_lyrics.write_text(song["lyrics"], encoding="utf-8")

        tags = f"{song['base']}, {song['extra']}, {STUDIO_MIX}"
        seed = random.randint(0, 2**31 - 1)
        wf = build_workflow(
            tags=tags, lyrics=song["lyrics"], seconds=duration_sec,
            seed=seed, bpm=song["bpm"], language=song["lang"],
            keyscale=song["keyscale"],
        )

        print(f"[{idx:2d}/10] {song['slug']}  lang={song['lang']}  bpm={song['bpm']}  key={song['keyscale']}  seed={seed}")
        t0 = time.time()
        try:
            free_vram()
            pid = submit(wf)
            print(f"        prompt_id={pid}")
            hist = wait_done(pid, timeout=1800)
            if collect(hist, out_audio):
                elapsed = time.time() - t0
                size_mb = out_audio.stat().st_size / 1e6
                print(f"        OK in {elapsed:.0f}s  ->  {out_audio.name} ({size_mb:.1f} MB)\n")
                results.append((song["slug"], "ok", elapsed))
            else:
                print(f"        FAIL no audio in outputs\n")
                results.append((song["slug"], "no_output", time.time() - t0))
        except Exception as e:
            print(f"        ERROR: {e}\n")
            results.append((song["slug"], f"error: {str(e)[:80]}", time.time() - t0))

    total = time.time() - overall_t0
    print(f"\n=== SUMMARY  (total {total/60:.1f} min) ===")
    for slug, status, secs in results:
        print(f"  {slug:35s} {status:20s} {secs:5.0f}s")
    ok = sum(1 for _, s, _ in results if s == "ok")
    print(f"\n{ok}/10 succeeded.  Outputs in: {DEST}")


if __name__ == "__main__":
    main()
