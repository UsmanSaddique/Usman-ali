"""
Generate 10 Islamic-themed kids songs (Cocomelon-style) via ACE-Step v1.5 XL SFT.
Sequential submission to ComfyUI — model is 18GB so parallel won't fit.
Each song: 50-step SFT, female lead vocal + kids choir backing, balanced mix.
"""
import json
import time
import uuid
import shutil
import random
import urllib.request
from pathlib import Path

COMFY = "http://127.0.0.1:8188"
COMFY_OUT = Path(r"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\output")
DEST = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\assets_generated\music\islamic_kids")
DEST.mkdir(parents=True, exist_ok=True)

# Cocomelon-style kid music tags — bright, simple, melodic, vocals forward
BASE_TAGS = (
    "children's music, kids song, nursery rhyme, female lead vocal, "
    "children choir backing, ukulele, glockenspiel, soft piano, "
    "gentle hand percussion, light strings, warm, happy, uplifting, "
    "clear vocals, balanced mix, middle eastern flavor, nasheed, "
    "duff frame drum, no instruments harsh, cheerful, sing-along"
)

SONGS = [
    {
        "slug": "01_bismillah",
        "extra": "playful, bouncy",
        "bpm": 96,
        "lyrics": """[verse]
Before I eat and before I play,
Bismillah is what I say.
In the name of Allah, kind and true,
Everything good begins with you.

[chorus]
Bismillah, Bismillah, say it with a smile,
Bismillah, Bismillah, every little while.
Bismillah, Bismillah, before I start my day,
In the name of Allah, that's the lovely way.

[verse]
When I wake and when I rest,
Bismillah makes the moment blessed.
Open hands and open heart,
Bismillah is the perfect start.

[chorus]
Bismillah, Bismillah, say it with a smile,
Bismillah, Bismillah, every little while.""",
    },
    {
        "slug": "02_alhamdulillah",
        "extra": "joyful, sunny",
        "bpm": 104,
        "lyrics": """[verse]
For the sun that warms the sky,
For the birds that learn to fly,
For my food and for my friends,
For the love that never ends.

[chorus]
Alhamdulillah, thank you Allah,
For every little thing.
Alhamdulillah, thank you Allah,
That's the song I sing.

[verse]
For my mom and for my dad,
For the days that make me glad,
For the rain and for the trees,
For the gentle morning breeze.

[chorus]
Alhamdulillah, thank you Allah,
For every little thing.
Alhamdulillah, thank you Allah,
That's the song I sing.""",
    },
    {
        "slug": "03_five_pillars",
        "extra": "educational, marching",
        "bpm": 100,
        "lyrics": """[verse]
One is Shahada, I believe,
There is no god but Allah I receive.
Two is Salah, five times each day,
I bow my head and softly pray.

[chorus]
Five little pillars, strong and tall,
Five little pillars hold them all.
Shahada, Salah, Zakat, Sawm, Hajj,
Five little pillars, that's our path.

[verse]
Three is Zakat, share what you can,
Help your neighbor, give a hand.
Four is Sawm in Ramadan,
Five is Hajj if able, then.

[chorus]
Five little pillars, strong and tall,
Five little pillars hold them all.""",
    },
    {
        "slug": "04_wudu_song",
        "extra": "gentle, splashing",
        "bpm": 92,
        "lyrics": """[verse]
Wash my hands one, two, three,
Rinse my mouth, so clean is me.
Sniff some water through my nose,
Wash my face from chin to brows.

[chorus]
Wudu, wudu, splish and splash,
Getting ready in a flash.
Wudu, wudu, clean and bright,
Ready now for Allah's light.

[verse]
Arms up to the elbows now,
Wipe my head and ears somehow.
Wash my feet up to the ankle,
Bismillah, the water sparkles.

[chorus]
Wudu, wudu, splish and splash,
Getting ready in a flash.""",
    },
    {
        "slug": "05_five_prayers",
        "extra": "soothing, melodic",
        "bpm": 88,
        "lyrics": """[verse]
Fajr in the morning, before the sun,
Whispering softly, day's begun.
Dhuhr when the sun is overhead,
Stand and bow with what is said.

[chorus]
Five times a day I talk to Allah,
Five times a day I say Subhanallah.
Fajr, Dhuhr, Asr, Maghrib, Isha,
Five little prayers, I love them all.

[verse]
Asr afternoon, golden and warm,
Maghrib at sunset, soft and calm.
Isha at night when stars appear,
Allah is close and always near.

[chorus]
Five times a day I talk to Allah,
Five times a day I say Subhanallah.""",
    },
    {
        "slug": "06_arabic_alphabet",
        "extra": "alphabet song, playful",
        "bpm": 108,
        "lyrics": """[verse]
Alif Baa Taa Thaa, sing along with me,
Jeem Haa Khaa, easy as can be.
Daal Dhaal Raa Zay, clap your little hands,
Seen Sheen Saad Daad, marching in the sand.

[chorus]
Arabic letters one by one,
Learning them is so much fun.
Alif Baa Taa Thaa, sing the song,
All my friends can sing along.

[verse]
Taa Dhaa Ayn Ghayn, lift your voice up high,
Faa Qaaf Kaaf Laam, reaching for the sky.
Meem Noon Haa Waw, almost at the end,
Yaa is last, hello my friend.

[chorus]
Arabic letters one by one,
Learning them is so much fun.""",
    },
    {
        "slug": "07_names_of_allah",
        "extra": "reverent, gentle, lullaby",
        "bpm": 76,
        "lyrics": """[verse]
Ar-Rahman, the most kind,
Ar-Raheem, the most merciful mind.
As-Salam, the giver of peace,
Al-Khaliq, whose making does not cease.

[chorus]
Beautiful names, beautiful names,
Allah has the loveliest names.
Whisper them softly, sing them sweet,
Beautiful names, our hearts repeat.

[verse]
Al-Wadud, the loving one,
An-Noor, the light like the sun.
Al-Hafiz, who keeps me safe,
Al-Karim, with endless grace.

[chorus]
Beautiful names, beautiful names,
Allah has the loveliest names.""",
    },
    {
        "slug": "08_prophets_song",
        "extra": "storytelling, gentle march",
        "bpm": 96,
        "lyrics": """[verse]
Adam was the very first man,
Nuh built a great big plan.
Ibrahim with a kind heart,
Musa parted seas apart.

[chorus]
Prophets of Allah, brave and true,
Showing kindness, me and you.
Peace be upon them, every one,
Their stories shine just like the sun.

[verse]
Yusuf in a coat of dreams,
Dawud sang by quiet streams.
Isa healed and showed the way,
Muhammad came to guide our day.

[chorus]
Prophets of Allah, brave and true,
Showing kindness, me and you.""",
    },
    {
        "slug": "09_manners_salaam",
        "extra": "polite, friendly, swing",
        "bpm": 112,
        "lyrics": """[verse]
When I see my friend I say,
Assalamu alaikum, have a lovely day.
If you bump and need to share,
Say excuse me, show you care.

[chorus]
Manners, manners, simple and sweet,
Manners, manners, make our hearts meet.
Please and thank you, smile and share,
Allah loves the kind and fair.

[verse]
Help your mama, hug your dad,
Cheer your sibling if they're sad.
Take your shoes off at the door,
Pick up toys from off the floor.

[chorus]
Manners, manners, simple and sweet,
Manners, manners, make our hearts meet.""",
    },
    {
        "slug": "10_ramadan_eid",
        "extra": "celebratory, festive, joyful",
        "bpm": 116,
        "lyrics": """[verse]
Ramadan is here at last,
Crescent moon is shining fast.
Daddy fasts and mama too,
Iftar dates for me and you.

[chorus]
Ramadan, Ramadan, blessings every night,
Ramadan, Ramadan, lanterns shining bright.
Eid Mubarak, hugs all around,
Sweetest day we ever found.

[verse]
Thirty days of giving more,
Sharing food with rich and poor.
Then comes Eid in colors new,
Hugs and gifts and laughter too.

[chorus]
Ramadan, Ramadan, blessings every night,
Ramadan, Ramadan, lanterns shining bright.""",
    },
]


def build_workflow(tags: str, lyrics: str, seconds: float, seed: int, bpm: int) -> dict:
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
            "timesignature": "4", "language": "en", "keyscale": "C major",
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
            "audio": ["7", 0], "filename_prefix": "islamic_kids"}},
    }


def http_post(url: str, payload: dict, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http_get(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(timeout: float = 300.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{COMFY}/system_stats", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("ComfyUI not reachable")


def submit(wf: dict) -> str:
    cid = uuid.uuid4().hex[:12]
    res = http_post(f"{COMFY}/prompt", {"prompt": wf, "client_id": cid})
    if "prompt_id" not in res:
        raise RuntimeError(f"Rejected: {res}")
    return res["prompt_id"]


def wait_done(pid: str, timeout: int = 1800) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            hist = http_get(f"{COMFY}/history/{pid}")
            if pid in hist:
                h = hist[pid]
                if h.get("status", {}).get("completed"):
                    return h
                if h.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(f"Error: {h['status'].get('messages')}")
        except urllib.error.HTTPError:
            pass
        time.sleep(3)
    raise TimeoutError(f"Timeout on {pid}")


def collect(hist: dict, dest: Path) -> bool:
    for nid, nout in hist.get("outputs", {}).items():
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
    print("[Setup] Waiting for ComfyUI...")
    wait_ready(timeout=300)
    print("[Setup] ComfyUI ready.\n")

    duration_sec = 60.0   # 1-minute song each
    results = []
    overall_t0 = time.time()

    for idx, song in enumerate(SONGS, 1):
        out_wav = DEST / f"{song['slug']}.flac"
        out_lyrics = DEST / f"{song['slug']}_lyrics.txt"
        out_lyrics.write_text(song["lyrics"], encoding="utf-8")

        tags = f"{BASE_TAGS}, {song['extra']}"
        seed = random.randint(0, 2**31 - 1)
        wf = build_workflow(tags, song["lyrics"], duration_sec, seed, song["bpm"])

        print(f"[{idx}/10] {song['slug']}  bpm={song['bpm']} seed={seed}")
        t0 = time.time()
        try:
            pid = submit(wf)
            print(f"        prompt_id={pid}")
            hist = wait_done(pid, timeout=1800)
            if collect(hist, out_wav):
                elapsed = time.time() - t0
                size_mb = out_wav.stat().st_size / 1e6
                print(f"        ✓ done in {elapsed:.0f}s  ->  {out_wav.name} ({size_mb:.1f} MB)\n")
                results.append((song["slug"], "ok", elapsed))
            else:
                print(f"        ✗ no audio in outputs\n")
                results.append((song["slug"], "no_output", time.time() - t0))
        except Exception as e:
            print(f"        ✗ ERROR: {e}\n")
            results.append((song["slug"], f"error: {e}", time.time() - t0))

    total = time.time() - overall_t0
    print(f"\n=== SUMMARY  (total {total/60:.1f} min) ===")
    for slug, status, secs in results:
        print(f"  {slug:25s} {status:15s} {secs:5.0f}s")
    print(f"\nOutputs in: {DEST}")


if __name__ == "__main__":
    main()
