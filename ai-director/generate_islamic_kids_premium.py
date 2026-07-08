"""
PREMIUM Islamic Kids Songs — ACE-Step v1.5 XL SFT
  - 5 English + 5 Urdu, ~4.5 min each (270s)
  - 200 diffusion steps for studio-grade quality
  - Full song structure: [intro] [hook] [verse] [chorus] [verse] [chorus] [bridge] [chorus] [outro]
  - Engineered for professional vocal-music balance
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
DEST = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\assets_generated\music\islamic_kids_premium")
DEST.mkdir(parents=True, exist_ok=True)

# Premium studio mix — mastered for kids content, broadcast-ready
STUDIO_PREMIUM = (
    "professional studio recording, crystal clear vocal mix, vocal forward, "
    "warm analog mastering, multiband compression, stereo width, "
    "soft plate reverb on vocals, dry tight percussion, "
    "balanced frequency spectrum, radio ready, high fidelity, "
    "24bit quality, no clipping, no distortion, "
    "vocals -3dB above instruments, clear diction, "
    "broadcast loudness standard, polished production"
)

EN_BASE = (
    "children's song, kids music, female lead vocal, "
    "children choir harmony backing vocals, "
    "acoustic guitar, ukulele, glockenspiel, soft grand piano, "
    "light hand claps, shaker, gentle tambourine, warm pad synth, "
    "light orchestral strings, music box accents, "
    "happy, uplifting, catchy melody, memorable hook, "
    "sing-along, clear english pronunciation, nasheed influence"
)

UR_BASE = (
    "urdu nasheed, islamic children song, female lead vocal, "
    "children choir harmony backing vocals, "
    "soft duff frame drum, ney flute, qanun plucks, "
    "soft grand piano, acoustic guitar, warm pad, "
    "light tabla, gentle sitar accents, warm strings, "
    "south asian melodic style, devotional yet playful, "
    "catchy melody, memorable hook, clear urdu diction, "
    "sing-along, melodious"
)

SONGS = [
    # ============ ENGLISH (5) ============
    {
        "slug": "en_01_bismillah_twinkle",
        "lang": "en",
        "base": EN_BASE,
        "extra": "twinkle twinkle little star melody style, gentle lullaby, sweet innocent",
        "bpm": 80,
        "keyscale": "C major",
        "lyrics": """[intro]
(gentle music box melody, soft piano)

[hook]
Bismillah, Bismillah, before I start my day,
Bismillah, Bismillah, the beautiful way.

[verse]
Bismillah before I eat my food,
Bismillah it puts me in the mood.
In the name of Allah kind and true,
Every single moment starts brand new.
When I open up a book to read,
Bismillah is all I really need.

[chorus]
Bismillah, Bismillah, say it with a smile,
Bismillah, Bismillah, every little while.
Bismillah, Bismillah, sweet and soft and true,
In the name of Allah, He takes care of you.
La la la la la la, He takes care of you.

[verse]
Bismillah when I wake up from my sleep,
Bismillah before I count my sheep.
When I tie my shoes and comb my hair,
Bismillah reminds me Allah's there.
Walking to the masjid hand in hand,
Bismillah across this beautiful land.

[chorus]
Bismillah, Bismillah, say it with a smile,
Bismillah, Bismillah, every little while.
Bismillah, Bismillah, sweet and soft and true,
In the name of Allah, He takes care of you.
La la la la la la, He takes care of you.

[bridge]
Every step I take, every breath I breathe,
Allah watches over you and me.
So whisper Bismillah, let your heart be light,
From the morning sun until the night.

[chorus]
Bismillah, Bismillah, say it with a smile,
Bismillah, Bismillah, every little while.
Bismillah, Bismillah, sweet and soft and true,
In the name of Allah, He takes care of you.

[outro]
Bismillah, Bismillah,
In the name of Allah.
(soft fade, music box)""",
    },
    {
        "slug": "en_02_alhamdulillah_happy",
        "lang": "en",
        "base": EN_BASE,
        "extra": "if you're happy and you know it melody style, clapping rhythm, joyful bounce",
        "bpm": 110,
        "keyscale": "G major",
        "lyrics": """[intro]
(cheerful ukulele strum, hand claps)

[hook]
Alhamdulillah for everything,
Alhamdulillah let's clap and sing!

[verse]
If you love Allah and you know it clap your hands,
If you love Allah and you know it clap your hands.
If you love Allah and you know it,
And you really want to show it,
If you love Allah and you know it clap your hands!

[chorus]
Alhamdulillah, Alhamdulillah,
Thank you Allah for the sun and the stars.
Alhamdulillah, Alhamdulillah,
Thank you Allah for just who we are.
We say Alhamdulillah!

[verse]
Thank you for the rain that helps the flowers grow,
Thank you for the wind that makes the breezes blow.
Thank you for my mama and my baba too,
Thank you for my friends and all the things we do.
Thank you for the food upon my plate,
Thank you Allah you are truly great.

[chorus]
Alhamdulillah, Alhamdulillah,
Thank you Allah for the sun and the stars.
Alhamdulillah, Alhamdulillah,
Thank you Allah for just who we are.
We say Alhamdulillah!

[bridge]
When I see a rainbow after rain,
When I hear the birds call out my name,
I know it's Allah's love that fills the earth,
Every blessing shows how much we're worth.

[chorus]
Alhamdulillah, Alhamdulillah,
Thank you Allah for the sun and the stars.
Alhamdulillah, Alhamdulillah,
Thank you Allah for just who we are.
We say Alhamdulillah!

[outro]
Alhamdulillah, Alhamdulillah,
Thank you, thank you Allah!
(claps fade, ukulele outro)""",
    },
    {
        "slug": "en_03_prophet_muhammad_love",
        "lang": "en",
        "base": EN_BASE,
        "extra": "baby shark melody style, catchy repetitive hook, bouncy energetic",
        "bpm": 115,
        "keyscale": "D major",
        "lyrics": """[intro]
(bright piano riff, kids laughing)

[hook]
Prophet Muhammad, peace be upon him,
Prophet Muhammad, we love him!
Prophet Muhammad, peace be upon him,
The best of all creation!

[verse]
He was gentle, he was kind,
The most beautiful heart you'd ever find.
He loved children, hugged them tight,
He would smile and be so bright.
Never angry, always fair,
Showed the whole world how to care.

[chorus]
Prophet Muhammad, peace be upon him,
Prophet Muhammad, we love him!
Prophet Muhammad, peace be upon him,
The best of all, the best of all,
The best of all creation!

[verse]
He would feed the hungry poor,
He would welcome at his door.
He was honest, spoke the truth,
Loved the old and loved the youth.
He would help his family too,
He is the example, me and you.

[chorus]
Prophet Muhammad, peace be upon him,
Prophet Muhammad, we love him!
Prophet Muhammad, peace be upon him,
The best of all, the best of all,
The best of all creation!

[bridge]
Sallallahu alaihi wasallam,
Sallallahu alaihi wasallam.
We send salawat with all our heart,
He taught us love right from the start.

[chorus]
Prophet Muhammad, peace be upon him,
Prophet Muhammad, we love him!
Prophet Muhammad, peace be upon him,
The best of all, the best of all,
The best of all creation!

[outro]
Peace be upon him,
Sallallahu alaihi wasallam.
(gentle fade)""",
    },
    {
        "slug": "en_04_five_pillars_fingers",
        "lang": "en",
        "base": EN_BASE,
        "extra": "five little ducks melody style, counting song, educational march",
        "bpm": 104,
        "keyscale": "F major",
        "lyrics": """[intro]
(playful glockenspiel counting notes)

[hook]
Five little pillars of Islam so grand,
Five little pillars help us understand!
Count on your fingers, one two three four five,
Five little pillars keep our faith alive!

[verse]
Number one is Shahada, say it loud and clear,
La ilaha illallah, nothing else I fear.
Muhammad is the messenger, I believe it's true,
Shahada is the first pillar, between me and you.

[chorus]
Five little pillars standing tall and strong,
Five little pillars singing this song.
Shahada, Salah, Zakat, Sawm, and Hajj,
Five little pillars walking Allah's path!

[verse]
Number two is Salah, I pray five times a day,
Fajr, Dhuhr, Asr, Maghrib, Isha on my way.
Number three is Zakat, sharing what I own,
Helping those in need so no one feels alone.

[chorus]
Five little pillars standing tall and strong,
Five little pillars singing this song.
Shahada, Salah, Zakat, Sawm, and Hajj,
Five little pillars walking Allah's path!

[bridge]
Number four is Sawm in Ramadan we fast,
Learning self-control that's meant to last.
Number five is Hajj, the journey to the land,
Where Ibrahim once stood with faithful hand.

[chorus]
Five little pillars standing tall and strong,
Five little pillars singing this song.
Shahada, Salah, Zakat, Sawm, and Hajj,
Five little pillars walking Allah's path!

[outro]
One, two, three, four, five,
Five little pillars keep our faith alive!
(counting chant fades)""",
    },
    {
        "slug": "en_05_allah_loves_me_abc",
        "lang": "en",
        "base": EN_BASE,
        "extra": "ABC alphabet song melody style, simple sweet lullaby ending, dreamy",
        "bpm": 76,
        "keyscale": "C major",
        "lyrics": """[intro]
(soft piano, warm strings swell)

[hook]
Allah loves me this I know,
Allah loves me told me so.
In the Quran I can read,
Allah gives me all I need.

[verse]
When the stars come out at night,
Twinkling softly, shining bright,
That's Allah showing me His care,
His beautiful signs are everywhere.
The moon up high, the oceans wide,
With Allah always by my side.

[chorus]
Allah loves me, yes He does,
Allah loves me, just because.
He made the sky, He made the sea,
And most of all, He made me.
Allah loves me, yes it's true,
Allah loves me, loves me and you.

[verse]
When the flowers bloom in spring,
When the little birdies sing,
When the rainbow paints the sky,
When the butterflies go by,
I see Allah in everything,
What a joy this life can bring.

[chorus]
Allah loves me, yes He does,
Allah loves me, just because.
He made the sky, He made the sea,
And most of all, He made me.
Allah loves me, yes it's true,
Allah loves me, loves me and you.

[bridge]
SubhanAllah, what a world,
Every leaf and petal curled.
Allah's mercy raining down,
Love and kindness all around.

[chorus]
Allah loves me, yes He does,
Allah loves me, just because.
He made the sky, He made the sea,
And most of all, He made me.
Allah loves me, yes it's true,
Allah loves me, loves me and you.

[outro]
Allah loves me,
Allah loves you,
Always and forever true.
(lullaby piano fade)""",
    },

    # ============ URDU (5) ============
    {
        "slug": "ur_01_lab_pe_dua_iqbal",
        "lang": "ur",
        "base": UR_BASE,
        "extra": "lab pe aati hai dua by allama iqbal classical melody, devotional taraana, reverent",
        "bpm": 78,
        "keyscale": "D minor",
        "lyrics": """[intro]
(soft ney flute, gentle duff drum roll)

[hook]
Lab pe aati hai dua ban ke tamanna meri,
Zindagi shama ki soorat ho khudaya meri.

[verse]
Subah uthta hoon to kehta hoon Bismillah,
Har qadam pe mera saathi hai khuda.
Door duniya ka mere dam se andhera ho jaye,
Har jagah mere chamakne se ujala ho jaye.

[chorus]
Ya Allah meri dua sun lijiye,
Mere dil ko apne noor se bhar dijiye.
Ya Allah mujhe achha bachha bana de,
Neikiyon ka raasta mujhe dikha de.
Ya Allah, ya Allah, sun lo meri fariyaad,
Rakh lo mujhe apni rehmat ki barsat mein yaad.

[verse]
Ho mera kaam gharibon ki himaayat karna,
Dard mando se zaeefon se mohabbat karna.
Mere Allah bura kisi se na ho mera,
Noor ho mera, ujala ho savera mera.

[chorus]
Ya Allah meri dua sun lijiye,
Mere dil ko apne noor se bhar dijiye.
Ya Allah mujhe achha bachha bana de,
Neikiyon ka raasta mujhe dikha de.
Ya Allah, ya Allah, sun lo meri fariyaad,
Rakh lo mujhe apni rehmat ki barsat mein yaad.

[bridge]
Lab pe aati hai dua ban ke tamanna meri,
Zindagi shama ki soorat ho khudaya meri.
Dil se nikalte hain yeh bol pyare pyare,
Allah ke naam se roshan hain sitare.

[chorus]
Ya Allah meri dua sun lijiye,
Mere dil ko apne noor se bhar dijiye.
Ya Allah mujhe achha bachha bana de,
Neikiyon ka raasta mujhe dikha de.

[outro]
Lab pe aati hai dua,
Ban ke tamanna meri.
(ney flute fades softly)""",
    },
    {
        "slug": "ur_02_bismillah_chanda",
        "lang": "ur",
        "base": UR_BASE,
        "extra": "chanda mama door ke melody style, gentle south asian lullaby, innocent sweet",
        "bpm": 84,
        "keyscale": "G major",
        "lyrics": """[intro]
(soft sitar plucks, gentle piano melody)

[hook]
Bismillah kaho, barkat pao,
Bismillah kaho, khushiyan lao.
Har kaam mein Bismillah kaho,
Allah ka naam sab se pehle lao!

[verse]
Subah uthoon to Bismillah kahoon,
Nashta karoon to Bismillah kahoon.
School ki taraf nikalta hoon jab,
Bismillah kehta ja raha hoon tab.
Kitab kholoon, qalam uthaaon,
Bismillah keh ke likhna shuru karaon.

[chorus]
Bismillah, Bismillah, pyara pyara naam,
Bismillah, Bismillah, har roz har shaam.
Bismillah kehne se barkat aati hai,
Allah ki rehmat ghar mein samati hai.
Bismillah, Bismillah, la la la la la,
Bismillah, Bismillah, MashaAllah!

[verse]
Khana pakaye ammi jab bhi,
Bismillah kehti pehle sabhi.
Abbu jab gaari chalate hain,
Bismillah keh ke nikaltay hain.
Neend aaye to bistar pe jaoon,
Bismillah keh ke aankhein band karaon.

[chorus]
Bismillah, Bismillah, pyara pyara naam,
Bismillah, Bismillah, har roz har shaam.
Bismillah kehne se barkat aati hai,
Allah ki rehmat ghar mein samati hai.
Bismillah, Bismillah, la la la la la,
Bismillah, Bismillah, MashaAllah!

[bridge]
Chote haath jod ke dua karo,
Bismillah keh ke har kaam shuru karo.
Allah dekh raha hai tum ko pyaar se,
Bismillah kaho apne dil ke taar se.

[chorus]
Bismillah, Bismillah, pyara pyara naam,
Bismillah, Bismillah, har roz har shaam.
Bismillah kehne se barkat aati hai,
Allah ki rehmat ghar mein samati hai.

[outro]
Bismillah, Bismillah,
Allah ka pyara naam.
(gentle piano outro)""",
    },
    {
        "slug": "ur_03_alhamdulillah_machli",
        "lang": "ur",
        "base": UR_BASE,
        "extra": "machli jal ki raani hai melody style, playful bouncy urdu rhyme, cheerful clapping",
        "bpm": 108,
        "keyscale": "C major",
        "lyrics": """[intro]
(playful duff rhythm, kids humming)

[hook]
Alhamdulillah, Alhamdulillah,
Shukriya tera sada, ya Allah!
Alhamdulillah, Alhamdulillah,
Tu hai mera, main hoon tera ya Allah!

[verse]
Suraj chamke aasmaan mein,
Taare jagmagayen raat mein.
Phool khilte hain baghon mein,
Titliyan udti hain raahon mein.
Yeh sab kis ne banaya hai,
Allah ne sab sajaya hai!

[chorus]
Alhamdulillah, Alhamdulillah,
Har neimat ke liye shukriya.
Alhamdulillah, Alhamdulillah,
Allah ka karam, Allah ka karam.
Taali bajao, shukriya karo,
Alhamdulillah dil se kaho!

[verse]
Ammi ki godi pyari hai,
Abbu ki ungali sahara hai.
Bhai behen ki hasi meethi,
Nani dadi ki dua hai seekhi.
Ghar mein rehmat barasti hai,
Allah ki meharbani hasti hai.

[chorus]
Alhamdulillah, Alhamdulillah,
Har neimat ke liye shukriya.
Alhamdulillah, Alhamdulillah,
Allah ka karam, Allah ka karam.
Taali bajao, shukriya karo,
Alhamdulillah dil se kaho!

[bridge]
Paani peeyoon, Alhamdulillah,
Roti khaoon, Alhamdulillah.
Sehat mile, Alhamdulillah,
Khushi mile, Alhamdulillah.
Har saans mein shukriya ada karo,
Allah ka ehsaan kabhi na bhulo!

[chorus]
Alhamdulillah, Alhamdulillah,
Har neimat ke liye shukriya.
Alhamdulillah, Alhamdulillah,
Allah ka karam, Allah ka karam.

[outro]
Alhamdulillah, Alhamdulillah,
Shukriya, shukriya, ya Allah!
(claps and duff fade)""",
    },
    {
        "slug": "ur_04_nabi_pyare_lakdi",
        "lang": "ur",
        "base": UR_BASE,
        "extra": "lakdi ki kaathi melody style, march tempo, storytelling cheerful",
        "bpm": 100,
        "keyscale": "F major",
        "lyrics": """[intro]
(rhythmic duff march, cheerful strings intro)

[hook]
Pyare Nabi, pyare Nabi, sab se pyare Nabi,
Sallallahu alaihi wasallam, sab se achhe Nabi!
Pyare Nabi, pyare Nabi, hum unko maante,
Sallallahu alaihi wasallam, dil se hum gaate!

[verse]
Nabi humare itne pyare thay,
Bachon ko gale lagate thay.
Muskurana unki aadat thi,
Sachhai unki ibadat thi.
Kabhi gussa na kisi pe kiya,
Sab ko maaf kar diya.

[chorus]
Sallallahu alaihi wasallam,
Sallallahu alaihi wasallam.
Pyare Nabi ka rasta apnayen,
Achhe bachon mein hum kehlayen.
Sach bolo, pyar karo,
Nabi jaise akhlaaq apnao!

[verse]
Bhookhe ko khana khilate thay,
Yatim bachon ko sambhalte thay.
Beemar ka haal poochte thay,
Garibon ko paisa dete thay.
Janwaron se bhi pyar karte,
Sabr ka daaman na chhorte.

[chorus]
Sallallahu alaihi wasallam,
Sallallahu alaihi wasallam.
Pyare Nabi ka rasta apnayen,
Achhe bachon mein hum kehlayen.
Sach bolo, pyar karo,
Nabi jaise akhlaaq apnao!

[bridge]
Agar hum bhi aise ban jayen,
Duniya ko khubsoorat bana jayen.
Nabi ki sunnat pe chalna seekho,
Dil mein mohabbat rakhna seekho.

[chorus]
Sallallahu alaihi wasallam,
Sallallahu alaihi wasallam.
Pyare Nabi ka rasta apnayen,
Achhe bachon mein hum kehlayen.

[outro]
Sallallahu alaihi wasallam,
Humare pyare pyare Nabi.
(gentle march fades)""",
    },
    {
        "slug": "ur_05_panch_namaz_ginti",
        "lang": "ur",
        "base": UR_BASE,
        "extra": "ek do teen char melody style, counting educational taraana, melodic march",
        "bpm": 96,
        "keyscale": "E minor",
        "lyrics": """[intro]
(soft qanun plucks, gentle ney melody)

[hook]
Ek do teen chaar paanch,
Paanch namazein yaad rakh!
Ek do teen chaar paanch,
Allah ki ibadat yaad rakh!

[verse]
Fajr ki azaan goonji subah savere,
Uthke namaz padho aankhein kholte dheere.
Suraj abhi nikla nahi aasmaan mein,
Allah se baatein karo sukoon ke gaane mein.
Fajr ki namaz se din shuru karo,
Allah ka shukriya har roz karo.

[chorus]
Paanch namazein, paanch khazane,
Allah ke darbar ke pyare nazrane.
Fajr, Zuhr, Asr, Maghrib, Isha,
Paanch namazein, paanch khazane.
Sab ko parho, har roz parho,
Allah ke qareeb raho, dil se kaho!

[verse]
Zuhr dopahar ki dhoop mein parhte hain,
Asr ko sunahri shaam mein karhte hain.
Maghrib jab suraj dhalta hai peeche,
Namaz parh lo apne ghar ke neeche.
Isha ki namaz jab taare chamkein,
Sajde mein gir ke dua mein damkein.

[chorus]
Paanch namazein, paanch khazane,
Allah ke darbar ke pyare nazrane.
Fajr, Zuhr, Asr, Maghrib, Isha,
Paanch namazein, paanch khazane.
Sab ko parho, har roz parho,
Allah ke qareeb raho, dil se kaho!

[bridge]
Namaz mein hum Allah se milte hain,
Dil ke saare dard unhe kehte hain.
Sajda karo, rukoo karo, qiyam mein khade raho,
Allah ke saath apna rishta mazboot banao.

[chorus]
Paanch namazein, paanch khazane,
Allah ke darbar ke pyare nazrane.
Fajr, Zuhr, Asr, Maghrib, Isha,
Paanch namazein, paanch khazane.

[outro]
Ek do teen chaar paanch,
Paanch namazein, paanch khazane.
Allah ke qareeb raho hamesha.
(ney flute outro, gentle fade)""",
    },
]


def build_workflow(tags, lyrics, seconds, seed, bpm, language, keyscale):
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
            "generate_audio_codes": True,
            "cfg_scale": 3.0,
            "temperature": 0.80,
            "top_p": 0.92,
            "top_k": 0, "min_p": 0.0}},
        "5": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {
            "seconds": float(seconds), "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["4", 0],
            "latent_image": ["5", 0], "seed": seed,
            "steps": 200,
            "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveAudio", "inputs": {
            "audio": ["7", 0], "filename_prefix": "islamic_premium"}},
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
        time.sleep(3)
    except Exception:
        pass

def submit(wf):
    cid = uuid.uuid4().hex[:12]
    res = http_post(f"{COMFY}/prompt", {"prompt": wf, "client_id": cid})
    if "prompt_id" not in res:
        raise RuntimeError(f"Rejected: {res}")
    return res["prompt_id"]

def wait_done(pid, timeout=3600):
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
        time.sleep(5)
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
    print("=" * 70)
    print("  PREMIUM Islamic Kids Songs — ACE-Step v1.5 XL SFT")
    print("  5 English + 5 Urdu | 270s each | 200 steps | Studio Mix")
    print("=" * 70)

    wait_ready(timeout=600)
    print("[Setup] ComfyUI ready.\n")

    duration_sec = 270.0   # 4.5 minutes
    results = []
    overall_t0 = time.time()

    for idx, song in enumerate(SONGS, 1):
        out_audio = DEST / f"{song['slug']}.flac"
        out_lyrics = DEST / f"{song['slug']}_lyrics.txt"
        out_lyrics.write_text(song["lyrics"], encoding="utf-8")

        tags = f"{song['base']}, {song['extra']}, {STUDIO_PREMIUM}"
        seed = random.randint(0, 2**31 - 1)
        wf = build_workflow(
            tags=tags, lyrics=song["lyrics"], seconds=duration_sec,
            seed=seed, bpm=song["bpm"], language=song["lang"],
            keyscale=song["keyscale"],
        )

        lang_label = "EN" if song["lang"] == "en" else "UR"
        print(f"[{idx:2d}/10] [{lang_label}] {song['slug']}")
        print(f"        bpm={song['bpm']}  key={song['keyscale']}  steps=200  dur=270s  seed={seed}")
        t0 = time.time()
        try:
            free_vram()
            pid = submit(wf)
            print(f"        prompt_id={pid}")
            hist = wait_done(pid, timeout=3600)
            if collect(hist, out_audio):
                elapsed = time.time() - t0
                size_mb = out_audio.stat().st_size / 1e6
                print(f"        OK in {elapsed:.0f}s ({elapsed/60:.1f}m)  ->  {out_audio.name} ({size_mb:.1f} MB)\n")
                results.append((song["slug"], "ok", elapsed))
            else:
                print(f"        FAIL — no audio in outputs\n")
                results.append((song["slug"], "no_output", time.time() - t0))
        except Exception as e:
            print(f"        ERROR: {e}\n")
            results.append((song["slug"], f"error: {str(e)[:80]}", time.time() - t0))

    total = time.time() - overall_t0
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY  (total {total/60:.1f} min)")
    print(f"{'=' * 70}")
    for slug, status, secs in results:
        print(f"  {slug:35s} {status:20s} {secs/60:5.1f}m")
    ok = sum(1 for _, s, _ in results if s == "ok")
    print(f"\n  {ok}/10 succeeded.  Outputs in: {DEST}")


if __name__ == "__main__":
    main()
