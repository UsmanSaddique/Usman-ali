import json, urllib.request

API = "http://127.0.0.1:8000"

lyrics = """[intro]
Wet your brush, here we go,
a tiny dot of paste, just so!
Open wide and show your smile,
brush along with us a while!

[chorus]
Brush, brush, brush your teeth,
up above and down beneath!
Round and round and clean and bright,
keep them sparkly, keep them white!

[verse]
Top right first, let's make a start,
little circles, that's so smart!
Crocodile has a hundred teeth,
brushes each one, up and beneath!

[chorus]
Brush, brush, brush your teeth,
up above and down beneath!
Round and round and clean and bright,
keep them sparkly, keep them white!

[verse]
Now up top on the other side,
brush them slow with happy pride!
Hippo grins a great big grin,
clean and shiny, cheek to chin!

[verse]
Down below, the right side now,
brush the back ones, show us how!
Little circles, soft and slow,
watch those bubbles grow and grow!

[verse]
Bottom left, we're almost through,
brush the tiny ones, me and you!
Brush your tongue to end the day,
sweep the sleepy germs away!

[chorus]
Brush, brush, brush your teeth,
up above and down beneath!
Round and round and clean and bright,
keep them sparkly, keep them white!

[outro]
Sip and swish and gently spit,
two whole minutes, you did it!
Morning sun or moon above,
brush your teeth with lots of love!"""

context = (
    "PLAYTIME lane. HERO (locked descriptor, EVERY scene): a chubby upright baby crocodile "
    "with soft mint-green scales, a big rounded head, huge friendly dark eyes, and tiny rounded "
    "NON-scary white teeth, holding a bright blue toothbrush. FRIEND (at most 1, some scenes): a "
    "small round dusty-pink baby hippo with a big happy smile holding a yellow toothbrush. "
    "SETTING: a sunny pastel toddler bathroom - mint and lemon tiles, a low white sink, a step "
    "stool, a rinse cup, soft morning light. STYLE: bouncy, sunny pastels, gentle slow toddler-safe "
    "actions (brushing motion, happy bounce), max 2 characters per frame, img2vid every scene, teeth "
    "always cute and friendly never scary. STRUCTURE = a functional 2-minute brushing timer following "
    "dentist quadrants: intro (wet brush, tiny paste) -> top-right -> top-left -> bottom-right -> "
    "bottom-left -> tongue -> rinse/spit outro. Each verse is one mouth zone so the finished video works "
    "as a real brush-along timer parents replay every night."
)

body = {
    "title": "Brush Your Teeth Song | 2 Minute Timer for Kids | Baby Pooem Animal Friends",
    "channel_slug": "baby-pooem",
    "duration": 120,
    "context": context,
    "lyrics": lyrics,
    "music_style": ("cheerful bouncy kids-pop nursery singalong, 100 bpm, bright ukulele glockenspiel "
                    "and light claps, clear solo female vocal, crisp playful enunciation, vocals loud and "
                    "up front, sunny and warm, simple catchy toddler melody"),
    "music_model": "auto",
    "video_engine": "clips",
    "project_type": "song",
}

req = urllib.request.Request(API + "/api/projects", data=json.dumps(body).encode(),
                             method="POST", headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=60) as r:
    print(r.read().decode())
