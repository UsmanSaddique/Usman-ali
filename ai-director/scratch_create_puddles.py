import json, urllib.request

API = "http://127.0.0.1:8000"

lyrics = """[intro]
Rain is done, the sun peeks through,
splashy puddles, me and you!

[verse]
One puddle blue, jump right in,
two puddles green, a happy grin!

[chorus]
Splash, splash, splashy day,
count and jump and shout hooray!

[verse]
Three puddles gold, our duckling too,
four puddles pink, we splash right through!

[verse]
Five puddles purple, biggest fun,
one big splash for everyone!

[outro]
Sunny garden, boots so bright,
splashy puddles, what a sight!"""

context = (
    "PLAYTIME lane. HERO (locked descriptor, EVERY scene): a chubby baby elephant with soft "
    "dove-grey skin, a big rounded head, huge friendly dark eyes, a tiny curled trunk, and bright "
    "yellow rain boots. FRIEND (at most 1, some scenes): a small fluffy yellow duckling in a tiny "
    "red raincoat. SETTING: a sunny pastel garden just after rain - mint-green grass, lemon-yellow "
    "flowers, five puddles each mirroring a different color (blue, green, gold, pink, purple), soft "
    "sparkly light, gentle drifting bubbles. STYLE: bouncy sunny pastels, gentle slow toddler-safe "
    "actions (happy splashing, little jumps, trunk sprinkles), max 2 characters per frame, cute "
    "never scary. STRUCTURE = a counting rhyme 1->5: intro (rain stops) -> puddle 1 (blue) -> "
    "puddle 2 (green) -> chorus -> puddle 3 (gold, duckling joins) -> puddle 4 (pink) -> puddle 5 "
    "(purple, big splash) -> outro. Each verse adds one puddle so toddlers count along 1 to 5 every "
    "replay. Keep it to about ten scenes so it fits the LTX multi-director engine."
)

body = {
    "title": "Five Little Puddles | Counting Song for Kids | Baby Pooem Animal Friends",
    "channel_slug": "baby-pooem",
    "duration": 60,
    "context": context,
    "lyrics": lyrics,
    "music_style": ("cheerful bouncy kids-pop nursery singalong, 100 bpm, bright ukulele glockenspiel "
                    "and light claps, playful splashy accents, clear solo female vocal, crisp playful "
                    "enunciation, vocals loud and up front, sunny and warm, simple catchy toddler melody"),
    "music_model": "auto",
    "video_engine": "ltx_director",
    "project_type": "song",
}

req = urllib.request.Request(API + "/api/projects", data=json.dumps(body).encode(),
                             method="POST", headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    print(r.read().decode())
