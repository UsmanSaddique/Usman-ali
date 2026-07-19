"""Create 'The Little Kitten Song' project on Baby Pooem (minimal-story direction)."""
import json, urllib.request

LYRICS = """[intro]
Meow meow! Here is the little white kitten!

[chorus]
The little kitten goes meow meow meow,
meow meow meow, meow meow meow,
the little kitten goes meow meow meow,
soft and sweet all day!

[verse]
The kitten sits on the windowsill,
blinking slow, sitting still,
her little tail goes swish swish swish,
meow meow meow, make a wish!

[verse]
The kitten stretches nice and slow,
paws out front, head down low,
a big soft yawn, a sleepy purr,
meow meow meow, fluffy fur!

[chorus]
The little kitten goes meow meow meow,
meow meow meow, meow meow meow,
the little kitten goes meow meow meow,
soft and sweet all day!

[verse]
The kitten taps a ball of yarn,
roll roll roll, soft and warm,
it wiggles left, it wiggles right,
meow meow meow, what a sight!

[verse]
A little butterfly floats by,
flutter flutter in the sky,
the kitten watches, eyes so wide,
meow meow meow, side to side!

[chorus]
The little kitten goes meow meow meow,
meow meow meow, meow meow meow,
the little kitten goes meow meow meow,
soft and sweet all day!

[verse]
The kitten licks her snowy paw,
clean and soft, purr purr purr,
she curls up in a cozy ball,
meow meow meow, best of all!

[outro]
The sun is warm, the day is done,
the little kitten purrs goodnight everyone,
meow meow... goodnight!"""

MUSIC_STYLE = (
    "classic children's nursery rhyme in the style of Pussy Cat Pussy Cat "
    "Where Have You Been, cheerful bouncy sing-along, clear solo female lead "
    "vocal with crisp enunciation, vocals loud and forward in the mix, playful "
    "call-and-response sounds sung sweetly (meow meow, purr purr, swish swish), "
    "soft kids choir echo on the hook only, acoustic guitar strums, ukulele, "
    "glockenspiel, marimba, gentle hand claps, light kick drum on the beat, "
    "bright major key, simple repetitive singable melody toddlers can hum, "
    "wholesome kids pop, professional studio mix, clean bright master, 100 bpm")

CONTEXT = (
    "PLAYTIME lane, MINIMAL-STORY format — the channel visual_direction applies "
    "with full force: NO story, no events, no plot, no arrivals, no interactions. "
    "The video is a loop of simple standalone clips. "
    "HERO (repeat this descriptor EXACTLY in every scene): KITTEN = a round "
    "fluffy white baby kitten with big sky-blue eyes, pink inner ears, a little "
    "red bow on its head. Side extra allowed in a few scenes only, tiny idle "
    "motion, never interacting: a small pastel-yellow butterfly. Each scene = "
    "the kitten doing ONE tiny slow movement (sitting and blinking, slow "
    "stretch, gentle tail swish, watching the butterfly, one soft tap on a ball "
    "of yarn, curling up to sleep). Vary only camera framing, daylight warmth "
    "and cozy backgrounds (sunny windowsill, flower garden, soft cushion, "
    "meadow). Little movement everywhere — slow, calm, plush.")

body = {
    "title": "The Little Kitten Song | Meow Meow Meow | Fun Animal Rhymes & Poems for Kids #kitten #babypoem",
    "channel_slug": "baby-pooem",
    "duration": 180,
    "project_type": "song",
    "video_engine": "ltx_director",
    "lyrics": LYRICS,
    "music_style": MUSIC_STYLE,
    "context": CONTEXT,
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/projects",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    print(r.read().decode())
