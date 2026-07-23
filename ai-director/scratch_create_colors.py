import json, urllib.request

API = "http://127.0.0.1:8000"

# "Learn Colors with Pip the Bunny" — demand-driven 4-min LTX colors song.
# 10 colors = 10 segments (~24s each). White hero so every color pops.
lyrics = """[intro]
Hello friends, it's Pip the bunny,
learning colors, bright and sunny!
Come along and play with me,
how many colors can we see?

[verse]
Red, red, what is red?
A shiny apple, that's what's red!
Pip holds it high, so round and bright,
red, red, red, what a sight!

[verse]
Orange, orange, what is orange?
A fluttery butterfly, that's what's orange!
It lands so soft on Pip's little nose,
orange, orange, off it goes!

[verse]
Yellow, yellow, what is yellow?
The great big sun, so warm and mellow!
It shines on Pip all through the day,
yellow, yellow, hip hooray!

[verse]
Green, green, what is green?
The soft green grass, the softest seen!
Pip hops along beside a frog,
green, green, leap and jog!

[verse]
Blue, blue, what is blue?
The splashy pond, so cool and new!
Pip dips a toe, then in they go,
blue, blue, splish and splosh!

[verse]
Purple, purple, what is purple?
The pretty flowers in a little circle!
Pip gives a sniff, they smell so sweet,
purple, purple, what a treat!

[verse]
Pink, pink, what is pink?
The blossom petals, soft as a wink!
They drift and swirl around his head,
pink, pink, a fluffy bed!

[verse]
Brown, brown, what is brown?
A wiggly puppy, the cutest in town!
He wags his tail and gives a hop,
brown, brown, never stop!

[verse]
Black, black, what is black?
The starry night, a cozy sky-sack!
The moon peeks out with a sleepy gleam,
black, black, time to dream!

[verse]
White, white, what is white?
The fluffy clouds and snowflakes light!
Pip blends right in, just his eyes show,
white, white, giggle and glow!

[outro]
We learned our colors, one by one,
red to white, wasn't that fun?
Wave goodbye to Pip the bunny,
colors make the world so sunny!"""

context = (
    "LEARN COLORS lane. HERO (locked descriptor, EVERY scene): Pip, a chubby soft snow-white baby "
    "bunny with a big round head, huge glossy dark eyes, a tiny pink nose, and small rounded ears, "
    "fluffy pure-white fur. Pip is deliberately PURE WHITE so every color pops against him. SETTING: a "
    "bright, clean, sunny toddler world - soft rounded meadow, gentle pastel sky, simple uncluttered "
    "background with soft depth. STYLE: adorable soft 3D Pixar-style cartoon render, plush toy-like, "
    "glossy, super-cute, bright saturated colors, SLOW gentle toddler-safe motion - exactly ONE calm "
    "continuous action per scene (ideal for one long smooth shot, never choppy), max 2 characters per "
    "frame, no scary elements, no readable text in frame. STRUCTURE = a color-learning song with ONE "
    "named color per scene, in this exact order: red, orange, yellow, green, blue, purple, pink, brown, "
    "black, white. In each scene Pip discovers a SINGLE vivid object of that exact color; that object is "
    "the clear focal point and unmistakably that color so toddlers learn the color name with the visible "
    "object (red apple, orange butterfly, yellow sun, green grass+frog, blue pond, purple flowers, pink "
    "blossoms, brown puppy, black starry night, white clouds/snow). Keep it to about ten scenes so it "
    "fits the LTX multi-director engine, one long smooth shot per color."
)

body = {
    "title": "Learn Colors with Pip the Bunny | 4 Minute Colors Song for Toddlers | Baby Pooem",
    "channel_slug": "baby-pooem",
    "duration": 240,
    "context": context,
    "lyrics": lyrics,
    "music_style": ("cheerful upbeat toddler learning song, 98 bpm, bright ukulele glockenspiel and "
                    "light claps, clear friendly solo female vocal, playful call-and-response, crisp "
                    "enunciation, vocals loud and up front, sunny warm and encouraging, simple catchy "
                    "repetitive melody for a colors singalong"),
    "music_model": "auto",
    "video_engine": "ltx_director",
    "project_type": "song",
}

req = urllib.request.Request(API + "/api/projects", data=json.dumps(body).encode(),
                             method="POST", headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    print(r.read().decode())
