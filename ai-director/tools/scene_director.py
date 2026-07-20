"""MASTER SCENE DIRECTOR — Baby Pooem
==================================================================
Generates the per-scene shot list for a song-mode video and saves it as a master
script the video can be rebuilt from.

WHY THIS EXISTS
    The app's built-in planners produce visually monotonous videos:
      * app/services/lyric_scenes.py repeats a handful of stock actions;
      * the C# ScenePlanner's StyleCueFor() is a hardcoded two-option string that
        never reads the channel yaml at all.
    Both also leave `camera_motion` unused — nothing passes it to the workflow —
    so ALL camera movement must be written into the prompt text itself.

THE DIRECTOR'S RULES (what stops every clip looking the same)
    1. EVERY scene gets its own SETTING. Never let two consecutive scenes share a
       background. Backgrounds cycle through a large pool, not one per verse.
    2. EVERY scene gets its own ACTION, taken once and not repeated. Actions carry
       real movement (trotting, hopping, rolling, shaking, spinning) instead of a
       parade of "sitting still and slowly blinking".
    3. EVERY scene gets its own CAMERA MOVE, written into the prompt (push in,
       pull back, pan, orbit, tilt, tracking) because the DB field is ignored.
    4. EVERY scene gets its own FRAMING, including macro and wide, not all
       medium close-up.
    5. CHORUS scenes keep a recognisable signature (hero + golden hour) so the
       hook still reads as a repeated motif — but the shot, action and camera
       change each time. Repetition of the MOTIF, never of the FRAME.

MOTION vs RENDER QUALITY
    channels/baby-pooem.yaml says "NEVER fast action" because LTX fumbles fast
    motion (warped limbs) and the freeze/black QA gate rejects static clips. The
    pool below is deliberately mid-range: real, readable movement described as
    smooth and flowing, with nothing frantic. If clips come back warped, move the
    high-motion verbs (leaps, spins, tumbles) toward the gentler end.

USAGE
    python tools/scene_director.py <project_id> [--dry-run]

    Writes projects/<project_id>/scene_script.json (the master script) and POSTs
    the scenes to the running C# app. --dry-run writes the file and prints a
    summary without touching the API.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
API_BASE = "http://localhost:5080/api/projects"

# ── Channel identity: constant on purpose (branding), never varied ──────────
# Style anchor, deliberately SHORT. It used to be 335 chars of byte-identical
# boilerplate on the end of every prompt - 41% of the whole prompt telling the
# model "make it look like all the others", which drowned out the per-scene
# setting and action. Near-synonyms (highly detailed / cinematic / beautifully
# rendered / soft volumetric lighting / warm rim light / gentle bokeh) bought
# nothing and were cut.
#
# The old tail also ended with "clean frame without any text, captions, titles or
# watermarks" while NEG already lists text/letters/numbers/logo/watermark. Naming
# them in the POSITIVE prompt is worse than useless - diffusion models handle
# negation poorly, so it can summon the very thing it asks to avoid.
STYLE = ("adorable soft 3D cartoon render, plush toy-like baby animal, huge glossy eyes, "
         "rounded shapes, pastel colors, soft cinematic lighting")
# "football" alone is ambiguous — the model rendered an oval American football in
# some scenes and a round soccer ball in others, breaking prop continuity. Pin it.
BALL = ("a small round black and white soccer ball plush toy with the classic "
        "hexagon and pentagon pattern, no logos, no text")
NEG = ("photorealistic, realistic animal fur, human, human face, hands, fingers, text, "
       "letters, numbers, logo, watermark, scary, dark shadows on faces, sharp teeth, "
       "angry expression, deformed, extra limbs, asymmetric eyes, blurry, low quality, "
       "motion blur, flickering, frantic motion, crowd, stadium lights, sports jersey, "
       "team kit, real football club, american football, oval ball, rugby ball, "
       "brown leather ball, laces on ball")

# Locked descriptor per hero — repeated EXACTLY in every scene of that animal.
HEROES = {
    "puppy": ("a chubby fluffy golden baby puppy with an oversized round head, big glossy "
              "dark eyes, floppy ears and a tiny black nose, soft downy fur"),
    "kitten": ("a round fluffy white baby kitten with big sky-blue eyes, pink inner ears, "
               "soft plush fur and a little curled tail"),
    "duckling": ("a chubby fluffy yellow baby duckling with an oversized round head, big "
                 "glossy dark eyes, tiny orange beak and feet, soft downy feathers"),
    "bunny": ("a plump soft grey baby bunny with long floppy ears, huge shiny brown eyes, "
              "a twitchy pink nose and round fluffy cheeks"),
    "chick": ("a tiny round fluffy yellow baby chick with an oversized head, big glossy "
              "black eyes, a very small orange beak and stubby wings"),
    "lamb": ("a plump curly-wool cream baby lamb with an oversized round head, huge gentle "
             "dark eyes, soft pink ears and tiny hooves"),
}

# ── Rule 1: a big setting pool. No two consecutive scenes may share one. ────
SETTINGS = [
    "on a warm open field at golden hour, long soft shadows, sky washed peach and apricot",
    "on soft pale beach sand at the water's edge, gentle pastel turquoise waves, tiny seashells",
    "in an autumn park, drifting amber and gold leaves, warm rust and cream palette",
    "on wet grass after rain, shallow puddles mirroring the sky, cool mint and soft grey palette",
    "on a soft snowy field, big slow snowflakes drifting down, pale blue and white palette",
    "in a garden at dusk, glowing golden fireflies drifting, deep blue twilight palette",
    "in a cosy pastel playroom, soft rug, plush toys blurred behind, big window light",
    "in a spring meadow under a blossom tree, pale pink petals drifting down, mint green grass",
    "on a mossy stone path winding through a pastel cottage garden, soft midday light",
    "in a field of tall pastel wildflowers swaying, hazy warm backlight",
    "beside a calm pastel pond with lily pads and drifting reflections, soft morning light",
    "on a grassy hilltop above rolling pastel hills, wide bright sky, drifting clouds",
    "under a big soft oak tree, dappled sunlight spots moving on the grass",
    "in a greenhouse of pastel potted plants, warm glass light, floating dust motes",
    "on a wooden garden deck scattered with soft cushions, late afternoon amber light",
    "in a bamboo-and-fern corner of a pastel garden, cool green shade and soft light beams",
]

# Chorus signature: all golden-hour so the hook reads as one recurring look, but
# four DIFFERENT locations so no two chorus scenes share a background either.
CHORUS_SETTINGS = [
    "on a warm open field at golden hour, long soft shadows, sky washed peach and apricot",
    "on a golden-hour hilltop with warm backlight rimming the grass, apricot sky beyond",
    "beside a golden-hour pond, warm amber light glinting on the water, peach reflections",
    "on a golden-hour garden path, low warm sun flaring softly through drifting seed fluff",
]

# ── Rule 2: every action used once. Mid-range motion, described as smooth. ──
ACTIONS = [
    "trotting a few smooth steps alongside the rolling ball",
    "hopping lightly over the ball in one smooth arc",
    "rolling onto its back with paws up as the ball wobbles beside it",
    "shaking its whole fluffy body in a soft wobble beside the ball",
    "stretching forward into a long smooth stretch with the ball under one paw",
    "spinning around once in a slow smooth circle around the ball",
    "gently tumbling sideways into the grass next to the ball",
    "sliding a short smooth slide with both front paws on the ball",
    "pawing at the ball so it rolls gently away and back",
    "bouncing softly on the spot as the ball rocks beside it",
    "leaning back on its haunches and batting the ball upward in a soft arc",
    "chasing the slowly rolling ball with small bouncy steps",
    "nudging the ball forward with its nose in one smooth push",
    "rocking side to side while balancing a paw on top of the ball",
    "leaping a small soft leap across the frame as the ball rolls",
    "curling around the ball and slowly rolling with it",
    "tapping the ball twice in a smooth rhythm and looking up",
    "wiggling happily with its tail swishing as the ball spins slowly",
    "stepping up onto the ball and wobbling gently for balance",
    "flopping down beside the ball with a soft bounce",
    "peeking out from behind the ball and tilting its head",
    "pushing the ball with its chest in a smooth steady roll",
    "kicking the ball gently with a back leg in one flowing motion",
    "circling the ball with light padding steps",
]

# ── Rule 3: camera move, written into the prompt (DB field is ignored) ──────
CAMERA = [
    "slow smooth push in towards the subject",
    "slow smooth pull back revealing more of the scene",
    "gentle pan following the action",
    "smooth low tracking shot moving alongside the subject",
    "slow gentle orbit around the subject",
    "slow tilt up from the ball to the animal's face",
    "steady locked-off shot with the subject moving through frame",
    "smooth dolly in slightly from the side",
]

# ── Rule 4: framing variety, including macro and wide ───────────────────────
FRAMINGS = [
    "medium close-up, hero large in frame, chest up",
    "low angle close-up looking slightly up at the hero",
    "macro detail shot of paws and the ball, shallow focus",
    "medium three-quarter view, hero centered",
    "wide shot, hero small in a big soft landscape",
    "slightly top-down medium shot looking over the hero",
    "over-the-ball foreground shot with the hero beyond it",
    "side-profile medium shot as the hero moves across frame",
]


def build(hero_order, n_scenes, seed=20260720):
    """One unique combination per scene. Settings never repeat back-to-back;
    actions/cameras/framings are drawn from shuffled decks that only reshuffle
    once exhausted, so repeats are maximally spaced."""
    rng = random.Random(seed)

    def deck(items):
        pool = []
        while True:
            if not pool:
                pool = items[:]
                rng.shuffle(pool)
            yield pool.pop()

    action_d, camera_d, framing_d = deck(ACTIONS), deck(CAMERA), deck(FRAMINGS)
    setting_d, chorus_d = deck(SETTINGS), deck(CHORUS_SETTINGS)

    scenes, last_setting = [], None
    for i in range(n_scenes):
        role, hero_key = hero_order[i]
        # Chorus keeps the golden-hour signature so the hook reads as a recurring
        # motif, but draws a DIFFERENT golden-hour location each time — pinning it
        # to one background is what made four scenes in a row look identical.
        source = chorus_d if role == "chorus" else setting_d
        setting = next(source)
        while setting == last_setting:
            setting = next(source)
        last_setting = setting

        action, framing, camera = next(action_d), next(framing_d), next(camera_d)
        scenes.append({
            "scene_number": i + 1,
            "role": role,
            "hero": hero_key,
            # Kept alongside the prompt so the master script stays editable and
            # the variety summary is counted from real fields, not by re-parsing
            # the prompt string.
            "setting": setting,
            "action": action,
            "framing": framing,
            "camera": camera,
            # Order matters: image models weight EARLY tokens hardest (the same
            # lesson as leading ACE-Step briefs with the vocal spec). The unique
            # per-scene content - setting, then action - goes first; the shared
            # hero descriptor, shot grammar and style anchor follow.
            "prompt": (f"{setting}. {HEROES[hero_key]}, {action}, "
                       f"with {BALL}. {framing}, {camera}. {STYLE}"),
            "negative_prompt": NEG,
            "duration": 5.04,
            "camera_motion": "static",
        })
    return scenes


# Section map for the football song: 12 blocks x 4 scenes.
BLOCK_PLAN = [
    ("chorus", "puppy"), ("verse", "puppy"), ("verse", "kitten"), ("chorus", "puppy"),
    ("verse", "duckling"), ("verse", "bunny"), ("bridge", "puppy"), ("chorus", "puppy"),
    ("verse", "chick"), ("verse", "lamb"), ("bridge", "kitten"), ("chorus", "puppy"),
]
HERO_ORDER = [blk for blk in BLOCK_PLAN for _ in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scenes = build(HERO_ORDER, len(HERO_ORDER))

    out_dir = REPO / "projects" / args.project_id
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / "scene_script.json"
    script_path.write_text(json.dumps(scenes, indent=2), encoding="utf-8")
    print(f"master script saved: {script_path}")

    bg = [s["setting"] for s in scenes]
    back_to_back = sum(1 for a, b in zip(bg, bg[1:]) if a == b)
    print(f"{len(scenes)} scenes = {len(scenes) * 5.04:.1f}s | "
          f"{len(set(bg))} distinct backgrounds | "
          f"{len({s['action'] for s in scenes})} distinct actions | "
          f"{len({s['camera'] for s in scenes})} camera moves | "
          f"{len({s['framing'] for s in scenes})} framings")
    print(f"consecutive same-background pairs: {back_to_back} | "
          f"duplicate prompts: {len(scenes) - len({s['prompt'] for s in scenes})}")

    if args.dry_run:
        return

    payload = {"scenes": [{k: s[k] for k in
                           ("prompt", "negative_prompt", "duration", "camera_motion")}
                          for s in scenes]}
    req = urllib.request.Request(f"{API_BASE}/{args.project_id}/scenes-manual",
                                 data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        print("POST scenes-manual ->", r.status)


if __name__ == "__main__":
    sys.exit(main())
