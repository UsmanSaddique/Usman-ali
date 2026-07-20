"""
AI Director — Lyric-to-Scene Prompt Builder
Turns parsed lyric segments into channel-styled visual prompts WITHOUT the LLM
(instant, deterministic, beat-synced). Ported from the proven song_to_video
template system so wizard "From Lyrics" scenes match the channel look.
"""
import random
import hashlib

from app.services import master_director

TEMPLATE_ACTIONS = {
    "intro": [
        "golden sunrise breaking over a peaceful {setting}, warm volumetric light rays, floating dust motes",
        "gentle morning light illuminating a beautiful {setting}, soft mist, serene atmosphere",
    ],
    "hook": [
        "{character} standing joyfully in a sunlit {setting}, arms gently open, welcoming smile",
        "{character} looking up with wonder at the sky in a bright {setting}, gentle breeze",
    ],
    "verse": [
        "{character} walking gently through a beautiful {setting}, soft golden hour light",
        "{character} sitting peacefully in a cozy {setting}, reading a small book",
        "{character} reaching out to help a small bird in a sunny {setting}",
        "{character} kneeling on a soft prayer rug in a warm {setting}, hands raised in dua",
        "{character} sharing food with friends in a cheerful {setting}, warm smiles",
        "{character} looking at flowers blooming in a colorful {setting}, gentle wonder",
        "{character} walking hand in hand with a friend in a peaceful {setting}",
        "{character} sitting under a large tree in a serene {setting}, dappled sunlight",
    ],
    "chorus": [
        "{character} joyfully singing with arms raised in a bright {setting}, sparkles floating",
        "{character} clapping hands happily in a sunlit {setting}, warm celebration",
        "{character} dancing gently in a beautiful {setting}, soft particles floating around",
    ],
    "bridge": [
        "{character} looking thoughtfully at the stars in a calm {setting}, moonlit atmosphere",
        "{character} sitting quietly by a gentle stream in a peaceful {setting}, reflective mood",
    ],
    "outro": [
        "beautiful sunset over a peaceful {setting}, warm orange and pink sky, silhouette of a mosque",
        "soft twilight over a serene {setting}, first stars appearing, gentle peace",
    ],
}

SETTINGS = [
    "garden with blooming flowers and soft grass",
    "cozy sunlit room with arched windows and cushions",
    "mosque courtyard with ornate fountain and gentle mist",
    "meadow with wildflowers under a blue sky with soft clouds",
    "ancient stone pathway lined with lanterns and greenery",
    "rooftop terrace overlooking a warm sunset cityscape",
    "quiet library corner with wooden shelves and golden light",
    "peaceful riverbank with stepping stones and willow trees",
]

SHOT_TYPES = ["wide establishing shot", "medium wide shot", "medium shot",
              "wide shot", "medium close-up", "medium wide shot"]

CAMERA_MOTIONS = ["static", "zoom_in", "pan_left", "pan_right", "static", "zoom_in"]

MOTION_CUES = [
    "gentle natural movement, soft breeze in clothing and hair",
    "the character moving with lively gentle motion, ambient particles drifting",
    "subtle camera push-in, warm light shimmering, natural motion",
]


def build_prompts(segments: list, channel_profile: dict, seed_key: str) -> list[dict]:
    """One channel-styled visual prompt per lyric segment. Deterministic for a
    given project (seed_key) so regenerating scenes gives the same storyboard."""
    profile = channel_profile or {}
    char_bible = profile.get("character_bible", [])
    art_style = profile.get("art_style_phrase", "soft 3D Pixar-style cartoon render")
    palette = profile.get("color_palette", "bright cheerful pastels")
    if isinstance(palette, dict):
        # lane palettes (e.g. {playtime: ..., dreamtime: ...}) — a raw dict
        # f-stringed into the prompt gets RENDERED AS TEXT by Z-Image.
        # Use the daytime lane (first value as fallback).
        palette = palette.get("playtime") or \
            next(iter(palette.values()), "bright cheerful pastels")
    elif isinstance(palette, list):
        palette = ", ".join(str(p) for p in palette)
    neg = profile.get("negative_prompt_additions",
                      "photorealistic, realistic skin, text, watermark, scary, deformed, extra limbs, blurry")
    if isinstance(neg, list):
        neg = ", ".join(neg)

    characters = char_bible if char_bible else ["a cute cartoon child in modest colorful clothing"]
    seed_hash = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)

    prompts = []
    for seg in segments:
        rng = random.Random(seed_hash + seg.index)
        templates = TEMPLATE_ACTIONS.get(seg.section_type, TEMPLATE_ACTIONS["verse"])
        template = rng.choice(templates)

        character = characters[seg.index % len(characters)]
        char_desc = character.split(":", 1)[1].strip() if ":" in character else character
        setting = rng.choice(SETTINGS)

        action = template.format(character=char_desc, setting=setting)
        motion = MOTION_CUES[seg.index % len(MOTION_CUES)]

        # Master-director pass: shot/camera/lighting/mood/composition follow a
        # dramatic arc across the whole video; saved with the scene so both
        # video engines (and the C# port) use the exact same storyboard.
        guidance = master_director.guidance_for(
            seg.index, len(segments), seg.section_type, seed_key)
        shot = guidance["shot"]

        prompts.append({
            "segment_index": seg.index,
            "prompt": (f"{shot}, {action}, {motion}, {art_style}, {palette}, "
                       f"highly detailed, cinematic, soft volumetric lighting, depth of field, "
                       f"beautifully rendered, warm rim light, gentle bokeh background, "
                       f"clean frame without any text, captions, titles or watermarks, "
                       f"camera: {guidance['camera']}, {guidance['lighting']}, "
                       f"{guidance['mood']} mood, {guidance['composition']}"),
            "negative_prompt": neg,
            "camera_motion": CAMERA_MOTIONS[seg.index % len(CAMERA_MOTIONS)],
            "narration_text": "" if seg.is_instrumental else seg.text,
            "duration": seg.duration,
            "director_guidance": guidance,
        })
    return prompts
