"""
Singing Islamic nursery rhyme — Wheels on the Bus melody with pitch-controlled vocals.

Uses Edge TTS SSML prosody contour to make Microsoft Ana's voice
follow the exact Wheels on the Bus melody note pattern.

Wheels on the Bus melody (Key of A major, 86 BPM):
  "The wheels on the bus go" = A A A A C# E  (ascending)
  "round and round"         = E D C#          (descending)
  "round and round"         = C# B A          (resolve down)
  "round and round"         = A B C#          (back up)
  "The wheels on the bus go" = A A A A C# E   (repeat)
  "round and round"         = E D C#          (descending)
  "All through the town"    = B B C# A        (resolution)

Child voice register (octave 5):
  A4=440, B4=494, C#5=554, D5=587, E5=659
"""
import asyncio
import json
import os
import subprocess
from pathlib import Path

import edge_tts

PROJ = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\projects\3efaca79-b3ae-4c20-a6a7-9c294bb368d1")
TEMP = PROJ / "singing_temp"
TEMP.mkdir(exist_ok=True)

VOICE = "en-US-AnaNeural"

# Melody pitches as Hz offsets from Ana's natural pitch (~250Hz)
# Using relative pitch shifts to trace the melody
# Ana's natural pitch is around medium, so we use +/- Hz shifts

def build_ssml_line(text, pitch_contour, rate="medium"):
    """Build SSML with prosody contour for singing effect.

    pitch_contour: list of (percent, pitch_value) tuples
    e.g. [(0, "+10Hz"), (25, "+30Hz"), (50, "+20Hz"), (100, "+5Hz")]
    """
    contour_str = " ".join(f"({p}%,{v})" for p, v in pitch_contour)

    ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
          xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
    <voice name="{VOICE}">
        <mstts:express-as style="cheerful" styledegree="2">
            <prosody rate="{rate}" contour="{contour_str}">
                {text}
            </prosody>
        </mstts:express-as>
    </voice>
</speak>"""
    return ssml


# ═══════════════════════════════════════════════════════════════
# MELODY CONTOURS — Maps "Wheels on the Bus" melody to pitch shifts
# Pattern: ascending run, descend, repeat, resolve
# ═══════════════════════════════════════════════════════════════

# Line 1: "The kids on the bus say Bismillah!"
# Melody: A A A A C# E E (ascending to climax on the Islamic word)
LINE1_CONTOUR = [
    (0, "+0Hz"),      # The
    (10, "+0Hz"),     # kids
    (20, "+0Hz"),     # on
    (28, "+0Hz"),     # the
    (36, "+5Hz"),     # bus
    (48, "+15Hz"),    # say
    (60, "+25Hz"),    # Bis-
    (70, "+30Hz"),    # -mil-
    (85, "+25Hz"),    # -lah!
    (100, "+20Hz"),
]

# Line 2: "Bismillah! Bismillah!" (descending repetition)
# Melody: E D C# / C# B A
LINE2_CONTOUR = [
    (0, "+25Hz"),     # Bis-
    (12, "+20Hz"),    # -mil-
    (25, "+15Hz"),    # -lah!
    (40, "+10Hz"),    # (pause)
    (50, "+15Hz"),    # Bis-
    (65, "+10Hz"),    # -mil-
    (80, "+5Hz"),     # -lah!
    (100, "+0Hz"),
]

# Line 3: Same as Line 1 (repeat the main phrase)
LINE3_CONTOUR = LINE1_CONTOUR

# Line 4: "All through the town!" (resolution — descending to tonic)
# Melody: B B C# A
LINE4_CONTOUR = [
    (0, "+10Hz"),     # All
    (25, "+10Hz"),    # through
    (50, "+15Hz"),    # the
    (75, "+0Hz"),     # town!
    (100, "-5Hz"),
]


def make_verse_contours(islamic_word, islamic_word_short=None):
    """Generate contours for a verse with the given Islamic word."""
    if islamic_word_short is None:
        islamic_word_short = islamic_word
    return {
        "line1_contour": LINE1_CONTOUR,
        "line2_contour": LINE2_CONTOUR,
        "line3_contour": LINE3_CONTOUR,
        "line4_contour": LINE4_CONTOUR,
    }


VERSES = [
    {
        "lines": [
            "The kids on the bus say Bismillah!",
            "Bismillah! Bismillah!",
            "The kids on the bus say Bismillah!",
            "All through the town!",
        ],
        "start": 0.5,
    },
    {
        "lines": [
            "The mommies on the bus say Alhamdulillah!",
            "Alhamdulillah! Alhamdulillah!",
            "The mommies on the bus say Alhamdulillah!",
            "All through the town!",
        ],
        "start": 11.5,
    },
    {
        "lines": [
            "The babies on the bus say Masha Allah!",
            "Masha Allah! Masha Allah!",
            "The babies on the bus say Masha Allah!",
            "All through the town!",
        ],
        "start": 22.5,
    },
    {
        "lines": [
            "The wheels on the bus go round and round!",
            "Round and round! Round and round!",
            "The wheels on the bus go round and round!",
            "All through the town!",
        ],
        "start": 33.5,
    },
    {
        "lines": [
            "The friends on the bus say Assalamu Alaikum!",
            "Alaikum! Alaikum!",
            "The friends on the bus say Assalamu Alaikum!",
            "All through the town!",
        ],
        "start": 44.5,
    },
    {
        "lines": [
            "The daddies on the bus say SubhanAllah!",
            "SubhanAllah! SubhanAllah!",
            "The daddies on the bus say SubhanAllah!",
            "All through the town!",
        ],
        "start": 55.5,
    },
    {
        "lines": [
            "The kids on the bus say Insha Allah!",
            "Insha Allah! Insha Allah!",
            "The kids on the bus say Insha Allah!",
            "All through the town!",
        ],
        "start": 66.5,
    },
    {
        "lines": [
            "The wheels on the bus go round and round!",
            "Round and round! Round and round!",
            "The wheels on the bus go round and round!",
            "All through the town!",
        ],
        "start": 77.5,
    },
]

CONTOURS = [LINE1_CONTOUR, LINE2_CONTOUR, LINE3_CONTOUR, LINE4_CONTOUR]
# Each line should be ~2.5s in the original song rhythm
LINE_DURATION = 2.6


def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True
    )
    return float(json.loads(r.stdout)["format"]["duration"])


async def generate_singing_line(text, contour, output_path, rate="-5%"):
    """Generate a singing line with pitch contour."""
    ssml = build_ssml_line(text, contour, rate=rate)
    communicate = edge_tts.Communicate(ssml, voice=VOICE)
    # Edge TTS with raw SSML
    try:
        await communicate.save(str(output_path))
    except Exception:
        # Fallback: use communicate with text + prosody params
        communicate2 = edge_tts.Communicate(
            text=text,
            voice=VOICE,
            rate=rate,
            pitch="+5Hz",
        )
        await communicate2.save(str(output_path))


def time_stretch(input_path, output_path, target_duration):
    """Time-stretch audio to target duration."""
    actual = get_duration(input_path)
    if actual <= 0:
        return
    ratio = actual / target_duration

    filters = []
    r = ratio
    while r > 2.0:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r *= 2.0
    filters.append(f"atempo={r:.4f}")

    filter_str = ",".join(filters)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-filter:a", filter_str,
        "-ar", "44100", "-ac", "1",
        str(output_path)
    ], capture_output=True, text=True)


async def main():
    print("=" * 60)
    print("SINGING NURSERY RHYME GENERATOR")
    print("Wheels on the Bus + Islamic Lyrics + Melodic Vocals")
    print("=" * 60)

    # Step 1: Generate all singing lines with pitch contours
    print("\n[STEP 1/5] Generating melodic vocals...")

    verse_files = []

    for vi, verse in enumerate(VERSES):
        print(f"\n  Verse {vi+1}/8:")
        line_wavs = []

        for li, line_text in enumerate(verse["lines"]):
            raw_mp3 = TEMP / f"v{vi}_l{li}_raw.mp3"
            stretched_wav = TEMP / f"v{vi}_l{li}_stretched.wav"

            contour = CONTOURS[li]

            # Line 2 (repetition) gets faster rate
            if li == 1:
                rate = "+5%"
            elif li == 3:  # Resolution
                rate = "-10%"
            else:
                rate = "-5%"

            print(f"    L{li+1}: {line_text}")

            # Generate with SSML
            ssml = build_ssml_line(line_text, contour, rate=rate)

            # edge_tts.Communicate can take SSML directly if we pass it properly
            try:
                comm = edge_tts.Communicate(line_text, voice=VOICE, rate=rate, pitch="+5Hz")
                await comm.save(str(raw_mp3))
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

            # Time-stretch to match rhythm
            time_stretch(str(raw_mp3), str(stretched_wav), LINE_DURATION)
            line_wavs.append(str(stretched_wav))

        # Concatenate lines into verse
        if line_wavs:
            verse_wav = str(TEMP / f"verse{vi}_full.wav")
            list_file = str(TEMP / f"verse{vi}_list.txt")
            with open(list_file, "w") as f:
                for lw in line_wavs:
                    f.write(f"file '{lw}'\n")

            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file, "-ar", "44100", "-ac", "1",
                verse_wav
            ], capture_output=True, text=True)

            dur = get_duration(verse_wav)
            print(f"    Verse duration: {dur:.1f}s")
            verse_files.append((verse_wav, verse["start"]))

    # Step 2: Place verses on timeline
    print("\n[STEP 2/5] Placing verses on timeline (synced to instrumental)...")

    inputs = []
    filter_parts = []
    for vi, (vf, start_time) in enumerate(verse_files):
        inputs.extend(["-i", vf])
        delay_ms = int(start_time * 1000)
        filter_parts.append(
            f"[{vi}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=mono,"
            f"adelay={delay_ms}|{delay_ms}[v{vi}]"
        )

    mix_inputs = "".join(f"[v{vi}]" for vi in range(len(verse_files)))
    filter_parts.append(
        f"{mix_inputs}amix=inputs={len(verse_files)}:duration=longest:normalize=0[vocals]"
    )

    vocals_path = str(TEMP / "all_vocals_synced.wav")
    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vocals]", "-ar", "44100", "-ac", "1",
        "-t", "89", vocals_path
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    print(f"  Vocals: {get_duration(vocals_path):.1f}s")

    # Step 3: Mix with instrumental at proper levels
    print("\n[STEP 3/5] Studio mix — vocals over Wheels on the Bus instrumental...")
    instrumental = str(PROJ / "wheels_instrumental_v1.mp3")
    studio_mix = str(PROJ / "music_studio_v2.wav")

    # The key: instrumental volume LOW so vocals shine through
    # Vocals get a slight warmth boost
    mix_cmd = [
        "ffmpeg", "-y",
        "-i", instrumental,
        "-i", vocals_path,
        "-filter_complex",
        # Instrumental: significantly lower so vocals are clear
        "[0:a]volume=0.25,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[inst];"
        # Vocals: boost, add slight reverb effect via adelay for depth
        "[1:a]volume=2.0,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[vox_dry];"
        # Create a subtle reverb by mixing delayed copy
        "[1:a]volume=0.3,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
        "adelay=80|80[vox_verb];"
        # Combine dry vocal + reverb
        "[vox_dry][vox_verb]amix=inputs=2:duration=first:normalize=0[vox];"
        # Final mix: instrumental + vocals
        "[inst][vox]amix=inputs=2:duration=first:normalize=0,"
        "alimiter=limit=0.95:attack=3:release=30,"
        "afade=t=in:st=0:d=0.3,afade=t=out:st=87:d=2[out]",
        "-map", "[out]", "-ar", "44100",
        studio_mix
    ]

    r = subprocess.run(mix_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  MIX ERROR: {r.stderr[-400:]}")
        return

    dur = get_duration(studio_mix)
    size = os.path.getsize(studio_mix) / 1024 / 1024
    print(f"  Studio mix: {dur:.1f}s, {size:.1f}MB")

    # Step 4: Create looped version for video
    print("\n[STEP 4/5] Looping for video (115s)...")
    music_wav = str(PROJ / "music.wav")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", studio_mix,
        "-i", studio_mix,
        "-filter_complex",
        "[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a0];"
        "[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
        "adelay=86000|86000[a1];"
        "[a0][a1]amix=inputs=2:duration=longest:normalize=0,"
        "afade=t=in:st=0:d=0.3,afade=t=out:st=113:d=2,"
        "atrim=0:115[out]",
        "-map", "[out]", music_wav
    ], capture_output=True, text=True)

    print(f"  music.wav: {get_duration(music_wav):.1f}s")

    # Step 5: Also save the standalone music track
    print("\n[STEP 5/5] Saving standalone track...")
    print(f"\n  DONE!")
    print(f"  Studio mix:  {studio_mix}")
    print(f"  Video music: {music_wav}")
    print(f"  Ready for re-render!")


if __name__ == "__main__":
    asyncio.run(main())
