"""
Studio-grade Islamic nursery rhyme: Wheels on the Bus with Islamic lyrics.

Pipeline:
1. Generate each verse as separate TTS audio using Microsoft Ana (child voice)
2. Time-stretch each line to match the Wheels on the Bus rhythm (86 BPM)
3. Mix vocals over the v1 instrumental (which has the melody)
4. Master the final mix with proper levels

Think: top 0.1% children's music studio.
"""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

# Edge TTS for child vocals
import edge_tts

PROJ = Path(r"C:\Users\PC\Desktop\VideoMaker\ai-director\projects\3efaca79-b3ae-4c20-a6a7-9c294bb368d1")
TEMP = PROJ / "vocal_temp"
TEMP.mkdir(exist_ok=True)

VOICE = "en-US-AnaNeural"  # Microsoft child voice ("Cute" personality)

# ═══════════════════════════════════════════════════════════════
# LYRICS — Each line timed to "Wheels on the Bus" rhythm
# Original song: 86 BPM, ~4 beats per phrase
# Each verse is ~11 seconds in v1
# Structure: line1 (2.5s) | repeat (2.5s) | line1 again (2.5s) | resolution (2.5s)
# ═══════════════════════════════════════════════════════════════

VERSES = [
    {
        "lines": [
            "The kids on the bus say Bismillah!",
            "Bismillah! Bismillah!",
            "The kids on the bus say Bismillah!",
            "All through the town!",
        ],
        "start": 0.0,
    },
    {
        "lines": [
            "The mommies on the bus say Alhamdulillah!",
            "Alhamdulillah! Alhamdulillah!",
            "The mommies on the bus say Alhamdulillah!",
            "All through the town!",
        ],
        "start": 11.0,
    },
    {
        "lines": [
            "The babies on the bus say MashaAllah!",
            "MashaAllah! MashaAllah!",
            "The babies on the bus say MashaAllah!",
            "All through the town!",
        ],
        "start": 22.0,
    },
    {
        "lines": [
            "The wheels on the bus go round and round!",
            "Round and round! Round and round!",
            "The wheels on the bus go round and round!",
            "All through the town!",
        ],
        "start": 33.0,
    },
    {
        "lines": [
            "The friends on the bus say Assalamu Alaikum!",
            "Alaikum! Alaikum!",
            "The friends on the bus say Assalamu Alaikum!",
            "All through the town!",
        ],
        "start": 44.0,
    },
    {
        "lines": [
            "The daddies on the bus say SubhanAllah!",
            "SubhanAllah! SubhanAllah!",
            "The daddies on the bus say SubhanAllah!",
            "All through the town!",
        ],
        "start": 55.0,
    },
    {
        "lines": [
            "The kids on the bus say InshaAllah!",
            "InshaAllah! InshaAllah!",
            "The kids on the bus say InshaAllah!",
            "All through the town!",
        ],
        "start": 66.0,
    },
    {
        "lines": [
            "The wheels on the bus go round and round!",
            "Round and round! Round and round!",
            "The wheels on the bus go round and round!",
            "All through the town!",
        ],
        "start": 77.0,
    },
]

LINE_DURATION = 2.7  # seconds per line in the original song


def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True
    )
    return float(json.loads(r.stdout)["format"]["duration"])


async def generate_line_audio(text, output_path, rate="+15%", pitch="+5Hz"):
    """Generate a single line using Edge TTS Ana voice."""
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=rate,
        pitch=pitch,
    )
    await communicate.save(str(output_path))


async def generate_all_vocals():
    """Generate TTS for every line of every verse."""
    print("Generating vocals with Microsoft Ana (child voice)...\n")

    all_line_files = []

    for vi, verse in enumerate(VERSES):
        verse_files = []
        for li, line in enumerate(verse["lines"]):
            out_path = TEMP / f"verse{vi}_line{li}.mp3"
            print(f"  v{vi+1} L{li+1}: {line}")

            # Adjust rate for different line types
            if li == 1:  # Repetition line (shorter text, needs to be faster)
                rate = "+20%"
            elif li == 3:  # Resolution line
                rate = "+10%"
            else:  # Main lines
                rate = "+15%"

            await generate_line_audio(line, out_path, rate=rate, pitch="+3Hz")
            verse_files.append(str(out_path))

        all_line_files.append(verse_files)
        print()

    return all_line_files


def time_stretch_line(input_path, output_path, target_duration):
    """Stretch/compress a line to fit the target duration using ffmpeg atempo."""
    actual = get_duration(input_path)
    if actual <= 0:
        return

    ratio = actual / target_duration

    # atempo only accepts 0.5-2.0, chain if needed
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
        "ffmpeg", "-y", "-i", input_path,
        "-filter:a", filter_str,
        "-ar", "44100", "-ac", "1",
        output_path
    ], capture_output=True, text=True)


def build_verse_audio(verse_idx, line_files):
    """Combine 4 lines into a single verse audio with proper timing."""
    stretched_files = []

    for li, lf in enumerate(line_files):
        stretched = str(TEMP / f"verse{verse_idx}_line{li}_stretched.wav")
        time_stretch_line(lf, stretched, LINE_DURATION)
        stretched_files.append(stretched)

    # Concatenate the 4 stretched lines into one verse
    verse_out = str(TEMP / f"verse{verse_idx}_full.wav")
    list_file = str(TEMP / f"verse{verse_idx}_list.txt")

    with open(list_file, "w") as f:
        for sf in stretched_files:
            f.write(f"file '{sf}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-ar", "44100", "-ac", "1",
        verse_out
    ], capture_output=True, text=True)

    return verse_out


def place_verses_on_timeline(verse_files, total_duration=89.0):
    """Place each verse at its correct time position on the timeline."""
    # Build a complex filter that places each verse at its start time
    inputs = []
    filter_parts = []

    for vi, (vf, verse) in enumerate(zip(verse_files, VERSES)):
        inputs.extend(["-i", vf])
        delay_ms = int(verse["start"] * 1000)
        filter_parts.append(
            f"[{vi}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=mono,"
            f"adelay={delay_ms}|{delay_ms}[v{vi}]"
        )

    # Mix all positioned verses together
    mix_inputs = "".join(f"[v{vi}]" for vi in range(len(verse_files)))
    filter_parts.append(
        f"{mix_inputs}amix=inputs={len(verse_files)}:duration=longest:normalize=0[vocals]"
    )

    filter_str = ";".join(filter_parts)
    vocals_out = str(TEMP / "all_vocals.wav")

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_str,
        "-map", "[vocals]",
        "-ar", "44100", "-ac", "1",
        "-t", str(total_duration),
        vocals_out
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR placing vocals: {r.stderr[-300:]}")
    return vocals_out


def final_mix(vocals_path, instrumental_path, output_path):
    """Mix vocals over instrumental with proper levels.

    Studio mix levels:
    - Instrumental (backing track): -6dB (sits underneath)
    - Vocals: 0dB (front and center)
    - Add slight reverb to vocals for polish
    - Limiter on master to prevent clipping
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", instrumental_path,   # 0: instrumental
        "-i", vocals_path,         # 1: vocals
        "-filter_complex",
        # Instrumental: lower volume, make stereo
        "[0:a]volume=0.35,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[inst];"
        # Vocals: boost slightly, add warmth, make stereo
        "[1:a]volume=1.8,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[vox];"
        # Mix together
        "[inst][vox]amix=inputs=2:duration=first:normalize=0,"
        # Master: soft limiter
        "alimiter=limit=0.95:attack=5:release=50,"
        # Fade in/out
        "afade=t=in:st=0:d=0.5,afade=t=out:st=87:d=2[out]",
        "-map", "[out]",
        "-ar", "44100",
        output_path
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR in final mix: {r.stderr[-500:]}")
        return False
    return True


async def main():
    print("=" * 60)
    print("STUDIO ISLAMIC NURSERY RHYME GENERATOR")
    print("Wheels on the Bus + Islamic Lyrics + Child Voice")
    print("=" * 60)

    # Step 1: Generate TTS vocals
    print("\n[STEP 1/4] Generating child vocals (Microsoft Ana)...")
    all_line_files = await generate_all_vocals()

    # Step 2: Time-stretch lines to match rhythm
    print("[STEP 2/4] Time-stretching to match 86 BPM rhythm...")
    verse_audio_files = []
    for vi, line_files in enumerate(all_line_files):
        print(f"  Verse {vi+1}/8...")
        vf = build_verse_audio(vi, line_files)
        verse_audio_files.append(vf)
        dur = get_duration(vf)
        print(f"    Duration: {dur:.1f}s")

    # Step 3: Place on timeline
    print("\n[STEP 3/4] Placing verses on timeline...")
    vocals_path = place_verses_on_timeline(verse_audio_files, total_duration=89.0)
    vocals_dur = get_duration(vocals_path)
    print(f"  Vocals track: {vocals_dur:.1f}s")

    # Step 4: Final mix with instrumental
    print("\n[STEP 4/4] Final studio mix...")
    instrumental = str(PROJ / "wheels_instrumental_v1.mp3")
    output = str(PROJ / "music_studio.wav")

    if final_mix(vocals_path, instrumental, output):
        dur = get_duration(output)
        size = os.path.getsize(output) / 1024 / 1024
        print(f"\n  SUCCESS: {output}")
        print(f"  Duration: {dur:.1f}s, Size: {size:.1f}MB")

        # Also create the looped version for the video (needs 115s)
        print("\n  Creating looped version for video...")
        looped = str(PROJ / "music.wav")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", output,
            "-i", output,
            "-filter_complex",
            "[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a0];"
            "[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            "adelay=86000|86000[a1];"
            "[a0][a1]amix=inputs=2:duration=longest:normalize=0,"
            "afade=t=in:st=0:d=0.5,afade=t=out:st=113:d=2,"
            "atrim=0:115[out]",
            "-map", "[out]",
            looped
        ], capture_output=True, text=True)
        looped_dur = get_duration(looped)
        print(f"  Looped music.wav: {looped_dur:.1f}s")
        print("\n  Ready for re-render!")
    else:
        print("\n  FAILED - check errors above")


if __name__ == "__main__":
    asyncio.run(main())
