"""
Karaoke Captions — CocoMelon-style word-synced lyric highlight.

Each lyric line is shown at the bottom of the frame; as the singer sings, the
words "fill" from the unsung color to the sung color (libass karaoke \\kf wipe)
so viewers can read along exactly like CocoMelon.

How the timing works:
  1. The assembled clip plan gives each lyric line its window on the RENDERED
     timeline (same crossfade-overlap math as final_render.srt).
  2. faster-whisper transcribes the song with word timestamps (same settings
     as lyric_sync — no VAD, sung vocals decode fine). Text content is NOT
     matched (Whisper writes Urdu in native script); only the word TIMINGS
     inside each line window are used.
  3. The displayed words of the line are mapped onto the heard vocal span:
     1:1 when the word counts match, otherwise proportionally by word length.
  4. An .ass subtitle file with per-word \\kf tags is written and burned into
     the final render with ffmpeg (libass). The caption-less master is kept
     as final_render_nocaptions.mp4.

Everything is best-effort: any failure leaves the render untouched.
"""
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Style (1080p PlayRes): big rounded bold text, white unsung, sunny yellow
# fill as sung, thick dark outline so it reads on any footage.
FONT = "Arial Rounded MT Bold"     # libass falls back to Arial if missing
FONT_SIZE = 64
SUNG_COLOR = "&H0000D7FF"          # BGR: gold/yellow  (#FFD700)
UNSUNG_COLOR = "&H00FFFFFF"        # white
OUTLINE_COLOR = "&H00201028"       # deep plum outline (BGR)
MARGIN_V = 56

ASS_HEADER = f"""[Script Info]
Title: AI Director karaoke captions
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{FONT},{FONT_SIZE},{SUNG_COLOR},{UNSUNG_COLOR},{OUTLINE_COLOR},&H7F000000,-1,0,0,0,100,100,0,0,1,5,2,2,80,80,{MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


# ── timing ─────────────────────────────────────────────────────────────────

def line_windows(cues: list[tuple[float, str]], transition_duration: float,
                 total_duration: float) -> list[tuple[float, float, str]]:
    """(start, end, text) per lyric line on the rendered timeline — the same
    crossfade-overlap math final_render.srt uses, with consecutive clips that
    share a line merged into one window."""
    td = float(transition_duration or 0)
    starts, t = [], 0.0
    for i, (dur, _) in enumerate(cues):
        starts.append(max(0.0, t - i * td))
        t += float(dur)
    video_end = float(total_duration or 0) or \
        max(0.0, t - max(0, len(cues) - 1) * td)

    entries: list[tuple[float, float, str]] = []
    for i, (dur, text) in enumerate(cues):
        text = (text or "").strip()
        end = starts[i + 1] if i + 1 < len(starts) else video_end
        if entries and entries[-1][2] == text:
            entries[-1] = (entries[-1][0], end, text)
        else:
            entries.append((starts[i], min(end, video_end), text))
    return [(s, e, txt) for s, e, txt in entries
            if txt and txt != "(instrumental)" and e > s]


def word_timings(audio_path: str,
                 language: Optional[str] = None) -> Optional[list[tuple[str, float, float]]]:
    """Heard words as (text, start, end) from the song. Same recipe as
    lyric_sync: small model on CPU, no VAD (silero deletes singing),
    low-confidence and no-speech segments dropped."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.info("[Karaoke] faster-whisper not installed — even spread only")
        return None
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(audio_path), word_timestamps=True, vad_filter=False,
        beam_size=1, condition_on_previous_text=False, language=language)
    words = []
    for seg in segments:
        if getattr(seg, "no_speech_prob", 0.0) > 0.8:
            continue
        for w in (seg.words or []):
            if getattr(w, "probability", 1.0) < 0.2:
                continue
            words.append((str(w.word).strip(), float(w.start), float(w.end)))
    logger.info(f"[Karaoke] {len(words)} vocal words heard")
    return words or None


def _norm_token(w: str) -> str:
    import re
    return re.sub(r"[^\w']+", "", w.lower())


def align_to_transcript(
    entries: list[tuple[float, float, str]],
    heard: list[tuple[str, float, float]],
    min_match_rate: float = 0.35,
):
    """TEXT-anchored karaoke timing: align the lyric words to the words
    whisper actually heard (order-preserving DP, fuzzy word match), so each
    line starts when it is really sung — independent of the clip plan.
    Whole-sentence drift disappears wherever the transcription is decent.
    Returns line_words like assign_word_times, or None when too little of
    the lyrics was matched (Urdu-script transcripts, mumbled vocals...)."""
    from difflib import SequenceMatcher

    disp = []                       # (entry_idx, word)
    for i, (_s, _e, text) in enumerate(entries):
        for w in text.split():
            disp.append((i, w))
    a = [_norm_token(w) for _, w in disp]
    b = [_norm_token(t) for t, _, _ in heard]
    n, m = len(a), len(b)
    if not n or not m:
        return None

    def sim(x: str, y: str) -> int:
        if not x or not y:
            return 0
        if x == y:
            return 10
        r = SequenceMatcher(None, x, y).ratio()
        return 8 if r >= 0.85 else (6 if r >= 0.7 else 0)

    # Needleman-Wunsch (gap penalty 0 = weighted LCS); O(n*m) is fine at
    # a few hundred words each side.
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        row, prev = score[i], score[i - 1]
        ai = a[i - 1]
        for j in range(1, m + 1):
            s = sim(ai, b[j - 1])
            row[j] = max(prev[j], row[j - 1],
                         prev[j - 1] + s if s else 0)
    # backtrack matched pairs
    matches: dict[int, int] = {}    # display index -> heard index
    i, j = n, m
    while i > 0 and j > 0:
        s = sim(a[i - 1], b[j - 1])
        if s and score[i][j] == score[i - 1][j - 1] + s:
            matches[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif score[i - 1][j] >= score[i][j - 1]:
            i -= 1
        else:
            j -= 1

    rate = len(matches) / n
    logger.info(f"[Karaoke] transcript alignment: {len(matches)}/{n} lyric "
                f"words matched ({rate:.0%})")
    if rate < min_match_rate:
        return None

    # anchor times per display word; interpolate the unmatched ones between
    # their nearest anchors (proportional to word length)
    times: list[Optional[tuple[float, float]]] = [None] * n
    for di, hj in matches.items():
        times[di] = (heard[hj][1], heard[hj][2])
    anchors = sorted(matches.keys())

    def interpolate(lo_i, lo_t, hi_i, hi_t):
        span = [k for k in range(lo_i + 1, hi_i)]
        if not span:
            return
        weights = [len(a[k]) + 1 for k in span]
        total = sum(weights) or 1
        t = lo_t
        avail = max(0.0, hi_t - lo_t)
        for k, w in zip(span, weights):
            dur = avail * w / total
            times[k] = (t, t + dur)
            t += dur

    for prev_a, next_a in zip(anchors, anchors[1:]):
        interpolate(prev_a, times[prev_a][1], next_a, times[next_a][0])
    if anchors:
        first, last = anchors[0], anchors[-1]
        # words before the first / after the last anchor: ~0.35s each
        t = times[first][0]
        for k in range(first - 1, -1, -1):
            times[k] = (max(0.0, t - 0.35), t)
            t = max(0.0, t - 0.35)
        t = times[last][1]
        for k in range(last + 1, n):
            times[k] = (t, t + 0.35)
            t += 0.35

    # regroup into lines; the line window hugs its own sung words
    out = []
    for i, (_s, _e, text) in enumerate(entries):
        words = [(w, round(times[k][0], 3), round(times[k][1], 3))
                 for k, (ei, w) in enumerate(disp) if ei == i]
        if not words:
            continue
        out.append([words[0][1], words[-1][2], words])

    # display windows: appear up to 0.8s before the first word, hold 1s after
    # the last — clamped so consecutive lines never overlap, and monotonic
    for idx, line in enumerate(out):
        lead = out[idx - 1][1] if idx > 0 else 0.0
        line[0] = max(lead, line[0] - 0.8)
        nxt = out[idx + 1][2][0][1] if idx + 1 < len(out) else line[1] + 1.0
        line[1] = max(line[0] + 0.2, min(line[1] + 1.0, nxt))
    return [tuple(line) for line in out]


def assign_word_times(
    entries: list[tuple[float, float, str]],
    vocal_words: Optional[list[tuple[float, float]]],
) -> list[tuple[float, float, list[tuple[str, float, float]]]]:
    """Per line: (line_start, line_end, [(word, w_start, w_end), ...]).
    Display words ride on the heard vocal span inside the line's window;
    without vocals they spread evenly (still reads fine, just less snappy)."""
    out = []
    for start, end, text in entries:
        display = text.split()
        if not display:
            continue
        heard = [(a, b) for a, b in (vocal_words or [])
                 if a >= start - 0.3 and a < end] or None
        if heard and len(heard) == len(display):
            times = [(max(start, a), min(end, b)) for a, b in heard]
        else:
            if heard:
                span_a = max(start, heard[0][0])
                span_b = min(end, max(b for _, b in heard))
                if span_b - span_a < 0.5:
                    span_a, span_b = start, end
            else:
                span_a, span_b = start, end
            weights = [len(w) + 1 for w in display]
            total_w = sum(weights)
            times, t = [], span_a
            for w in weights:
                dur = (span_b - span_a) * w / total_w
                times.append((t, t + dur))
                t += dur
        words = [(w, round(a, 3), round(b, 3))
                 for w, (a, b) in zip(display, times)]
        out.append((start, end, words))
    return out


# ── ASS writing ────────────────────────────────────────────────────────────

def _ass_ts(sec: float) -> str:
    cs = int(round(max(0.0, sec) * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def write_ass(line_words, out_path) -> Optional[Path]:
    """One Dialogue per lyric line with per-word \\kf fills. The wait between
    the line appearing and the vocal starting is consumed by a leading no-text
    \\kf tag so the fill starts exactly on the voice."""
    events = []
    for line_start, line_end, words in line_words:
        if not words:
            continue
        parts = []
        wait_cs = int(round((words[0][1] - line_start) * 100))
        if wait_cs > 0:
            parts.append(f"{{\\kf{wait_cs}}}")
        for i, (word, w_start, w_end) in enumerate(words):
            # fill runs until the NEXT word starts so the wipe is continuous
            until = words[i + 1][1] if i + 1 < len(words) else w_end
            dur_cs = max(1, int(round((until - w_start) * 100)))
            sep = "" if i == len(words) - 1 else " "
            parts.append(f"{{\\kf{dur_cs}}}{word}{sep}")
        text = "".join(parts).replace("\n", " ")
        events.append(
            f"Dialogue: 0,{_ass_ts(line_start)},{_ass_ts(line_end)},"
            f"Karaoke,,0,0,0,,{text}")
    if not events:
        return None
    out_path = Path(out_path)
    out_path.write_text(ASS_HEADER + "\n".join(events) + "\n",
                        encoding="utf-8-sig")
    logger.info(f"[Karaoke] {len(events)} karaoke lines -> {out_path}")
    return out_path


def write_srt(line_words, out_path) -> Optional[Path]:
    """Plain .srt with the SAME aligned line timings as the karaoke — for
    editors that import SRT (CapCut etc.) so manual tweaks start from the
    correct sync, not the clip plan."""
    def ts(sec: float) -> str:
        ms = int(round(max(0.0, sec) * 1000))
        h, rem = divmod(ms, 3600000)
        m, rem = divmod(rem, 60000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for start, end, words in line_words:
        if not words:
            continue
        text = " ".join(w for w, _a, _b in words)
        blocks.append(f"{len(blocks) + 1}\n{ts(start)} --> {ts(end)}\n{text}\n")
    if not blocks:
        return None
    out_path = Path(out_path)
    out_path.write_text("\n".join(blocks), encoding="utf-8")
    logger.info(f"[Karaoke] aligned SRT: {out_path} ({len(blocks)} cues)")
    return out_path


# ── burn-in ────────────────────────────────────────────────────────────────

def burn_in(ffmpeg_bin, video_path, ass_path, keep_plain: bool = True) -> str:
    """Burn the .ass into the video IN PLACE (final_render.mp4 keeps its name,
    the caption-less master is kept as *_nocaptions.mp4)."""
    video = Path(video_path)
    plain = video.with_name(video.stem + "_nocaptions" + video.suffix)
    tmp = video.with_name(video.stem + "_karaoke_tmp" + video.suffix)
    # libass filter path: forward slashes, escaped drive colon
    ass_arg = Path(ass_path).as_posix().replace(":", r"\:")
    subprocess.run(
        [str(ffmpeg_bin), "-y", "-i", str(video),
         "-vf", f"ass='{ass_arg}'",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-c:a", "copy", str(tmp)],
        check=True, capture_output=True)
    if keep_plain:
        if plain.exists():
            plain.unlink()
        video.rename(plain)
    else:
        video.unlink()
    tmp.rename(video)
    logger.info(f"[Karaoke] captions burned into {video}"
                + (f" (plain master: {plain.name})" if keep_plain else ""))
    return str(video)


# ── top-level ──────────────────────────────────────────────────────────────

def add_to_render(ffmpeg_bin, video_path, music_path,
                  cues: list[tuple[float, str]], transition_duration: float,
                  total_duration: float,
                  language: Optional[str] = None) -> Optional[str]:
    """Full pass: line windows → whisper word timings → .ass → burn-in.
    Returns the .ass path, or None when there was nothing to caption."""
    entries = line_windows(cues, transition_duration, total_duration)
    if not entries:
        return None
    heard = None
    try:
        heard = word_timings(music_path, language=language)
    except Exception as e:
        logger.warning(f"[Karaoke] transcription failed (even spread used): {e}")

    # Best: TEXT-anchored — lines follow the actually-sung words, immune to
    # clip-plan drift. Falls back to clip-plan windows + heard-word rhythm,
    # then to an even spread.
    line_words = None
    if heard:
        try:
            line_words = align_to_transcript(entries, heard)
        except Exception as e:
            logger.warning(f"[Karaoke] transcript alignment failed: {e}")
    if line_words is None:
        line_words = assign_word_times(
            entries, [(s, e) for _t, s, e in (heard or [])] or None)
    video = Path(video_path)
    ass_path = write_ass(line_words, video.with_suffix(".ass"))
    if not ass_path:
        return None
    # sidecar with the same aligned timings, for CapCut/editor tweaking
    try:
        write_srt(line_words, video.with_suffix(".srt"))
    except Exception as e:
        logger.warning(f"[Karaoke] aligned .srt failed (non-fatal): {e}")
    burn_in(ffmpeg_bin, video_path, ass_path)
    return str(ass_path)
