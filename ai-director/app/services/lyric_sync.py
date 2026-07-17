"""
Lyric Sync — snap lyric-segment boundaries to real vocal phrase onsets.

The lyrics parser distributes lines evenly across the song, so chorus visuals
can drift seconds off the actual vocals. This module transcribes the generated
song with faster-whisper (CPU, word timestamps), derives vocal PHRASE ONSETS
(a word after a >=0.6s gap starts a phrase), and snaps each estimated segment
boundary to the nearest onset. Text content is deliberately ignored — Whisper
writes Urdu/Hindi in native script while our lyrics are Roman, so text
matching is unreliable; onset timing is language-agnostic and is what actually
makes cuts feel edited-to-the-music.

Everything here is best-effort: any failure returns None and the caller keeps
the evenly-distributed estimates.
"""
import copy
import logging
import math
from typing import Optional

from app.services.lyrics_parser import LyricSegment

logger = logging.getLogger(__name__)

# A new vocal phrase starts after at least this much wordless audio
PHRASE_GAP_SEC = 0.6
# Never snap a boundary further than this from its estimated position
MAX_SNAP_SEC = 2.5
# Scenes shorter than this get merged into their predecessor
MIN_SCENE_SEC = 1.5


def whisper_language(profile_language: str) -> Optional[str]:
    """Map a channel profile's language to a Whisper language code. Roman
    Urdu / Urdu / Hindi all decode best as 'hi' (Hindustani); auto-detection
    on sung vocals is unreliable (it guessed Sanskrit and heard 4x fewer
    words on a real track)."""
    lang = (profile_language or "").lower()
    if "urdu" in lang or "hindi" in lang or "hindustani" in lang:
        return "hi"
    if "english" in lang:
        return "en"
    return None


def _vocal_phrase_onsets(audio_path: str,
                         language: Optional[str] = None) -> Optional[list[float]]:
    """Transcribe the song and return sorted phrase-onset times (seconds)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.info("[LyricSync] faster-whisper not installed — skipping sync")
        return None

    # small = good word timing at ~1-2 min CPU cost per song; runs while the
    # GPU is free (scenes aren't generating yet).
    model = WhisperModel("small", device="cpu", compute_type="int8")
    # NO vad_filter: silero VAD classifies singing-over-music as non-speech
    # and deletes virtually the whole song. Whisper itself hears sung vocals
    # fine; hallucinated instrumental-break "words" are dropped below via
    # no_speech_prob / word confidence instead.
    segments, _info = model.transcribe(
        str(audio_path), word_timestamps=True, vad_filter=False,
        beam_size=1, condition_on_previous_text=False,
        language=language,
    )

    words = []
    for seg in segments:
        if getattr(seg, "no_speech_prob", 0.0) > 0.8:
            continue
        for w in (seg.words or []):
            if getattr(w, "probability", 1.0) < 0.2:
                continue
            words.append((float(w.start), float(w.end)))
    if len(words) < 4:
        logger.info(f"[LyricSync] Only {len(words)} words heard — skipping sync")
        return None

    onsets = [words[0][0]]
    for (prev_s, prev_e), (cur_s, cur_e) in zip(words, words[1:]):
        if cur_s - prev_e >= PHRASE_GAP_SEC:
            onsets.append(cur_s)
    logger.info(f"[LyricSync] {len(words)} words -> {len(onsets)} vocal phrase onsets")
    return onsets


def sync_segments_to_audio(
    segments: list[LyricSegment],
    audio_path: str,
    max_clip_sec: float,
    language: Optional[str] = None,
) -> Optional[list[LyricSegment]]:
    """Return a NEW segment list whose boundaries sit on vocal phrase onsets,
    with every duration inside [MIN_SCENE_SEC, max_clip_sec] (long segments
    are split into equal parts sharing the same text; tiny ones are merged).
    Returns None if the audio can't be analyzed — caller keeps estimates."""
    if not segments:
        return None
    onsets = _vocal_phrase_onsets(audio_path, language=language)
    if not onsets:
        return None

    song_end = segments[-1].end_sec
    # Snap each internal boundary (start of segment i>0) to the nearest onset,
    # keeping boundaries monotonic and scenes >= MIN_SCENE_SEC apart.
    boundaries = [0.0]
    snapped = 0
    for seg in segments[1:]:
        est = seg.start_sec
        lo = boundaries[-1] + MIN_SCENE_SEC
        candidates = [o for o in onsets if abs(o - est) <= MAX_SNAP_SEC and o >= lo]
        if candidates:
            best = min(candidates, key=lambda o: abs(o - est))
            if abs(best - est) > 0.05:
                snapped += 1
            boundaries.append(round(best, 2))
        else:
            boundaries.append(round(max(est, lo), 2))
    boundaries.append(round(song_end, 2))

    # Rebuild segments from the snapped boundaries, enforcing the clip cap.
    out: list[LyricSegment] = []
    for seg, start, end in zip(segments, boundaries, boundaries[1:]):
        dur = round(end - start, 2)
        if dur < MIN_SCENE_SEC and out:
            out[-1].end_sec = end
            out[-1].duration = round(end - out[-1].start_sec, 2)
            continue
        n_parts = max(1, math.ceil(dur / max_clip_sec))
        part = dur / n_parts
        for k in range(n_parts):
            piece = copy.copy(seg)
            piece.start_sec = round(start + k * part, 2)
            piece.end_sec = round(start + (k + 1) * part, 2)
            piece.duration = round(piece.end_sec - piece.start_sec, 2)
            out.append(piece)

    # A split piece can still be a hair under MIN_SCENE_SEC only if the whole
    # segment was; already handled above. Renumber for build_prompts.
    for i, seg in enumerate(out):
        seg.index = i

    logger.info(
        f"[LyricSync] {len(segments)} segments -> {len(out)} scenes; "
        f"{snapped} boundaries snapped to vocal onsets")
    return out
