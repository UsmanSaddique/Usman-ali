"""
Lyrics Parser — convert structured lyrics with section markers into timed segments.

Parses [intro], [verse], [chorus], [bridge], [outro], [hook] markers and
distributes them proportionally across a song's duration.
"""
import re
from dataclasses import dataclass


@dataclass
class LyricSegment:
    index: int
    section_type: str       # intro, verse, chorus, bridge, outro, hook
    text: str
    start_sec: float
    end_sec: float
    duration: float
    is_instrumental: bool


# Estimated seconds per lyric line by section type
_LINE_DURATIONS = {
    "intro": 4.0,
    "outro": 4.0,
    "hook": 3.5,
    "chorus": 3.5,
    "verse": 3.0,
    "bridge": 3.0,
}

_SECTION_RE = re.compile(
    r"\[(intro|verse|chorus|hook|bridge|outro)\]",
    re.IGNORECASE,
)
_PAREN_RE = re.compile(r"^\s*\(.*\)\s*$")


def parse_lyrics(
    lyrics: str,
    song_duration: float,
    max_segments: int = 0,
    target_segment_sec: float = 5.0,
) -> list[LyricSegment]:
    """Parse structured lyrics into timed segments.

    Args:
        lyrics: Full lyrics with [section] markers.
        song_duration: Actual song duration in seconds.
        max_segments: If >0, cap the total number of segments by grouping lines
                      more aggressively. 0 = natural grouping.
        target_segment_sec: Target duration per segment (used for grouping).

    Returns:
        List of LyricSegment with timestamps spanning the full song duration.
    """
    sections = _split_sections(lyrics)
    if not sections:
        return [LyricSegment(0, "verse", lyrics.strip() or "(instrumental)",
                             0.0, song_duration, song_duration, not lyrics.strip())]

    raw_segments = []
    for sec_type, sec_text in sections:
        lines = _extract_lines(sec_text)
        if not lines:
            raw_segments.append((sec_type, "(instrumental)", True))
            continue
        for line in lines:
            is_inst = bool(_PAREN_RE.match(line))
            raw_segments.append((sec_type, line, is_inst))

    if not raw_segments:
        return [LyricSegment(0, "verse", "(instrumental)",
                             0.0, song_duration, song_duration, True)]

    grouped = _group_segments(raw_segments, target_segment_sec, max_segments)
    return _assign_timestamps(grouped, song_duration)


def _split_sections(lyrics: str) -> list[tuple[str, str]]:
    """Split lyrics into (section_type, text_block) pairs."""
    parts = _SECTION_RE.split(lyrics)
    # parts alternates: [pre-text, section_name, text, section_name, text, ...]
    sections = []
    i = 1
    while i < len(parts) - 1:
        sec_type = parts[i].lower()
        sec_text = parts[i + 1]
        sections.append((sec_type, sec_text))
        i += 2
    return sections


def _extract_lines(text: str) -> list[str]:
    """Extract non-empty lines from a section text block."""
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def _estimate_duration(sec_type: str, is_instrumental: bool) -> float:
    if is_instrumental:
        return 4.0
    return _LINE_DURATIONS.get(sec_type, 3.0)


def _group_segments(
    raw: list[tuple[str, str, bool]],
    target_sec: float,
    max_segments: int,
) -> list[tuple[str, str, bool, float]]:
    """Group raw lines into segments of approximately target_sec duration.

    Returns: [(section_type, combined_text, is_instrumental, estimated_duration)]
    """
    if max_segments > 0 and len(raw) <= max_segments:
        return [(t, txt, inst, _estimate_duration(t, inst)) for t, txt, inst in raw]

    groups: list[tuple[str, str, bool, float]] = []
    buf_type = raw[0][0]
    buf_lines: list[str] = []
    buf_dur = 0.0
    buf_inst = raw[0][2]

    for sec_type, line, is_inst in raw:
        line_dur = _estimate_duration(sec_type, is_inst)

        # Start new group if: section type changed, or accumulated duration
        # would exceed target (allow up to 1.5x before splitting)
        if buf_lines and (
            sec_type != buf_type
            or buf_dur + line_dur > target_sec * 1.5
        ):
            groups.append((buf_type, "\n".join(buf_lines), buf_inst, buf_dur))
            buf_type = sec_type
            buf_lines = []
            buf_dur = 0.0
            buf_inst = is_inst

        buf_lines.append(line)
        buf_dur += line_dur
        if not is_inst:
            buf_inst = False

    if buf_lines:
        groups.append((buf_type, "\n".join(buf_lines), buf_inst, buf_dur))

    # If max_segments set and we still have too many, merge smallest adjacent pairs
    if max_segments > 0:
        while len(groups) > max_segments:
            min_dur = float("inf")
            min_idx = 0
            for i in range(len(groups) - 1):
                combined = groups[i][3] + groups[i + 1][3]
                if combined < min_dur:
                    min_dur = combined
                    min_idx = i
            a, b = groups[min_idx], groups[min_idx + 1]
            merged = (
                a[0],
                a[1] + "\n" + b[1],
                a[2] and b[2],
                a[3] + b[3],
            )
            groups[min_idx:min_idx + 2] = [merged]

    return groups


def _assign_timestamps(
    groups: list[tuple[str, str, bool, float]],
    song_duration: float,
) -> list[LyricSegment]:
    """Scale group durations to fit song_duration exactly, assign timestamps."""
    total_est = sum(g[3] for g in groups)
    if total_est <= 0:
        total_est = len(groups) * 3.0

    scale = song_duration / total_est
    segments = []
    t = 0.0

    for i, (sec_type, text, is_inst, est_dur) in enumerate(groups):
        dur = est_dur * scale
        segments.append(LyricSegment(
            index=i,
            section_type=sec_type,
            text=text,
            start_sec=round(t, 2),
            end_sec=round(t + dur, 2),
            duration=round(dur, 2),
            is_instrumental=is_inst,
        ))
        t += dur

    # Fix rounding: ensure last segment ends exactly at song_duration
    if segments:
        segments[-1].end_sec = round(song_duration, 2)
        segments[-1].duration = round(segments[-1].end_sec - segments[-1].start_sec, 2)

    return segments
