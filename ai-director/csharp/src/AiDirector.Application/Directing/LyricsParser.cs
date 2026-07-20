using System.Text.RegularExpressions;

namespace AiDirector.Application.Directing;

/// Port of app/services/lyrics_parser.py. Parses [intro]/[verse]/[chorus]/
/// [hook]/[bridge]/[outro] markers into timed segments spanning the song.
public static class LyricsParser
{
    public sealed record LyricSegment(
        int Index, string SectionType, string Text,
        double StartSec, double EndSec, double Duration, bool IsInstrumental);

    private static readonly Dictionary<string, double> LineDurations = new()
    {
        ["intro"] = 4.0, ["outro"] = 4.0, ["hook"] = 3.5,
        ["chorus"] = 3.5, ["verse"] = 3.0, ["bridge"] = 3.0,
    };
    private static readonly Regex SectionRe =
        new(@"\[(intro|verse|chorus|hook|bridge|outro)\]", RegexOptions.IgnoreCase);
    private static readonly Regex ParenRe = new(@"^\s*\(.*\)\s*$");

    public static List<LyricSegment> Parse(string lyrics, double songDuration,
        int maxSegments = 0, double targetSegmentSec = 5.0)
    {
        var sections = SplitSections(lyrics);
        if (sections.Count == 0)
        {
            var t = lyrics.Trim();
            return [new(0, "verse", string.IsNullOrEmpty(t) ? "(instrumental)" : t,
                0.0, songDuration, songDuration, string.IsNullOrEmpty(t))];
        }

        var raw = new List<(string Type, string Text, bool Inst)>();
        foreach (var (secType, secText) in sections)
        {
            var lines = secText.Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0).ToList();
            if (lines.Count == 0) { raw.Add((secType, "(instrumental)", true)); continue; }
            foreach (var line in lines) raw.Add((secType, line, ParenRe.IsMatch(line)));
        }
        if (raw.Count == 0) return [new(0, "verse", "(instrumental)", 0.0, songDuration, songDuration, true)];

        var grouped = Group(raw, targetSegmentSec, maxSegments);
        return AssignTimestamps(grouped, songDuration);
    }

    private static List<(string, string)> SplitSections(string lyrics)
    {
        var parts = SectionRe.Split(lyrics);
        var sections = new List<(string, string)>();
        for (var i = 1; i < parts.Length - 1; i += 2)
            sections.Add((parts[i].ToLowerInvariant(), parts[i + 1]));
        return sections;
    }

    private static double EstDuration(string secType, bool inst) =>
        inst ? 4.0 : LineDurations.GetValueOrDefault(secType, 3.0);

    private static List<(string Type, string Text, bool Inst, double Dur)> Group(
        List<(string Type, string Text, bool Inst)> raw, double target, int maxSegments)
    {
        if (maxSegments > 0 && raw.Count <= maxSegments)
            return raw.Select(r => (r.Type, r.Text, r.Inst, EstDuration(r.Type, r.Inst))).ToList();

        var groups = new List<(string, string, bool, double)>();
        var bufType = raw[0].Type;
        var bufLines = new List<string>();
        var bufDur = 0.0;
        var bufInst = raw[0].Inst;

        foreach (var (secType, line, isInst) in raw)
        {
            var lineDur = EstDuration(secType, isInst);
            if (bufLines.Count > 0 && (secType != bufType || bufDur + lineDur > target * 1.5))
            {
                groups.Add((bufType, string.Join("\n", bufLines), bufInst, bufDur));
                bufType = secType; bufLines = []; bufDur = 0.0; bufInst = isInst;
            }
            bufLines.Add(line); bufDur += lineDur;
            if (!isInst) bufInst = false;
        }
        if (bufLines.Count > 0) groups.Add((bufType, string.Join("\n", bufLines), bufInst, bufDur));

        while (maxSegments > 0 && groups.Count > maxSegments)
        {
            var minDur = double.MaxValue; var minIdx = 0;
            for (var i = 0; i < groups.Count - 1; i++)
            {
                var combined = groups[i].Item4 + groups[i + 1].Item4;
                if (combined < minDur) { minDur = combined; minIdx = i; }
            }
            var a = groups[minIdx]; var b = groups[minIdx + 1];
            groups[minIdx] = (a.Item1, a.Item2 + "\n" + b.Item2, a.Item3 && b.Item3, a.Item4 + b.Item4);
            groups.RemoveAt(minIdx + 1);
        }
        return groups;
    }

    private static List<LyricSegment> AssignTimestamps(
        List<(string Type, string Text, bool Inst, double Dur)> groups, double songDuration)
    {
        var totalEst = groups.Sum(g => g.Dur);
        if (totalEst <= 0) totalEst = groups.Count * 3.0;
        var scale = songDuration / totalEst;

        var segments = new List<LyricSegment>();
        var t = 0.0;
        for (var i = 0; i < groups.Count; i++)
        {
            var (type, text, inst, est) = groups[i];
            var dur = est * scale;
            segments.Add(new(i, type, text, Math.Round(t, 2), Math.Round(t + dur, 2), Math.Round(dur, 2), inst));
            t += dur;
        }
        if (segments.Count > 0)
        {
            var last = segments[^1];
            segments[^1] = last with { EndSec = Math.Round(songDuration, 2), Duration = Math.Round(songDuration - last.StartSec, 2) };
        }
        return segments;
    }
}
