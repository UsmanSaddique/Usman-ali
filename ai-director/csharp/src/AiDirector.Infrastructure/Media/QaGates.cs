using System.Text.RegularExpressions;
using AiDirector.Application.Abstractions;

namespace AiDirector.Infrastructure.Media;

/// Port of the clip checks in app/services/qa.py: freeze-frame, black-frame,
/// and duration sanity via ffmpeg detect filters. Returns a verdict per clip.
public sealed partial class QaGates(IMediaRunner media)
{
    public sealed record ClipQa(bool Passed, double Duration, bool Frozen, bool Black, List<string> Notes);

    public async Task<ClipQa> CheckClipAsync(string path, double expectedDuration, CancellationToken ct = default)
    {
        var notes = new List<string>();
        var duration = await media.GetDurationSecAsync(path, ct);

        // freezedetect + blackdetect both write to stderr.
        var r = await media.FfmpegAsync(
        [
            "-i", path,
            "-vf", "freezedetect=n=0.003:d=1.0,blackdetect=d=0.5:pic_th=0.98",
            "-map", "0:v", "-f", "null", "-",
        ], ct);

        var frozen = FreezeRe().IsMatch(r.StdErr);
        var black = r.StdErr.Contains("black_start");
        if (frozen) notes.Add("freeze-frame detected (static clip)");
        if (black) notes.Add("black frames detected");
        if (expectedDuration > 0 && Math.Abs(duration - expectedDuration) > 1.0)
            notes.Add($"duration {duration:F2}s off from expected {expectedDuration:F2}s");

        var passed = !frozen && !black && duration > 0.1;
        return new ClipQa(passed, duration, frozen, black, notes);
    }

    [GeneratedRegex(@"freeze_start")]
    private static partial Regex FreezeRe();
}
