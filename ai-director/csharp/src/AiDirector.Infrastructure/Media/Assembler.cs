using AiDirector.Application.Abstractions;
using AiDirector.Application.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace AiDirector.Infrastructure.Media;

/// Port of app/services/assembler.py. Concatenates clips with xfade crossfades,
/// mixes narration + music, muxes the final mp4. ffmpeg args go through the
/// MediaRunner (ArgumentList), and the filter graph is written to a script file
/// (Windows caps the command line at ~32k chars — inline graphs overflow with
/// 100+ clips, WinError 206).
public sealed class Assembler(IMediaRunner media, IOptions<AiDirectorOptions> options, ILogger<Assembler> log)
{
    private readonly AiDirectorOptions _o = options.Value;

    public sealed record Clip(string Path, double Duration);
    public sealed record Result(string OutputPath, double TotalDuration, string Resolution, double FileSizeMb, double RenderSec);

    private static readonly Dictionary<string, (int W, int H)> ResMap = new()
    {
        ["1080p"] = (1920, 1080), ["2k"] = (2560, 1440), ["1440p"] = (2560, 1440),
        ["4k"] = (3840, 2160), ["2160p"] = (3840, 2160), ["720p"] = (1280, 720),
    };

    public async Task<Result> AssembleAsync(IReadOnlyList<Clip> clips, string outputPath,
        string? narrationPath = null, string? musicPath = null,
        double musicVolume = 0.3, double narrationVolume = 1.0,
        double transitionDuration = 0.5, string resolution = "1080p", int fps = 24,
        CancellationToken ct = default)
    {
        if (clips.Count == 0) throw new ArgumentException("No clips to assemble");
        var t0 = DateTime.UtcNow;
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        var (w, h) = ResMap.GetValueOrDefault(resolution, (1920, 1080));

        var tempDir = Path.Combine(Path.GetTempPath(), "aidir_concat_" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(tempDir);
        var concat = await ConcatClipsAsync(clips, w, h, fps, transitionDuration, tempDir, ct);

        string? audio = null;
        if (!string.IsNullOrEmpty(narrationPath) || !string.IsNullOrEmpty(musicPath))
        {
            var videoDur = await media.GetDurationSecAsync(concat, ct);
            audio = await MixAudioAsync(narrationPath, musicPath, narrationVolume, musicVolume, videoDur, tempDir, ct);
        }

        await MuxFinalAsync(concat, audio, outputPath, fps, ct);
        TryDelete(tempDir);

        var duration = await media.GetDurationSecAsync(outputPath, ct);
        var sizeMb = new FileInfo(outputPath).Length / (1024.0 * 1024.0);
        log.LogInformation("[Assembler] Done: {Dur:F1}s, {Size:F1}MB", duration, sizeMb);
        return new Result(outputPath, duration, resolution, sizeMb, (DateTime.UtcNow - t0).TotalSeconds);
    }

    private async Task<string> ConcatClipsAsync(IReadOnlyList<Clip> clips, int w, int h, int fps,
        double td, string tempDir, CancellationToken ct)
    {
        var output = Path.Combine(tempDir, "concat.mp4");
        var pad = $"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2";

        if (clips.Count == 1)
        {
            var r = await media.FfmpegAsync(
            [
                "-y", "-i", clips[0].Path, "-vf", pad,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-r", fps.ToString(), "-pix_fmt", "yuv420p", "-an", output,
            ], ct);
            if (!r.Ok) throw new InvalidOperationException($"ffmpeg concat(1) failed: {Tail(r.StdErr)}");
            return output;
        }

        // Real per-clip durations drive xfade offsets (a short clip freezes its
        // last frame until the fade instead of running past end-of-stream).
        var durations = new List<double>();
        foreach (var c in clips)
        {
            var d = await media.GetDurationSecAsync(c.Path, ct);
            durations.Add(d > 0 ? d : c.Duration);
        }

        var filters = new List<string>();
        for (var i = 0; i < clips.Count; i++)
            filters.Add($"[{i}:v]{pad},setsar=1,fps={fps}[v{i}]");

        if (clips.Count == 2)
        {
            var off = Math.Max(0, durations[0] - td);
            filters.Add($"[v0][v1]xfade=transition=fade:duration={td}:offset={off}[vout]");
        }
        else
        {
            var off = Math.Max(0, durations[0] - td);
            filters.Add($"[v0][v1]xfade=transition=fade:duration={td}:offset={off}[x0]");
            var running = off + durations[1] - td;
            for (var i = 2; i < clips.Count; i++)
            {
                var prev = $"x{i - 2}";
                var outLabel = i < clips.Count - 1 ? $"x{i - 1}" : "vout";
                off = Math.Max(0, running);
                filters.Add($"[{prev}][v{i}]xfade=transition=fade:duration={td}:offset={off}[{outLabel}]");
                running = off + durations[i] - td;
            }
        }

        var script = Path.Combine(tempDir, "filter_complex.txt");
        await File.WriteAllTextAsync(script, string.Join(";\n", filters), ct);

        var args = new List<string> { "-y" };
        foreach (var c in clips) { args.Add("-i"); args.Add(c.Path); }
        args.AddRange(["-filter_complex_script", script, "-map", "[vout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-an", output]);

        var res = await media.FfmpegAsync(args, ct);
        if (!res.Ok)
        {
            log.LogWarning("[Assembler] xfade failed, simple concat fallback: {Err}", Tail(res.StdErr));
            return await SimpleConcatAsync(clips, w, h, fps, pad, tempDir, ct);
        }
        return output;
    }

    private async Task<string> SimpleConcatAsync(IReadOnlyList<Clip> clips, int w, int h, int fps,
        string pad, string tempDir, CancellationToken ct)
    {
        var listFile = Path.Combine(tempDir, "list.txt");
        var lines = new List<string>();
        for (var i = 0; i < clips.Count; i++)
        {
            var norm = Path.Combine(tempDir, $"norm_{i:D4}.mp4");
            var r = await media.FfmpegAsync(
            [
                "-y", "-i", clips[i].Path, "-vf", pad,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-r", fps.ToString(), "-pix_fmt", "yuv420p", "-an", norm,
            ], ct);
            if (r.Ok) lines.Add($"file '{norm.Replace("\\", "/")}'");
        }
        await File.WriteAllTextAsync(listFile, string.Join("\n", lines), ct);
        var output = Path.Combine(tempDir, "concat.mp4");
        await media.FfmpegAsync(["-y", "-f", "concat", "-safe", "0", "-i", listFile, "-c", "copy", output], ct);
        return output;
    }

    private async Task<string> MixAudioAsync(string? narration, string? music,
        double narrVol, double musVol, double videoDur, string tempDir, CancellationToken ct)
    {
        var output = Path.Combine(tempDir, "mixed.wav");
        var args = new List<string> { "-y" };
        var filters = new List<string>();
        var srcs = new List<string>();
        var n = 0;

        if (!string.IsNullOrEmpty(narration) && File.Exists(narration))
        {
            args.Add("-i"); args.Add(narration);
            filters.Add($"[{n}:a]aresample=48000,volume={narrVol}[narr]"); srcs.Add("[narr]"); n++;
        }
        if (!string.IsNullOrEmpty(music) && File.Exists(music))
        {
            args.Add("-i"); args.Add(music);
            filters.Add($"[{n}:a]aresample=48000,volume={musVol}[mus]"); srcs.Add("[mus]"); n++;
        }
        if (srcs.Count == 0) return output;

        filters.Add($"{string.Join("", srcs)}amix=inputs={srcs.Count}:normalize=0:duration=first,loudnorm=I=-14:TP=-1.5:LRA=11[out]");
        args.AddRange(["-filter_complex", string.Join(";", filters), "-map", "[out]",
            "-t", videoDur.ToString("F3"), "-c:a", "pcm_s16le", "-ar", "48000", output]);
        var r = await media.FfmpegAsync(args, ct);
        if (!r.Ok) throw new InvalidOperationException($"ffmpeg mix failed: {Tail(r.StdErr)}");
        return output;
    }

    private async Task MuxFinalAsync(string video, string? audio, string output, int fps, CancellationToken ct)
    {
        var args = new List<string> { "-y", "-i", video };
        if (audio is not null) { args.Add("-i"); args.Add(audio); }
        args.AddRange(["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", fps.ToString()]);
        if (audio is not null) args.AddRange(["-c:a", "aac", "-b:a", "192k", "-map", "0:v", "-map", "1:a", "-shortest"]);
        args.Add(output);
        var r = await media.FfmpegAsync(args, ct);
        if (!r.Ok) throw new InvalidOperationException($"ffmpeg mux failed: {Tail(r.StdErr)}");
    }

    private static string Tail(string s) => s.Length <= 400 ? s : s[^400..];
    private static void TryDelete(string dir) { try { Directory.Delete(dir, true); } catch { /* best effort */ } }
}
