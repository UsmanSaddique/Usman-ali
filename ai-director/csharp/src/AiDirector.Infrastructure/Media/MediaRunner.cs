using System.Diagnostics;
using System.Globalization;
using AiDirector.Application.Abstractions;
using AiDirector.Application.Configuration;
using Microsoft.Extensions.Options;

namespace AiDirector.Infrastructure.Media;

/// Runs ffmpeg/ffprobe as child processes. Args are passed via ArgumentList
/// (never a shell string) so Windows quoting can't corrupt them.
public sealed class MediaRunner(IOptions<AiDirectorOptions> options) : IMediaRunner
{
    private readonly AiDirectorOptions _o = options.Value;

    public Task<ProcessResult> FfmpegAsync(IReadOnlyList<string> args, CancellationToken ct = default) =>
        RunAsync(_o.Paths.FfmpegBin, args, ct);

    public Task<ProcessResult> FfprobeAsync(IReadOnlyList<string> args, CancellationToken ct = default) =>
        RunAsync(_o.Paths.FfprobeBin, args, ct);

    public async Task<double> GetDurationSecAsync(string mediaPath, CancellationToken ct = default)
    {
        var r = await FfprobeAsync(
        [
            "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", mediaPath,
        ], ct);
        return double.TryParse(r.StdOut.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var d)
            ? d : 0.0;
    }

    private static async Task<ProcessResult> RunAsync(string exe, IReadOnlyList<string> args, CancellationToken ct)
    {
        var psi = new ProcessStartInfo
        {
            FileName = exe,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (var a in args) psi.ArgumentList.Add(a);

        using var proc = new Process { StartInfo = psi };
        proc.Start();
        var stdout = proc.StandardOutput.ReadToEndAsync(ct);
        var stderr = proc.StandardError.ReadToEndAsync(ct);
        await proc.WaitForExitAsync(ct);
        return new ProcessResult(proc.ExitCode, await stdout, await stderr);
    }
}
