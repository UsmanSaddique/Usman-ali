using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using AiDirector.Application.Abstractions;
using AiDirector.Application.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace AiDirector.Infrastructure.ComfyUi;

/// HttpClient port of app/services/comfyui_client.py (ComfyUIClient).
public sealed class ComfyUiClient(
    HttpClient http,
    IOptions<AiDirectorOptions> options,
    ILogger<ComfyUiClient> log) : IComfyUiClient
{
    private readonly AiDirectorOptions _o = options.Value;
    private static readonly SemaphoreSlim LaunchLock = new(1, 1);
    private static DateTime _launchStartedAt = DateTime.MinValue;

    private string Base => _o.ComfyUi.BaseUrl.TrimEnd('/');

    public async Task<bool> PingAsync(CancellationToken ct = default)
    {
        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(TimeSpan.FromSeconds(3));
            var resp = await http.GetAsync($"{Base}/system_stats", cts.Token);
            return resp.IsSuccessStatusCode;
        }
        catch { return false; }
    }

    public async Task<bool> EnsureRunningAsync(CancellationToken ct = default)
    {
        if (await PingAsync(ct)) return true;
        await LaunchLock.WaitAsync(ct);
        try
        {
            if (await PingAsync(ct) || IsStarting()) return true;
            if (SpawnComfyui()) { _launchStartedAt = DateTime.UtcNow; return true; }
        }
        finally { LaunchLock.Release(); }
        return false;
    }

    public async Task<bool> WaitReadyAsync(double timeoutSec = 60, bool autoLaunch = true, CancellationToken ct = default)
    {
        if (await PingAsync(ct)) return true;
        if (autoLaunch && await EnsureRunningAsync(ct))
            timeoutSec = Math.Max(timeoutSec, _o.ComfyUi.ColdStartTimeoutSec);
        var t0 = DateTime.UtcNow;
        while ((DateTime.UtcNow - t0).TotalSeconds < timeoutSec)
        {
            if (await PingAsync(ct)) return true;
            await Task.Delay(TimeSpan.FromSeconds(_o.ComfyUi.PollSec), ct);
        }
        return false;
    }

    public async Task<string> SubmitAsync(JsonObject workflow, CancellationToken ct = default)
    {
        var clientId = Guid.NewGuid().ToString("N")[..12];
        var payload = new JsonObject { ["prompt"] = workflow, ["client_id"] = clientId };
        using var content = new StringContent(payload.ToJsonString(), Encoding.UTF8, "application/json");
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(TimeSpan.FromSeconds(_o.ComfyUi.SubmitTimeoutSec));

        var resp = await http.PostAsync($"{Base}/prompt", content, cts.Token);
        var body = await resp.Content.ReadAsStringAsync(cts.Token);
        if (!resp.IsSuccessStatusCode)
            throw new InvalidOperationException($"ComfyUI HTTP {(int)resp.StatusCode}: {body}");

        var result = JsonNode.Parse(body)?.AsObject();
        var promptId = result?["prompt_id"]?.GetValue<string>();
        if (string.IsNullOrEmpty(promptId))
        {
            var error = result?["error"] ?? result?["node_errors"] ?? result;
            throw new InvalidOperationException($"ComfyUI rejected workflow: {error}");
        }
        log.LogInformation("[ComfyUI] Submitted prompt {PromptId}", promptId);
        return promptId;
    }

    public async Task<JsonObject?> GetHistoryAsync(string promptId, CancellationToken ct = default)
    {
        try
        {
            var resp = await http.GetAsync($"{Base}/history/{promptId}", ct);
            if (!resp.IsSuccessStatusCode) return null;
            var data = JsonNode.Parse(await resp.Content.ReadAsStringAsync(ct))?.AsObject();
            return data?[promptId]?.AsObject();
        }
        catch { return null; }
    }

    public async Task<JsonObject> WaitForCompletionAsync(string promptId, int? timeoutSec = null, CancellationToken ct = default)
    {
        var timeout = timeoutSec ?? _o.ComfyUi.CompletionTimeoutSec;
        var t0 = DateTime.UtcNow;
        var lastLog = DateTime.MinValue;
        var missing = 0;

        while ((DateTime.UtcNow - t0).TotalSeconds < timeout)
        {
            var hist = await GetHistoryAsync(promptId, ct);
            if (hist is not null)
            {
                var status = hist["status"]?.AsObject();
                if (status?["completed"]?.GetValue<bool>() == true) return hist;
                if (status?["status_str"]?.GetValue<string>() == "error")
                    throw new InvalidOperationException($"ComfyUI error: {status["messages"]}");
            }

            if ((DateTime.UtcNow - lastLog).TotalSeconds > 10.0)
            {
                var queue = await GetQueueAsync(ct);
                if (queue is not null)
                {
                    var running = HasId(queue["queue_running"], promptId);
                    var pending = HasId(queue["queue_pending"], promptId);
                    if (running || pending) { missing = 0; log.LogInformation("[ComfyUI] Generating..."); }
                    else if (++missing >= 3)
                        throw new InvalidOperationException(
                            $"ComfyUI job disappeared (queue cleared or restarted): {promptId}");
                }
                lastLog = DateTime.UtcNow;
            }
            await Task.Delay(TimeSpan.FromSeconds(_o.ComfyUi.PollSec), ct);
        }
        throw new TimeoutException($"ComfyUI timed out after {timeout}s");
    }

    public Task<string> CollectOutputAsync(JsonObject history, string destPath, CancellationToken ct = default)
    {
        var outputs = history["outputs"]?.AsObject();
        if (outputs is not null)
        {
            foreach (var (_, nodeOut) in outputs)
                foreach (var key in new[] { "gifs", "images", "videos", "audio" })
                {
                    if (nodeOut?[key] is not JsonArray arr) continue;
                    foreach (var entry in arr)
                    {
                        var fname = entry?["filename"]?.GetValue<string>();
                        if (string.IsNullOrEmpty(fname)) continue;
                        var subfolder = entry?["subfolder"]?.GetValue<string>() ?? "";
                        var src = string.IsNullOrEmpty(subfolder)
                            ? Path.Combine(_o.Paths.ComfyOutput, fname)
                            : Path.Combine(_o.Paths.ComfyOutput, subfolder, fname);
                        if (File.Exists(src))
                        {
                            Directory.CreateDirectory(Path.GetDirectoryName(destPath)!);
                            File.Copy(src, destPath, overwrite: true);
                            log.LogInformation("[ComfyUI] Copied output {Src} -> {Dest}", src, destPath);
                            return Task.FromResult(destPath);
                        }
                    }
                }
        }
        throw new FileNotFoundException($"No output file found in ComfyUI history for {destPath}");
    }

    public async Task<bool> FreeVramAsync(CancellationToken ct = default)
    {
        try
        {
            var payload = new JsonObject { ["unload_models"] = true, ["free_memory"] = true };
            using var content = new StringContent(payload.ToJsonString(), Encoding.UTF8, "application/json");
            var resp = await http.PostAsync($"{Base}/free", content, ct);
            return resp.IsSuccessStatusCode;
        }
        catch { return false; }
    }

    public async Task<JsonObject> GetObjectInfoAsync(CancellationToken ct = default)
    {
        var resp = await http.GetAsync($"{Base}/object_info", ct);
        resp.EnsureSuccessStatusCode();
        var data = JsonNode.Parse(await resp.Content.ReadAsStringAsync(ct))?.AsObject();
        return data ?? throw new InvalidOperationException("ComfyUI /object_info returned no data");
    }

    private async Task<JsonObject?> GetQueueAsync(CancellationToken ct)
    {
        try
        {
            var resp = await http.GetAsync($"{Base}/queue", ct);
            return resp.IsSuccessStatusCode
                ? JsonNode.Parse(await resp.Content.ReadAsStringAsync(ct))?.AsObject()
                : null;
        }
        catch { return null; }
    }

    // queue entries are [number, prompt_id, ...] arrays.
    private static bool HasId(JsonNode? queueList, string promptId)
    {
        if (queueList is not JsonArray arr) return false;
        foreach (var item in arr)
            if (item is JsonArray entry && entry.Count > 1 &&
                entry[1]?.GetValue<string>() == promptId) return true;
        return false;
    }

    private bool IsStarting() =>
        _launchStartedAt != DateTime.MinValue &&
        (DateTime.UtcNow - _launchStartedAt).TotalSeconds < _o.ComfyUi.ColdStartTimeoutSec;

    private bool SpawnComfyui()
    {
        var python = _o.Paths.ComfyPython;
        var mainPy = _o.Paths.ComfyMain;
        if (!File.Exists(python) || !File.Exists(mainPy))
        {
            log.LogWarning("[ComfyUI] Cannot auto-launch — portable install not found at {Root}", _o.Paths.ComfyRoot);
            return false;
        }
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = python,
                WorkingDirectory = _o.Paths.ComfyRoot,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            foreach (var a in new[] { "-s", mainPy, "--windows-standalone-build", "--enable-manager" })
                psi.ArgumentList.Add(a);
            Process.Start(psi);
            log.LogInformation("[ComfyUI] Auto-launched");
            return true;
        }
        catch (Exception e)
        {
            log.LogError(e, "[ComfyUI] Auto-launch failed");
            return false;
        }
    }
}
