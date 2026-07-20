using System.Text.Json.Nodes;

namespace AiDirector.Application.Abstractions;

/// Port for the ComfyUI HTTP server (app/services/comfyui_client.py).
/// A workflow is the API-format node graph as a JsonObject.
public interface IComfyUiClient
{
    Task<bool> PingAsync(CancellationToken ct = default);

    /// Ping; if offline, auto-launch the portable server (once). True if running or booting.
    Task<bool> EnsureRunningAsync(CancellationToken ct = default);

    /// Block until ComfyUI answers, up to timeout (cold start handled).
    Task<bool> WaitReadyAsync(double timeoutSec = 60, bool autoLaunch = true, CancellationToken ct = default);

    /// POST /prompt — returns the prompt_id.
    Task<string> SubmitAsync(JsonObject workflow, CancellationToken ct = default);

    /// GET /history/{promptId} — null until present.
    Task<JsonObject?> GetHistoryAsync(string promptId, CancellationToken ct = default);

    /// Poll /history + /queue until the job completes; returns the history entry.
    Task<JsonObject> WaitForCompletionAsync(string promptId, int? timeoutSec = null, CancellationToken ct = default);

    /// Locate the produced output file in history and copy it to destPath.
    Task<string> CollectOutputAsync(JsonObject history, string destPath, CancellationToken ct = default);

    /// POST /free — unload models / free VRAM (best effort).
    Task<bool> FreeVramAsync(CancellationToken ct = default);

    /// GET /object_info — node class metadata (widget input ordering for the
    /// UI-graph → API-prompt conversion).
    Task<JsonObject> GetObjectInfoAsync(CancellationToken ct = default);
}
