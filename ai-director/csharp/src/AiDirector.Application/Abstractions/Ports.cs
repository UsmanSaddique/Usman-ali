using System.Text.Json.Nodes;
using AiDirector.Domain.Entities;

namespace AiDirector.Application.Abstractions;

/// Builds ComfyUI API-format workflows (port of the build_* functions in
/// app/services/comfyui_client.py). Pure graph construction — no I/O.
public interface IWorkflowBuilder
{
    JsonObject ZImage(string prompt, int width, int height, int steps, double cfg,
        long seed, double shift, string outputPrefix = "zimage");

    JsonObject LtxText2Video(string model, string prompt, string negativePrompt,
        int width, int height, int numFrames, int steps, double cfg, long seed, int fps,
        IReadOnlyList<(string File, double Weight)>? loras = null, string outputPrefix = "ai_director");

    JsonObject LtxImage2Video(string model, string imageFilename, string prompt, string negativePrompt,
        int width, int height, int numFrames, int steps, double cfg, long seed, int fps, double strength,
        IReadOnlyList<(string File, double Weight)>? loras = null, string outputPrefix = "ai_director_i2v");

    JsonObject EsrganVideoUpscale(string videoFilename, int targetWidth, int targetHeight,
        double fps, string modelName, string outputPrefix = "ai_director_hd");

    JsonObject AceStep15Xl(string styleTags, string lyrics, double seconds, long seed,
        int steps, double cfg, int bpm, string language, string keyscale,
        string unetName = "acestep_v1.5_xl_turbo_bf16.safetensors",
        string weightDtype = "default",
        string outputPrefix = "ai_director_music");
}

/// Runs ffmpeg/ffprobe as child processes (app/services/assembler.py, qa.py).
public interface IMediaRunner
{
    Task<ProcessResult> FfmpegAsync(IReadOnlyList<string> args, CancellationToken ct = default);
    Task<ProcessResult> FfprobeAsync(IReadOnlyList<string> args, CancellationToken ct = default);
    Task<double> GetDurationSecAsync(string mediaPath, CancellationToken ct = default);
}

public sealed record ProcessResult(int ExitCode, string StdOut, string StdErr)
{
    public bool Ok => ExitCode == 0;
}

/// Pushes pipeline progress to connected clients (the /ws/pipeline WebSocket).
public interface IProgressNotifier
{
    Task PublishAsync(string projectId, ProgressUpdate update, CancellationToken ct = default);
}

public sealed record ProgressUpdate(
    string Stage, string Status, double? Percent = null, string? Message = null,
    string? SceneId = null, int? SceneNumber = null);

/// Data access for the pipeline and endpoints (implemented over EF Core).
public interface IProjectRepository
{
    Task<Project?> GetAsync(string id, bool includeGraph = false, CancellationToken ct = default);
    Task<List<Project>> ListAsync(CancellationToken ct = default);
    Task AddAsync(Project project, CancellationToken ct = default);
    Task SaveAsync(CancellationToken ct = default);
}
