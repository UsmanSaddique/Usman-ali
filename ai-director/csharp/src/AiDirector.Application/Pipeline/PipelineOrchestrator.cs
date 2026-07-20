using AiDirector.Application.Abstractions;
using AiDirector.Application.Configuration;
using AiDirector.Domain.Entities;
using AiDirector.Domain.Enums;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace AiDirector.Application.Pipeline;

/// Port of the "clips" video engine in app/services/pipeline.py. For each
/// pending scene it generates a still (Z-Image) then animates it (LTX i2v),
/// records a Generation, and advances scene/project status — emitting progress
/// over IProgressNotifier the whole time. External heavy work stays in ComfyUI.
public sealed class PipelineOrchestrator(
    IProjectRepository projects,
    IComfyUiClient comfy,
    IWorkflowBuilder workflows,
    IProgressNotifier progress,
    IOptions<AiDirectorOptions> options,
    ILogger<PipelineOrchestrator> log)
{
    private readonly AiDirectorOptions _o = options.Value;

    public async Task RunAsync(string projectId, CancellationToken ct = default)
    {
        var project = await projects.GetAsync(projectId, includeGraph: true, ct);
        if (project is null) { log.LogWarning("Pipeline: project {Id} not found", projectId); return; }

        await Notify(projectId, "startup", "running", 0, "Ensuring ComfyUI is up");
        if (!await comfy.WaitReadyAsync(_o.ComfyUi.ColdStartTimeoutSec, _o.ComfyUi.AutoLaunch, ct))
        {
            await Fail(project, "ComfyUI did not become ready");
            return;
        }

        project.Status = ProjectStatus.Generating;
        await projects.SaveAsync(ct);

        // Generating is included deliberately: the GPU queue is single-tenant and
        // this run has just started, so any scene still marked Generating is a
        // ghost left by a killed process. Excluding it stranded those scenes —
        // they were never retried and the project always ended up Failed.
        var pending = project.Scenes
            .Where(s => s.Status is SceneStatus.Pending or SceneStatus.Queued
                                 or SceneStatus.Failed or SceneStatus.Generating)
            .OrderBy(s => s.SceneNumber)
            .ToList();

        var done = 0;
        try
        {
            foreach (var scene in pending)
            {
                ct.ThrowIfCancellationRequested();
                if (await GenerateWithRetryAsync(project, scene, ct))
                {
                    done++;
                    project.CompletedScenes = project.Scenes.Count(s => s.Status == SceneStatus.Generated);
                    await projects.SaveAsync(ct);
                    await Notify(projectId, "generate", "running",
                        pending.Count == 0 ? 100 : 100.0 * done / pending.Count,
                        $"Scene {scene.SceneNumber} done", scene.Id, scene.SceneNumber);
                }
            }
        }
        catch (Exception)
        {
            // The run is dying (app shutdown, cancellation, or a non-scene error).
            // Roll back to the checkpoint so nothing is ghosted: a scene stuck in
            // Generating would otherwise show "generating" forever, and the
            // project would need manual surgery before resume worked.
            await RollbackToCheckpointAsync(project);
            throw;
        }

        project.Status = project.Scenes.All(s => s.Status == SceneStatus.Generated)
            ? ProjectStatus.Generated
            : ProjectStatus.Failed;
        await projects.SaveAsync(ct);
        await Notify(projectId, "generate", "completed", 100, $"Generated {done}/{pending.Count} scenes");
    }

    /// Attempt a scene up to its MaxRetries budget (parity with the Python
    /// pipeline). Each attempt draws a fresh random seed inside
    /// GenerateSceneAsync — the "retry seed fix": retrying a bad seed verbatim
    /// just reproduces the failure. Returns true when the scene generated.
    private async Task<bool> GenerateWithRetryAsync(Project project, Scene scene, CancellationToken ct)
    {
        var budget = Math.Max(1, scene.MaxRetries);
        while (true)
        {
            try
            {
                await GenerateSceneAsync(project, scene, ct);
                return true;
            }
            catch (OperationCanceledException) { throw; }
            catch (Exception e)
            {
                scene.RetryCount++;
                if (scene.RetryCount >= budget)
                {
                    log.LogError(e, "Scene {Num} failed permanently after {N} attempts",
                        scene.SceneNumber, scene.RetryCount);
                    scene.Status = SceneStatus.Failed;
                    await projects.SaveAsync(ct);
                    return false;
                }

                log.LogWarning(e, "Scene {Num} attempt {N}/{Budget} failed — retrying with a new seed",
                    scene.SceneNumber, scene.RetryCount, budget);
                scene.Status = SceneStatus.Pending;
                await projects.SaveAsync(ct);
                // Transient ComfyUI errors are usually VRAM pressure; give the
                // server a moment and ask it to free memory before retrying.
                await comfy.FreeVramAsync(ct);
                await Task.Delay(TimeSpan.FromSeconds(5), ct);
            }
        }
    }

    /// Roll a dying run back to its checkpoint: in-flight scenes return to
    /// Pending (their clips never completed), and the project returns to
    /// Approved so a later Resume continues cleanly. Uses CancellationToken.None
    /// deliberately — this often runs while the app is shutting down, and the
    /// rollback write must not be cancelled with it.
    private async Task RollbackToCheckpointAsync(Project project)
    {
        try
        {
            foreach (var s in project.Scenes.Where(s => s.Status == SceneStatus.Generating))
                s.Status = SceneStatus.Pending;
            project.Status = ProjectStatus.Approved;
            await projects.SaveAsync(CancellationToken.None);
            log.LogInformation("Rolled {Id} back to checkpoint (approved) after interrupted run", project.Id);
        }
        catch (Exception e)
        {
            log.LogError(e, "Checkpoint rollback failed for {Id}", project.Id);
        }
    }

    private async Task GenerateSceneAsync(Project project, Scene scene, CancellationToken ct)
    {
        scene.Status = SceneStatus.Generating;
        await projects.SaveAsync(ct);
        await Notify(project.Id, "generate", "running", null,
            $"Generating scene {scene.SceneNumber}", scene.Id, scene.SceneNumber);

        var seed = Random.Shared.NextInt64(1, int.MaxValue);
        var (w, h) = ResolveResolution(project, scene);
        var loras = ZipLoras(scene.LoraIds, scene.LoraWeights);
        var started = DateTime.UtcNow;

        // 1. Still (Z-Image) -> placed into ComfyUI/input for the i2v stage.
        var stillPrefix = $"aidir_{scene.Id[..8]}_still";
        var stillWf = workflows.ZImage(scene.Prompt, w, h,
            _o.Image.ZimageSteps, 1.0, seed, _o.Image.ZimageShift, stillPrefix);
        var stillId = await comfy.SubmitAsync(stillWf, ct);
        var stillHist = await comfy.WaitForCompletionAsync(stillId, ct: ct);
        var stillDest = Path.Combine(_o.Paths.ComfyInput, $"{stillPrefix}.png");
        await comfy.CollectOutputAsync(stillHist, stillDest, ct);

        // 2. Image-to-video (LTX) using the still.
        var clipPrefix = $"aidir_{scene.Id[..8]}_clip";
        var frames = FramesFor(scene.Duration, _o.Video.DefaultFps, _o.Video.MaxNumFrames);
        var clipWf = workflows.LtxImage2Video(project.VideoModel, Path.GetFileName(stillDest),
            scene.Prompt, scene.NegativePrompt, w, h, frames,
            _o.Video.DefaultSteps, _o.Video.DefaultCfg, seed, _o.Video.DefaultFps,
            _o.Video.Img2VidStrength, loras, clipPrefix);
        var clipId = await comfy.SubmitAsync(clipWf, ct);
        var clipHist = await comfy.WaitForCompletionAsync(clipId, ct: ct);

        var outDir = Path.Combine(_o.Paths.AssetsDir, project.Id);
        var clipDest = Path.Combine(outDir, $"{clipPrefix}.mp4");
        await comfy.CollectOutputAsync(clipHist, clipDest, ct);

        // 3. Record the generation and advance the scene.
        var version = (scene.Generations.Count == 0 ? 0 : scene.Generations.Max(g => g.Version)) + 1;
        var gen = new Generation
        {
            SceneId = scene.Id,
            Version = version,
            ModelUsed = project.VideoModel,
            OutputPath = clipDest,
            PromptUsed = scene.Prompt,
            NegativePromptUsed = scene.NegativePrompt,
            Seed = seed,
            Status = GenerationStatus.Completed,
            GenerationTimeSec = (DateTime.UtcNow - started).TotalSeconds,
            Parameters = new()
            {
                ["width"] = w, ["height"] = h, ["num_frames"] = frames,
                ["fps"] = _o.Video.DefaultFps, ["steps"] = _o.Video.DefaultSteps,
            },
        };
        scene.Generations.Add(gen);
        scene.ActiveGenerationId = gen.Id;
        scene.Status = SceneStatus.Generated;
        await projects.SaveAsync(ct);
    }

    private (int W, int H) ResolveResolution(Project project, Scene scene)
    {
        // Premium opening: scenes starting inside the hook window render larger.
        var startsInHook = scene.SceneNumber <= 2;
        return startsInHook
            ? (_o.Video.PremiumWidth, _o.Video.PremiumHeight)
            : (_o.Video.DefaultWidth, _o.Video.DefaultHeight);
    }

    // LTX requires frame counts of the form 8n+1. Cap by the VRAM-safe ceiling.
    private static int FramesFor(double durationSec, int fps, int maxFrames)
    {
        var raw = (int)Math.Round(durationSec * fps);
        var snapped = ((raw - 1) / 8) * 8 + 1;
        return Math.Clamp(snapped, 9, maxFrames);
    }

    private static IReadOnlyList<(string, double)> ZipLoras(List<string> ids, List<double> weights)
    {
        var list = new List<(string, double)>();
        for (var i = 0; i < ids.Count; i++)
            list.Add((ids[i], i < weights.Count ? weights[i] : 0.7));
        return list;
    }

    private async Task Fail(Project project, string message)
    {
        log.LogError("Pipeline failed for {Id}: {Msg}", project.Id, message);
        project.Status = ProjectStatus.Failed;
        project.ErrorLog = message;
        await projects.SaveAsync();
        await Notify(project.Id, "startup", "failed", null, message);
    }

    private Task Notify(string projectId, string stage, string status,
        double? pct = null, string? msg = null, string? sceneId = null, int? sceneNumber = null) =>
        progress.PublishAsync(projectId,
            new ProgressUpdate(stage, status, pct, msg, sceneId, sceneNumber));
}
