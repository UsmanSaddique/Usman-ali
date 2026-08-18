using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using AiDirector.Application.Abstractions;
using AiDirector.Application.Archetypes;
using AiDirector.Application.Configuration;
using AiDirector.Application.Directing;
using AiDirector.Domain.Entities;
using AiDirector.Domain.Enums;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace AiDirector.Application.Pipeline;

/// Port of the "clips" video engine in app/services/pipeline.py. For each
/// pending scene it generates a still (Z-Image) then animates it (LTX i2v),
/// records a Generation, and advances scene/project status — emitting progress
/// over IProgressNotifier the whole time. External heavy work stays in ComfyUI.
/// Projects with VideoEngine == "ltx_director" instead route to the
/// multi-director engine (stills → resumable director parts → concat).
public sealed class PipelineOrchestrator(
    IProjectRepository projects,
    IComfyUiClient comfy,
    IWorkflowBuilder workflows,
    ILtxDirectorEngine ltxDirector,
    IProgressNotifier progress,
    ArchetypeResolver archetypes,
    IOptions<AiDirectorOptions> options,
    ILogger<PipelineOrchestrator> log)
{
    private readonly AiDirectorOptions _o = options.Value;

    public async Task RunAsync(string projectId, CancellationToken ct = default)
    {
        var project = await projects.GetAsync(projectId, includeGraph: true, ct);
        if (project is null) { log.LogWarning("Pipeline: project {Id} not found", projectId); return; }

        // Content Archetype gate — the single GPU choke point (parity with
        // pipeline.ensure_safety → ensure_archetype_allowed / ensure_script_review).
        var recipe = archetypes.Resolve(project, project.Channel);
        if (recipe.IsBlocked)
        {
            await Fail(project, $"Archetype blocked: {recipe.BlockReason()}");
            return;
        }
        if (recipe.ScriptReview == "required" && !project.Reviewed)
        {
            await Fail(project, $"Human review REQUIRED for archetype '{recipe.ArchetypeId}' " +
                $"(Tier {recipe.Tier}) — verify the script and approve before generating.");
            return;
        }

        await Notify(projectId, "startup", "running", 0, "Ensuring ComfyUI is up");
        if (!await comfy.WaitReadyAsync(_o.ComfyUi.ColdStartTimeoutSec, _o.ComfyUi.AutoLaunch, ct))
        {
            await Fail(project, "ComfyUI did not become ready");
            return;
        }

        project.Status = ProjectStatus.Generating;
        await projects.SaveAsync(ct);

        // The archetype's engine is authoritative when one is set (e.g.
        // ai_dreamscape forces ltx_director); otherwise the project's field.
        var engine = recipe.ArchetypeId is not null ? recipe.VideoEngine : project.VideoEngine;
        if (engine == "ltx_director")
        {
            // Routing parity with pipeline.py: ltx_director projects render as
            // one long multi-director video instead of clip-by-clip.
            try
            {
                await RunLtxDirectorAsync(project, ct);
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                await Fail(project, $"LTX Director failed: {e.Message}");
            }
            catch (Exception)
            {
                await RollbackToCheckpointAsync(project);
                throw;
            }
            return;
        }

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

    /// LTX Director engine (port of pipeline.generate_ltx_director +
    /// ltx_director.generate_for_project): ensure a reference still per scene
    /// (resumable — existing stills are reused), apply master-director
    /// guidance, then render via ILtxDirectorEngine. The engine itself is
    /// resumable at DIRECTOR granularity: finished ltx_parts/part_XX.mp4
    /// files are skipped after a crash or power outage.
    private async Task RunLtxDirectorAsync(Project project, CancellationToken ct)
    {
        var scenes = project.Scenes.OrderBy(s => s.SceneNumber).ToList();
        var imagesDir = Path.Combine(_o.Paths.ProjectsDir, project.Id, "images");
        Directory.CreateDirectory(imagesDir);

        var segments = new List<LtxSegment>();
        var guidancePlan = new List<Dictionary<string, object?>>();
        for (var idx = 0; idx < scenes.Count; idx++)
        {
            var scene = scenes[idx];
            ct.ThrowIfCancellationRequested();

            // Master-director guidance: deterministic on (project, index, section),
            // so recomputing here matches what scene planning saved — and scenes
            // created before the guidance feature get theirs assigned now.
            var section = scene.DirectorNotes.TryGetValue("section", out var sec)
                ? sec?.ToString() : null;
            var guidance = MasterDirector.GuidanceFor(idx, scenes.Count, section, project.Id);
            if (!scene.DirectorNotes.ContainsKey("director_guidance"))
            {
                scene.DirectorNotes = new Dictionary<string, object?>(scene.DirectorNotes)
                {
                    ["director_guidance"] = guidance.ToNotes(),
                };
            }
            guidancePlan.Add(guidance.ToNotes());

            var still = await EnsureSceneStillAsync(project, scene, imagesDir, ct);
            if (still is null)
            {
                log.LogWarning("[LTXDirector] scene {N}: no still available — skipped", scene.SceneNumber);
                continue;
            }

            // Song projects: the song is the soundtrack — characters must NOT
            // speak the lyric lines; keep native ambience only.
            var dialogue = "";
            if (!string.IsNullOrWhiteSpace(scene.NarrationText) && project.ProjectType != "song")
                dialogue = $"The narrator says: \"{scene.NarrationText!.Trim()}\"";

            segments.Add(new LtxSegment(
                MasterDirector.ApplyCue(scene.Prompt, guidance), dialogue,
                still, Math.Max(scene.Duration, 1.0)));
        }
        await projects.SaveAsync(ct);   // persist any newly assigned guidance

        // persist the storyboard next to the renders (director_guidance.json)
        try
        {
            var projectDir = Path.Combine(_o.Paths.ProjectsDir, project.Id);
            Directory.CreateDirectory(projectDir);
            await File.WriteAllTextAsync(Path.Combine(projectDir, "director_guidance.json"),
                JsonSerializer.Serialize(new Dictionary<string, object> { ["scenes"] = guidancePlan },
                    new JsonSerializerOptions { WriteIndented = true }), ct);
        }
        catch (Exception e)
        {
            log.LogWarning(e, "[LTXDirector] guidance save failed");
        }

        if (segments.Count < 2)
        {
            await Fail(project, "LTX Director needs at least 2 scenes with a still image each " +
                                "(generate stills first, or pin images per scene)");
            return;
        }

        await Notify(project.Id, "generate", "running", 5,
            "LTX Director: rendering the full video (finished directors are checkpointed)");
        var final = await ltxDirector.GenerateForProjectAsync(project, segments, ct);

        project.OutputPath = final;
        project.Status = ProjectStatus.Rendered;
        await projects.SaveAsync(ct);
        await Notify(project.Id, "generate", "completed", 100,
            $"LTX Director render complete (1080p): {final}");
    }

    /// Reference still for a scene, in priority order: user-pinned still →
    /// latest generation image output/thumbnail → previously rendered batch
    /// still → freshly generated Z-Image still (saved so resume reuses it).
    private async Task<string?> EnsureSceneStillAsync(
        Project project, Scene scene, string imagesDir, CancellationToken ct)
    {
        static bool IsImage(string? p) =>
            p is not null && File.Exists(p) &&
            Path.GetExtension(p).ToLowerInvariant() is ".png" or ".jpg" or ".jpeg" or ".webp";

        if (scene.DirectorNotes.TryGetValue("pinned_still", out var pin) && IsImage(pin?.ToString()))
            return pin!.ToString();
        if (scene.DirectorNotes.TryGetValue("pinned_image", out var pim) && IsImage(pim?.ToString()))
            return pim!.ToString();
        foreach (var g in scene.Generations.OrderByDescending(g => g.Version))
        {
            if (IsImage(g.OutputPath)) return g.OutputPath;
            if (IsImage(g.ThumbnailPath)) return g.ThumbnailPath;
        }

        var still = Path.Combine(imagesDir, $"scene_{scene.SceneNumber:000}_still.png");
        if (File.Exists(still)) return still;   // resume: reuse the earlier still

        // One seed per project so every still shares the same visual "world"
        // (parity with the Python pipeline's project-locked seed).
        var seed = System.Convert.ToInt64(
            System.Convert.ToHexString(MD5.HashData(Encoding.UTF8.GetBytes(project.Id)))[..7], 16);
        var prefix = $"aidir_{scene.Id[..8]}_ltxstill";
        var wf = workflows.ZImage(scene.Prompt, _o.Image.DefaultWidth, _o.Image.DefaultHeight,
            _o.Image.ZimageSteps, 1.0, seed, _o.Image.ZimageShift, prefix);
        var id = await comfy.SubmitAsync(wf, ct);
        var hist = await comfy.WaitForCompletionAsync(id, ct: ct);
        await comfy.CollectOutputAsync(hist, still, ct);
        return File.Exists(still) ? still : null;
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
