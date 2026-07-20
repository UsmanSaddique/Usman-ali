using AiDirector.Application.Directing;
using AiDirector.Application.Safety;
using AiDirector.Domain.Entities;
using AiDirector.Domain.Enums;
using AiDirector.Infrastructure.Audio;
using AiDirector.Infrastructure.Media;
using AiDirector.Infrastructure.Persistence;
using AiDirector.WebApi.Contracts;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace AiDirector.WebApi.Endpoints;

/// Safety, scene planning, music, and render — ports the matching main.py routes.
public static class ProductionEndpoints
{
    public static void MapProductionEndpoints(this IEndpointRouteBuilder app)
    {
        var g = app.MapGroup("/api/projects");

        // Safety gate (rules layer): scan, persist SafetyReport, return verdict.
        g.MapPost("/{id}/safety-check",
            async (string id, AiDirectorDbContext db, YtSafetyGate gate, CancellationToken ct) =>
        {
            var project = await db.Projects.Include(p => p.Channel)
                .FirstOrDefaultAsync(p => p.Id == id, ct);
            if (project is null) return Results.NotFound();
            var scenes = await db.Scenes.Where(s => s.ProjectId == id)
                .OrderBy(s => s.SceneNumber).ToListAsync(ct);

            var result = gate.Run(project, scenes, project.Channel?.MadeForKids ?? false);
            var report = new SafetyReport
            {
                ProjectId = id,
                Verdict = result.Verdict,
                Issues = result.Issues,
                CheckedFields = result.CheckedFields,
                LlmUsed = false,
            };
            db.SafetyReports.Add(report);
            await db.SaveChangesAsync(ct);
            return Results.Ok(new
            {
                verdict = result.Verdict.ToString().ToLowerInvariant(),
                issues = result.Issues,
                checked_fields = result.CheckedFields,
            });
        });

        g.MapGet("/{id}/safety-report", async (string id, AiDirectorDbContext db, CancellationToken ct) =>
        {
            var latest = await db.SafetyReports.Where(r => r.ProjectId == id)
                .OrderByDescending(r => r.CreatedAt).FirstOrDefaultAsync(ct);
            return latest is null
                ? Results.NotFound()
                : Results.Ok(new { verdict = latest.Verdict.ToString().ToLowerInvariant(), issues = latest.Issues });
        });

        // Build scenes from the project's lyrics.
        g.MapPost("/{id}/scenes-from-lyrics",
            async (string id, AiDirectorDbContext db, CancellationToken ct) =>
        {
            var project = await db.Projects.Include(p => p.Channel)
                .FirstOrDefaultAsync(p => p.Id == id, ct);
            if (project is null) return Results.NotFound();
            if (string.IsNullOrWhiteSpace(project.Lyrics))
                return Results.BadRequest(new { error = "project has no lyrics" });

            var styleCue = StyleCueFor(project.Channel);
            var scenes = ScenePlanner.FromLyrics(project, styleCue, project.DurationTarget);

            await ClearScenesAsync(db, id, ct);
            db.Scenes.AddRange(scenes);
            project.TotalScenes = scenes.Count;
            project.Status = ProjectStatus.Scripted;
            await db.SaveChangesAsync(ct);
            return Results.Ok(new { count = scenes.Count, scenes = scenes.Select(SceneDto.From) });
        });

        // Generate the backing song (ACE-Step via ComfyUI).
        g.MapPost("/{id}/generate-music",
            async (string id, AiDirectorDbContext db, MusicEngine music, CancellationToken ct) =>
        {
            var project = await db.Projects.FirstOrDefaultAsync(p => p.Id == id, ct);
            if (project is null) return Results.NotFound();
            var style = project.MusicStyle ?? "gentle acoustic children's song, warm, simple melody";
            var path = await music.GenerateAsync(id, style, project.Lyrics ?? "",
                project.DurationTarget, ct: ct);

            foreach (var t in db.MusicTracks.Where(t => t.ProjectId == id)) t.IsActive = false;
            var track = new MusicTrack
            {
                ProjectId = id, StylePrompt = style, OutputPath = path,
                Duration = project.DurationTarget, IsActive = true,
            };
            db.MusicTracks.Add(track);
            await db.SaveChangesAsync(ct);
            return Results.Ok(new { track_id = track.Id, output_path = path });
        });

        // Generate N song variants in the background (frontend polls music_variants).
        g.MapPost("/{id}/generate-music-variants",
            async (string id, MusicVariantsRequest req, AiDirectorDbContext db,
                   IServiceScopeFactory scopes, ILoggerFactory logs, CancellationToken ct) =>
        {
            var logger = logs.CreateLogger("MusicVariants");
            var project = await db.Projects.FirstOrDefaultAsync(p => p.Id == id, ct);
            if (project is null) return Results.NotFound();
            var baseStyle = req.Style ?? project.MusicStyle ?? "gentle acoustic children's song, warm, simple melody";
            var baseLyrics = (req.Vocals ?? true) ? (req.Lyrics ?? project.Lyrics ?? "") : "";
            var seconds = project.DurationTarget;

            // Explicit variant specs win; otherwise fan Count out over the base
            // style. Each take gets its own seed — appending "variation N" to the
            // style barely changes the output, a different seed genuinely does.
            var specs = req.Variants is { Count: > 0 }
                ? req.Variants.Take(8).ToList()
                : Enumerable.Range(0, Math.Clamp(req.Count ?? 1, 1, 5))
                    .Select(i => new MusicVariantSpec($"v{i + 1}", null, null, null, null))
                    .ToList();
            var count = specs.Count;

            // Detached so the HTTP call returns immediately; each variant gets a
            // fresh DI scope (its own DbContext + MusicEngine).
            _ = Task.Run(async () =>
            {
                for (var i = 0; i < specs.Count; i++)
                {
                    var spec = specs[i];
                    try
                    {
                        using var scope = scopes.CreateScope();
                        var music = scope.ServiceProvider.GetRequiredService<MusicEngine>();
                        var sdb = scope.ServiceProvider.GetRequiredService<AiDirectorDbContext>();

                        var style = spec.Style ?? baseStyle;
                        var takeLyrics = spec.Lyrics ?? baseLyrics;
                        var label = string.IsNullOrWhiteSpace(spec.Label) ? $"v{i + 1}" : spec.Label!;
                        var seed = spec.Seed ?? Random.Shared.NextInt64(1, int.MaxValue);

                        var path = await music.GenerateAsync(id, style, takeLyrics, seconds,
                            language: spec.Language ?? "en", seed: seed,
                            variantLabel: $"{label}_{DateTime.UtcNow.Ticks}");
                        sdb.MusicTracks.Add(new MusicTrack
                        {
                            ProjectId = id, StylePrompt = style, OutputPath = path,
                            Duration = seconds, IsActive = false,
                        });
                        await sdb.SaveChangesAsync();
                    }
                    catch (Exception ex)
                    {
                        // One variant failing shouldn't kill the batch, but a silent
                        // catch made a whole empty run look like success.
                        logger.LogError(ex, "[Music] Variant {Label} failed for {Project}",
                            spec.Label ?? $"v{i + 1}", id);
                    }
                }
            });
            return Results.Accepted($"/api/projects/{id}", new { count });
        });

        // Pick a variant as the active soundtrack.
        g.MapPost("/{id}/select-music/{trackId}",
            async (string id, string trackId, AiDirectorDbContext db, CancellationToken ct) =>
        {
            var tracks = await db.MusicTracks.Where(t => t.ProjectId == id).ToListAsync(ct);
            if (tracks.All(t => t.Id != trackId)) return Results.NotFound();
            foreach (var t in tracks) t.IsActive = t.Id == trackId;
            await db.SaveChangesAsync(ct);
            return Results.Ok(new { selected = trackId });
        });

        g.MapPost("/{id}/pause", async (string id, AiDirectorDbContext db, CancellationToken ct) =>
        {
            var p = await db.Projects.FirstOrDefaultAsync(x => x.Id == id, ct);
            if (p is null) return Results.NotFound();
            return Results.Ok(new { status = "pause requested" });
        });

        g.MapPost("/{id}/cancel", async (string id, AiDirectorDbContext db, CancellationToken ct) =>
        {
            var p = await db.Projects.FirstOrDefaultAsync(x => x.Id == id, ct);
            if (p is null) return Results.NotFound();
            p.Status = ProjectStatus.Failed;
            await db.SaveChangesAsync(ct);
            return Results.Ok(new { status = "cancelled" });
        });

        // Assemble the final video from generated clips + music.
        g.MapPost("/{id}/render",
            async (string id, RenderRequest? req, AiDirectorDbContext db, Assembler assembler,
                   Microsoft.Extensions.Options.IOptions<AiDirector.Application.Configuration.AiDirectorOptions> opts,
                   CancellationToken ct) =>
        {
            var project = await db.Projects
                .Include(p => p.Scenes).ThenInclude(s => s.Generations)
                .Include(p => p.MusicTracks)
                .FirstOrDefaultAsync(p => p.Id == id, ct);
            if (project is null) return Results.NotFound();

            // Prefer the ESRGAN-upscaled clip when one exists; scene.Duration is
            // the real clip length (5.04s) — the old hardcoded 5.0 drifted the
            // crossfade offsets by ~2s over a 48-scene video.
            var clips = project.Scenes.OrderBy(s => s.SceneNumber)
                .Select(s =>
                {
                    var gen = s.ActiveGeneration ?? s.Generations.LastOrDefault();
                    var path = (gen?.UpscaledPath is { } up && File.Exists(up)) ? up : gen?.OutputPath;
                    return (Path: path, s.Duration);
                })
                .Where(c => !string.IsNullOrEmpty(c.Path) && File.Exists(c.Path))
                .Select(c => new Assembler.Clip(c.Path!, c.Duration))
                .ToList();
            if (clips.Count == 0)
                return Results.BadRequest(new { error = "no rendered clips to assemble" });

            // Explicit track wins; else the active one. A song video's music is
            // the master audio (volume 1.0), not background under narration.
            var track = req?.TrackId is { } tid
                ? project.MusicTracks.FirstOrDefault(t => t.Id == tid)
                : project.MusicTracks.FirstOrDefault(t => t.IsActive);
            if (req?.TrackId is not null && track is null)
                return Results.BadRequest(new { error = $"unknown track_id {req.TrackId}" });
            var musicVolume = req?.MusicVolume ?? 1.0;

            var outName = string.IsNullOrWhiteSpace(req?.OutputName) ? "final.mp4" : req!.OutputName!;
            if (!outName.EndsWith(".mp4", StringComparison.OrdinalIgnoreCase)) outName += ".mp4";
            var outPath = Path.Combine(opts.Value.Paths.AssetsDir, id, outName);

            project.Status = ProjectStatus.Assembling;
            await db.SaveChangesAsync(ct);
            var result = await assembler.AssembleAsync(clips, outPath,
                musicPath: track?.OutputPath, musicVolume: musicVolume,
                resolution: req?.Resolution ?? opts.Value.Upscale.Target, ct: ct);
            project.OutputPath = result.OutputPath;
            project.Status = ProjectStatus.Rendered;
            await db.SaveChangesAsync(ct);
            return Results.Ok(new { output_path = result.OutputPath, duration = result.TotalDuration, size_mb = result.FileSizeMb });
        });

        // Real-ESRGAN upscale of every finished clip via ComfyUI (port of
        // main.py /start-upscale). Resumable: clips whose generation already has
        // an UpscaledPath on disk are skipped, so re-POSTing after a crash
        // continues where it left off. Runs detached — poll the project status.
        g.MapPost("/{id}/start-upscale",
            async (string id, AiDirectorDbContext db, IServiceScopeFactory scopes,
                   ILoggerFactory logs, CancellationToken ct) =>
        {
            var exists = await db.Projects.AnyAsync(p => p.Id == id, ct);
            if (!exists) return Results.NotFound();
            var logger = logs.CreateLogger("Upscale");

            _ = Task.Run(async () =>
            {
                using var scope = scopes.CreateScope();
                var sdb = scope.ServiceProvider.GetRequiredService<AiDirectorDbContext>();
                var comfy = scope.ServiceProvider.GetRequiredService<AiDirector.Application.Abstractions.IComfyUiClient>();
                var workflows = scope.ServiceProvider.GetRequiredService<AiDirector.Application.Abstractions.IWorkflowBuilder>();
                var o = scope.ServiceProvider
                    .GetRequiredService<Microsoft.Extensions.Options.IOptions<AiDirector.Application.Configuration.AiDirectorOptions>>().Value;

                var project = await sdb.Projects
                    .Include(p => p.Scenes).ThenInclude(s => s.Generations)
                    .FirstAsync(p => p.Id == id);
                var prior = project.Status;
                project.Status = ProjectStatus.Upscaling;
                await sdb.SaveChangesAsync();

                var done = 0; var failed = 0;
                try
                {
                    if (!await comfy.WaitReadyAsync(o.ComfyUi.ColdStartTimeoutSec, o.ComfyUi.AutoLaunch))
                        throw new InvalidOperationException("ComfyUI not ready for upscaling");

                    var model = o.Upscale.UseAnimeModel ? o.Upscale.AnimeModelName : o.Upscale.ModelName;
                    foreach (var scene in project.Scenes.OrderBy(s => s.SceneNumber))
                    {
                        var gen = scene.ActiveGeneration ?? scene.Generations.LastOrDefault();
                        if (gen?.OutputPath is null || !File.Exists(gen.OutputPath)) continue;
                        if (gen.UpscaledPath is { } up && File.Exists(up)) { done++; continue; }

                        try
                        {
                            // VHS_LoadVideo reads from ComfyUI/input.
                            var inputName = $"aidir_up_{scene.Id[..8]}.mp4";
                            File.Copy(gen.OutputPath, Path.Combine(o.Paths.ComfyInput, inputName), overwrite: true);

                            var prefix = $"aidir_{scene.Id[..8]}_hd";
                            var wf = workflows.EsrganVideoUpscale(inputName, 1920, 1080,
                                o.Video.DefaultFps, model, prefix);
                            var promptId = await comfy.SubmitAsync(wf);
                            var hist = await comfy.WaitForCompletionAsync(promptId);

                            var dest = Path.Combine(o.Paths.AssetsDir, id, "upscaled",
                                $"scene_{scene.SceneNumber:D3}_hd.mp4");
                            await comfy.CollectOutputAsync(hist, dest);
                            gen.UpscaledPath = dest;
                            await sdb.SaveChangesAsync();
                            done++;
                            logger.LogInformation("[Upscale] {N}: scene {Scene} -> {Dest}", done, scene.SceneNumber, dest);
                        }
                        catch (Exception ex)
                        {
                            failed++;
                            logger.LogError(ex, "[Upscale] scene {Scene} failed", scene.SceneNumber);
                        }
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "[Upscale] batch failed for {Id}", id);
                }
                finally
                {
                    project.Status = prior == ProjectStatus.Rendered ? ProjectStatus.Rendered : ProjectStatus.Generated;
                    await sdb.SaveChangesAsync();
                    logger.LogInformation("[Upscale] {Id} finished: {Done} upscaled, {Failed} failed", id, done, failed);
                }
            }, CancellationToken.None);

            return Results.Accepted($"/api/projects/{id}", new { status = "upscaling" });
        });

        // Create scenes explicitly (ports main.py /scenes-manual). Full control
        // over count/prompts/duration — used for the clips-method flow.
        g.MapPost("/{id}/scenes-manual",
            async (string id, ManualScenesRequest req, AiDirectorDbContext db, CancellationToken ct) =>
        {
            var project = await db.Projects.FirstOrDefaultAsync(p => p.Id == id, ct);
            if (project is null) return Results.NotFound();
            if (req.Scenes.Count == 0) return Results.BadRequest(new { error = "no scenes provided" });

            await ClearScenesAsync(db, id, ct);
            var n = 1;
            var scenes = req.Scenes.Select(s => new Scene
            {
                ProjectId = id,
                SceneNumber = n++,
                SceneType = SceneType.Img2Vid,
                Prompt = s.Prompt,
                NegativePrompt = s.NegativePrompt ?? "blurry, low quality, distorted, extra limbs, text watermark",
                Duration = s.Duration ?? 5.0,
                CameraMotion = s.CameraMotion ?? "slow zoom in",
                Status = SceneStatus.Pending,
            }).ToList();
            db.Scenes.AddRange(scenes);
            project.TotalScenes = scenes.Count;
            project.Status = ProjectStatus.Scripted;
            await db.SaveChangesAsync(ct);
            return Results.Ok(new { count = scenes.Count, scenes = scenes.Select(SceneDto.From) });
        });

        // Minimal scene edit.
        app.MapPut("/api/scenes/{sceneId}",
            async (string sceneId, UpdateSceneRequest req, AiDirectorDbContext db, CancellationToken ct) =>
        {
            var scene = await db.Scenes.FirstOrDefaultAsync(s => s.Id == sceneId, ct);
            if (scene is null) return Results.NotFound();
            if (req.Prompt is not null) scene.Prompt = req.Prompt;
            if (req.NegativePrompt is not null) scene.NegativePrompt = req.NegativePrompt;
            if (req.Duration is not null) scene.Duration = req.Duration.Value;
            await db.SaveChangesAsync(ct);
            return Results.Ok(SceneDto.From(scene));
        });
    }

    /// Drop a project's scenes AND the generations hanging off them. Deleting only
    /// the scenes trips "FOREIGN KEY constraint failed" the moment a project has
    /// generated a single clip, which made re-planning an in-flight project impossible.
    /// active_generation_id is cleared first so the scene rows don't point at
    /// generations that are already gone.
    private static async Task ClearScenesAsync(AiDirectorDbContext db, string projectId,
        CancellationToken ct)
    {
        var scenes = await db.Scenes
            .Include(s => s.Generations)
            .Where(s => s.ProjectId == projectId)
            .ToListAsync(ct);
        if (scenes.Count == 0) return;

        // Three separate saves, in this order. Batching them lets EF emit the
        // generation DELETEs before the active_generation_id UPDATE, which trips
        // the FK again.
        foreach (var scene in scenes) scene.ActiveGenerationId = null;
        await db.SaveChangesAsync(ct);

        foreach (var scene in scenes) db.Generations.RemoveRange(scene.Generations);
        await db.SaveChangesAsync(ct);

        db.Scenes.RemoveRange(scenes);
        await db.SaveChangesAsync(ct);
    }

    private static string StyleCueFor(Channel? channel) =>
        channel?.MadeForKids == true
            ? "3D cartoon, soft pastel colors, cute rounded characters, golden-hour lighting"
            : "cinematic, richly detailed, dramatic lighting";
}

public sealed record UpdateSceneRequest(string? Prompt, string? NegativePrompt, double? Duration);

public sealed record ManualScene(string Prompt, string? NegativePrompt, double? Duration, string? CameraMotion);
public sealed record ManualScenesRequest(List<ManualScene> Scenes);

/// Optional render controls. TrackId picks the soundtrack without flipping
/// is_active (lets one project render an English and an Urdu version side by
/// side); OutputName keeps those renders from overwriting each other.
public sealed record RenderRequest(
    string? TrackId, string? OutputName, double? MusicVolume, string? Resolution);

public sealed record MusicVariantsRequest(
    int? Count, string? Engine, string? Style, string? Lyrics, bool? Vocals, bool? Resume,
    // Explicit per-variant specs. When present these win over Count/Style/Lyrics
    // and let one call mix languages (e.g. 2 English + 2 Urdu takes).
    List<MusicVariantSpec>? Variants);

/// One requested take. Style/Lyrics fall back to the project's when omitted.
/// Language is the ACE-Step vocal language hint ("en", "ur", "hi"); it biases
/// pronunciation and must match the language the Lyrics are actually written in.
public sealed record MusicVariantSpec(
    string? Label, string? Style, string? Lyrics, string? Language, long? Seed);
