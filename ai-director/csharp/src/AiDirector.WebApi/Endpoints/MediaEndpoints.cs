using System.Text.RegularExpressions;
using AiDirector.Infrastructure.Persistence;
using Microsoft.EntityFrameworkCore;

namespace AiDirector.WebApi.Endpoints;

/// Serves generated media files to the frontend the way the Python app's static
/// mount does: /projects/{id}/final_render.mp4, /projects/{id}/clips/scene_NNN_vN.mp4,
/// /projects/{id}/music/{trackId}. Files are located via the DB output paths.
public static partial class MediaEndpoints
{
    public static void MapMediaEndpoints(this IEndpointRouteBuilder app)
    {
        // Final rendered video.
        app.MapGet("/projects/{id}/final_render.mp4",
            async (string id, AiDirectorDbContext db, CancellationToken ct) =>
        {
            var project = await db.Projects.FirstOrDefaultAsync(p => p.Id == id, ct);
            var path = Resolve(project?.OutputPath);
            return path is null ? Results.NotFound() : ServeFile(path, "video/mp4");
        });

        // A scene clip: /clips/scene_003_v1.mp4 -> scene 3's active/latest clip.
        app.MapGet("/projects/{id}/clips/{clip}",
            async (string id, string clip, AiDirectorDbContext db, CancellationToken ct) =>
        {
            var m = SceneClipRe().Match(clip);
            if (!m.Success) return Results.NotFound();
            var sceneNo = int.Parse(m.Groups[1].Value);

            var scene = await db.Scenes
                .Include(s => s.Generations)
                .FirstOrDefaultAsync(s => s.ProjectId == id && s.SceneNumber == sceneNo, ct);
            if (scene is null) return Results.NotFound();

            var gen = scene.Generations.FirstOrDefault(g => g.Id == scene.ActiveGenerationId)
                      ?? scene.Generations.OrderByDescending(g => g.Version).FirstOrDefault();
            var path = Resolve(gen?.OutputPath);
            return path is null ? Results.NotFound() : ServeFile(path, "video/mp4");
        });

        // A music track's audio.
        app.MapGet("/projects/{id}/music/{trackId}",
            async (string id, string trackId, AiDirectorDbContext db, CancellationToken ct) =>
        {
            var track = await db.MusicTracks
                .FirstOrDefaultAsync(t => t.Id == trackId && t.ProjectId == id, ct);
            var path = Resolve(track?.OutputPath);
            return path is null ? Results.NotFound() : ServeFile(path, "audio/wav");
        });
    }

    // Stored paths may be relative (to the WebApi CWD) or absolute; resolve + verify.
    private static string? Resolve(string? stored)
    {
        if (string.IsNullOrWhiteSpace(stored)) return null;
        var full = Path.GetFullPath(stored);
        return File.Exists(full) ? full : null;
    }

    private static IResult ServeFile(string path, string contentType) =>
        Results.File(path, contentType, enableRangeProcessing: true);

    [GeneratedRegex(@"scene_0*(\d+)")]
    private static partial Regex SceneClipRe();
}
