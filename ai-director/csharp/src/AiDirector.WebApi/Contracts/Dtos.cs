using AiDirector.Domain.Entities;

namespace AiDirector.WebApi.Contracts;

public sealed record ProjectSummary(
    string Id, string Title, string ChannelId, string ProjectType,
    int DurationTarget, int Duration,   // frontend list reads `duration`; detail reads `duration_target`
    string Status, int TotalScenes, int CompletedScenes, DateTime CreatedAt)
{
    public static ProjectSummary From(Project p) => new(
        p.Id, p.Title, p.ChannelId, p.ProjectType, p.DurationTarget, p.DurationTarget,
        p.Status.ToString().ToLowerInvariant(), p.TotalScenes, p.CompletedScenes, p.CreatedAt);
}

public sealed record SceneDto(
    string Id, int SceneNumber, string SceneType, string Prompt, double Duration,
    string Status, string? OutputPath, string? ClipPath, string? ClipUrl,
    int Versions, int? ActiveVersion, bool Upscaled)
{
    // Mirrors the Python scene dict (main.py project detail): the frontend keys
    // per-scene actions off `id`, shows the HD badge off `upscaled`, and plays
    // `clip_url` — which must point at the upscaled clip when one exists.
    public static SceneDto From(Scene s)
    {
        var gen = s.ActiveGeneration ?? s.Generations.OrderByDescending(g => g.Version).FirstOrDefault();
        var upscaled = gen?.UpscaledPath is { Length: > 0 } up && up != gen.OutputPath;
        var clipPath = upscaled ? gen!.UpscaledPath : gen?.OutputPath;
        return new(
            s.Id, s.SceneNumber, s.SceneType.ToString().ToLowerInvariant(), s.Prompt, s.Duration,
            s.Status.ToString().ToLowerInvariant(), gen?.OutputPath, clipPath,
            clipPath is null ? null : $"/projects/{s.ProjectId}/clips/scene_{s.SceneNumber:D3}_v{gen?.Version ?? 1}.mp4",
            s.Generations.Count, gen?.Version, upscaled);
    }
}

public sealed record MusicVariant(string Id, string Url, string? Style, bool Active);

public sealed record ProjectDetail(
    string Id, string Title, string ChannelId, string ProjectType, int DurationTarget,
    string Status, string VideoEngine, string VideoModel,
    string? Lyrics, string? Context,
    IReadOnlyList<SceneDto> Scenes, IReadOnlyList<MusicVariant> MusicVariants)
{
    public static ProjectDetail From(Project p) => new(
        p.Id, p.Title, p.ChannelId, p.ProjectType, p.DurationTarget,
        p.Status.ToString().ToLowerInvariant(), p.VideoEngine, p.VideoModel,
        p.Lyrics, p.Context,
        p.Scenes.OrderBy(s => s.SceneNumber).Select(SceneDto.From).ToList(),
        p.MusicTracks
            .Where(t => !string.IsNullOrEmpty(t.OutputPath))
            .Select(t => new MusicVariant(t.Id, $"/projects/{p.Id}/music/{t.Id}", t.StylePrompt, t.IsActive))
            .ToList());
}

public sealed record CreateProjectRequest(
    string Title, string ChannelId, int DurationTarget,
    string ProjectType = "song", string? Context = null, string? Lyrics = null);

public sealed record ChannelSummary(
    string Id, string Name, string Slug, bool MadeForKids, string? ContentArchetype)
{
    public static ChannelSummary From(Channel c) =>
        new(c.Id, c.Name, c.Slug, c.MadeForKids, c.ContentArchetype);
}
