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
    int SceneNumber, string SceneType, string Prompt, double Duration,
    string Status, string? OutputPath)
{
    public static SceneDto From(Scene s) => new(
        s.SceneNumber, s.SceneType.ToString().ToLowerInvariant(), s.Prompt, s.Duration,
        s.Status.ToString().ToLowerInvariant(), s.ActiveGeneration?.OutputPath);
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

public sealed record ChannelSummary(string Id, string Name, string Slug, bool MadeForKids)
{
    public static ChannelSummary From(Channel c) => new(c.Id, c.Name, c.Slug, c.MadeForKids);
}
