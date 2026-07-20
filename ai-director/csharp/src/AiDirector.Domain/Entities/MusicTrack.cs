namespace AiDirector.Domain.Entities;

/// Maps table "music_tracks" (app/database.py MusicTrack).
public class MusicTrack
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string ProjectId { get; set; } = null!;
    public string? StylePrompt { get; set; }
    public string? OutputPath { get; set; }
    public double? Duration { get; set; }
    public bool IsActive { get; set; } = true;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public Project Project { get; set; } = null!;
}
