using AiDirector.Domain.Enums;

namespace AiDirector.Domain.Entities;

/// Maps table "render_jobs" (app/database.py RenderJob).
public class RenderJob
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string ProjectId { get; set; } = null!;
    public string Resolution { get; set; } = "1080p";
    public string? OutputPath { get; set; }
    public RenderStatus Status { get; set; } = RenderStatus.Queued;
    public double ProgressPct { get; set; }
    public Dictionary<string, object?> RenderSettings { get; set; } = [];  // transitions, audio mix levels
    public string? ErrorLog { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public Project Project { get; set; } = null!;
}
