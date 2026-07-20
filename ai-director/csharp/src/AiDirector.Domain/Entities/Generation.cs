using AiDirector.Domain.Enums;

namespace AiDirector.Domain.Entities;

/// Maps table "generations" (app/database.py Generation).
public class Generation
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string SceneId { get; set; } = null!;
    public int Version { get; set; }
    public string ModelUsed { get; set; } = null!;          // "ltx-2.3", "sdxl", "wan-14b"
    public string? OutputPath { get; set; }                 // raw clip/image path
    public string? UpscaledPath { get; set; }               // HD version path
    public string? ThumbnailPath { get; set; }
    public string? PromptUsed { get; set; }                 // actual prompt (with channel prefix etc.)
    public string? NegativePromptUsed { get; set; }
    public long? Seed { get; set; }
    public Dictionary<string, object?> Parameters { get; set; } = [];  // steps, cfg, resolution, fps, etc.
    public double? QualityScore { get; set; }               // 0-100
    public GenerationStatus Status { get; set; } = GenerationStatus.Queued;
    public string? ErrorLog { get; set; }
    public double? GenerationTimeSec { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public Scene Scene { get; set; } = null!;
}
