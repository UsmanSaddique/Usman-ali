namespace AiDirector.Domain.Entities;

/// Maps table "channels" (app/database.py Channel).
public class Channel
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Name { get; set; } = null!;
    public string Slug { get; set; } = null!;              // "little-fairy-dreams"
    public string? ProfilePath { get; set; }               // path to YAML config
    public string? SystemPrompt { get; set; }              // preloaded LLM context
    public List<string> DefaultLoras { get; set; } = [];
    public double StillRatio { get; set; } = 0.4;
    public string TargetResolution { get; set; } = "1080p";
    public bool MadeForKids { get; set; }
    public string? ContentArchetype { get; set; }          // archetype id (archetypes/*.yaml); null=legacy
    public string DefaultVideoModel { get; set; } = "ltx-2.3";
    public string DefaultImageModel { get; set; } = "sdxl";
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public List<Project> Projects { get; set; } = [];
}
