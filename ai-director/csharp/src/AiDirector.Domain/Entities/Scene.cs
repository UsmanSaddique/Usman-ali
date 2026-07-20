using AiDirector.Domain.Enums;

namespace AiDirector.Domain.Entities;

/// Maps table "scenes" (app/database.py Scene).
public class Scene
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string ProjectId { get; set; } = null!;
    public int SceneNumber { get; set; }
    public SceneType SceneType { get; set; }
    public string Prompt { get; set; } = null!;
    public string NegativePrompt { get; set; } = "";
    public double Duration { get; set; } = 5.0;              // seconds
    public string CameraMotion { get; set; } = "static";
    public string? VideoModel { get; set; }                 // per-scene model override
    public List<string> LoraIds { get; set; } = [];
    public List<double> LoraWeights { get; set; } = [];
    public SceneStatus Status { get; set; } = SceneStatus.Pending;
    public string? ActiveGenerationId { get; set; }         // FK set after generation
    public int RetryCount { get; set; }
    public int MaxRetries { get; set; } = 3;
    public Dictionary<string, object?> DirectorNotes { get; set; } = [];  // style cues, transition hints
    public string? NarrationText { get; set; }
    public double? NarrationStart { get; set; }             // narration mode: beat start (s) in master WAV
    public double? NarrationEnd { get; set; }               // narration mode: beat end (s)
    public string? VisualType { get; set; }                 // broll|still|diagram|code|map|title_card
    public string? SfxPrompt { get; set; }                  // sound-design intent for this beat
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public Project Project { get; set; } = null!;
    public List<Generation> Generations { get; set; } = [];

    /// The generation pointed to by ActiveGenerationId, if any.
    public Generation? ActiveGeneration =>
        ActiveGenerationId is null
            ? null
            : Generations.FirstOrDefault(g => g.Id == ActiveGenerationId);

    /// Highest-version generation (Generations is ordered by version).
    public Generation? LatestGeneration =>
        Generations.Count > 0 ? Generations[^1] : null;
}
