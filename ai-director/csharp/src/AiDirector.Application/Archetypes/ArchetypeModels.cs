namespace AiDirector.Application.Archetypes;

/// Raw archetype as parsed from archetypes/<id>.yaml. Mirrors the Python raw
/// dict in app/services/archetypes.py. Optional fields fall back to defaults in
/// the resolver so a terse YAML still produces a complete recipe.
public sealed class ContentArchetype
{
    public string? Id { get; set; }
    public int Tier { get; set; } = 1;
    public string Label { get; set; } = "";
    public bool Enabled { get; set; } = true;

    public string AudioMode { get; set; } = "song";          // song | narration | ambient
    public string VideoEngine { get; set; } = "clips";        // clips | ltx_director
    public string VisualMode { get; set; } = "character_panels";
    public bool CharacterConsistency { get; set; } = true;
    public string ScenePlanner { get; set; } = "lyric_scenes";
    public string Source { get; set; } = "llm";               // llm | scrape | mixed

    public HitlPolicy? Hitl { get; set; }
    public string SeoProfile { get; set; } = "default";

    public List<string> IpDenylist { get; set; } = [];
}

public sealed class HitlPolicy
{
    public string? ScriptReview { get; set; }   // optional | required | blocked
    public string? SafetyGate { get; set; }     // standard | strict | blocked
}

/// The effective pipeline wiring for a project after merge. Parity with the
/// Python ResolvedRecipe dataclass.
public sealed record ResolvedRecipe
{
    public string? ArchetypeId { get; init; }   // null => legacy (no archetype)
    public int Tier { get; init; } = 1;
    public string Label { get; init; } = "";
    public bool Enabled { get; init; } = true;

    public string AudioMode { get; init; } = "song";
    public string VideoEngine { get; init; } = "clips";
    public string VisualMode { get; init; } = "character_panels";
    public bool CharacterConsistency { get; init; } = true;
    public string ScenePlanner { get; init; } = "lyric_scenes";
    public string Source { get; init; } = "llm";

    public string ScriptReview { get; init; } = "optional";
    public string SafetyGate { get; init; } = "standard";
    public string SeoProfile { get; init; } = "default";
    public IReadOnlyList<string> IpDenylist { get; init; } = [];

    /// Internal lane. audio_mode maps to project_type so legacy code that
    /// branches on ProjectType keeps working: song => "song", else "narration".
    public string ProjectType => AudioMode == "song" ? "song" : "narration";

    /// Tier-3 / disabled archetypes must refuse at start-generation.
    public bool IsBlocked =>
        !Enabled || Tier == 3 || ScriptReview == "blocked" || SafetyGate == "blocked";

    public string BlockReason() =>
        $"Archetype '{ArchetypeId ?? "unknown"}' ({(string.IsNullOrEmpty(Label) ? "Tier 3" : Label)}) is " +
        "not automatable: this niche relies on real-world footage, exact physics, " +
        "copyrighted material, or human authenticity that local AI generation " +
        "cannot produce credibly. Enable it only with explicit human production.";
}
