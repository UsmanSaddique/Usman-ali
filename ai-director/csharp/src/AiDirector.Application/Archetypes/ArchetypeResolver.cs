using AiDirector.Domain.Entities;

namespace AiDirector.Application.Archetypes;

/// Loads the archetypes/*.yaml registry (implemented in Infrastructure with
/// YamlDotNet). Mirrors load_archetypes()/get_archetype() in the Python service.
public interface IArchetypeRegistry
{
    /// All archetypes keyed by id (cached).
    IReadOnlyDictionary<string, ContentArchetype> All();

    ContentArchetype? Get(string? id);
}

/// Pure resolution logic (no I/O). Parity with resolve()/_coerce_recipe()/
/// _legacy_recipe() in app/services/archetypes.py.
public sealed class ArchetypeResolver(IArchetypeRegistry registry)
{
    // Tier -> default HITL policy. An archetype's hitl block overrides per-field.
    private static readonly Dictionary<int, (string Review, string Gate)> TierHitl = new()
    {
        [1] = ("optional", "standard"),
        [2] = ("required", "strict"),
        [3] = ("blocked", "blocked"),
    };

    /// Resolve the effective recipe for a project.
    /// Precedence: project.ContentArchetype > channel.ContentArchetype >
    /// channelProfile["content_archetype"] > legacy.
    public ResolvedRecipe Resolve(
        Project project,
        Channel? channel = null,
        IReadOnlyDictionary<string, object?>? channelProfile = null)
    {
        var aid = project.ContentArchetype;
        if (string.IsNullOrEmpty(aid) && channel is not null)
            aid = channel.ContentArchetype;
        if (string.IsNullOrEmpty(aid) && channelProfile is not null
            && channelProfile.TryGetValue("content_archetype", out var v))
            aid = v?.ToString();

        var raw = registry.Get(aid);
        if (raw is null)
            return Legacy(project);   // unknown or unset => identical to pre-archetype behavior

        return Coerce(raw);
    }

    /// Every archetype as a resolved recipe (tier→HITL defaults applied),
    /// ordered by tier then id. Powers the wizard's /api/archetypes list.
    public IReadOnlyList<ResolvedRecipe> Describe() =>
        registry.All().Values
            .Select(Coerce)
            .OrderBy(r => r.Tier).ThenBy(r => r.ArchetypeId, StringComparer.Ordinal)
            .ToList();

    private static ResolvedRecipe Coerce(ContentArchetype raw)
    {
        var tier = raw.Tier;
        var (defReview, defGate) = TierHitl.TryGetValue(tier, out var d) ? d : TierHitl[1];
        var review = raw.Hitl?.ScriptReview ?? defReview;
        var gate = raw.Hitl?.SafetyGate ?? defGate;

        return new ResolvedRecipe
        {
            ArchetypeId = raw.Id,
            Tier = tier,
            Label = raw.Label,
            Enabled = raw.Enabled,
            AudioMode = raw.AudioMode,
            VideoEngine = raw.VideoEngine,
            VisualMode = raw.VisualMode,
            CharacterConsistency = raw.CharacterConsistency,
            ScenePlanner = raw.ScenePlanner,
            Source = raw.Source,
            ScriptReview = review,
            SafetyGate = gate,
            SeoProfile = raw.SeoProfile,
            IpDenylist = raw.IpDenylist,
        };
    }

    private static ResolvedRecipe Legacy(Project project)
    {
        var ptype = string.IsNullOrEmpty(project.ProjectType) ? "song" : project.ProjectType;
        var audio = ptype == "song" ? "song" : "narration";
        return new ResolvedRecipe
        {
            ArchetypeId = null,
            Tier = 1,
            Label = "(legacy)",
            Enabled = true,
            AudioMode = audio,
            VideoEngine = string.IsNullOrEmpty(project.VideoEngine) ? "clips" : project.VideoEngine,
            VisualMode = audio == "song" ? "character_panels" : "voice_over_bg",
            CharacterConsistency = audio == "song",
            ScenePlanner = audio == "song" ? "lyric_scenes" : "narration_scenes",
            Source = "llm",
        };
    }
}
