using AiDirector.Application.Archetypes;
using AiDirector.Domain.Entities;
using FluentAssertions;

namespace AiDirector.Application.Tests;

/// In-memory registry for testing the pure resolver without file I/O.
file sealed class FakeRegistry(params ContentArchetype[] items) : IArchetypeRegistry
{
    private readonly Dictionary<string, ContentArchetype> _map =
        items.ToDictionary(a => a.Id!, a => a);
    public IReadOnlyDictionary<string, ContentArchetype> All() => _map;
    public ContentArchetype? Get(string? id) =>
        string.IsNullOrEmpty(id) ? null : (_map.TryGetValue(id, out var a) ? a : null);
}

public sealed class ArchetypeResolverTests
{
    private static readonly ContentArchetype KidsPoem = new()
    {
        Id = "kids_poem", Tier = 1, AudioMode = "song", VideoEngine = "clips",
        VisualMode = "character_panels", CharacterConsistency = true,
    };
    private static readonly ContentArchetype Dreamscape = new()
    {
        Id = "ai_dreamscape", Tier = 1, AudioMode = "ambient",
        VideoEngine = "ltx_director", VisualMode = "surreal_loop",
        CharacterConsistency = false,
    };
    private static readonly ContentArchetype EduFacts = new()
    {
        Id = "edu_facts", Tier = 2, AudioMode = "narration", VideoEngine = "clips",
    };
    private static readonly ContentArchetype Trap = new()
    {
        Id = "authenticity_trap", Tier = 3, Enabled = false, AudioMode = "narration",
    };

    private static ArchetypeResolver Resolver() =>
        new(new FakeRegistry(KidsPoem, Dreamscape, EduFacts, Trap));

    [Fact]
    public void Legacy_project_with_no_archetype_falls_back_to_song_lane()
    {
        var p = new Project { Title = "x", ChannelId = "c", ProjectType = "song" };
        var r = Resolver().Resolve(p);
        r.ArchetypeId.Should().BeNull();
        r.ProjectType.Should().Be("song");
        r.IsBlocked.Should().BeFalse();
    }

    [Fact]
    public void Channel_archetype_is_inherited_by_project()
    {
        var p = new Project { Title = "x", ChannelId = "c" };
        var ch = new Channel { Name = "n", Slug = "s", ContentArchetype = "ai_dreamscape" };
        var r = Resolver().Resolve(p, ch);
        r.ArchetypeId.Should().Be("ai_dreamscape");
        r.AudioMode.Should().Be("ambient");
        r.VideoEngine.Should().Be("ltx_director");
        r.ProjectType.Should().Be("narration");   // ambient maps to narration lane
    }

    [Fact]
    public void Project_override_beats_channel()
    {
        var p = new Project { Title = "x", ChannelId = "c", ContentArchetype = "kids_poem" };
        var ch = new Channel { Name = "n", Slug = "s", ContentArchetype = "ai_dreamscape" };
        var r = Resolver().Resolve(p, ch);
        r.ArchetypeId.Should().Be("kids_poem");
        r.AudioMode.Should().Be("song");
    }

    [Fact]
    public void Channel_profile_yaml_supplies_archetype_when_db_null()
    {
        var p = new Project { Title = "x", ChannelId = "c" };
        var profile = new Dictionary<string, object?> { ["content_archetype"] = "kids_poem" };
        var r = Resolver().Resolve(p, channel: null, channelProfile: profile);
        r.ArchetypeId.Should().Be("kids_poem");
    }

    [Fact]
    public void Tier2_defaults_to_required_review_and_strict_gate()
    {
        var p = new Project { Title = "x", ChannelId = "c", ContentArchetype = "edu_facts" };
        var r = Resolver().Resolve(p);
        r.Tier.Should().Be(2);
        r.ScriptReview.Should().Be("required");
        r.SafetyGate.Should().Be("strict");
        r.IsBlocked.Should().BeFalse();
    }

    [Fact]
    public void Tier3_trap_is_blocked()
    {
        var p = new Project { Title = "x", ChannelId = "c", ContentArchetype = "authenticity_trap" };
        var r = Resolver().Resolve(p);
        r.IsBlocked.Should().BeTrue();
        r.BlockReason().Should().Contain("not automatable");
    }

    [Fact]
    public void Unknown_archetype_falls_back_to_legacy_without_throwing()
    {
        var p = new Project { Title = "x", ChannelId = "c", ContentArchetype = "nope", ProjectType = "narration" };
        var r = Resolver().Resolve(p);
        r.ArchetypeId.Should().BeNull();
        r.ProjectType.Should().Be("narration");
    }
}
