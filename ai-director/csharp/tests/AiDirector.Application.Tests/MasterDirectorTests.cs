using AiDirector.Application.Directing;
using FluentAssertions;

namespace AiDirector.Application.Tests;

/// Pins MasterDirector to the Python master_director.py algorithm: identical
/// picks for the same (seed_key, index, section), idempotent cue application.
public sealed class MasterDirectorTests
{
    [Fact]
    public void Guidance_matches_the_python_reference_vector()
    {
        // Reference produced by app/services/master_director.py:
        //   guidance_for(2, 8, 'chorus', 'parity-project')
        var g = MasterDirector.GuidanceFor(2, 8, "chorus", "parity-project");

        g.Phase.Should().Be("peak");
        g.Shot.Should().Be("hero low-angle medium shot");
        g.Camera.Should().Be("confident push-in");
        g.Lighting.Should().Be("cool serene twilight key with warm practical accents");
        g.Mood.Should().Be("joyful celebration");
        g.Composition.Should().Be("subject on the right third with leading room");
        g.PromptCue.Should().Be(
            "hero low-angle medium shot, camera: confident push-in, " +
            "cool serene twilight key with warm practical accents, " +
            "joyful celebration mood, subject on the right third with leading room");
    }

    [Fact]
    public void Guidance_is_deterministic_and_varies_by_project()
    {
        var a = Enumerable.Range(0, 10)
            .Select(i => MasterDirector.GuidanceFor(i, 10, "verse", "projA").PromptCue).ToList();
        var b = Enumerable.Range(0, 10)
            .Select(i => MasterDirector.GuidanceFor(i, 10, "verse", "projB").PromptCue).ToList();

        a.Should().Equal(Enumerable.Range(0, 10)
            .Select(i => MasterDirector.GuidanceFor(i, 10, "verse", "projA").PromptCue));
        a.Zip(b).Count(p => p.First != p.Second).Should().BeGreaterThanOrEqualTo(5,
            "storyboards must vary per project, not clone a template");
    }

    [Fact]
    public void Arc_phases_follow_opening_build_peak_finale()
    {
        MasterDirector.GuidanceFor(0, 10, "intro", "p").Phase.Should().Be("opening");
        MasterDirector.GuidanceFor(2, 10, "verse", "p").Phase.Should().Be("build");
        MasterDirector.GuidanceFor(4, 10, "chorus", "p").Phase.Should().Be("peak");
        MasterDirector.GuidanceFor(7, 10, "verse", "p").Phase.Should().Be("resolve");
        MasterDirector.GuidanceFor(9, 10, "outro", "p").Phase.Should().Be("finale");
    }

    [Fact]
    public void ApplyCue_is_idempotent_across_resume_and_retry()
    {
        var g = MasterDirector.GuidanceFor(0, 10, "intro", "projA");
        var once = MasterDirector.ApplyCue("a cat in a garden", g);
        once.Should().Contain(g.PromptCue);
        MasterDirector.ApplyCue(once, g).Should().Be(once, "re-applying must not double the cue");
    }
}
