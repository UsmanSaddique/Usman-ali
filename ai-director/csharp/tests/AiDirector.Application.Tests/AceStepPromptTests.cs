using AiDirector.Application.Music;
using FluentAssertions;

namespace AiDirector.Application.Tests;

/// Parity guards for the user-mandated ACE-Step master prompt
/// (app/services/music_gen.py :: ACESTEP_MASTER_TEMPLATE and friends).
public sealed class AceStepPromptTests
{
    // The approved nursery-rhyme brief from the "Animal Parade" song.
    private const string NurseryBrief =
        "classic children's nursery rhyme in the style of Old MacDonald Had a Farm, " +
        "cheerful bouncy sing-along march, clear solo female lead vocal with crisp enunciation, " +
        "acoustic guitar strums, ukulele, glockenspiel, marimba, bright major key, " +
        "professional studio mix, clean bright master, 105 bpm";

    [Fact]
    public void Style_brief_rides_verbatim_in_the_genre_slot()
    {
        // The original bug: the brief was condensed to ~6 generic words and the
        // channel's identity was thrown away.
        var prompt = AceStepPrompt.Build(NurseryBrief, bpm: 105, instrumental: false);

        prompt.Should().Contain("Genre:");
        prompt.Should().Contain(NurseryBrief.TrimEnd('.'),
            "the full brief must survive into the Genre slot");
    }

    [Fact]
    public void Explicit_bpm_in_the_brief_beats_the_caller_default()
    {
        // "upbeat" used to force 130 even when the author wrote 105.
        AceStepPrompt.ExplicitBpm(NurseryBrief, fallback: 130).Should().Be(105);
        AceStepPrompt.ExplicitBpm("no tempo stated", fallback: 90).Should().Be(90);
        AceStepPrompt.ExplicitBpm("900 bpm nonsense", fallback: 90).Should().Be(90);
    }

    [Fact]
    public void Explicit_key_wins_over_mood_default()
    {
        AceStepPrompt.ExtractKey("gentle waltz in D minor", "calm").Should().Be("D minor");
        AceStepPrompt.ExtractKey("bright and cheerful", "happy").Should().Be("C major");
        AceStepPrompt.ExtractKey("brooding score", "dark").Should().Be("A minor");
    }

    [Fact]
    public void Instrumental_swaps_the_vocals_block()
    {
        var sung = AceStepPrompt.Build(NurseryBrief, 105, instrumental: false);
        var instrumental = AceStepPrompt.Build(NurseryBrief, 105, instrumental: true);

        sung.Should().Contain("Layered backing vocals.");
        instrumental.Should().Contain("Instrumental only");
        instrumental.Should().NotContain("Layered backing vocals.");
    }

    [Fact]
    public void Instruments_named_in_the_brief_drive_the_instrumentation_block()
    {
        var prompt = AceStepPrompt.Build(NurseryBrief, 105, instrumental: false);

        prompt.Should().Contain("- Ukulele.");
        prompt.Should().Contain("- Marimba.");
        // The generic cinematic fallback must not appear when real instruments exist.
        prompt.Should().NotContain("- Deep sub bass.");
    }

    [Fact]
    public void Falls_back_to_default_instrumentation_when_none_named()
    {
        var prompt = AceStepPrompt.Build("moody atmospheric score", 90, instrumental: true);
        prompt.Should().Contain("- Deep sub bass.");
    }

    [Fact]
    public void Contains_the_mandated_structural_sections()
    {
        var prompt = AceStepPrompt.Build(NurseryBrief, 105, instrumental: false);

        foreach (var section in new[]
                 {
                     "Song Structure:", "Production Quality:", "Instrumentation:", "Melody:",
                     "Harmony:", "Rhythm:", "Mix:", "Master:", "Creative Direction:",
                     "Reference Style:", "Negative Prompt:", "Output:",
                 })
            prompt.Should().Contain(section);
    }
}
