using AiDirector.Application.Directing;
using AiDirector.Application.Safety;
using AiDirector.Domain.Entities;
using AiDirector.Domain.Enums;
using FluentAssertions;

namespace AiDirector.Application.Tests;

public sealed class LyricsParserTests
{
    [Fact]
    public void Sections_span_full_song_duration()
    {
        const string lyrics = "[verse]\nLine one\nLine two\n[chorus]\nSing along\nEveryone";
        var segs = LyricsParser.Parse(lyrics, songDuration: 30.0);

        segs.Should().NotBeEmpty();
        segs[0].StartSec.Should().Be(0.0);
        segs[^1].EndSec.Should().Be(30.0);          // exact span
        segs.Should().BeInAscendingOrder(s => s.StartSec);
    }

    [Fact]
    public void No_markers_yields_one_segment()
    {
        var segs = LyricsParser.Parse("just a plain line", 10.0);
        segs.Should().ContainSingle();
        segs[0].Duration.Should().Be(10.0);
    }

    [Fact]
    public void MaxSegments_caps_count()
    {
        var lyrics = "[verse]\n" + string.Join("\n", Enumerable.Range(1, 20).Select(i => $"line {i}"));
        var segs = LyricsParser.Parse(lyrics, 60.0, maxSegments: 5);
        segs.Count.Should().BeLessThanOrEqualTo(5);
    }
}

public sealed class YtSafetyGateTests
{
    private static Project Project(string? lyrics = null, string title = "Nice Song") =>
        new() { Title = title, ChannelId = "c", Lyrics = lyrics, DurationTarget = 60 };

    [Fact]
    public void Clean_content_passes()
    {
        var gate = new YtSafetyGate();
        var r = gate.Run(Project(lyrics: "twinkle twinkle little star"), [], madeForKids: true);
        r.Verdict.Should().Be(SafetyVerdict.Pass);
        r.Issues.Should().BeEmpty();
    }

    [Fact]
    public void Graphic_violence_blocks()
    {
        var gate = new YtSafetyGate();
        var r = gate.Run(Project(lyrics: "a graphic massacre scene"), [], madeForKids: false);
        r.Verdict.Should().Be(SafetyVerdict.Block);
        r.Issues.Should().Contain(i => i.Category == "violence");
    }

    [Fact]
    public void Kids_channel_flags_otherwise_ok_words()
    {
        var gate = new YtSafetyGate();
        // "gun" is fine for adults but fails made-for-kids review -> high -> revise.
        var r = gate.Run(Project(lyrics: "he had a gun"), [], madeForKids: true);
        r.Verdict.Should().Be(SafetyVerdict.Revise);
        r.Issues.Should().Contain(i => i.Category == "kids");
    }
}
