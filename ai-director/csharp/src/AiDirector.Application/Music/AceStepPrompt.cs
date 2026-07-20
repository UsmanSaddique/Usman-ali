using System.Text;
using System.Text.RegularExpressions;

namespace AiDirector.Application.Music;

/// Port of the user-mandated ACE-Step master prompt in app/services/music_gen.py
/// (ACESTEP_MASTER_TEMPLATE + _acestep_master_prompt/_explicit_bpm/_extract_key).
///
/// Every ACE-Step generation is wrapped in this cinematic producer brief. The
/// caller's full style brief rides in the Genre slot VERBATIM so channel identity
/// (nursery-rhyme, nasheed, lullaby...) is never condensed away — that condensing
/// was the original "muddy generic songs" bug.
public static partial class AceStepPrompt
{
    /// Build the full master prompt for a style brief.
    public static string Build(string styleBrief, int bpm, bool instrumental)
    {
        var brief = (styleBrief ?? "").Trim().TrimEnd('.');
        var genre = brief.Length > 0 ? brief : "cinematic";
        var mood = ExtractMood(brief);

        return $"""
            Create a cinematic, commercially releasable, studio-quality track with flawless production.

            Genre:
            {genre}

            Mood:
            {mood}

            Tempo:
            {bpm} BPM

            Key:
            {ExtractKey(brief, mood)}

            Song Structure:
            Intro (8 bars) → Verse → Pre-Chorus → Chorus → Verse 2 → Bridge → Final Chorus → Outro

            Production Quality:
            - Modern 2026 chart-quality mix.
            - Crystal-clear stereo imaging.
            - Ultra-clean mastering.
            - Wide dynamic range.
            - Punchy yet controlled low end.
            - Warm analog saturation.
            - High-end sparkle without harshness.
            - No clipping or distortion.
            - Professional loudness suitable for Spotify, Apple Music and YouTube.

            Instrumentation:
            {BuildInstrumentation(brief)}

            Melody:
            - Emotional.
            - Memorable.
            - Strong hook.
            - Catchy motifs.
            - Natural progression.
            - Dynamic variation.
            - Professional songwriting.

            Harmony:
            - Rich chord extensions.
            - Emotional transitions.
            - Modern cinematic harmony.
            - Layered textures.

            Rhythm:
            - Humanized timing.
            - Groove-focused.
            - Natural swing.
            - Professional drum programming.

            {(instrumental ? InstrumentalBlock : VocalsBlock)}

            Mix:
            - Every instrument occupies its own frequency space.
            - Excellent instrument separation.
            - Clean transient response.
            - Balanced frequencies.
            - Powerful but controlled bass.
            - Wide stereo field.
            - Mono-compatible.

            Master:
            - Radio-ready.
            - Streaming-ready.
            - Commercial loudness.
            - Premium mastering chain.

            Creative Direction:
            - Build tension gradually.
            - Emotional climax in final chorus.
            - Cinematic transitions.
            - Smooth automation.
            - Professional arrangement.
            - Avoid repetition.
            - Introduce subtle variations every section.
            - Add tasteful ear candy throughout.

            Reference Style:
            Inspired by the production quality of Hans Zimmer, Illenium, Martin Garrix, Alan Walker, and modern Hollywood soundtrack engineering, while remaining completely original and non-derivative.

            Negative Prompt:
            Avoid muddy mix, distorted bass, clipping, harsh highs, robotic vocals, repetitive melodies, weak transitions, poor mastering, thin sound, off-key notes, timing issues, excessive compression, low-quality synths, amateur arrangement, or generic loops.

            Output:
            Generate a premium, emotionally engaging, cinematic, high-fidelity production suitable for commercial release and professional streaming platforms.
            """;
    }

    /// Honor an explicit "NNN bpm" in the brief over a coarse fallback — "upbeat"
    /// used to force 130 even when the author wrote 100.
    public static int ExplicitBpm(string? styleBrief, int fallback)
    {
        var m = BpmRegex().Match(styleBrief ?? "");
        if (m.Success && int.TryParse(m.Groups[1].Value, out var bpm) && bpm is >= 40 and <= 220)
            return bpm;
        return fallback;
    }

    /// Explicit key in the brief wins ("D major", "F# minor"); else derive from mood.
    public static string ExtractKey(string? styleBrief, string mood)
    {
        var m = KeyRegex().Match(styleBrief ?? "");
        if (m.Success)
        {
            var quality = m.Groups[2].Value.StartsWith("maj", StringComparison.OrdinalIgnoreCase)
                ? "major" : "minor";
            return $"{m.Groups[1].Value.ToUpperInvariant()} {quality}";
        }
        return mood is "dark" or "melancholic" or "mysterious" or "sad" or "epic"
            ? "A minor" : "C major";
    }

    private static string ExtractMood(string brief)
    {
        var s = brief.ToLowerInvariant();
        foreach (var (kw, val) in MoodMap)
            if (s.Contains(kw)) return val;
        return "uplifting";
    }

    private static string BuildInstrumentation(string brief)
    {
        var s = brief.ToLowerInvariant();
        var found = Instruments.Where(i => s.Contains(i)).ToList();
        if (found.Count == 0) return DefaultInstrumentation;

        var sb = new StringBuilder();
        foreach (var i in found)
            sb.AppendLine($"- {char.ToUpperInvariant(i[0])}{i[1..]}.");
        sb.AppendLine("- Rich atmospheric pads.");
        sb.Append("- Textural ambience.");
        return sb.ToString();
    }

    // Ordered: first match wins, mirroring the Python dict iteration order.
    private static readonly (string Keyword, string Mood)[] MoodMap =
    [
        ("peaceful", "peaceful"), ("calm", "calm"), ("gentle", "calm"),
        ("warm", "uplifting"), ("reverent", "peaceful"), ("happy", "happy"),
        ("cheerful", "happy"), ("dreamy", "dreamy"), ("playful", "playful"),
        ("energetic", "energetic"), ("upbeat", "uplifting"), ("sad", "melancholic"),
        ("epic", "epic"), ("dark", "dark"), ("romantic", "romantic"),
        ("nostalgic", "nostalgic"), ("mysterious", "mysterious"),
    ];

    private static readonly string[] Instruments =
    [
        "piano", "guitar", "ukulele", "xylophone", "flute", "violin",
        "drums", "bass", "synthesizer", "bells", "harp", "daf", "tabla",
        "sitar", "oud", "accordion", "trumpet", "cello", "marimba",
        "strings", "organ", "synth",
    ];

    private const string VocalsBlock = """
        Vocals:
        - Expressive.
        - Emotional.
        - Natural breathing.
        - Perfect pronunciation.
        - Studio-quality recording.
        - Rich harmonies.
        - Layered backing vocals.
        - Wide chorus doubles.
        - Smooth vocal automation.
        - Professional vocal chain with EQ, compression, de-essing, saturation and reverb.
        """;

    private const string InstrumentalBlock = """
        Vocals:
        - Instrumental only — no vocals, no humming, no voice of any kind.
        """;

    private const string DefaultInstrumentation = """
        - Layered cinematic synths.
        - Rich atmospheric pads.
        - Deep sub bass.
        - Tight punchy kick.
        - Clean snare.
        - Crisp hi-hats.
        - Organic percussion.
        - Modern FX.
        - Reverse impacts.
        - Risers.
        - Downlifters.
        - Textural ambience.
        - Live strings and piano where appropriate.
        """;

    [GeneratedRegex(@"(\d{2,3})\s*bpm", RegexOptions.IgnoreCase)]
    private static partial Regex BpmRegex();

    [GeneratedRegex(@"\b([A-G][#b]?)\s*(major|minor|maj|min)\b", RegexOptions.IgnoreCase)]
    private static partial Regex KeyRegex();
}
