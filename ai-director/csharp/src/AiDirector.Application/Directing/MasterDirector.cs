using System.Security.Cryptography;
using System.Text;

namespace AiDirector.Application.Directing;

/// Master-director cinematography guidance (port of app/services/master_director.py).
/// Every scene gets a deterministic guidance block (shot, camera move, lighting,
/// mood, composition, continuity) that follows a dramatic arc across the video
/// (opening → build → peak → resolve → finale). The algorithm — MD5 slot hashes
/// reduced with pure modular arithmetic — matches the Python module exactly, so
/// both backends produce identical storyboards for the same project.
public static class MasterDirector
{
    private static readonly Dictionary<string, string[]> Shots = new()
    {
        ["opening"] = ["wide establishing shot", "slow aerial establishing shot", "wide master shot"],
        ["build"] = ["medium shot", "medium wide tracking shot", "over-the-shoulder medium shot"],
        ["peak"] = ["dynamic medium close-up", "hero low-angle medium shot", "sweeping circular medium shot"],
        ["resolve"] = ["medium close-up", "gentle wide shot", "profile medium shot"],
        ["finale"] = ["slow pull-back wide shot", "closing crane-up wide shot", "fading wide twilight shot"],
    };

    private static readonly Dictionary<string, string[]> Cameras = new()
    {
        ["opening"] = ["slow push-in", "gentle crane down", "drifting lateral glide"],
        ["build"] = ["smooth tracking follow", "slow arc around the subject", "steady glide forward"],
        ["peak"] = ["energetic slow orbit", "rising crane-up", "confident push-in"],
        ["resolve"] = ["static camera with subtle drift", "slow pan", "soft push-in"],
        ["finale"] = ["slow pull-back", "tilt up toward the sky", "locked-off wide"],
    };

    private static readonly string[] Lighting =
    [
        "warm golden-hour key light with a soft rim",
        "soft diffused daylight with gentle bounce fill",
        "bright airy high-key lighting",
        "amber sunset backlight with long soft shadows",
        "cool serene twilight key with warm practical accents",
    ];

    private static readonly Dictionary<string, string> Moods = new()
    {
        ["intro"] = "calm anticipation",
        ["hook"] = "wonder and delight",
        ["verse"] = "warm storytelling",
        ["chorus"] = "joyful celebration",
        ["bridge"] = "quiet reflection",
        ["outro"] = "peaceful resolution",
    };

    private static readonly string[] Compositions =
    [
        "subject on the left third with leading room",
        "subject centered with symmetrical framing",
        "subject on the right third with leading room",
        "foreground framing with soft depth of field",
    ];

    public const string Continuity =
        "keep the same character design, outfit, color palette and setting style as the previous shot";

    public sealed record Guidance(
        int Scene, string Phase, string Shot, string Camera, string Lighting,
        string Mood, string Composition, string PromptCue)
    {
        /// Shape stored in scene.director_notes["director_guidance"] — key
        /// names must match the Python dict for DB/JSON parity.
        public Dictionary<string, object?> ToNotes() => new()
        {
            ["scene"] = Scene,
            ["phase"] = Phase,
            ["shot"] = Shot,
            ["camera"] = Camera,
            ["lighting"] = Lighting,
            ["mood"] = Mood,
            ["composition"] = Composition,
            ["continuity"] = Continuity,
            ["prompt_cue"] = PromptCue,
        };
    }

    /// Guidance for scene `index` (0-based) of `total`. Deterministic per
    /// (seedKey, index) — parity with master_director.guidance_for.
    public static Guidance GuidanceFor(int index, int total, string? sectionType, string seedKey)
    {
        var section = string.IsNullOrWhiteSpace(sectionType) ? "verse" : sectionType.ToLowerInvariant();
        var phase = PhaseFor(index, Math.Max(total, 1), section);
        var shot = Pick(Shots[phase], seedKey, index, 1);
        var camera = Pick(Cameras[phase], seedKey, index, 2);
        var lighting = Pick(Lighting, seedKey, index, 3);
        var mood = Moods.GetValueOrDefault(section, Moods["verse"]);
        var composition = Pick(Compositions, seedKey, index, 4);
        var cue = $"{shot}, camera: {camera}, {lighting}, {mood} mood, {composition}";
        return new Guidance(index + 1, phase, shot, camera, lighting, mood, composition, cue);
    }

    /// Append the director cue to a generation prompt. Idempotent: skips when
    /// the cue (or an inlined "camera: <move>" marker from scene planning) is
    /// already present, so resume/retry never doubles the cue.
    public static string ApplyCue(string prompt, Guidance guidance)
    {
        prompt ??= "";
        var marker = $"camera: {guidance.Camera}";
        if (prompt.Contains(guidance.PromptCue) || prompt.Contains(marker)) return prompt;
        return $"{prompt}, {guidance.PromptCue}";
    }

    private static string PhaseFor(int index, int total, string section)
    {
        if (section is "chorus" or "hook") return "peak";
        if (index <= 0) return "opening";
        if (total > 1 && index >= total - 1) return "finale";
        if (total > 1 && (double)index / (total - 1) < 0.5) return "build";
        return "resolve";
    }

    private static string Pick(string[] items, string seedKey, int index, int salt) =>
        items[(int)((SlotHash(seedKey, salt) + (long)index * 31) % items.Length)];

    /// Python: int(md5(f"{seed_key}|{salt}").hexdigest()[:8], 16) — the first
    /// 4 MD5 bytes read big-endian as an unsigned 32-bit value.
    private static long SlotHash(string seedKey, int salt)
    {
        var digest = MD5.HashData(Encoding.UTF8.GetBytes($"{seedKey}|{salt}"));
        return ((long)digest[0] << 24) | ((long)digest[1] << 16) | ((long)digest[2] << 8) | digest[3];
    }
}
