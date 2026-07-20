using System.Text.RegularExpressions;
using AiDirector.Domain.Entities;
using AiDirector.Domain.Enums;
using AiDirector.Domain.ValueObjects;

namespace AiDirector.Application.Safety;

/// Port of the rule layer of app/services/yt_safety.py. Deterministic, instant,
/// no model — lexicon scans + quote-length limits + made-for-kids strictness.
/// (The Python LLM critic layer is intentionally omitted here; rules stand.)
public sealed class YtSafetyGate
{
    public sealed record GateResult(
        SafetyVerdict Verdict, List<SafetyIssue> Issues, Dictionary<string, object?> CheckedFields);

    private static readonly string[] Profanity =
    [
        "fuck", "fucking", "shit", "bitch", "asshole", "bastard", "dick",
        "cunt", "motherfucker", "nigger", "faggot", "slut", "whore",
    ];
    private static readonly string[] ViolenceGraphic =
    [
        "beheading", "decapitat", "dismember", "torture", "massacre", "mutilat",
        "gore", "bloodbath", "execution video", "mass shooting", "school shooting", "killed on camera",
    ];
    private static readonly string[] DangerousActs =
    [
        "how to make a bomb", "build a gun", "3d printed gun", "make explosives",
        "poison someone", "hard drug", "how to overdose", "self-harm tutorial",
        "suicide method", "choking challenge", "blackout challenge", "tide pod", "drink bleach",
    ];
    private static readonly string[] MedicalMisinfo =
    [
        "vaccines cause autism", "cure cancer with", "cures cancer", "miracle cure",
        "doctors don't want you to know", "big pharma is hiding", "covid is a hoax",
        "5g causes", "drink your urine",
    ];
    private static readonly string[] MisleadingClickbait =
    [
        "you won't believe", "gone wrong gone sexual", "(not clickbait)", "free robux",
        "free v-bucks", "get rich quick", "guaranteed profit", "100% guaranteed returns",
    ];
    private static readonly string[] KidsUnsafe =
    [
        "kill", "gun", "knife", "blood", "die", "dead", "death", "scary", "horror",
        "demon", "devil", "drug", "beer", "cigarette", "kidnap", "gambling", "casino",
    ];
    private static readonly string[] CopyrightMarkers =
    [
        "lyrics by", "official lyrics", "©", "(c) 20", "all rights reserved",
        "originally performed by", "cover of the song",
    ];

    public GateResult Run(Project project, IReadOnlyList<Scene> scenes, bool madeForKids)
    {
        var fields = Collect(project, scenes);
        var issues = new List<SafetyIssue>();

        foreach (var (where, text) in fields)
        {
            if (string.IsNullOrEmpty(text)) continue;

            foreach (var hit in Scan(text, Profanity))
                issues.Add(new("high", "profanity", where, $"Profanity '{hit}' — limits ads, trips age-restriction.", "Remove or replace the word."));
            foreach (var hit in Scan(text, ViolenceGraphic, wholeWord: false))
                issues.Add(new("block", "violence", where, $"Graphic-violence phrase '{hit}' — advertiser-unfriendly / possible strike.", "Describe events without graphic detail."));
            foreach (var hit in Scan(text, DangerousActs, wholeWord: false))
                issues.Add(new("block", "dangerous", where, $"Dangerous-acts phrase '{hit}' — harmful/dangerous-content policy.", "Remove the instructional/dangerous framing entirely."));
            foreach (var hit in Scan(text, MedicalMisinfo, wholeWord: false))
                issues.Add(new("block", "medical", where, $"Medical-misinformation phrase '{hit}'.", "State only mainstream-consensus health information."));
            foreach (var hit in Scan(text, MisleadingClickbait, wholeWord: false))
                issues.Add(new("medium", "clickbait", where, $"Clickbait/scam phrase '{hit}' — misleading-metadata risk.", "Rewrite as an honest, specific hook."));
            foreach (var hit in Scan(text, CopyrightMarkers, wholeWord: false))
                issues.Add(new("high", "copyright", where, $"Copyright marker '{hit}' — suggests third-party lyrics/content.", "All lyrics and text must be original to this project."));

            if (madeForKids && where != "context")
                foreach (var hit in Scan(text, KidsUnsafe))
                    issues.Add(new("high", "kids", where, $"'{hit}' on a made-for-kids channel — fails kids review.", "Keep made-for-kids content gentle: no weapons/death/scary/substances."));

            foreach (Match m in Regex.Matches(text, "[\"“]([^\"”]{90,})[\"”]"))
                issues.Add(new("medium", "copyright", where, $"Verbatim quote of {m.Groups[1].Value.Length} chars — keep quotes under ~90 chars.", "Paraphrase in original words."));
        }

        var sev = issues.Select(i => i.Severity).ToHashSet();
        var verdict = sev.Contains("block") ? SafetyVerdict.Block
                    : sev.Contains("high") ? SafetyVerdict.Revise
                    : SafetyVerdict.Pass;

        var checkedFields = fields.ToDictionary(kv => kv.Key, kv => (object?)(kv.Value?.Length ?? 0));
        return new GateResult(verdict, issues, checkedFields);
    }

    private static Dictionary<string, string?> Collect(Project project, IReadOnlyList<Scene> scenes)
    {
        var f = new Dictionary<string, string?>();
        if (!string.IsNullOrEmpty(project.Title)) f["title"] = project.Title;
        if (!string.IsNullOrEmpty(project.Context)) f["context"] = project.Context;
        if (!string.IsNullOrEmpty(project.Lyrics)) f["lyrics"] = project.Lyrics;
        if (!string.IsNullOrEmpty(project.MusicStyle)) f["music_style"] = project.MusicStyle;
        if (!string.IsNullOrEmpty(project.NarrationScript)) f["narration"] = project.NarrationScript;
        foreach (var s in scenes)
        {
            f[$"scene {s.SceneNumber} prompt"] = s.Prompt;
            if (!string.IsNullOrEmpty(s.NarrationText)) f[$"scene {s.SceneNumber} narration"] = s.NarrationText;
        }
        return f;
    }

    // Whole-word for single tokens that aren't substrings (matches Python heuristic).
    private static IEnumerable<string> Scan(string text, string[] terms, bool wholeWord = true)
    {
        var low = text.ToLowerInvariant();
        foreach (var t in terms)
        {
            var single = !t.Contains(' ') && !t.EndsWith("at") && !t.EndsWith("ing");
            if (wholeWord && single)
            {
                if (Regex.IsMatch(low, $@"\b{Regex.Escape(t)}\b")) yield return t;
            }
            else if (low.Contains(t)) yield return t;
        }
    }
}
