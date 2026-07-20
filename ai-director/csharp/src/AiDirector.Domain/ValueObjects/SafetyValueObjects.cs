namespace AiDirector.Domain.ValueObjects;

// Shapes taken from the JSON stored by app/services/yt_safety.py.

/// One flagged policy issue inside a SafetyReport.issues array.
public sealed record SafetyIssue(
    string Severity,     // e.g. "high" | "medium" | "low"
    string Category,     // policy bucket
    string Where,        // field the issue was found in (lyrics/narration/prompt/seo)
    string Detail,       // human-readable explanation
    string? Suggestion   // proposed fix, if any
);

/// One auto-applied revision recorded in SafetyReport.auto_revisions.
public sealed record AutoRevision(
    string Where,
    string Before,
    string After
);
