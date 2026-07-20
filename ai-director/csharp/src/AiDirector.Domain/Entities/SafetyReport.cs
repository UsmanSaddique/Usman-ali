using AiDirector.Domain.Enums;
using AiDirector.Domain.ValueObjects;

namespace AiDirector.Domain.Entities;

/// Maps table "safety_reports" (app/database.py SafetyReport).
/// YT-policy safety gate result. One row per gate run, per project. The LATEST
/// row is authoritative: start-generation refuses to run unless its verdict is
/// pass/override (override = recorded human sign-off).
public class SafetyReport
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string ProjectId { get; set; } = null!;
    public SafetyVerdict Verdict { get; set; }
    public List<SafetyIssue> Issues { get; set; } = [];
    public Dictionary<string, object?> CheckedFields { get; set; } = [];   // what was scanned
    public List<AutoRevision> AutoRevisions { get; set; } = [];            // applied by the gate
    public string? OverrideNote { get; set; }                              // human reason when verdict=override
    public bool LlmUsed { get; set; }                                      // LLM critic layer ran (vs rules-only)
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}
