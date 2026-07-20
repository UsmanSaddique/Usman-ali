using AiDirector.Domain.Enums;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;

namespace AiDirector.Infrastructure.Persistence.Conversions;

/// EF Core value converters that store each enum exactly as the Python /
/// SQLAlchemy app does: the enum's UPPERCASE member NAME (e.g. "STILL_PAN",
/// "TXT2VID", "DRAFT"), which is what the SAEnum column actually contains.
///
/// Reads are deliberately tolerant: the live ai_director.db has mixed rows —
/// most hold the Python name ("PENDING") but some hold the lowercase value
/// ("draft"). The reverse map therefore accepts BOTH spellings, case-insensitive.
public static class PythonEnumConversions
{
    // (enum value -> exact Python member name written to the DB)
    private static readonly (ProjectStatus V, string Name)[] ProjectStatusMap =
    [
        (ProjectStatus.Draft, "DRAFT"), (ProjectStatus.Scripted, "SCRIPTED"),
        (ProjectStatus.Approved, "APPROVED"), (ProjectStatus.Generating, "GENERATING"),
        (ProjectStatus.Generated, "GENERATED"), (ProjectStatus.Upscaling, "UPSCALING"),
        (ProjectStatus.Music, "MUSIC"), (ProjectStatus.Assembling, "ASSEMBLING"),
        (ProjectStatus.Rendered, "RENDERED"), (ProjectStatus.Uploaded, "UPLOADED"),
        (ProjectStatus.Failed, "FAILED"),
    ];

    private static readonly (SceneType V, string Name)[] SceneTypeMap =
    [
        (SceneType.Txt2Vid, "TXT2VID"), (SceneType.Img2Vid, "IMG2VID"),
        (SceneType.StillPan, "STILL_PAN"), (SceneType.NarrationOnly, "NARRATION_ONLY"),
        (SceneType.Template, "TEMPLATE"), (SceneType.UserAsset, "USER_ASSET"),
    ];

    private static readonly (SceneStatus V, string Name)[] SceneStatusMap =
    [
        (SceneStatus.Draft, "DRAFT"), (SceneStatus.Pending, "PENDING"),
        (SceneStatus.Queued, "QUEUED"), (SceneStatus.Generating, "GENERATING"),
        (SceneStatus.Generated, "GENERATED"), (SceneStatus.Approved, "APPROVED"),
        (SceneStatus.Failed, "FAILED"), (SceneStatus.Skipped, "SKIPPED"),
    ];

    private static readonly (GenerationStatus V, string Name)[] GenerationStatusMap =
    [
        (GenerationStatus.Queued, "QUEUED"), (GenerationStatus.Running, "RUNNING"),
        (GenerationStatus.Completed, "COMPLETED"), (GenerationStatus.Failed, "FAILED"),
        (GenerationStatus.Superseded, "SUPERSEDED"),
    ];

    private static readonly (RenderStatus V, string Name)[] RenderStatusMap =
    [
        (RenderStatus.Queued, "QUEUED"), (RenderStatus.Rendering, "RENDERING"),
        (RenderStatus.Completed, "COMPLETED"), (RenderStatus.Failed, "FAILED"),
    ];

    private static readonly (SafetyVerdict V, string Name)[] SafetyVerdictMap =
    [
        (SafetyVerdict.Pass, "PASS"), (SafetyVerdict.Revise, "REVISE"),
        (SafetyVerdict.Block, "BLOCK"), (SafetyVerdict.Override, "OVERRIDE"),
    ];

    public static ValueConverter<ProjectStatus, string> ProjectStatusConverter() => Build(ProjectStatusMap);
    public static ValueConverter<SceneType, string> SceneTypeConverter() => Build(SceneTypeMap);
    public static ValueConverter<SceneStatus, string> SceneStatusConverter() => Build(SceneStatusMap);
    public static ValueConverter<GenerationStatus, string> GenerationStatusConverter() => Build(GenerationStatusMap);
    public static ValueConverter<RenderStatus, string> RenderStatusConverter() => Build(RenderStatusMap);
    public static ValueConverter<SafetyVerdict, string> SafetyVerdictConverter() => Build(SafetyVerdictMap);

    private static ValueConverter<TEnum, string> Build<TEnum>((TEnum V, string Name)[] map)
        where TEnum : struct, Enum
    {
        var toDb = map.ToDictionary(x => x.V, x => x.Name);

        // Reverse map keyed case-insensitively, accepting the Python NAME
        // ("PENDING"), the lowercase value ("pending"/"draft"), and — since
        // some values differ from the lowercased name (NARRATION_ONLY vs
        // "narration") — both forms are registered explicitly below.
        var fromDb = new Dictionary<string, TEnum>(StringComparer.OrdinalIgnoreCase);
        foreach (var (v, name) in map)
        {
            fromDb[name] = v;              // "STILL_PAN"
            fromDb[name.ToLowerInvariant()] = v; // "still_pan"
        }

        return new ValueConverter<TEnum, string>(
            v => toDb[v],
            s => FromDb(fromDb, s));
    }

    private static TEnum FromDb<TEnum>(Dictionary<string, TEnum> map, string s)
        where TEnum : struct, Enum
        => map.TryGetValue(s, out var e)
            ? e
            : throw new InvalidOperationException(
                $"Unmapped {typeof(TEnum).Name} value in DB: '{s}'");
}
