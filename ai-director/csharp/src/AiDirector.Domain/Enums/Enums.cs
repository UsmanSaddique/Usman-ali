using System.Text.Json.Serialization;

namespace AiDirector.Domain.Enums;

// String-backed enums mirroring app/database.py. JsonStringEnumConverter +
// EF Core value conversion (configured in Infrastructure) persist these as the
// exact lowercase strings the Python app writes, so the same ai_director.db and
// the same API payloads keep working.

[JsonConverter(typeof(JsonStringEnumConverter<ProjectStatus>))]
public enum ProjectStatus
{
    [JsonStringEnumMemberName("draft")] Draft,
    [JsonStringEnumMemberName("scripted")] Scripted,
    [JsonStringEnumMemberName("approved")] Approved,
    [JsonStringEnumMemberName("generating")] Generating,
    [JsonStringEnumMemberName("generated")] Generated,
    [JsonStringEnumMemberName("upscaling")] Upscaling,
    [JsonStringEnumMemberName("music")] Music,
    [JsonStringEnumMemberName("assembling")] Assembling,
    [JsonStringEnumMemberName("rendered")] Rendered,
    [JsonStringEnumMemberName("uploaded")] Uploaded,
    [JsonStringEnumMemberName("failed")] Failed,
}

[JsonConverter(typeof(JsonStringEnumConverter<SceneType>))]
public enum SceneType
{
    [JsonStringEnumMemberName("txt2vid")] Txt2Vid,
    [JsonStringEnumMemberName("img2vid")] Img2Vid,
    [JsonStringEnumMemberName("still_pan")] StillPan,        // Ken Burns effect on still image
    [JsonStringEnumMemberName("narration")] NarrationOnly,   // black/gradient screen with narration
    [JsonStringEnumMemberName("template")] Template,         // headless-browser rendered
    [JsonStringEnumMemberName("user_asset")] UserAsset,      // user-provided file from assets_in/
}

[JsonConverter(typeof(JsonStringEnumConverter<SceneStatus>))]
public enum SceneStatus
{
    [JsonStringEnumMemberName("draft")] Draft,
    [JsonStringEnumMemberName("pending")] Pending,
    [JsonStringEnumMemberName("queued")] Queued,
    [JsonStringEnumMemberName("generating")] Generating,
    [JsonStringEnumMemberName("generated")] Generated,
    [JsonStringEnumMemberName("approved")] Approved,
    [JsonStringEnumMemberName("failed")] Failed,
    [JsonStringEnumMemberName("skipped")] Skipped,
}

[JsonConverter(typeof(JsonStringEnumConverter<GenerationStatus>))]
public enum GenerationStatus
{
    [JsonStringEnumMemberName("queued")] Queued,
    [JsonStringEnumMemberName("running")] Running,
    [JsonStringEnumMemberName("completed")] Completed,
    [JsonStringEnumMemberName("failed")] Failed,
    [JsonStringEnumMemberName("superseded")] Superseded,   // replaced by a newer version
}

[JsonConverter(typeof(JsonStringEnumConverter<RenderStatus>))]
public enum RenderStatus
{
    [JsonStringEnumMemberName("queued")] Queued,
    [JsonStringEnumMemberName("rendering")] Rendering,
    [JsonStringEnumMemberName("completed")] Completed,
    [JsonStringEnumMemberName("failed")] Failed,
}

[JsonConverter(typeof(JsonStringEnumConverter<SafetyVerdict>))]
public enum SafetyVerdict
{
    [JsonStringEnumMemberName("pass")] Pass,
    [JsonStringEnumMemberName("revise")] Revise,
    [JsonStringEnumMemberName("block")] Block,
    [JsonStringEnumMemberName("override")] Override,   // human signed off despite a non-pass verdict
}
