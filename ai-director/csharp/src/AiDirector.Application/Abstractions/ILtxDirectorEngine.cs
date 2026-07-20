using AiDirector.Domain.Entities;

namespace AiDirector.Application.Abstractions;

/// One LTX Director segment: a reference still + rich prompt (+ optional
/// spoken line) rendered as part of one continuous native-audio video.
public sealed record LtxSegment(string Prompt, string Dialogue, string ImagePath, double Seconds);

/// Port of app/services/ltx_director.py — the "ltx_director" video engine.
/// Renders the whole project as one long video via chained LTXDirector nodes,
/// ONE DIRECTOR PER CHUNK, each director's output saved to
/// projects/<id>/ltx_parts/part_XX.mp4 the moment it finishes. A crash or
/// power cut only costs the director that was mid-render: resume skips every
/// finished part instead of restarting from director 1.
public interface ILtxDirectorEngine
{
    /// Missing model files, if any (empty = ready to run).
    IReadOnlyList<string> CheckModels();

    /// Full run for a project: renders (or resumes) all director parts,
    /// concats them, upscales to 1080p and returns the final video path.
    Task<string> GenerateForProjectAsync(Project project,
        IReadOnlyList<LtxSegment> segments, CancellationToken ct = default);
}
