using AiDirector.Application.Abstractions;
using AiDirector.Application.Configuration;
using AiDirector.Application.Music;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace AiDirector.Infrastructure.Audio;

/// Port of the ACE-Step path in app/services/music_gen.py. Generates a song via
/// the ComfyUI ACE-Step 1.5 XL workflow and collects the WAV. The vocal-clarity
/// levers (cfg 2.5 / temp 0.75, style passthrough) live in the workflow builder.
public sealed class MusicEngine(
    IComfyUiClient comfy,
    IWorkflowBuilder workflows,
    IOptions<AiDirectorOptions> options,
    ILogger<MusicEngine> log)
{
    private readonly AiDirectorOptions _o = options.Value;

    public async Task<string> GenerateAsync(string projectId, string styleTags, string lyrics,
        double seconds, int bpm = 90, string language = "en", string keyscale = "C major",
        long? seed = null, string variantLabel = "", CancellationToken ct = default)
    {
        if (!await comfy.WaitReadyAsync(_o.ComfyUi.ColdStartTimeoutSec, _o.ComfyUi.AutoLaunch, ct))
            throw new InvalidOperationException("ComfyUI not ready for music generation");

        var s = seed ?? Random.Shared.NextInt64(1, int.MaxValue);
        // User-mandated: every ACE-Step generation rides in the master producer
        // brief, with the caller's style string verbatim in the Genre slot. An
        // explicit "NNN bpm" in the brief beats the caller's coarse default.
        var effectiveBpm = AceStepPrompt.ExplicitBpm(styleTags, bpm);
        var masterPrompt = AceStepPrompt.Build(styleTags, effectiveBpm,
            instrumental: string.IsNullOrWhiteSpace(lyrics));
        // Unique prefix per variant so multiple songs don't overwrite each other.
        var suffix = string.IsNullOrEmpty(variantLabel) ? "" : $"_{variantLabel}";
        var prefix = $"aidir_{projectId[..8]}_music{suffix}";
        // Use the SFT model (what the Python app uses): the turbo weights on this
        // box are an incomplete download (.tmp sibling) and error with a shape
        // mismatch. SFT is 18.6GB bf16 -> load as fp8 to fit 16GB, 50 steps.
        var wf = workflows.AceStep15Xl(masterPrompt, lyrics, seconds, s,
            steps: 50, cfg: 1.0, bpm: effectiveBpm, language: language, keyscale: keyscale,
            unetName: "acestep_v1.5_xl_sft_bf16.safetensors", weightDtype: "fp8_e4m3fn",
            outputPrefix: prefix);

        var promptId = await comfy.SubmitAsync(wf, ct);
        var hist = await comfy.WaitForCompletionAsync(promptId, ct: ct);

        var dest = Path.Combine(_o.Paths.AssetsDir, projectId, $"{prefix}.wav");
        await comfy.CollectOutputAsync(hist, dest, ct);
        log.LogInformation("[Music] Generated {Path} ({Sec}s)", dest, seconds);
        return dest;
    }
}
