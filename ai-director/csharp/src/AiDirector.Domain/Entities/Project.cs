using AiDirector.Domain.Enums;

namespace AiDirector.Domain.Entities;

/// Maps table "projects" (app/database.py Project).
public class Project
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Title { get; set; } = null!;
    public string ChannelId { get; set; } = null!;
    public string ProjectType { get; set; } = "song";        // "song" | "narration" (lane; chosen by archetype)
    public string? ContentArchetype { get; set; }            // archetype id override; null=inherit channel
    public bool Reviewed { get; set; }                       // human script sign-off (Tier-2 gate); only the approve endpoint sets it
    public int DurationTarget { get; set; }                  // seconds

    public string? Context { get; set; }                     // user-provided notes
    public string? NarrationScript { get; set; }             // narration mode: JSON {chapters, seo}
    public string? NarrationVoice { get; set; }              // kokoro voice id or engine override
    public string? NarrationAudioPath { get; set; }          // master narration WAV
    public string? Lyrics { get; set; }                      // drives music vocals + scene timing
    public string? LyricsUrdu { get; set; }                  // Hindi/Urdu lyric version
    public string? MusicStyle { get; set; }                  // style tags for the music engine
    public string MusicModel { get; set; } = "auto";         // auto|sft|turbo|heartmula|ace1
    public bool UpscaleInline { get; set; } = true;          // upscale each clip after generation

    // "clips" = one 5-6s ComfyUI job per scene; "ltx_director" = multi-director
    // single workflow (20-30s per node, native audio, reference stills per segment).
    public string VideoEngine { get; set; } = "clips";
    public int? NumScenesTarget { get; set; }                // user-customized target scene count
    public string VideoModel { get; set; } = "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf";
    public List<string> DefaultLoraIds { get; set; } = [];   // ["lora_file.safetensors", ...]
    public List<double> DefaultLoraWeights { get; set; } = [];

    public string? ScriptRaw { get; set; }                   // raw LLM output
    public ProjectStatus Status { get; set; } = ProjectStatus.Draft;
    public int TotalScenes { get; set; }
    public int CompletedScenes { get; set; }
    public string? OutputPath { get; set; }                  // final rendered video
    public string? ThumbnailPath { get; set; }
    public string? ErrorLog { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    public Channel Channel { get; set; } = null!;
    public List<Scene> Scenes { get; set; } = [];
    public List<MusicTrack> MusicTracks { get; set; } = [];
    public List<RenderJob> RenderJobs { get; set; } = [];
}
