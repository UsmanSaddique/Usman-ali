namespace AiDirector.Application.Configuration;

/// Strongly-typed config mirroring app/config.py. Bound from appsettings.json
/// (section "AiDirector") + AIDIR_* env vars in the WebApi composition root.
public sealed class AiDirectorOptions
{
    public const string SectionName = "AiDirector";

    public string AppName { get; set; } = "AI Director";
    public string Host { get; set; } = "0.0.0.0";
    public int Port { get; set; } = 8000;
    public bool AutoResume { get; set; }        // recovered projects wait for explicit Resume

    public PathsOptions Paths { get; set; } = new();
    public ComfyUiOptions ComfyUi { get; set; } = new();
    public VideoOptions Video { get; set; } = new();
    public ImageOptions Image { get; set; } = new();
    public UpscaleOptions Upscale { get; set; } = new();
    public GenerationOptions Generation { get; set; } = new();
}

public sealed class PathsOptions
{
    public string ComfyRoot { get; set; } =
        @"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable";
    public string ModelsDir { get; set; } =
        @"C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI\models";
    public string AssetsDir { get; set; } = "assets_generated";
    public string ProjectsDir { get; set; } = "projects";
    public string ChannelsDir { get; set; } = "channels";
    public string Database { get; set; } = "ai_director.db";
    public string FfmpegBin { get; set; } = "ffmpeg";
    public string FfprobeBin { get; set; } = "ffprobe";

    public string ComfyOutput => System.IO.Path.Combine(ComfyRoot, "ComfyUI", "output");
    public string ComfyInput => System.IO.Path.Combine(ComfyRoot, "ComfyUI", "input");
    public string ComfyPython => System.IO.Path.Combine(ComfyRoot, "python_embeded", "python.exe");
    public string ComfyMain => System.IO.Path.Combine(ComfyRoot, "ComfyUI", "main.py");
}

public sealed class ComfyUiOptions
{
    public string BaseUrl { get; set; } = "http://127.0.0.1:8188";
    public double ColdStartTimeoutSec { get; set; } = 240.0;
    public int SubmitTimeoutSec { get; set; } = 30;
    public int CompletionTimeoutSec { get; set; } = 600;
    public double PollSec { get; set; } = 2.0;
    public bool AutoLaunch { get; set; } = true;
}

public sealed class VideoOptions
{
    public string Model { get; set; } = "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf";
    public int DefaultWidth { get; set; } = 768;
    public int DefaultHeight { get; set; } = 512;
    public int DefaultNumFrames { get; set; } = 121;   // 5.04s @ 24fps, VRAM-safe on 16GB
    public int DefaultFps { get; set; } = 24;
    public int DefaultSteps { get; set; } = 8;
    public double DefaultCfg { get; set; } = 1.0;
    public int MaxNumFrames { get; set; } = 121;
    public double Img2VidStrength { get; set; } = 0.75;
    public double PremiumOpenSeconds { get; set; } = 20.0;
    public int PremiumWidth { get; set; } = 960;
    public int PremiumHeight { get; set; } = 544;
    public int PremiumSteps { get; set; } = 16;
}

public sealed class ImageOptions
{
    public string Engine { get; set; } = "zimage";       // "zimage" | "sdxl"
    public string ZimageUnet { get; set; } = "z_image_turbo_bf16.safetensors";
    public int ZimageSteps { get; set; } = 9;
    public double ZimageShift { get; set; } = 3.0;
    public int ZimageMinHeight { get; set; } = 720;
    public int DefaultWidth { get; set; } = 1280;
    public int DefaultHeight { get; set; } = 720;
}

public sealed class UpscaleOptions
{
    public string ModelName { get; set; } = "4x-UltraSharp.pth";
    public string AnimeModelName { get; set; } = "RealESRGAN_x4plus_anime_6B.pth";
    public bool UseAnimeModel { get; set; } = true;
    public string Target { get; set; } = "1080p";
}

public sealed class GenerationOptions
{
    public int MaxRetries { get; set; } = 3;
    public double TransitionDuration { get; set; } = 0.5;   // crossfade seconds
    public bool SaveIntermediate { get; set; } = true;
}
