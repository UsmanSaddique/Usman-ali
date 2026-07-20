using System.Text.Json.Nodes;
using AiDirector.Application.Abstractions;

namespace AiDirector.Infrastructure.ComfyUi;

/// Port of the build_* workflow constructors in comfyui_client.py. Builds
/// API-format ComfyUI graphs as JsonObject. Node ids are sequential strings,
/// matching the Python numbering so submissions are byte-comparable.
public sealed class WorkflowBuilder : IWorkflowBuilder
{
    private sealed class Graph
    {
        public readonly JsonObject Root = new();
        private int _n;
        public string Nid() => (++_n).ToString();
        public void Add(string id, string classType, JsonObject inputs) =>
            Root[id] = new JsonObject { ["class_type"] = classType, ["inputs"] = inputs };
    }

    public JsonObject ZImage(string prompt, int width, int height, int steps, double cfg,
        long seed, double shift, string outputPrefix = "zimage")
    {
        var g = new Graph();
        var unet = g.Nid(); g.Add(unet, "UNETLoader",
            new() { ["unet_name"] = "z_image_turbo_bf16.safetensors", ["weight_dtype"] = "default" });
        var ms = g.Nid(); g.Add(ms, "ModelSamplingAuraFlow",
            new() { ["model"] = Ref(unet, 0), ["shift"] = shift });
        var clip = g.Nid(); g.Add(clip, "CLIPLoader",
            new() { ["clip_name"] = "qwen_3_4b.safetensors", ["type"] = "lumina2", ["device"] = "default" });
        var vae = g.Nid(); g.Add(vae, "VAELoader", new() { ["vae_name"] = "z_image_ae.safetensors" });
        var pos = g.Nid(); g.Add(pos, "CLIPTextEncode", new() { ["text"] = prompt, ["clip"] = Ref(clip, 0) });
        var zero = g.Nid(); g.Add(zero, "ConditioningZeroOut", new() { ["conditioning"] = Ref(pos, 0) });
        var lat = g.Nid(); g.Add(lat, "EmptySD3LatentImage",
            new() { ["width"] = width, ["height"] = height, ["batch_size"] = 1 });
        var samp = g.Nid(); g.Add(samp, "KSampler", new()
        {
            ["model"] = Ref(ms, 0), ["positive"] = Ref(pos, 0), ["negative"] = Ref(zero, 0),
            ["latent_image"] = Ref(lat, 0), ["seed"] = seed, ["steps"] = steps, ["cfg"] = cfg,
            ["sampler_name"] = "res_multistep", ["scheduler"] = "simple", ["denoise"] = 1.0,
        });
        var dec = g.Nid(); g.Add(dec, "VAEDecode", new() { ["samples"] = Ref(samp, 0), ["vae"] = Ref(vae, 0) });
        var save = g.Nid(); g.Add(save, "SaveImage",
            new() { ["images"] = Ref(dec, 0), ["filename_prefix"] = outputPrefix });
        return g.Root;
    }

    public JsonObject LtxText2Video(string model, string prompt, string negativePrompt,
        int width, int height, int numFrames, int steps, double cfg, long seed, int fps,
        IReadOnlyList<(string File, double Weight)>? loras = null, string outputPrefix = "ai_director")
    {
        var g = new Graph();
        var isGguf = model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
        var modelNode = g.Nid();
        g.Add(modelNode, isGguf ? "UnetLoaderGGUF" : "UNETLoader",
            isGguf ? new() { ["unet_name"] = model }
                   : new() { ["unet_name"] = model, ["weight_dtype"] = "default" });
        var clip = g.Nid(); g.Add(clip, "DualCLIPLoader", new()
        {
            ["clip_name1"] = "gemma_3_12B_it_fp4_mixed.safetensors",
            ["clip_name2"] = "ltx-2.3_text_projection_bf16.safetensors", ["type"] = "ltxv",
        });
        var vae = g.Nid(); g.Add(vae, "VAELoader", new() { ["vae_name"] = "LTX23_video_vae_bf16.safetensors" });

        var (curModel, curClip) = ApplyLoras(g, Ref(modelNode, 0), Ref(clip, 0), loras);

        var pos = g.Nid(); g.Add(pos, "CLIPTextEncode", new() { ["text"] = prompt, ["clip"] = curClip.DeepClone() });
        var neg = g.Nid(); g.Add(neg, "CLIPTextEncode", new() { ["text"] = negativePrompt, ["clip"] = curClip.DeepClone() });
        var lat = g.Nid(); g.Add(lat, "EmptyLTXVLatentVideo",
            new() { ["width"] = width, ["height"] = height, ["length"] = numFrames, ["batch_size"] = 1 });
        var samp = g.Nid(); g.Add(samp, "KSampler", new()
        {
            ["model"] = curModel.DeepClone(), ["positive"] = Ref(pos, 0), ["negative"] = Ref(neg, 0),
            ["latent_image"] = Ref(lat, 0), ["seed"] = seed, ["steps"] = steps, ["cfg"] = cfg,
            ["sampler_name"] = "euler", ["scheduler"] = "normal", ["denoise"] = 1.0,
        });
        var dec = g.Nid(); g.Add(dec, "VAEDecode", new() { ["samples"] = Ref(samp, 0), ["vae"] = Ref(vae, 0) });
        AddVideoCombine(g, Ref(dec, 0), fps, outputPrefix);
        return g.Root;
    }

    public JsonObject LtxImage2Video(string model, string imageFilename, string prompt, string negativePrompt,
        int width, int height, int numFrames, int steps, double cfg, long seed, int fps, double strength,
        IReadOnlyList<(string File, double Weight)>? loras = null, string outputPrefix = "ai_director_i2v")
    {
        var g = new Graph();
        var isGguf = model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
        var modelNode = g.Nid();
        g.Add(modelNode, isGguf ? "UnetLoaderGGUF" : "UNETLoader",
            isGguf ? new() { ["unet_name"] = model }
                   : new() { ["unet_name"] = model, ["weight_dtype"] = "default" });
        var clip = g.Nid(); g.Add(clip, "DualCLIPLoader", new()
        {
            ["clip_name1"] = "gemma_3_12B_it_fp4_mixed.safetensors",
            ["clip_name2"] = "ltx-2.3_text_projection_bf16.safetensors", ["type"] = "ltxv",
        });
        var vae = g.Nid(); g.Add(vae, "VAELoader", new() { ["vae_name"] = "LTX23_video_vae_bf16.safetensors" });

        var (curModel, curClip) = ApplyLoras(g, Ref(modelNode, 0), Ref(clip, 0), loras);

        var pos = g.Nid(); g.Add(pos, "CLIPTextEncode", new() { ["text"] = prompt, ["clip"] = curClip.DeepClone() });
        var neg = g.Nid(); g.Add(neg, "CLIPTextEncode", new() { ["text"] = negativePrompt, ["clip"] = curClip.DeepClone() });
        var img = g.Nid(); g.Add(img, "LoadImage", new() { ["image"] = imageFilename });
        var scaled = g.Nid(); g.Add(scaled, "ImageScale", new()
        {
            ["image"] = Ref(img, 0), ["upscale_method"] = "lanczos",
            ["width"] = width, ["height"] = height, ["crop"] = "disabled",
        });
        var i2v = g.Nid(); g.Add(i2v, "LTXVImgToVideo", new()
        {
            ["positive"] = Ref(pos, 0), ["negative"] = Ref(neg, 0), ["vae"] = Ref(vae, 0),
            ["image"] = Ref(scaled, 0), ["width"] = width, ["height"] = height,
            ["length"] = numFrames, ["batch_size"] = 1, ["strength"] = strength,
        });
        var samp = g.Nid(); g.Add(samp, "KSampler", new()
        {
            ["model"] = curModel.DeepClone(), ["positive"] = Ref(i2v, 0), ["negative"] = Ref(i2v, 1),
            ["latent_image"] = Ref(i2v, 2), ["seed"] = seed, ["steps"] = steps, ["cfg"] = cfg,
            ["sampler_name"] = "euler", ["scheduler"] = "normal", ["denoise"] = 1.0,
        });
        var dec = g.Nid(); g.Add(dec, "VAEDecode", new() { ["samples"] = Ref(samp, 0), ["vae"] = Ref(vae, 0) });
        AddVideoCombine(g, Ref(dec, 0), fps, outputPrefix);
        return g.Root;
    }

    public JsonObject EsrganVideoUpscale(string videoFilename, int targetWidth, int targetHeight,
        double fps, string modelName, string outputPrefix = "ai_director_hd")
    {
        var wf = new JsonObject
        {
            ["1"] = Node("VHS_LoadVideo", new()
            {
                ["video"] = videoFilename, ["force_rate"] = 0, ["custom_width"] = 0, ["custom_height"] = 0,
                ["frame_load_cap"] = 0, ["skip_first_frames"] = 0, ["select_every_nth"] = 1,
            }),
            ["2"] = Node("UpscaleModelLoader", new() { ["model_name"] = modelName }),
            ["3"] = Node("ImageUpscaleWithModel", new() { ["upscale_model"] = Ref("2", 0), ["image"] = Ref("1", 0) }),
            ["4"] = Node("ImageScale", new()
            {
                ["image"] = Ref("3", 0), ["upscale_method"] = "lanczos",
                ["width"] = targetWidth, ["height"] = targetHeight, ["crop"] = "disabled",
            }),
            ["5"] = Node("VHS_VideoCombine", new()
            {
                ["images"] = Ref("4", 0), ["frame_rate"] = fps, ["loop_count"] = 0,
                ["filename_prefix"] = outputPrefix, ["format"] = "video/h264-mp4",
                ["pingpong"] = false, ["save_output"] = true,
            }),
        };
        return wf;
    }

    public JsonObject AceStep15Xl(string styleTags, string lyrics, double seconds, long seed,
        int steps, double cfg, int bpm, string language, string keyscale,
        string unetName = "acestep_v1.5_xl_turbo_bf16.safetensors",
        string weightDtype = "default",
        string outputPrefix = "ai_director_music")
    {
        return new JsonObject
        {
            ["1"] = Node("UNETLoader", new()
                { ["unet_name"] = unetName, ["weight_dtype"] = weightDtype }),
            ["2"] = Node("DualCLIPLoader", new()
                { ["clip_name1"] = "qwen_0.6b_ace15.safetensors", ["clip_name2"] = "qwen_1.7b_ace15.safetensors", ["type"] = "ace" }),
            ["3"] = Node("VAELoader", new() { ["vae_name"] = "ace_1.5_vae.safetensors" }),
            ["4"] = Node("TextEncodeAceStepAudio1.5", new()
            {
                ["clip"] = Ref("2", 0), ["tags"] = styleTags, ["lyrics"] = lyrics, ["seed"] = seed,
                ["bpm"] = bpm, ["duration"] = seconds, ["timesignature"] = "4", ["language"] = language,
                ["keyscale"] = keyscale, ["generate_audio_codes"] = true,
                ["cfg_scale"] = 2.5, ["temperature"] = 0.75, ["top_p"] = 0.85, ["top_k"] = 0, ["min_p"] = 0.0,
            }),
            ["5"] = Node("EmptyAceStep1.5LatentAudio", new() { ["seconds"] = seconds, ["batch_size"] = 1 }),
            ["6"] = Node("KSampler", new()
            {
                ["model"] = Ref("1", 0), ["positive"] = Ref("4", 0), ["negative"] = Ref("4", 0),
                ["latent_image"] = Ref("5", 0), ["seed"] = seed, ["steps"] = steps, ["cfg"] = cfg,
                ["sampler_name"] = "euler", ["scheduler"] = "simple", ["denoise"] = 1.0,
            }),
            ["7"] = Node("VAEDecodeAudio", new() { ["samples"] = Ref("6", 0), ["vae"] = Ref("3", 0) }),
            ["8"] = Node("SaveAudio", new() { ["audio"] = Ref("7", 0), ["filename_prefix"] = outputPrefix }),
        };
    }

    // ── helpers ──────────────────────────────────────────────────────────
    private static (JsonArray Model, JsonArray Clip) ApplyLoras(
        Graph g, JsonArray model, JsonArray clip, IReadOnlyList<(string File, double Weight)>? loras)
    {
        var curModel = model; var curClip = clip;
        foreach (var (file, weight) in loras ?? [])
        {
            var ln = g.Nid();
            g.Add(ln, "LoraLoader", new()
            {
                ["model"] = curModel.DeepClone(), ["clip"] = curClip.DeepClone(),
                ["lora_name"] = file, ["strength_model"] = weight, ["strength_clip"] = weight,
            });
            curModel = Ref(ln, 0); curClip = Ref(ln, 1);
        }
        return (curModel, curClip);
    }

    private static void AddVideoCombine(Graph g, JsonArray images, int fps, string prefix)
    {
        var save = g.Nid();
        g.Add(save, "VHS_VideoCombine", new()
        {
            ["images"] = images, ["frame_rate"] = fps, ["loop_count"] = 0,
            ["filename_prefix"] = prefix, ["format"] = "video/h264-mp4",
            ["save_output"] = true, ["pingpong"] = false,
        });
    }

    private static JsonArray Ref(string node, int slot) => new(node, slot);
    private static JsonObject Node(string classType, JsonObject inputs) =>
        new() { ["class_type"] = classType, ["inputs"] = inputs };
}
