using AiDirector.Infrastructure.ComfyUi;
using FluentAssertions;

namespace AiDirector.Infrastructure.Tests.ComfyUi;

/// Structural parity checks against the Python build_* functions in
/// comfyui_client.py. Full deep-equality vs recorded submissions comes in
/// Phase 2 (golden fixtures); these lock the node graph shape now.
public sealed class WorkflowBuilderTests
{
    private readonly WorkflowBuilder _b = new();

    [Fact]
    public void ZImage_matches_python_recipe()
    {
        var wf = _b.ZImage("a cat", 1024, 1024, 8, 1.0, 42, 3.0);

        // 10 nodes: UNETLoader, ModelSamplingAuraFlow, CLIPLoader, VAELoader,
        // CLIPTextEncode, ConditioningZeroOut, EmptySD3LatentImage, KSampler,
        // VAEDecode, SaveImage.
        wf.Count.Should().Be(10);
        ClassTypes(wf).Should().Contain(new[]
        {
            "UNETLoader", "ModelSamplingAuraFlow", "CLIPLoader", "VAELoader",
            "ConditioningZeroOut", "EmptySD3LatentImage", "KSampler", "SaveImage",
        });

        // KSampler uses the turbo recipe: res_multistep / simple / cfg 1.
        var ksampler = wf.First(kv => wf[kv.Key]!["class_type"]!.GetValue<string>() == "KSampler");
        var inputs = wf[ksampler.Key]!["inputs"]!;
        inputs["sampler_name"]!.GetValue<string>().Should().Be("res_multistep");
        inputs["scheduler"]!.GetValue<string>().Should().Be("simple");
        inputs["cfg"]!.GetValue<double>().Should().Be(1.0);
    }

    [Fact]
    public void LtxImage2Video_uses_gguf_loader_and_i2v_node()
    {
        var wf = _b.LtxImage2Video("LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf", "still.png",
            "prompt", "", 832, 480, 121, 8, 1.0, 42, 24, 0.75);

        var types = ClassTypes(wf);
        types.Should().Contain("UnetLoaderGGUF");   // .gguf model -> GGUF loader
        types.Should().Contain("LTXVImgToVideo");   // image-to-video node
        types.Should().Contain("ImageScale");       // still downscaled to clip res
        types.Should().Contain("VHS_VideoCombine"); // mp4 output
    }

    [Fact]
    public void Loras_are_chained_onto_model_and_clip()
    {
        var wf = _b.LtxText2Video("model.gguf", "p", "", 768, 512, 97, 8, 1.0, 1, 24,
            loras: [("style.safetensors", 0.8), ("char.safetensors", 0.7)]);
        ClassTypes(wf).Count(t => t == "LoraLoader").Should().Be(2);
    }

    private static List<string> ClassTypes(System.Text.Json.Nodes.JsonObject wf) =>
        wf.Select(kv => wf[kv.Key]!["class_type"]!.GetValue<string>()).ToList();
}
