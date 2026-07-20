using System.Text.Json.Nodes;
using AiDirector.Application.Abstractions;
using AiDirector.Application.Configuration;
using AiDirector.Domain.Entities;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace AiDirector.Infrastructure.ComfyUi;

/// Port of app/services/ltx_director.py LTXDirectorService — the
/// "ltx_director" multi-segment engine: long continuous video with native
/// audio via chained LTXDirector nodes. Renders ONE DIRECTOR PER CHUNK and
/// saves each finished director to projects/<id>/ltx_parts/part_XX.mp4 so a
/// crash or power outage only costs the director that was mid-render.
public sealed class LtxDirectorEngine(
    IComfyUiClient comfy,
    IMediaRunner media,
    IOptions<AiDirectorOptions> options,
    ILogger<LtxDirectorEngine> log) : ILtxDirectorEngine
{
    private readonly AiDirectorOptions _o = options.Value;

    // GGUF the template actually loads (proven 720p config).
    private static readonly string[] DevGgufCandidates =
        ["LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf"];

    // Extra LoRAs chained into the director's MODEL path when present.
    private static readonly (string File, double Weight)[] ExtraLoras =
    [
        ("LTX2.3_CameraControls.safetensors", 0.7),
        ("LTX2.3_Crisp_Enhance.safetensors", 0.5),
    ];

    private static readonly (string Sub, string File)[] RequiredModels =
    [
        ("loras", "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors"),
        ("loras", "ltx2.3-transition.safetensors"),
        ("loras", "ltx-2-19b-ic-lora-detailer.safetensors"),
        ("vae", "LTX23_video_vae_bf16.safetensors"),
        ("vae", "LTX23_audio_vae_bf16.safetensors"),
        ("vae", "taeltx2_3.safetensors"),
        ("text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors"),
        ("text_encoders", "ltx-2.3_text_projection_bf16.safetensors"),
        ("latent_upscale_models", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
    ];

    public IReadOnlyList<string> CheckModels()
    {
        var missing = new List<string>();
        foreach (var (sub, file) in RequiredModels)
            if (!File.Exists(Path.Combine(_o.Paths.ModelsDir, sub, file)))
                missing.Add($"{sub}/{file}");
        if (!DevGgufCandidates.Any(c =>
                File.Exists(Path.Combine(_o.Paths.ModelsDir, "diffusion_models", c))))
            missing.Add($"diffusion_models/{DevGgufCandidates[0]}");
        return missing;
    }

    public async Task<string> GenerateForProjectAsync(Project project,
        IReadOnlyList<LtxSegment> segments, CancellationToken ct = default)
    {
        if (segments.Count < 2)
            throw new InvalidOperationException(
                "LTX Director needs at least 2 scenes with a still image each");

        var projLoras = new List<(string, double)>();
        for (var i = 0; i < project.DefaultLoraIds.Count; i++)
            projLoras.Add((Path.GetFileName(project.DefaultLoraIds[i]),
                i < project.DefaultLoraWeights.Count ? project.DefaultLoraWeights[i] : 0.7));

        var projectDir = Path.Combine(_o.Paths.ProjectsDir, project.Id);
        Directory.CreateDirectory(projectDir);
        var rawOut = Path.Combine(projectDir, "ltx_director_render.mp4");

        if (segments.Count <= _o.LtxDirector.ChunkSize)
        {
            // single director — one workflow, nothing to checkpoint between
            await GenerateAsync(segments, rawOut, projLoras, ct);
        }
        else
        {
            // MULTI-DIRECTOR: one director node per chunk, each saved to disk
            // as ltx_parts/part_XX.mp4 the moment it finishes. Resume skips
            // every finished part instead of restarting from director 1.
            await GenerateChunkedAsync(segments, projectDir, rawOut, projLoras, ct);
        }

        // Upscale to 1080p via FFmpeg Lanczos (parity with generate_ltx_director).
        var finalOut = Path.Combine(projectDir, "final_render.mp4");
        var up = await media.FfmpegAsync(
            ["-y", "-i", rawOut, "-vf", "scale=1920:1080:flags=lanczos",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-c:a", "copy", finalOut], ct);
        if (!up.Ok)
            throw new InvalidOperationException($"LTX Director 1080p upscale failed: {up.StdErr}");
        return finalOut;
    }

    // ── orchestration ──────────────────────────────────────────────────

    /// Full run of ONE workflow: stage images → build → submit → collect.
    private async Task GenerateAsync(IReadOnlyList<LtxSegment> segments,
        string outputPath, IReadOnlyList<(string File, double Weight)> extraLoras,
        CancellationToken ct)
    {
        var missing = CheckModels();
        if (missing.Count > 0)
            throw new InvalidOperationException(
                $"LTX Director models missing: {string.Join(", ", missing)}");

        if (!await comfy.WaitReadyAsync(_o.ComfyUi.ColdStartTimeoutSec, _o.ComfyUi.AutoLaunch, ct))
            throw new InvalidOperationException("ComfyUI is not running");

        // stage reference images into ComfyUI/input
        Directory.CreateDirectory(_o.Paths.ComfyInput);
        foreach (var seg in segments)
        {
            if (!File.Exists(seg.ImagePath))
                throw new FileNotFoundException($"Segment image missing: {seg.ImagePath}");
            var dst = Path.Combine(_o.Paths.ComfyInput, Path.GetFileName(seg.ImagePath));
            if (!File.Exists(dst) || new FileInfo(dst).Length != new FileInfo(seg.ImagePath).Length)
                File.Copy(seg.ImagePath, dst, overwrite: true);
        }

        // everything else out of VRAM — this is the biggest job we run
        await comfy.FreeVramAsync(ct);
        await Task.Delay(TimeSpan.FromSeconds(2), ct);
        await comfy.FreeVramAsync(ct);

        var wf = await BuildWorkflowAsync(segments, extraLoras, ct);
        log.LogInformation("[LTXDirector] Submitting {N} segments ({S:F0}s total) as {C} API nodes",
            segments.Count, segments.Sum(s => s.Seconds), wf.Count);
        var promptId = await comfy.SubmitAsync(wf, ct);
        var history = await comfy.WaitForCompletionAsync(promptId, _o.LtxDirector.TimeoutSec, ct);
        await comfy.CollectOutputAsync(history, outputPath, ct);
    }

    /// One director node per chunk; finished part files are skipped on resume.
    private async Task GenerateChunkedAsync(IReadOnlyList<LtxSegment> segments,
        string projectDir, string outPath,
        IReadOnlyList<(string File, double Weight)> extraLoras, CancellationToken ct)
    {
        var chunkSize = Math.Max(2, _o.LtxDirector.ChunkSize);
        var chunks = segments.Chunk(chunkSize).Select(c => c.ToList()).ToList();
        if (chunks.Count > 1 && chunks[^1].Count == 1)
        {
            // a lone trailing segment renders poorly — fold it into the
            // previous chunk (7 segments splits 4+3 across the two directors)
            chunks[^2].AddRange(chunks[^1]);
            chunks.RemoveAt(chunks.Count - 1);
        }

        var partsDir = Path.Combine(projectDir, "ltx_parts");
        Directory.CreateDirectory(partsDir);
        var partFiles = new List<string>();
        for (var i = 0; i < chunks.Count; i++)
        {
            var part = Path.Combine(partsDir, $"part_{i:00}.mp4");
            if (File.Exists(part) && new FileInfo(part).Length > 100_000)
            {
                log.LogInformation("[LTXDirector] director {I}/{N} already rendered — resuming past it",
                    i + 1, chunks.Count);
            }
            else
            {
                log.LogInformation("[LTXDirector] director {I}/{N}: {S} segments ({Sec:F0}s)",
                    i + 1, chunks.Count, chunks[i].Count, chunks[i].Sum(s => s.Seconds));
                // render to a temp name and swap in atomically: a power cut
                // mid-copy must never leave a corrupt part_XX.mp4 behind that
                // a later resume would silently trust
                var tmp = Path.Combine(partsDir, $"part_{i:00}.tmp.mp4");
                if (File.Exists(tmp)) File.Delete(tmp);
                await GenerateAsync(chunks[i], tmp, extraLoras, ct);
                File.Move(tmp, part, overwrite: true);
            }
            partFiles.Add(part);
        }

        var lst = Path.Combine(partsDir, "concat.txt");
        await File.WriteAllTextAsync(lst,
            string.Concat(partFiles.Select(p => $"file '{p.Replace('\\', '/')}'\n")), ct);
        var res = await media.FfmpegAsync(
            ["-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", outPath], ct);
        if (!res.Ok)
            throw new InvalidOperationException($"LTX Director concat failed: {res.StdErr}");
    }

    // ── segment building ───────────────────────────────────────────────

    /// segments → (timeline_json, local_prompts, lengths_csv, guides_csv, total_frames)
    internal static (string Timeline, string Prompts, string Lengths, string Guides, int Total)
        BuildSegments(IReadOnlyList<LtxSegment> segments, int fps)
    {
        var tl = new JsonArray();
        var prompts = new List<string>();
        var lengths = new List<string>();
        var start = 0.0;
        for (var i = 0; i < segments.Count; i++)
        {
            var seg = segments[i];
            var frames = Math.Max(24, (int)Math.Round(seg.Seconds * fps));
            var text = seg.Prompt.Trim();
            if (!string.IsNullOrWhiteSpace(seg.Dialogue)) text += "\n" + seg.Dialogue.Trim();
            var fname = Path.GetFileName(seg.ImagePath);
            tl.Add(new JsonObject
            {
                ["id"] = $"aidir{i}{Random.Shared.Next(1000, 10000)}",
                ["start"] = start,
                ["length"] = (double)frames,
                ["prompt"] = text,
                ["type"] = "image",
                ["imageFile"] = fname,
                ["imageB64"] = $"/api/view?filename={fname}&type=input&subfolder=",
            });
            prompts.Add(text);
            lengths.Add(frames.ToString());
            start += frames;
        }
        var timeline = new JsonObject { ["segments"] = tl, ["audioSegments"] = new JsonArray() }
            .ToJsonString();
        var guides = string.Join(",", segments.Select(_ => "1.00"));
        return (timeline, string.Join(" | ", prompts), string.Join(",", lengths), guides, (int)start);
    }

    // ── workflow assembly ──────────────────────────────────────────────

    private async Task<JsonObject> BuildWorkflowAsync(IReadOnlyList<LtxSegment> segments,
        IReadOnlyList<(string File, double Weight)> perProjectLoras, CancellationToken ct)
    {
        if (segments.Count > 12)
            throw new InvalidOperationException(
                $"LTX Director supports at most 12 scenes per workflow (6 per director node) " +
                $"— got {segments.Count}. Reduce the chunk size.");

        var templatePath = _o.LtxDirector.TemplatePath;
        if (!File.Exists(templatePath))
            throw new FileNotFoundException($"LTX Director template not found: {templatePath}");
        var graph = (JsonObject)JsonNode.Parse(await File.ReadAllTextAsync(templatePath, ct))!;

        // chain in the motion/detail LoRAs (+ any per-project ones)
        InjectExtraLoras(graph, [.. ExtraLoras, .. perProjectLoras]);

        var fps = _o.LtxDirector.Fps;
        var nodes = (JsonArray)graph["nodes"]!;
        var directors = nodes.OfType<JsonObject>()
            .Where(n => n["type"]!.GetValue<string>() == "LTXDirector")
            .OrderBy(n => n["id"]!.GetValue<long>())
            .ToList();

        List<List<LtxSegment>> parts;
        if (segments.Count <= 6)
        {
            // SINGLE-DIRECTOR mode: drop the second director + its sampler
            // subgraph, bypass the joiners so branch 1 passes straight through.
            parts = [segments.ToList()];
            var d2 = directors[1];
            var d2Links = new HashSet<long>();
            foreach (var o in (d2["outputs"] as JsonArray ?? []).OfType<JsonObject>())
                foreach (var l in (o["links"] as JsonArray ?? []))
                    if (l is not null) d2Links.Add(l.GetValue<long>());

            var keep = new List<JsonObject>();
            foreach (var n in nodes.OfType<JsonObject>().ToList())
            {
                if (ReferenceEquals(n, d2)) continue;
                var consumesD2 = (n["inputs"] as JsonArray ?? []).OfType<JsonObject>()
                    .Any(i => i["link"] is JsonNode l &&
                              l.GetValueKind() == System.Text.Json.JsonValueKind.Number &&
                              d2Links.Contains(l.GetValue<long>()));
                if (consumesD2) continue;      // director 2's sampler subgraph
                var t = n["type"]!.GetValue<string>();
                if (t is "ImageBatchMulti" or "AudioConcatenate")
                    n["mode"] = 4;             // bypass: image_1/audio1 pass through
                keep.Add(n);
            }
            nodes.Clear();
            foreach (var n in keep) nodes.Add(n);
            directors = directors.Take(1).ToList();
        }
        else
        {
            // split across the two chained director nodes
            var half = (segments.Count + 1) / 2;
            parts = [segments.Take(half).ToList(), segments.Skip(half).ToList()];
        }

        for (var d = 0; d < directors.Count && d < parts.Count; d++)
        {
            var (timeline, prompts, lengths, guides, total) = BuildSegments(parts[d], fps);
            var wv = (JsonArray)directors[d]["widgets_values"]!;
            wv[1] = total;                    // duration_frames
            wv[2] = (int)Math.Round(total / (double)fps);   // duration_seconds
            wv[3] = timeline;                 // timeline_data
            wv[4] = prompts;                  // local_prompts
            wv[5] = lengths;                  // segment_lengths
            // guide_strength MUST have exactly one entry per segment
            wv[7] = guides;
            wv[9] = fps;                      // frame_rate
            // width/height left at the template's native resolution on purpose
        }

        // unique output prefix
        foreach (var n in nodes.OfType<JsonObject>())
            if (n["type"]!.GetValue<string>() == "VHS_VideoCombine" &&
                n["widgets_values"] is JsonObject vwv)
                vwv["filename_prefix"] = $"aidir_ltxdir/{DateTimeOffset.UtcNow.ToUnixTimeSeconds()}";

        var objectInfo = await comfy.GetObjectInfoAsync(ct);
        return new LtxGraphConverter(graph, objectInfo).Convert();
    }

    /// Chain additional LoraLoaderModelOnly nodes into the top-level MODEL path
    /// (between the template's last LoRA and the 'Anything Everywhere' MODEL
    /// broadcast) so every sampler downstream sees them.
    private void InjectExtraLoras(JsonObject graph, IReadOnlyList<(string File, double Weight)> extra)
    {
        var lorasDir = Path.Combine(_o.Paths.ModelsDir, "loras");
        var present = extra.Where(e => File.Exists(Path.Combine(lorasDir, e.File))).ToList();
        if (present.Count == 0) return;

        var nodes = (JsonArray)graph["nodes"]!;
        var ae = nodes.OfType<JsonObject>().FirstOrDefault(n =>
            n["type"]!.GetValue<string>().StartsWith("Anything Everywhere") &&
            (n["inputs"] as JsonArray ?? []).OfType<JsonObject>()
                .Any(i => i["type"]?.GetValue<string>() == "MODEL" && i["link"] is not null));
        if (ae is null)
        {
            log.LogWarning("[LTXDirector] no MODEL broadcast node — extra LoRAs skipped");
            return;
        }
        var aeInput = (ae["inputs"] as JsonArray)!.OfType<JsonObject>()
            .First(i => i["type"]?.GetValue<string>() == "MODEL" && i["link"] is not null);
        var tailLink = aeInput["link"]!.GetValue<long>();
        var links = (JsonArray)graph["links"]!;
        var rec = links.OfType<JsonArray>().First(l => l[0]!.GetValue<long>() == tailLink);
        var originId = rec[1]!.GetValue<long>();
        var originSlot = rec[2]!.GetValue<int>();
        var originNode = nodes.OfType<JsonObject>().First(n => n["id"]!.GetValue<long>() == originId);

        var nid = graph["last_node_id"]?.GetValue<long>() ?? 0;
        var lid = graph["last_link_id"]?.GetValue<long>() ?? 0;
        var prevId = originId;
        var prevSlot = originSlot;
        JsonObject? lastNew = null;
        foreach (var (fname, weight) in present)
        {
            nid++;
            lid++;
            if (prevId == originId)
            {
                // rewire the origin's output away from the AE link
                var outp = ((JsonArray)originNode["outputs"]!)[originSlot]!.AsObject();
                var newLinks = new JsonArray();
                foreach (var l in (outp["links"] as JsonArray ?? []))
                    if (l is not null && l.GetValue<long>() != tailLink)
                        newLinks.Add(l.GetValue<long>());
                newLinks.Add(lid);
                outp["links"] = newLinks;
            }
            else
            {
                lastNew!["outputs"]![0]!["links"] = new JsonArray(lid);
            }
            links.Add(new JsonArray(lid, prevId, prevSlot, nid, 0, "MODEL"));
            lastNew = new JsonObject
            {
                ["id"] = nid,
                ["type"] = "LoraLoaderModelOnly",
                ["pos"] = new JsonArray(0, 0),
                ["size"] = new JsonArray(270, 82),
                ["flags"] = new JsonObject(),
                ["order"] = 0,
                ["mode"] = 0,
                ["inputs"] = new JsonArray(new JsonObject
                {
                    ["name"] = "model", ["type"] = "MODEL", ["link"] = lid,
                }),
                ["outputs"] = new JsonArray(new JsonObject
                {
                    ["name"] = "MODEL", ["type"] = "MODEL", ["links"] = new JsonArray(),
                }),
                ["properties"] = new JsonObject { ["Node name for S&R"] = "LoraLoaderModelOnly" },
                ["widgets_values"] = new JsonArray(fname, weight),
            };
            nodes.Add(lastNew);
            log.LogInformation("[LTXDirector] + LoRA {File} @{Weight}", fname, weight);
            prevId = nid;
            prevSlot = 0;
        }
        // last new node feeds the AE broadcast via the original link id
        lastNew!["outputs"]![0]!["links"] = new JsonArray(tailLink);
        rec[1] = prevId;
        rec[2] = 0;
        graph["last_node_id"] = nid;
        graph["last_link_id"] = lid;
    }
}
