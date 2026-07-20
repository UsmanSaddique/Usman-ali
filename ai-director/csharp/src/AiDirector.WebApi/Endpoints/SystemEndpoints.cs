using System.Diagnostics;
using AiDirector.Application.Abstractions;
using AiDirector.Infrastructure.Persistence;
using AiDirector.WebApi.Contracts;

namespace AiDirector.WebApi.Endpoints;

public static class SystemEndpoints
{
    public static void MapSystemEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapGet("/api/channels", async (AiDirectorDbContext db, CancellationToken ct) =>
        {
            var channels = await Microsoft.EntityFrameworkCore.EntityFrameworkQueryableExtensions
                .ToListAsync(db.Channels, ct);
            return Results.Ok(channels.Select(ChannelSummary.From));
        });

        app.MapGet("/api/system/health", () => Results.Ok(new { status = "ok" }));

        // Shape must match app/main.py::engine_status — the frontend reads
        // `.running` / `.starting`, not a status string.
        app.MapGet("/api/system/engine-status", async (IComfyUiClient comfy, CancellationToken ct) =>
        {
            var running = await comfy.PingAsync(ct);
            return Results.Ok(new
            {
                running,
                starting = false,
                engine = "ComfyUI API",
                url = "http://127.0.0.1:8188",
            });
        });

        // The "ComfyUI: Offline - click to start" tile posts here.
        app.MapPost("/api/system/engine-start", async (IComfyUiClient comfy, CancellationToken ct) =>
        {
            if (await comfy.PingAsync(ct))
                return Results.Ok(new { running = true, starting = false, message = "already running" });

            // Fire-and-forget: the cold start takes minutes, the tile polls engine-status.
            _ = Task.Run(() => comfy.WaitReadyAsync(240, autoLaunch: true, CancellationToken.None));
            return Results.Ok(new { running = false, starting = true, message = "starting" });
        });

        // Shape must match app/main.py::gpu_status — the frontend reads
        // `.gpu.device` / `.gpu.allocated_mb` / `.gpu.total_mb` / `.loaded_model.name`.
        app.MapGet("/api/system/gpu-status", async (CancellationToken ct) =>
        {
            var gpu = await QueryGpuAsync(ct) ?? new { available = false } as object;
            return Results.Ok(new
            {
                gpu,
                loaded_model = new { type = (string?)null, name = (string?)null, vram_mb = 0 },
            });
        });
    }

    private static async Task<object?> QueryGpuAsync(CancellationToken ct)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true,
            };
            foreach (var a in new[]
            {
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            }) psi.ArgumentList.Add(a);

            using var proc = Process.Start(psi)!;
            var outText = await proc.StandardOutput.ReadToEndAsync(ct);
            await proc.WaitForExitAsync(ct);
            var parts = outText.Trim().Split(',', StringSplitOptions.TrimEntries);
            if (parts.Length < 4) return null;
            var usedMb = int.Parse(parts[1]);
            var totalMb = int.Parse(parts[2]);
            return new
            {
                available = true,
                device = parts[0],
                total_mb = totalMb,
                allocated_mb = usedMb,
                reserved_mb = usedMb,
                free_mb = totalMb - usedMb,
                utilization_pct = int.Parse(parts[3]),
            };
        }
        catch { return null; }
    }
}
